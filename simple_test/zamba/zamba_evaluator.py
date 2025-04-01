import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import pytrec_eval

class ZambaEvaluator:
    def __init__(self, 
                 model_name: str = "Zyphra/Zamba-7B-v1", 
                 score_type: str = "binary", 
                 device: str = "cuda", 
                 max_new_tokens: int = 5):
        """
        Initialize the evaluator with the Zamba model.

        Parameters:
            model_name (str): Name or path of the Zamba model.
            score_type (str): 'binary' for yes/no (score 1/0) 
                              or 'ordered' for graded relevance ('high':1, 'mid':0.5, 'low':0).
            device (str): The device to run the model on (e.g., "cuda" or "cpu").
            max_new_tokens (int): Maximum number of tokens to generate for an answer.
        """
        self.score_type = score_type.lower()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model_name, device_map="auto", torch_dtype=torch.bfloat16
        # )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.float16
        )
        self.device = device
        self.model.to(self.device)
        self.max_new_tokens = max_new_tokens
        logging.info(f"Initialized ZambaEvaluator with model {model_name} in {score_type} mode.")

    def get_prompt(self, query: str, document: str) -> str:
        """
        Compose a prompt for Zamba based on the desired scoring scheme.
        Place the query after the document in the prompt.
        """
        if self.score_type == "binary":
            prompt = (
                f"Document: '''{document}'''\n\n"
                f"Query: '''{query}'''\n\n"
                "Is the document relevant enough to answer the query?\n\n"
                "Directly answer with exactly one word: 'yes' or 'no'.\n"
                "Do not add any additional text or commentary.\n"
                "Answer:"                
            )
        elif self.score_type == "ordered":
            prompt = (
                f"Document: '''{document}'''\n"
                f"Query: '''{query}'''\n"
                "Rate the relevance of the document to answer the query as 'high', 'mid', or 'low'.\n\n"
                "Directly answer with exactly one word: 'high', 'mid', or 'low'.\n" 
                "Do not add any additional text or commentary.\n"
                "Answer:"
            )
        else:
            raise ValueError("Unknown score_type. Please use 'binary' or 'ordered'.")
        return prompt

    def parse_response(self, response: str) -> float:
        """
        Parse Zamba's response to extract a numerical score.
        """
        response_lower = response.lower()
        if self.score_type == "binary":
            return 1.0 if "yes" in response_lower else 0.0
        elif self.score_type == "ordered":
            if "high" in response_lower:
                return 1.0
            elif "mid" in response_lower:
                return 0.5
            elif "low" in response_lower:
                return 0.0
            else:
                # If response is unclear, default to 0.
                return 0.0

    def ask_zamba(self, query: str, document: str) -> tuple[float, str, float]:
        """
        Ask Zamba the prompt and return:
            - a binary/graded relevance score (based on the parsed response),
            - the raw answer string,
            - and the logit for the token 'yes' (as a continuous confidence signal).
        """
        prompt = self.get_prompt(query, document)
        logging.info(f"Full Prompt:\n{prompt}")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        # Use generate with output_scores to extract token logits
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=self.max_new_tokens,
            do_sample=False,  # Greedy decoding
            num_beams=1,
            temperature=0.0,  # Eliminates randomness (if do_sample=True)
            output_scores=True,
            return_dict_in_generate=True
        )
        # Extract the logit for the first generated token (i.e., the next token after the prompt)
        first_token_logits = outputs.scores[0][0]  # shape: (vocab_size,)
        yes_token_id = self.tokenizer.encode("yes", add_special_tokens=False)[0]
        yes_logit = first_token_logits[yes_token_id].item()
        
        full_response = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        # Assume Zamba's answer follows immediately after the prompt.
        answer_raw = full_response[len(prompt):].strip()
        logging.info(f"Full Answer:\n{answer_raw}")

        if self.score_type == "binary":
            if 'yes' in answer_raw.lower():
                answer = 'yes'
            elif 'no' in answer_raw.lower():
                answer = 'no'
            else:
                answer = 'unknown'
        elif self.score_type == "ordered":
            if 'high' in answer_raw.lower():
                answer = 'high'
            elif 'mid' in answer_raw.lower():
                answer = 'mid'
            elif 'low' in answer_raw.lower():
                answer = 'low'
            else:
                answer = 'unknown'

        score = self.parse_response(answer)

        return score, answer, yes_logit

    def evaluate(self, 
                 corpus: dict[str, dict[str, str]], 
                 queries: dict[str, str],
                ) -> dict[str, dict[str, float]]:
        """
        Evaluate all query-document pairs from the BEIR dataset using Zamba.

        Parameters:
            corpus (dict): Dictionary mapping document IDs to document details (with keys 'title' and 'text').
            queries (dict): Dictionary mapping query IDs to query text.

        Returns:
            dict: A nested dictionary with scores for each document per query.
                  Format: {query_id: {doc_id: score, ...}, ...}
        """
        results = {}

        for qid, query in tqdm(queries.items(), desc="Evaluating queries"):
            results[qid] = {}
            for doc_id, doc in corpus.items():
                # Combine title and text for a more complete document context.
                document_text = f"{doc.get('title', '')}\n{doc.get('text', '')}"
                score, answer, yes_logit = self.ask_zamba(query, document_text)
                # results[qid][doc_id] = score
                results[qid][doc_id] = yes_logit
                logging.info(f"Query ID: {qid} | Document ID: {doc_id} | Score: {score} | Answer:   {answer} | Yes Logit: {yes_logit}")

        return results
    
    def evaluate_metrics(self, 
                         qrels: dict[str, dict[str, int]], 
                         results: dict[str, dict[str, float]], 
                         k_values: list[int] = [1, 3, 5, 10, 100, 1000]
                        ) -> dict[str, float]:
        """
        Calculate evaluation metrics (NDCG@k, MAP@k, Recall@k, Precision@k) using pytrec_eval.

        Parameters:
            qrels (dict): Ground truth relevance judgments.
            results (dict): Scores from ZambaEvaluator.
            k_values (list): List of cutoff values.

        Returns:
            A dictionary with the aggregated metric scores.
        """
        # Build metric keys for pytrec_eval.
        map_key = "map_cut." + ",".join([str(k) for k in k_values])
        ndcg_key = "ndcg_cut." + ",".join([str(k) for k in k_values])
        recall_key = "recall." + ",".join([str(k) for k in k_values])
        precision_key = "P." + ",".join([str(k) for k in k_values])
        
        evaluator = pytrec_eval.RelevanceEvaluator(
            qrels, {map_key, ndcg_key, recall_key, precision_key}
        )
        scores = evaluator.evaluate(results)

        ndcg, _map, recall, precision = {}, {}, {}, {}
        # Initialize metrics for each k value.
        for k in k_values:
            ndcg[f"NDCG@{k}"] = 0.0
            _map[f"MAP@{k}"] = 0.0
            recall[f"Recall@{k}"] = 0.0
            precision[f"P@{k}"] = 0.0

        for query_id in scores.keys():
            for k in k_values:
                ndcg[f"NDCG@{k}"] += scores[query_id][f"ndcg_cut_{k}"]
                _map[f"MAP@{k}"] += scores[query_id][f"map_cut_{k}"]
                recall[f"Recall@{k}"] += scores[query_id][f"recall_{k}"]
                precision[f"P@{k}"] += scores[query_id][f"P_{k}"]

        num_queries = len(scores)
        for k in k_values:
            ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / num_queries, 5)
            _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / num_queries, 5)
            recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / num_queries, 5)
            precision[f"P@{k}"] = round(precision[f"P@{k}"] / num_queries, 5)

        metrics = {}
        metrics.update(ndcg)
        metrics.update(_map)
        metrics.update(recall)
        metrics.update(precision)
        return metrics
