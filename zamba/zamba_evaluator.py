import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

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
                "Answer with exactly one word: 'yes' or 'no'.\n"
                "Do not add any additional text or commentary.\n"
                "Answer:"                
            )
        elif self.score_type == "ordered":
            prompt = (
                f"Document: '''{document}'''\n"
                f"Query: '''{query}'''\n"
                "Rate the relevance of the document to answer the query as 'high', 'mid', or 'low'.\n\n"
                "Answer with exactly one word: 'high', 'mid', or 'low'.\n" 
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

    def ask_zamba(self, query: str, document: str) -> tuple[float, str]:
        """
        Ask Zamba the prompt and return the relevance score along with the raw answer.
        """
        prompt = self.get_prompt(query, document)
        logging.info(f"Full Prompt:\n{prompt}")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=self.max_new_tokens,
            do_sample=False, # Greedy decoding
            num_beams=1,     # If Beam search
            temperature=0.0    # Eliminates randomness (if do_sample=True)
        )
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

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

        return score, answer

    def evaluate(self, 
                 corpus: dict[str, dict[str, str]], 
                 queries: dict[str, str],
                 num_test_samples: int = 100
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
        if num_test_samples > 0: # -1 for all samples
            queries = dict(list(queries.items())[:num_test_samples])

        for qid, query in tqdm(queries.items(), desc="Evaluating queries"):
            results[qid] = {}
            for doc_id, doc in corpus.items():
                # Combine title and text for a more complete document context.
                document_text = f"{doc.get('title', '')}\n{doc.get('text', '')}"
                score, answer = self.ask_zamba(query, document_text)
                results[qid][doc_id] = score
                logging.info(f"Query ID: {qid} | Document ID: {doc_id} | Score: {score} | Answer: {answer}")

        return results
