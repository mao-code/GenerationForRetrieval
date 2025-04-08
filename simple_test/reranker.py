import time
import torch
from transformers import LlamaTokenizer
from MLA_CDR.configuration_mla import MLAConfig
from MLA_CDR.modeling_mla import MLAForSequenceScoring
from MLA_CDR.tokenizer_utils import get_tokenizer as get_tokenizer_mla
from GFR.configuration_GFR import GFRConfig
from GFR.modeling_GFR import GFRForSequenceScoring
from GFR.tokenizer_utils import get_tokenizer as get_tokenizer_gfr
from CDR.modeling_cdr import ScoringWrapper
from CDR.tokenizer_utils import get_tokenizer as get_tokenizer_cdr
from Mamba2_CDR.tokenizer_utils import get_tokenizer as get_tokenizer_mamba2
from Mamba2_CDR.configuration_mamba2 import Mamba2Config
from Mamba2_CDR.modeling_mamba2 import Mamba2ForSequenceScoring
from cache.cache import (
    get_documents_cache,
    move_cache_to_cpu,
    move_cache_to_gpu,
    score_with_cache,
)
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sentence_transformers import CrossEncoder

from pympler import asizeof
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("reranker_simpletest.log", mode="w")
    ],
    force=True
)
logger = logging.getLogger()

def compute_cache_size(cache):
    """
    Compute the total memory footlogger.info (in bytes) of the key/value cache.
    Supports both DynamicCache (with key_cache/value_cache attributes)
    and legacy tuple format.
    """
    # total_bytes = 0
    # if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
    #     for tensor in cache.key_cache:
    #         total_bytes += tensor.numel() * tensor.element_size()
    #     for tensor in cache.value_cache:
    #         total_bytes += tensor.numel() * tensor.element_size()
    # elif isinstance(cache, tuple):
    #     for layer in cache:
    #         key, value = layer
    #         total_bytes += key.numel() * key.element_size()
    #         total_bytes += value.numel() * value.element_size()
    # else:
    #     logger.info("Warning: Unrecognized cache format.")
    # return total_bytes

    cache_size = asizeof.asizeof(cache)
    return cache_size

def measure_ttft_no_cache(model, full_input, device):
    """
    Measure time-to-first-token (TTFT) for processing the full input (document + query)
    without caching.
    """
    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        _ = model(full_input, use_cache=False)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start_time
    return elapsed

def test_noncache_batch_scoring(model, tokenizer, device, batch_size=8, cache_size=True):
    # Define sample texts.
    doc_text = "word " * 512    # ~512 tokens per document
    query_text = "query " * 15   # ~15 tokens per query

    # --- Non-cached inference (full input) ---
    # We don't care input and performance here, so we don't use prepare_input function.
    doc_tokens = tokenizer(doc_text, return_tensors= "pt")["input_ids"].to(device)
    query_tokens = tokenizer(query_text, return_tensors="pt")["input_ids"].to(device)
    if batch_size > 1:
        doc_tokens = doc_tokens.repeat(batch_size, 1)
        query_tokens = query_tokens.repeat(batch_size, 1)

    full_input = torch.cat((doc_tokens, query_tokens), dim=1)

    # Get cache to get size
    documents = {f"doc_{i}": doc_text for i in range(batch_size)}
    
    # Warm-up cache generation.
    if cache_size:
        with torch.no_grad():
            _ = get_documents_cache(model, documents, tokenizer, device, batch_size=batch_size)
        
        # Measure cache generation time.
        start = time.time()
        cache = get_documents_cache(model, documents, tokenizer, device, batch_size=batch_size)
        if device.type == "cuda":
            torch.cuda.synchronize()
        cache_gen_time = time.time() - start

        # Compute cache size.
        cache_size_bytes = compute_cache_size(cache)
        cache_size_mb = cache_size_bytes / (1024 * 1024)

    # Warm-up.
    with torch.no_grad():
        _ = model(input_ids=full_input, use_cache=False)

    # Timed run.
    noncache_time = measure_ttft_no_cache(model, full_input, device)

    logger.info(f"Non-cached inference time (full input):   {noncache_time * 1000} ms")

    if cache_size:
        logger.info(f"Cache size:                               {cache_size_mb:.2f} MB\n")
        logger.info(f"Cache generation time:                    {cache_gen_time*1000:.2f} ms")


def test_cache_vs_noncache_batch_scoring(model, tokenizer, device, batch_size=8):
    """
    Compare non-cached (full concatenated input) vs. cached scoring on a batch of documents.
    Also, measure the time spent for cache generation and movement between GPU and CPU,
    and report the cache size.
    """
    # Define sample texts.
    doc_text = "word " * 512    # ~512 tokens per document
    query_text = "query " * 15   # ~15 tokens per query

    # --- Non-cached inference (full input) ---
    # Tokenize document and query.
    doc_tokens = tokenizer.encode(doc_text, return_tensors="pt").to(device)
    query_tokens = tokenizer.encode(query_text, return_tensors="pt").to(device)
    if batch_size > 1:
        doc_tokens = doc_tokens.repeat(batch_size, 1)
        query_tokens = query_tokens.repeat(batch_size, 1)
    full_input = torch.cat((doc_tokens, query_tokens), dim=1)
    # Warm-up.
    with torch.no_grad():
        _ = model(full_input, use_cache=True)
    # Timed run.
    noncache_time = measure_ttft_no_cache(model, full_input, device)

    # --- Cached inference ---
    # Build a dict of documents for batch scoring.
    documents = {f"doc_{i}": doc_text for i in range(batch_size)}
    
    # Warm-up cache generation.
    with torch.no_grad():
        _ = get_documents_cache(model, documents, tokenizer, device, batch_size=batch_size)
    
    # Measure cache generation time.
    start = time.time()
    cache = get_documents_cache(model, documents, tokenizer, device, batch_size=batch_size)
    if device.type == "cuda":
        torch.cuda.synchronize()
    cache_gen_time = time.time() - start

    # Measure movement: GPU → CPU.
    start = time.time()
    cache_cpu = move_cache_to_cpu(cache)
    if device.type == "cuda":
        torch.cuda.synchronize()
    cpu_move_time = time.time() - start

    # Measure movement: CPU → GPU.
    start = time.time()
    cache_gpu = move_cache_to_gpu(cache_cpu, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    gpu_move_time = time.time() - start

    # Warm-up cached scoring.
    with torch.no_grad():
        _ = score_with_cache(model, cache_gpu, query_text, tokenizer, device, batch_size=batch_size)
    
    # Measure cached scoring time (query inference using precomputed cache).
    start = time.time()
    scores, cache_forward_time = score_with_cache(model, cache_gpu, query_text, tokenizer, device, batch_size=batch_size)
    if device.type == "cuda":
        torch.cuda.synchronize()
    total_cached_time = time.time() - start

    # Compute overhead in the score_with_cache function (if any).
    movement_overhead_in_score = total_cached_time - cache_forward_time

    # Compute cache size.
    cache_size_bytes = compute_cache_size(cache)
    cache_size_mb = cache_size_bytes / (1024 * 1024)

    # --- Report results ---
    logger.info("=== Batch Scoring Speed Comparison ===")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Non-cached inference time (full input): {noncache_time*1000:.2f} ms")
    logger.info(f"Cache generation time:               {cache_gen_time*1000:.2f} ms")
    logger.info(f"Cache move to CPU time:              {cpu_move_time*1000:.2f} ms")
    logger.info(f"Cache move to GPU time:              {gpu_move_time*1000:.2f} ms")
    logger.info(f"Cached query inference (forward):    {cache_forward_time*1000:.2f} ms")
    logger.info(f"Total cached scoring time:           {total_cached_time*1000:.2f} ms")
    logger.info(f"Movement overhead in score_with_cache:{movement_overhead_in_score*1000:.2f} ms")
    logger.info(f"Cache size:                          {cache_size_mb:.2f} MB\n")

def test_cross_encoder_batch_scoring(model, batch_size=8):
    # Define sample texts.
    doc_text = "word " * 512    # ~512 tokens per document
    query_text = "query " * 15   # ~15 tokens per query

    pairs = []
    pairs = [(query_text, doc_text) for _ in range(batch_size)]

    # Warm-up.
    with torch.no_grad():
        scores = model.predict(pairs)

    # Timed run.
    with torch.no_grad():
        start_time = time.time()
        scores = model.predict(pairs)
        elapsed = time.time() - start_time

    logger.info("Non-cached inference time (full input):", elapsed * 1000, "ms")

def main():
    # Device selection: GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model.
    tokenizer_mla = get_tokenizer_mla()
    logger.info("Initializing MLA model...")
    config_mla = MLAConfig(vocab_size=len(tokenizer_mla), num_hidden_layers=16)
    mla = MLAForSequenceScoring(config_mla)
    mla.resize_token_embeddings(len(tokenizer_mla))
    mla.to(device)
    num_params = sum(p.numel() for p in mla.parameters())
    logger.info(f"Number of parameters: {num_params}")
    mla.eval()  # Set model to evaluation mode.

    logger.info("Initializing GFR1 model...")
    tokenizer_gfr = get_tokenizer_gfr()
    config = GFRConfig(
        vocab_size=len(tokenizer_gfr), 
        hidden_size=1024,
        intermediate_size=1024 * 4,
        num_attention_heads=16,
        num_key_value_heads=16,
        n_mamba_heads=2,
        num_hidden_blocks=3, 
        num_layers_per_block=8,
        max_position_embeddings=1024, # no real effect
        pad_token_id=tokenizer_gfr.pad_token_id,
        bos_token_id=tokenizer_gfr.bos_token_id,
        eos_token_id=tokenizer_gfr.eos_token_id,
    )
    gfr = GFRForSequenceScoring(config)
    gfr.resize_token_embeddings(len(tokenizer_gfr))
    gfr.to(device)
    num_params = sum(p.numel() for p in gfr.parameters())
    logger.info(f"Number of parameters: {num_params}")
    gfr.eval()  # Set model to evaluation mode.

    logger.info("Initializing CDR-Pythia...")
    pythia_model_name = "EleutherAI/pythia-410m"
    tokenizer_cdr_pythia = get_tokenizer_cdr(pythia_model_name)
    config = AutoConfig.from_pretrained(pythia_model_name)
    decoder = AutoModel.from_pretrained(pythia_model_name)
    cdr_pythia = ScoringWrapper(config, decoder)
    num_params = sum(p.numel() for p in cdr_pythia.parameters())
    logger.info(f"Number of parameters: {num_params}")
    cdr_pythia.resize_token_embeddings(len(tokenizer_cdr_pythia))
    cdr_pythia.to(device) 
    cdr_pythia.eval()

    logger.info("Initializing CDR-Mamba2...")
    tokenizer_cdr_mamba2 = get_tokenizer_mamba2()
    mamba2_config = Mamba2Config(
        num_heads=16,
        head_dim=64,
        vocab_size=len(tokenizer_cdr_mamba2),
        hidden_size=1024,
        state_size=128,
        num_hidden_layers=48,
    )
    mamba2 = Mamba2ForSequenceScoring(mamba2_config)
    mamba2.resize_token_embeddings(len(tokenizer_cdr_mamba2))
    mamba2.to(device)
    num_params = sum(p.numel() for p in mamba2.parameters())
    logger.info(f"Number of parameters: {num_params}")
    mamba2.eval()  # Set model to evaluation mode.

    logger.info("Initializing Cross-encoder model (msmarco-minilm)...")
    msmarco_minilm = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-12-v2", 
        device=device, 
        automodel_args={
            "torch_dtype": "auto",
            "attn_implementation": "eager"  # Explicitly disable Flash Attention
        }, 
        trust_remote_code=True
    )
    num_params = sum(p.numel() for p in msmarco_minilm.model.parameters())
    logger.info(f"Number of parameters: {num_params}")

    logger.info("Initializing Cross-encoder model (BGE)...")
    bge = CrossEncoder(
        "BAAI/bge-reranker-v2-m3", 
        device=device, 
        automodel_args={
            "torch_dtype": "auto",
            "attn_implementation": "eager"  # Explicitly disable Flash Attention
        }, 
        trust_remote_code=True
    )
    num_params = sum(p.numel() for p in bge.model.parameters())
    logger.info(f"Number of parameters: {num_params}")    

    # Test for various batch sizes.
    for batch_size in [1, 8, 16]:
        logger.info(f"Testing MLA model... {"-"* 20}")
        test_cache_vs_noncache_batch_scoring(mla, tokenizer_mla, device, batch_size=batch_size)

        logger.info(f"Testing GFR model... {"-"* 20}") 
        test_noncache_batch_scoring(gfr, tokenizer_gfr, device, batch_size=batch_size, cache_size=False) # No Mamba Cache available, now

        logger.info(f"Testing CDR-pythia model... {"-"* 20}")
        test_cache_vs_noncache_batch_scoring(cdr_pythia, tokenizer_cdr_pythia, device, batch_size=batch_size)

        logger.info(f"Testing CDR-mamba2 model... {"-"* 20}")
        test_noncache_batch_scoring(mamba2, tokenizer_cdr_mamba2, device, batch_size=batch_size) # No Mamba Cache available, now

        logger.info(f"Testing Cross-encoder model (msmarco-minilm)... {"-"* 20}")
        test_cross_encoder_batch_scoring(msmarco_minilm, batch_size=batch_size)

        logger.info(f"Testing Cross-encoder model (bge))... {"-"* 20}")
        test_cross_encoder_batch_scoring(bge, batch_size=batch_size)

if __name__ == "__main__":
    main()
