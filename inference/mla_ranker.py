import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def measure_ttft_no_cache(model, full_input, device):
    """Measure TTFT (without caching) for the full input (document + query)."""
    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        _ = model(full_input, use_cache=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start_time
    return elapsed

def measure_ttft_cache(model, doc_tokens, query_tokens, device):
    """Measure TTFT (with caching) by processing the document then the query."""
    with torch.no_grad():
        # Process document to get cached key/value states.
        doc_outputs = model(doc_tokens, use_cache=True)
        cache = doc_outputs.past_key_values
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        _ = model(query_tokens, past_key_values=cache, use_cache=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start_time
    return elapsed

def run_experiment(model, tokenizer, doc_text, query_text, batch_size, device):
    """
    Create batched inputs, perform warm-up runs, and measure timings with and without caching.
    Returns TTFT (in seconds) for full-input and cached inference.
    """
    # Tokenize a single sample.
    doc_tokens_single = tokenizer.encode(doc_text, return_tensors="pt").to(device)
    query_tokens_single = tokenizer.encode(query_text, return_tensors="pt").to(device)
    
    # For batch inference, replicate the single sample along the batch dimension (dimension 0).
    if batch_size > 1:
        doc_tokens = doc_tokens_single.repeat(batch_size, 1)
        query_tokens = query_tokens_single.repeat(batch_size, 1)
    else:
        doc_tokens = doc_tokens_single
        query_tokens = query_tokens_single

    # Warm-up runs to avoid initial overhead.
    with torch.no_grad():
        full_input = torch.cat((doc_tokens, query_tokens), dim=1)
        _ = model(full_input, use_cache=True)
        _ = model(doc_tokens, use_cache=True)
        doc_out = model(doc_tokens, use_cache=True)
        _ = model(query_tokens, past_key_values=doc_out.past_key_values, use_cache=True)

    # Measure TTFT without caching (processing full input).
    full_input = torch.cat((doc_tokens, query_tokens), dim=1)
    time_no_cache = measure_ttft_no_cache(model, full_input, device)

    # Measure TTFT with caching (processing document then query).
    time_cache = measure_ttft_cache(model, doc_tokens, query_tokens, device)
    
    return time_no_cache, time_cache

def main():
    # Device selection: GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model and tokenizer name (using GPT-2 as an example).
    model_name = "gpt2"
    
    # Define dummy inputs:
    # - Document: ~512 tokens (by repeating "word ")
    # - Query: ~15 tokens (by repeating "query ")
    doc_text = "word " * 512
    query_text = "query " * 15
    
    # Conditions to test.
    batch_sizes = [1, 8]                # Single sample and batch inference.
    flash_attention_options = [False, True]  # Flash Attention2 disabled vs enabled.
    
    # Print header for results.
    print("Comparison of TTFT (in ms) and time improvement rate:")
    print("{:<20}{:<12}{:<20}{:<20}{:<20}".format(
        "FlashAttention2", "BatchSize", "NoCache (ms)", "WithCache (ms)", "Improvement (%)"
    ))
    print("-" * 92)
    
    # Loop over flash attention and batch size conditions.
    for flash_attention in flash_attention_options:
        # Load the model and tokenizer according to the flash attention setting.
        if flash_attention:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                use_cache=True,
                attn_implementation="flash_attention_2",
            ).to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                use_cache=True,
            ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.eval()  # Set evaluation mode
        
        for batch_size in batch_sizes:
            # Run experiment for the current configuration.
            time_no_cache, time_cache = run_experiment(
                model, tokenizer, doc_text, query_text, batch_size, device
            )
            # Calculate improvement rate (percentage reduction using cache).
            improvement_rate = (time_no_cache - time_cache) / time_no_cache * 100
            
            # Convert seconds to milliseconds.
            time_no_cache_ms = time_no_cache * 1000
            time_cache_ms = time_cache * 1000
            
            # Print results.
            print("{:<20}{:<12}{:<20.3f}{:<20.3f}{:<20.2f}".format(
                "Enabled" if flash_attention else "Disabled",
                batch_size,
                time_no_cache_ms,
                time_cache_ms,
                improvement_rate
            ))

if __name__ == "__main__":
    main()