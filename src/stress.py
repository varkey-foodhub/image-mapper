import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import os
import psutil
import time
import random
import platform
import torch

# --- Configuration ---
BENCHMARK_DURATION_SEC = 10
BATCH_SIZE = 512  # <--- CRITICAL: Increase this to 256 or 512 if GPU is still bored
TOP_K = 5

# --- System Info ---
def get_system_info():
    uname = platform.uname()
    device = "CPU"
    gpu_name = "N/A"
    if torch.cuda.is_available():
        device = "GPU (CUDA)"
        gpu_name = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        device = "GPU (MPS/Mac)"
    return f"{device} | GPU: {gpu_name}"

print("\n" + "="*60)
print("⚙️  INITIALIZING HIGH-THROUGHPUT BATCH BENCHMARK")
print("="*60)

if not os.path.exists("menu_index.faiss") or not os.path.exists("metadata.pkl"):
    print("❌ Error: Missing index files.")
    exit()

# Load Resources
try:
    print("   -> Loading Model...")
    # device='cuda' ensures model loads directly on GPU
    model = SentenceTransformer("all-MiniLM-L6-v2", device='cuda' if torch.cuda.is_available() else 'cpu')
    
    print("   -> Loading Index...")
    index = faiss.read_index("menu_index.faiss")
    
    # Move FAISS to GPU if available (Optional, usually CPU FAISS is fast enough for small indices)
    # res = faiss.StandardGpuResources()
    # index = faiss.index_cpu_to_gpu(res, 0, index) 

    print("   -> Loading Data...")
    df = pd.read_pickle("metadata.pkl")
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

# --- Synthetic Data Generation ---
# We need a LOT of queries to feed the batch processor
base_queries = [
    "coca cola can", "pepsi bottle", "water 1l", "cheese burger", 
    "chicken biryani", "french fries", "chocolate brownie", 
    "diet coke", "spicy paneer", "cold coffee"
]
# Create a massive list of 10,000 random queries
large_query_pool = [random.choice(base_queries) + f" {i}" for i in range(10000)]

# --- Batch Processing Function ---
def process_batch(batch_queries):
    # 1. Encode Batch (GPU Heavy Lifting)
    # This sends 128 strings to the GPU at once
    embeddings = model.encode(batch_queries, normalize_embeddings=True, batch_size=BATCH_SIZE, convert_to_numpy=True)
    
    # 2. Search Batch (FAISS is optimized for matrix searches)
    embeddings = embeddings.astype("float32")
    D, I = index.search(embeddings, TOP_K)
    
    return len(batch_queries)

# --- Main Loop ---
if __name__ == "__main__":
    process = psutil.Process(os.getpid())
    
    print(f"\n🔥 WARMING UP (Batch Size: {BATCH_SIZE})...")
    process_batch(large_query_pool[:BATCH_SIZE])
    
    print(f"\n🚀 STARTING GPU STRESS TEST ({BENCHMARK_DURATION_SEC}s)...")
    
    start_time = time.perf_counter()
    end_time = start_time + BENCHMARK_DURATION_SEC
    
    total_queries = 0
    batches_processed = 0
    gpu_utilization = []

    while time.perf_counter() < end_time:
        # Grab a random batch
        batch = random.sample(large_query_pool, BATCH_SIZE)
        
        process_batch(batch)
        
        total_queries += BATCH_SIZE
        batches_processed += 1
        
        # Log GPU Usage (Requires torch or nvidia-smi logic)
        if torch.cuda.is_available():
            # Note: getting accurate instantaneous GPU util in python is tricky without pynvml
            # This is a proxy using memory, actual util usually monitored externally
            pass 

    total_time = time.perf_counter() - start_time
    rps = total_queries / total_time
    
    print("\n\n")
    print("████████████████████████████████████████████████████████")
    print("       ⚡ BATCH PERFORMANCE SCORECARD ⚡")
    print("████████████████████████████████████████████████████████")
    print(f"\n🖥️  SYSTEM: {get_system_info()}")
    print(f"📦 BATCH SIZE: {BATCH_SIZE}")
    
    print(f"\n🏆 THROUGHPUT (RPS)")
    print(f"   ▶ {rps:,.2f} Requests Per Second")
    if rps > 1000:
        print("   (🚀 GPU is fast!)")
    
    print(f"\n📊 TOTALS")
    print(f"   • Queries: {total_queries:,}")
    print(f"   • Batches: {batches_processed:,}")
    print(f"   • Time:    {total_time:.2f}s")
    
    