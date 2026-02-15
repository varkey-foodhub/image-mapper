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
BENCHMARK_DURATION_SEC = 10  # Run test for exactly 10 seconds
WARMUP_QUERIES = 20          # Run these first to load caches (don't measure them)
TOP_K = 5

# --- System Info Helper ---
def get_system_info():
    uname = platform.uname()
    cpus = psutil.cpu_count(logical=False)
    total_ram = round(psutil.virtual_memory().total / (1024.0 ** 3), 1)
    
    device = "CPU"
    gpu_name = "N/A"
    if torch.cuda.is_available():
        device = "GPU (CUDA)"
        gpu_name = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        device = "GPU (MPS/Mac)"

    return f"{uname.system} {uname.release} | {device} | {cpus} Phys Cores | {total_ram} GB RAM | GPU: {gpu_name}"

# --- Load Resources ---
print("\n" + "="*60)
print("⚙️  INITIALIZING BENCHMARK ENVIRONMENT...")
print("="*60)

if not os.path.exists("menu_index.faiss") or not os.path.exists("metadata.pkl"):
    print("❌ CRITICAL ERROR: Index files missing.")
    print("   Please run your indexing script first to generate 'menu_index.faiss' and 'metadata.pkl'")
    exit()

# Load Assets
try:
    print("   -> Loading Embedding Model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    print("   -> Loading Vector Index (FAISS)...")
    index = faiss.read_index("menu_index.faiss")
    
    print("   -> Loading Metadata (Pandas)...")
    df = pd.read_pickle("metadata.pkl")
    print("✅ Resources Loaded Successfully.")
except Exception as e:
    print(f"❌ Error loading resources: {e}")
    exit()

# --- Core Logic ---
def search_func(query):
    # 1. Encode
    query_vec = model.encode([query], normalize_embeddings=True)
    query_vec = np.array(query_vec).astype("float32")
    
    # 2. Vector Search
    D, I = index.search(query_vec, TOP_K)
    
    # 3. Data Retrieval
    results = []
    for idx in I[0]:
        if idx != -1:
            _ = df.iloc[idx] # Simulate fetch
    return len(results)

# --- Synthetic Data ---
test_queries = [
    "coca cola can 330ml", "pepsi bottle", "water 1l", "cheese burger", 
    "chicken biryani", "french fries large", "chocolate brownie", 
    "diet coke", "spicy paneer burger", "cold coffee", "masala dosai", 
    "garlic naan", "butter chicken", "vanilla ice cream", "fanta orange"
]

# --- Main Benchmark Loop ---
if __name__ == "__main__":
    process = psutil.Process(os.getpid())
    
    # 1. Warmup Phase
    print("\n🔥 WARMING UP ENGINE...")
    for _ in range(WARMUP_QUERIES):
        search_func(random.choice(test_queries))
    print("   -> Warmup complete. Caches are hot.")

    # 2. Stress Test
    print(f"\n🚀 STARTING STRESS TEST ({BENCHMARK_DURATION_SEC} seconds)...")
    print("   -> Hammering the system with queries...")
    
    start_time = time.perf_counter()
    end_time = start_time + BENCHMARK_DURATION_SEC
    
    query_count = 0
    latencies = []
    cpu_readings = []
    ram_readings = []

    while time.perf_counter() < end_time:
        q_start = time.perf_counter()
        
        # The unit of work
        search_func(random.choice(test_queries))
        
        q_end = time.perf_counter()
        
        # Metrics
        query_count += 1
        latencies.append((q_end - q_start) * 1000) # Convert to ms
        
        # Log resources every 50 queries to avoid overhead slowing down the test
        if query_count % 50 == 0:
            cpu_readings.append(process.cpu_percent(interval=None))
            ram_readings.append(process.memory_info().rss / (1024 * 1024))

    total_time = time.perf_counter() - start_time
    rps = query_count / total_time
    
    # 3. Statistics
    avg_latency = sum(latencies) / len(latencies)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    avg_cpu = sum(cpu_readings) / len(cpu_readings) if cpu_readings else 0
    avg_ram = sum(ram_readings) / len(ram_readings) if ram_readings else 0

    # --- THE FINAL REPORT ---
    print("\n\n")
    print("████████████████████████████████████████████████████████")
    print("       ⚡ SYSTEM PERFORMANCE SCORECARD ⚡")
    print("████████████████████████████████████████████████████████")
    print(f"\n🖥️  HARDWARE: {get_system_info()}")
    
    print(f"\n🏆 THROUGHPUT (The Golden Metric)")
    print(f"   ▶ {rps:,.2f} Requests Per Second (RPS)")
    
    print(f"\n⏱️  LATENCY (Speed per Request)")
    print(f"   • Average:      {avg_latency:.2f} ms")
    print(f"   • Fastest:      {min(latencies):.2f} ms")
    print(f"   • Slowest:      {max(latencies):.2f} ms")
    print(f"   • 95th %ile:    {p95_latency:.2f} ms (Most users see this)")
    print(f"   • 99th %ile:    {p99_latency:.2f} ms (Worst 1% cases)")

    print(f"\n💻 RESOURCE UTILIZATION")
    print(f"   • CPU Usage:    {avg_cpu:.1f}% (Single Core Load)")
    print(f"   • RAM Usage:    {avg_ram:.1f} MB")
    
    print(f"\n📦 TOTAL PROCESSED")
    print(f"   • Queries:      {query_count} in {total_time:.2f} seconds")
    print("========================================================")
    
    # Simple Interpretation
    rating = ""
    if rps > 100: rating = "EXCELLENT (Production Ready)"
    elif rps > 50: rating = "GOOD (Capable)"
    else: rating = "LOW (Optimization Recommended)"
    print(f"📝 VERDICT: {rating}")
    print("========================================================")