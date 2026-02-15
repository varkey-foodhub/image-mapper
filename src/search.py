import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import os
import psutil
import time
import random

# --- Configuration ---
NUM_QUERIES = 1000  # Number of queries to simulate
TOP_K = 5

# --- Resource Monitor Class ---
class PerformanceMonitor:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.cpu_log = []
        self.ram_log = []
        self.latency_log = []
    
    def snapshot(self):
        # Record current resources
        self.cpu_log.append(self.process.cpu_percent(interval=None))
        self.ram_log.append(self.process.memory_info().rss / (1024 * 1024)) # MB

    def log_latency(self, start_time):
        self.latency_log.append((time.time() - start_time) * 1000) # ms

    def report(self):
        avg_latency = sum(self.latency_log) / len(self.latency_log)
        avg_cpu = sum(self.cpu_log) / len(self.cpu_log) if self.cpu_log else 0
        avg_ram = sum(self.ram_log) / len(self.ram_log) if self.ram_log else 0
        
        print("\n" + "="*40)
        print(f"📊 BENCHMARK REPORT ({len(self.latency_log)} Queries)")
        print("="*40)
        print(f"⏱️  Avg Latency:   {avg_latency:.2f} ms")
        print(f"🚀 Max Latency:   {max(self.latency_log):.2f} ms")
        print(f"⚡ Min Latency:   {min(self.latency_log):.2f} ms")
        print("-" * 40)
        print(f"💻 Avg CPU Usage: {avg_cpu:.1f}%")
        print(f"💾 Avg RAM Usage: {avg_ram:.1f} MB")
        print("="*40)

# --- Load Resources (Setup Phase) ---
print("⚙️  Loading Model & Index (Warmup)...")
if not os.path.exists("menu_index.faiss") or not os.path.exists("metadata.pkl"):
    print("❌ Error: Missing index/metadata files.")
    exit()

# Load everything once (Simulate production server state)
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("menu_index.faiss")
df = pd.read_pickle("metadata.pkl")

# --- Search Function ---
def search(query, top_k=TOP_K):
    # This is the critical path we are measuring
    query_vec = model.encode([query], normalize_embeddings=True)
    query_vec = np.array(query_vec).astype("float32")
    
    D, I = index.search(query_vec, top_k)
    
    results = []
    for idx in I[0]:
        if idx != -1:
            # Fetching data is part of the latency cost
            _ = df.iloc[idx] 
    return results

# --- Synthetic Test Data ---
# A mix of easy, hard, and misspelled queries
test_queries = [
    "coca cola can 330ml", "pepsi bottle", "water 1l", "cheese burger", 
    "chicken biryani", "french fries large", "chocolate brownie", 
    "diet coke", "spicy paneer burger", "cold coffee",
    "coka colla", "pepsi 300ml", "bisleri water", "aloo tikki", 
    "masala dosai", "garlic naan", "butter chicken", "vanilla ice cream",
    "fanta orange", "sprite plastic bottle"
]

# --- Main Execution ---
if __name__ == "__main__":
    monitor = PerformanceMonitor()
    
    print(f"\n🚀 Starting Stress Test: {NUM_QUERIES} queries...")
    
    total_start = time.time()
    
    for i in range(NUM_QUERIES):
        # Pick a random query
        q = random.choice(test_queries)
        
        # Start Timer
        q_start = time.time()
        
        # Execute Search
        search(q)
        
        # Stop Timer & Log
        monitor.log_latency(q_start)
        monitor.snapshot()
        
        # Print progress every 10 queries
        if (i + 1) % 10 == 0:
            print(f"   -> Completed {i + 1}/{NUM_QUERIES} queries...")

    total_end = time.time()
    
    # Final Output
    monitor.report()
    print(f"\n✅ Total Wall Time: {total_end - total_start:.2f} seconds")