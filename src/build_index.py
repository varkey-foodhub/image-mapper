import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import psutil
import os
import time
import torch

# --- Resource Monitoring Helper ---
class ResourceMonitor:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_time = time.time()
        self.step_start_time = time.time()
        
        # Check for CUDA availability for VRAM logging
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Running on: {self.device.upper()}")
        if self.device == "cuda":
            print(f"   GPU: {torch.cuda.get_device_name(0)}")

    def log(self, step_name):
        # Time calculations
        now = time.time()
        step_duration = now - self.step_start_time
        total_duration = now - self.start_time
        self.step_start_time = now

        # CPU & RAM (System Memory)
        ram_usage_mb = self.process.memory_info().rss / (1024 * 1024)  # Convert Bytes to MB
        cpu_percent = self.process.cpu_percent(interval=None) # Non-blocking

        # VRAM (GPU Memory) - Only if using CUDA
        vram_info = ""
        if self.device == "cuda":
            allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)
            vram_info = f"| VRAM Used: {allocated:.2f} MB (Reserved: {reserved:.2f} MB)"

        print(f"[{step_name}]")
        print(f"   ⏱️  Time: {step_duration:.2f}s (Total: {total_duration:.2f}s)")
        print(f"   💻 RAM:  {ram_usage_mb:.2f} MB | CPU: {cpu_percent}% {vram_info}")
        print("-" * 60)

# Initialize Monitor
monitor = ResourceMonitor()

try:
    # 1. Load CSV
    print("Reading CSV...")
    # Assuming your header is 'menu_item_name' based on previous context, 
    # but keeping 'name' as per your provided code snippet.
    # If using my generated dataset, change 'name' to 'menu_item_name' below.
    df = pd.read_csv("data.csv") 
    
    # Validation to ensure column exists
    column_name = "menu_item_name" if "menu_item_name" in df.columns else "name"
    texts = df[column_name].tolist()
    
    monitor.log("CSV Load")

    # 2. Load model
    print("Loading Model...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device=monitor.device)
    monitor.log("Model Load")

    # 3. Create embeddings
    print(f"Encoding {len(texts)} items...")
    # batch_size=32 is standard; increase if you have high VRAM
    embeddings = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    monitor.log("Encoding Generation")

    # 4. Convert to float32 (required by faiss)
    embeddings = np.array(embeddings).astype("float32")
    
    # 5. Create FAISS index
    print("Building Index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product = cosine when normalized
    index.add(embeddings)
    monitor.log("FAISS Index Build")

    # 6. Save index and metadata
    print("Saving to disk...")
    faiss.write_index(index, "menu_index.faiss")
    df.to_pickle("metadata.pkl")
    monitor.log("Save to Disk")

    print("\n✅ Index built successfully!")

except Exception as e:
    print(f"\n❌ Error: {e}")