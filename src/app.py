import os
import time
import psutil
import faiss
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer

# --- Configuration ---
app = Flask(__name__)
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5

# --- Global Resources ---
print("⚙️  Booting FoodHub AI Engine...")
try:
    # 1. Load AI Model
    model = SentenceTransformer(MODEL_NAME)
    
    # 2. Load Vector Database
    if os.path.exists("menu_index.faiss") and os.path.exists("metadata.pkl"):
        index = faiss.read_index("menu_index.faiss")
        meta_df = pd.read_pickle("metadata.pkl")
    else:
        print("❌ CRITICAL: 'menu_index.faiss' or 'metadata.pkl' not found.")
        # Create dummy data if files missing (prevents crash during testing)
        index = None
        meta_df = pd.DataFrame()

    # 3. Load Master Image List (The "Clean" Names)
    # We convert this to a Dictionary for O(1) ultra-fast lookups
    if os.path.exists("master_image_list.csv"):
        master_df = pd.read_csv("master_image_list.csv")
        # Create dict: {1: "Coca Cola 200ml", 2: "Pepsi..."}
        master_map = dict(zip(master_df.image_id, master_df.name))
    else:
        master_map = {}
        print("⚠️ Warning: master_image_list.csv not found.")

except Exception as e:
    print(f"❌ Error during startup: {e}")

# --- Helper: Resource Monitor ---
def get_system_metrics():
    process = psutil.Process(os.getpid())
    return {
        "cpu": process.cpu_percent(interval=None),
        "ram": round(process.memory_info().rss / (1024 * 1024), 2) # MB
    }

# --- Route: Home ---
@app.route('/')
def home():
    return render_template('index.html')

# --- Route: Search API ---
@app.route('/search', methods=['POST'])
def search():
    start_time = time.time()
    data = request.json
    query = data.get('query', '').strip()

    if not query:
        return jsonify({"error": "Empty query"}), 400

    if index is None:
        return jsonify({"error": "Vector DB not loaded"}), 500

    # 1. Vector Search
    query_vec = model.encode([query], normalize_embeddings=True).astype("float32")
    D, I = index.search(query_vec, TOP_K)

    results = []
    seen_ids = set()

    # 2. Process Results
    for score, idx in zip(D[0], I[0]):
        if idx != -1:
            # Get Image ID from vector metadata
            # Use .get() to avoid crashes if column names differ
            raw_img_id = meta_df.iloc[idx].get("image_id")
            
            # Ensure ID is a standard Python int
            img_id = int(raw_img_id)

            # Deduplicate (Don't show the same image twice)
            if img_id in seen_ids:
                continue
            seen_ids.add(img_id)

            # 3. Fetch Canonical Name from Master CSV
            # Default to "Unknown Item" if ID not in master list
            canonical_name = master_map.get(img_id, f"Unknown Item (ID: {img_id})")

            results.append({
                "image_id": img_id,
                "name": canonical_name,
                "score": round(float(score), 4),
                "original_match": meta_df.iloc[idx].get("menu_item_name", "N/A") # Debug info
            })

    # 3. Calculate Performance Metrics
    latency_ms = round((time.time() - start_time) * 1000, 2)
    metrics = get_system_metrics()

    return jsonify({
        "results": results,
        "stats": {
            "latency": f"{latency_ms} ms",
            "cpu": f"{metrics['cpu']}%",
            "ram": f"{metrics['ram']} MB"
        }
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)