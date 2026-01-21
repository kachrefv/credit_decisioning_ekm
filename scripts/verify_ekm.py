import numpy as np
import time
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from ekm.core.models import Episode
from ekm.core.engine import EKM

def generate_synthetic_data(num_episodes=1100, d=768):
    episodes = []
    for i in range(num_episodes):
        group = i // 100
        base = np.random.randn(d)
        embedding = base + np.random.randn(d) * 0.1
        embedding /= np.linalg.norm(embedding)
        
        episodes.append(Episode(
            id=f"ep_{i}",
            content=f"Synthetic content for episode {i} in group {group}",
            embedding=embedding,
            metadata={"timestamp": time.time() - i * 100, "group": group}
        ))
    return episodes

def run_benchmark():
    print("Initializing EKM Corporate Logic...")
    ekm = EKM(mesh_threshold=1000)
    
    print("\n--- Testing Cold Start Mode ---")
    data_small = generate_synthetic_data(num_episodes=100)
    ekm.ingest_episodes(data_small)
    print(f"Mode: {ekm.mode}, AKUs: {len(ekm.akus)}")
    
    query_ep = data_small[0]
    results = ekm.retrieve("Test Query", query_ep.embedding)
    print(f"Retrieved {len(results)} items")

if __name__ == "__main__":
    run_benchmark()
