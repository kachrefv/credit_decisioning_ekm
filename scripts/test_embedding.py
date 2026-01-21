#!/usr/bin/env python3
"""
Simple test script to verify the new SentenceTransformer embedding service works correctly.
"""

import asyncio
import numpy as np
from src.ekm.services.embedding_service import SentenceTransformerEmbeddingService

def test_sentence_transformer_embedding():
    print("Testing SentenceTransformer embedding service...")
    
    # Initialize the embedding service
    embedding_service = SentenceTransformerEmbeddingService(model_name="all-MiniLM-L6-v2")
    
    # Test text embedding
    test_text = "This is a test sentence for embedding."
    print(f"Input text: {test_text}")
    
    embedding = embedding_service.generate_text_embedding(test_text)
    print(f"Generated embedding shape: {embedding.shape}")
    print(f"Embedding dtype: {embedding.dtype}")
    print(f"Sample of embedding: {embedding[:10]}...")  # Show first 10 elements
    
    # Test with another text to ensure consistency
    test_text2 = "Another test sentence for comparison."
    embedding2 = embedding_service.generate_text_embedding(test_text2)
    print(f"\nSecond embedding shape: {embedding2.shape}")
    
    # Calculate similarity between embeddings
    similarity = np.dot(embedding, embedding2) / (np.linalg.norm(embedding) * np.linalg.norm(embedding2))
    print(f"Cosine similarity between embeddings: {similarity}")
    
    print("\n✓ SentenceTransformer embedding service test completed successfully!")
    
    return embedding, embedding2

if __name__ == "__main__":
    test_sentence_transformer_embedding()