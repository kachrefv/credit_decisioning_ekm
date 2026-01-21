import os
import numpy as np
from typing import Union, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Batch
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SentenceTransformerEmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", qdrant_url: Optional[str] = None):
        self.model = SentenceTransformer(model_name)
        self.qdrant_client = QdrantClient(url=qdrant_url) if qdrant_url else QdrantClient()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup if needed
        pass

    def generate_text_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for text using Sentence Transformers
        """
        embedding = self.model.encode([text])
        return np.array(embedding[0])

    def generate_image_embedding(self, image_path: str) -> np.ndarray:
        """
        Generate embedding for image using Sentence Transformers
        Note: Sentence Transformers primarily handles text embeddings.
        For image embeddings, you might want to use a different model like CLIP.
        """
        raise NotImplementedError("Image embeddings are not supported with Sentence Transformers. Consider using a vision model like CLIP.")

    def store_embedding_in_qdrant(self, collection_name: str, embedding: np.ndarray, point_id: Union[int, str], metadata: Optional[dict] = None):
        """
        Store embedding in Qdrant collection
        """
        # Convert point_id to integer if it's a string
        if isinstance(point_id, str):
            # Use hash to convert string to int for Qdrant compatibility
            point_id = abs(hash(point_id)) % (2**63)  # Ensure positive int within range

        self.qdrant_client.upsert(
            collection_name=collection_name,
            points=Batch(
                ids=[point_id],
                vectors=[embedding.tolist()],
                payloads=[metadata or {}]
            )
        )

    def generate_and_store_text_embedding(self, text: str, collection_name: str, point_id: Union[int, str], metadata: Optional[dict] = None) -> np.ndarray:
        """
        Generate embedding for text and store it in Qdrant
        """
        embedding = self.generate_text_embedding(text)
        self.store_embedding_in_qdrant(collection_name, embedding, point_id, metadata)
        return embedding

    def generate_and_store_image_embedding(self, image_path: str, collection_name: str, point_id: Union[int, str], metadata: Optional[dict] = None) -> np.ndarray:
        """
        Generate embedding for image and store it in Qdrant
        """
        embedding = self.generate_image_embedding(image_path)
        self.store_embedding_in_qdrant(collection_name, embedding, point_id, metadata)
        return embedding