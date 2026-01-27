"""
ACU Embedding Generator Module.

Generates unique, semantically meaningful embeddings for each Atomic Credit Unit (ACU).
This solves the "Embedding Identity Problem" where all risk factors from the same
application previously shared identical embeddings.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
import hashlib


class ACUEmbeddingGenerator(ABC):
    """Abstract base class for ACU embedding generation strategies."""
    
    @abstractmethod
    def generate(self, risk_factor: str, risk_level: str, metadata: Dict[str, Any]) -> np.ndarray:
        """
        Generate a unique embedding for an ACU.
        
        Args:
            risk_factor: The description of the risk factor (e.g., "low_credit_score").
            risk_level: The severity level ("low", "medium", "high", "critical").
            metadata: Additional context like timestamps, borrower_id, etc.
            
        Returns:
            A numpy array representing the embedding vector.
        """
        pass


class TextTemplateEmbeddingGenerator(ACUEmbeddingGenerator):
    """
    Generates embeddings using a sentence transformer model.
    
    Creates a textual description of the ACU and encodes it,
    ensuring that different risk factors have distinct vectors.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", embedding_dim: int = 768):
        """
        Initialize the generator.
        
        Args:
            model_name: The sentence-transformers model to use.
            embedding_dim: The target embedding dimension (will be padded/truncated if needed).
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self._model = None
        
    @property
    def model(self):
        """Lazy-load the sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                print("Warning: sentence-transformers not installed. Using fallback embedding.")
                self._model = None
        return self._model
    
    def generate(self, risk_factor: str, risk_level: str, metadata: Dict[str, Any]) -> np.ndarray:
        """
        Generate an embedding by encoding a templated description of the ACU.
        """
        # Create a semantic description
        text = f"Credit Risk Factor: {risk_factor.replace('_', ' ')}. Severity: {risk_level}."
        
        if self.model is not None:
            # Use the sentence transformer
            embedding = self.model.encode(text, convert_to_numpy=True)
            
            # Pad or truncate to target dimension
            if len(embedding) < self.embedding_dim:
                embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)))
            elif len(embedding) > self.embedding_dim:
                embedding = embedding[:self.embedding_dim]
                
            return embedding.astype(np.float32)
        else:
            # Fallback: Hash-based deterministic embedding
            return self._fallback_embedding(text)
    
    def _fallback_embedding(self, text: str) -> np.ndarray:
        """
        Generate a deterministic embedding based on text hash.
        Used when sentence-transformers is not available.
        """
        # Create a deterministic seed from the text
        hash_bytes = hashlib.sha256(text.encode()).digest()
        seed = int.from_bytes(hash_bytes[:4], byteorder='big')
        
        # Generate reproducible random vector
        rng = np.random.default_rng(seed)
        embedding = rng.standard_normal(self.embedding_dim).astype(np.float32)
        
        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-9)
        return embedding


class HybridEmbeddingGenerator(ACUEmbeddingGenerator):
    """
    Combines a base application embedding with a risk-factor-specific offset.
    
    This preserves some contextual information from the source application
    while still differentiating between risk factors.
    """
    
    def __init__(self, text_generator: Optional[TextTemplateEmbeddingGenerator] = None, 
                 blend_ratio: float = 0.3, embedding_dim: int = 768):
        """
        Initialize the hybrid generator.
        
        Args:
            text_generator: The text-based generator to use for offsets.
            blend_ratio: How much of the final embedding comes from the risk-specific part (0.0-1.0).
            embedding_dim: The embedding dimension.
        """
        self.text_generator = text_generator or TextTemplateEmbeddingGenerator(embedding_dim=embedding_dim)
        self.blend_ratio = blend_ratio
        self.embedding_dim = embedding_dim
        
    def generate(self, risk_factor: str, risk_level: str, metadata: Dict[str, Any]) -> np.ndarray:
        """
        Generate a hybrid embedding.
        
        If a base_embedding is provided in metadata, blend it with the risk-specific embedding.
        Otherwise, use only the risk-specific embedding.
        """
        # Get the risk-specific embedding
        risk_embedding = self.text_generator.generate(risk_factor, risk_level, metadata)
        
        # Check for base embedding in metadata
        base_embedding = metadata.get("base_embedding")
        
        if base_embedding is not None:
            base_embedding = np.array(base_embedding, dtype=np.float32)
            
            # Ensure same dimension
            if len(base_embedding) != self.embedding_dim:
                if len(base_embedding) < self.embedding_dim:
                    base_embedding = np.pad(base_embedding, (0, self.embedding_dim - len(base_embedding)))
                else:
                    base_embedding = base_embedding[:self.embedding_dim]
            
            # Blend: (1 - ratio) * base + ratio * risk_specific
            blended = (1 - self.blend_ratio) * base_embedding + self.blend_ratio * risk_embedding
            
            # Normalize
            blended = blended / (np.linalg.norm(blended) + 1e-9)
            return blended.astype(np.float32)
        else:
            return risk_embedding


# Default generator instance for easy import
default_generator = TextTemplateEmbeddingGenerator()
