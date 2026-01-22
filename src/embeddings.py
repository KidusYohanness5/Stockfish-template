"""
Image embedding utilities using CLIP model.
Converts images into high-dimensional vector representations for similarity search.
"""

from sentence_transformers import SentenceTransformer
from typing import List


class ImageEmbedder:
    """Handles image embedding generation using CLIP model."""

    def __init__(self, model_name: str, dimension: int):
        """
        Initialize the Sentence Transformers model for image embeddings.

        Args:
            model_name: Name of the sentence-transformers model
            dimension: Dimension of the embedding
        """
        self.model = SentenceTransformer(model_name)
        self.dimension = dimension
    
    def get_image_embedding(self, image_path: str) -> List[float]:
        """
        Generate embedding vector for an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of float values representing the image embedding
        """
        # TODO