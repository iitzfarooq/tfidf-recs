# Concrete implementation of SimilarityStrategy
# Using scikit-learn's cosine_similarity function (for now)

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from .interfaces import SimilarityStrategy


class CosineSimilarityStrategy(SimilarityStrategy):
    def compute(self, features: np.ndarray) -> np.ndarray:
        """Compute cosine similarity scores for the given features."""
        return cosine_similarity(features)

    def compute_pairwise(
        self, 
        features_A: np.ndarray, 
        features_B: np.ndarray
    ) -> np.ndarray:
        """Compute pairwise cosine similarity scores between two sets of features."""
        return cosine_similarity(features_A, features_B)


def create_similarity_strategy(strategy_name: str = 'cosine') -> SimilarityStrategy:
    """
    Factory function to create similarity strategies.
    """
    if strategy_name.lower() == 'cosine':
        return CosineSimilarityStrategy()
    else:
        raise ValueError(f"Unknown similarity strategy: {strategy_name}")