import numpy as np
from typing import List, Dict
from .interfaces import Recommender, SimilarityStrategy

class ContentBasedRecommender(Recommender):
    def __init__(
        self,
        item_ids: List[int],
        features: np.ndarray,
        similarity_strategy: SimilarityStrategy
    ):
        """Initialize the Content-Based Recommender.
        
        Pre: 
        - item_ids: List of unique item identifiers. (in the same order as features),
        - features: 2D numpy array, where row i -> feature vector of item i,
        - similarity_strategy: An instance of SimilarityStrategy to compute item similarities.
        - Takes ownership of all inputs.
        """


        self.item_ids = item_ids
        self.features = features
        self.similarity_strategy = similarity_strategy

        self.id_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
        self.idx_to_id = {idx: item_id for idx, item_id in enumerate(item_ids)}

        self.similarity_matrix = self.similarity_strategy.compute(features)

    def recommend(self, id: int, k: int = 10) -> List[Dict]:
        """Generate recommendations for a single item ID."""

        if id not in self.id_to_idx:
            return []

        idx = self.id_to_idx[id]
        similarity_scores = self.similarity_matrix[idx]

        # Get top-k indices (excluding self)
        similar_indices = np.argsort(similarity_scores)[::-1][1:k+1]
        
        return [{
            "item_id": self.idx_to_id[sim_idx],
            "score": float(similarity_scores[sim_idx]),
            "rank": rank + 1
        } for rank, sim_idx in enumerate(similar_indices)]
    
    def recommend_batch(self, ids: List[int], k: int = 10) -> Dict[int, List[Dict]]:
        """Generate recommendations for a batch of item IDs."""

        all_recommendations = {}
        for id in ids:
            all_recommendations[id] = self.recommend(id, k)
        return all_recommendations
    
    def find_similar_items(self, query_feat: np.ndarray, k: int = 10) -> List[Dict]:
        """Find similar items for given query feature vectors."""

        similarities = (
            self.similarity_strategy.compute_pairwise(
                query_feat.reshape(1, -1), self.features
            )
        )[0]

        similar_indices = np.argsort(similarities)[::-1][:k]

        return [{
            "item_id": self.idx_to_id[idx],
            "score": float(similarities[idx]),
            "rank": rank + 1
        } for rank, idx in enumerate(similar_indices)]
    
    