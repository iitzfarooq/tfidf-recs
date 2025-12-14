import joblib
import numpy as np
from scipy import sparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

from features.vectorizer import Vectorizer

class FeatureStorage:
    """
    Class to handle storage of feature matrices and vectorizers.

    FeatureStorage(base_path: str, latest_pointer: str)
    - base_path: Directory to store feature matrices and vectorizers.
    - latest_pointer: File to store the pointer to the latest feature set.
    """
    
    def __init__(
        self, 
        base_path: str ="data/features", 
        latest_pointer: str ="data/features/latest_pointer.json"
    ):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.latest_pointer = Path(latest_pointer)

    def save_features(
        self, 
        matrix: sparse.spmatrix, 
        vectorizer: Vectorizer,
        metadata: Dict
    ) -> None:
        """Save the matrix, vectorizer, and metadata."""

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        matrix_path = self.base_path / f"feature_matrix_{timestamp}.npz"
        vectorizer_path = self.base_path / f"vectorizer_{timestamp}.joblib"
        metadata_path = self.base_path / f"metadata_{timestamp}.json"

        sparse.save_npz(matrix_path, matrix)
        joblib.dump(vectorizer, vectorizer_path)

        metadata['timestamp'] = timestamp
        metadata['shape'] = matrix.shape
        with open(metadata_path, 'w') as f: 
            json.dump(metadata, f, indent=4)

        self.update_latest_pointer(timestamp)

    def update_latest_pointer(self, timestamp: str) -> None:
        pointer_data = {'latest_timestamp': timestamp}
        with open(self.latest_pointer, 'w') as f:
            json.dump(pointer_data, f, indent=4)

    def load_latest(self) -> Tuple[sparse.spmatrix, Vectorizer]:
        with open(self.latest_pointer, 'r') as f:
            pointer_data = json.load(f)
        timestamp = pointer_data['latest_timestamp']
        matrix_path = self.base_path / f"feature_matrix_{timestamp}.npz"
        vectorizer_path = self.base_path / f"vectorizer_{timestamp}.joblib"

        matrix = sparse.load_npz(matrix_path)
        vectorizer = joblib.load(vectorizer_path)

        return matrix, vectorizer
    
def create_feature_storage(
    base_path: str ="data/features", 
    latest_pointer: str ="data/features/latest_pointer.json"
) -> FeatureStorage:
    return FeatureStorage(base_path=base_path, latest_pointer=latest_pointer)