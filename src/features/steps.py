from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
from scipy import sparse
import os
from src.features.vectorizer import create_vectorizer
from src.features.storage import FeatureStorage

class FeatureStep(ABC):
    """Abstract base class for feature generation steps."""
    
    @abstractmethod
    def execute(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Execute the step.
        
        Args:
            data: The input data for this step.
            context: A dictionary to share state/metadata between steps.
            
        Returns:
            The transformed data to be passed to the next step.
        """
        pass

class LoadDataStep(FeatureStep):
    """Step to load processed data."""
    
    def execute(self, input_path: str, context: Dict[str, Any]) -> pd.DataFrame:
        print(f"Loading data from {input_path}...")
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        df = pd.read_parquet(input_path)
        
        if 'combined_text' not in df.columns:
            raise ValueError("Input data missing 'combined_text' column")
            
        context['input_path'] = str(input_path)
        context['num_documents'] = len(df)
        return df

class VectorizationStep(FeatureStep):
    """Step to vectorize text data."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vectorizer = create_vectorizer(
            config.get('type', 'tfidf'),
            **config.get('params', {})
        )
        
    def execute(self, df: pd.DataFrame, context: Dict[str, Any]) -> sparse.csr_matrix:
        print("Fitting and transforming vectorizer...")
        feature_matrix = self.vectorizer.fit_transform(df['combined_text'])
        
        print(f"Generated feature matrix with shape: {feature_matrix.shape}")
        
        # Store vectorizer and config in context for later use (e.g. saving)
        context['vectorizer'] = self.vectorizer
        context['vectorizer_config'] = self.config
        
        return feature_matrix

class SaveFeaturesStep(FeatureStep):
    """Step to save features and vectorizer."""
    
    def __init__(self, config: Dict[str, Any]):
        self.storage = FeatureStorage(
            base_path=config.get('base_path', 'data/features'),
            latest_pointer=config.get('latest_pointer', 'data/features/latest_pointer.json')
        )
        
    def execute(self, matrix: sparse.csr_matrix, context: Dict[str, Any]) -> sparse.csr_matrix:
        print("Saving features and vectorizer...")
        
        vectorizer = context.get('vectorizer')
        if not vectorizer:
            raise ValueError("Vectorizer not found in context. Ensure VectorizationStep has run.")
            
        metadata = {
            'input_path': context.get('input_path'),
            'num_documents': context.get('num_documents'),
            'vectorizer_type': context.get('vectorizer_config', {}).get('type'),
            'vectorizer_params': vectorizer.config
        }
        
        self.storage.save_features(matrix, vectorizer, metadata)
        print("Feature generation completed successfully.")
        
        return matrix

def get_step(step_type: str, config: Dict[str, Any]) -> FeatureStep:
    """Factory method to create steps."""
    if step_type == 'load_data':
        return LoadDataStep()
    elif step_type == 'vectorize':
        return VectorizationStep(config)
    elif step_type == 'save':
        return SaveFeaturesStep(config)
    else:
        raise ValueError(f"Unknown step type: {step_type}")
