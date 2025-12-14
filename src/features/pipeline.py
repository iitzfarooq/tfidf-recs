import sys
import os
from typing import Dict, Any, List

# Add project root to path if running as script
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.steps import (
    FeatureStep, LoadDataStep, VectorizationStep, SaveFeaturesStep
    )
from src.utils.config_loader import ConfigLoader

class FeaturePipeline:
    """
    Pipeline for generating features from processed data using a chain of steps.
    """
    
    def __init__(self, steps: List[FeatureStep]):
        self.steps = steps

    def run(self, initial_data: Any) -> Any:
        """
        Run the pipeline steps in sequence.
        """
        data = initial_data
        context = {}
        
        for step in self.steps:
            data = step.execute(data, context)
            
        return data

def create_feature_pipeline(config: Dict[str, Any]) -> FeaturePipeline:
    """
    Factory function to create a FeaturePipeline from configuration.
    """
    steps = []
    
    # 1. Load Data Step
    steps.append(LoadDataStep())
    
    # 2. Vectorization Step
    vectorizer_config = config.get('vectorizer', {})
    steps.append(VectorizationStep(vectorizer_config))
    
    # 3. Save Step
    storage_config = config.get('storage', {})
    steps.append(SaveFeaturesStep(storage_config))
    
    return FeaturePipeline(steps)

def run_feature_pipeline():
    """Helper function to run the pipeline using default configs."""
    # Load configuration
    config_loader = ConfigLoader()
    config_loader.load_all()
    features_config = config_loader.get('features')
    data_config = config_loader.get('data')
    
    if not features_config:
        raise ValueError("Features configuration not found")
        
    input_path = (
        data_config.get('output_path', "data/processed/movies_processed.parquet") 
        if data_config else "data/processed/movies_processed.parquet"
    )
    
    pipeline = create_feature_pipeline(features_config)
    pipeline.run(input_path)

if __name__ == "__main__":
    run_feature_pipeline()
