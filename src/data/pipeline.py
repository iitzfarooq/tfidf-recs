import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
from typing import List, Optional, Dict, Any
from src.data.transform import BaseTransformer, get_transformer
from src.data.loaders import BaseLoader, CSVLoader
from src.utils.config_loader import ConfigLoader
from pathlib import Path

class DataProcessingPipeline:
    def __init__(
        self, 
        loader: BaseLoader,
        transformers: List[BaseTransformer],
        output_path: str = 'data/processed/movies_processed.parquet'
    ):
        self.loader = loader
        self.transformers = transformers
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def run(self) -> pd.DataFrame:
        """Load data, apply transformations, and optionally save the result."""
        df = self.loader.load()
        for transformer in self.transformers:
            df = transformer.transform(df)
        if self.output_path:
            df.to_parquet(self.output_path)     # faster saving format

        return df
    
    def validate(self, df: pd.DataFrame) -> bool:
        """Basic validation to check if DataFrame is not empty."""
        required_columns = ['movieId', 'combined_text']
        return not df.empty and all(col in df.columns for col in required_columns)
    
def create_pipeline(
    config: Dict[str, Any]
):
    loader_config = config.get("loader", {})
    transformer_configs = config.get("transformers", [])
    output_path = config.get("output_path", None)

    # Initialize loader
    loader_type = loader_config.get("type")
    if loader_type == "csv":
        loader = CSVLoader(loader_config)
    else:
        raise ValueError(f"Unsupported loader type: {loader_type}")

    # Initialize transformers
    transformers = []
    for t_config in transformer_configs:
        t_type = t_config.get("type")
        t_params = t_config.get("params", {})
        transformer = get_transformer(t_type, **t_params)
        transformers.append(transformer)

    return DataProcessingPipeline(
        loader=loader,
        transformers=transformers,
        output_path=output_path
    )

if __name__ == "__main__":
    # Load configuration
    config_loader = ConfigLoader()
    config_loader.load_all()
    data_config = config_loader.get("data")
    
    if not data_config:
        raise ValueError("Data configuration not found!")

    # Create and run pipeline
    pipeline = create_pipeline(data_config)
    
    print("Starting pipeline...")
    df_result = pipeline.run()
    print(f"Pipeline finished. Output shape: {df_result.shape}")
    
    if pipeline.validate(df_result):
        print("Validation passed.")
    else:
        print("Validation failed.")