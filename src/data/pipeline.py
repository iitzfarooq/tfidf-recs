import pandas as pd
from typing import List, Optional, Dict, Any
from data.transform import BaseTransformer, get_transformer
from data.loaders import BaseLoader

class DataProcessingPipeline:
    def __init__(
        self, 
        loader: BaseLoader,
        transformers: List[BaseTransformer],
        output_path: Optional[str] = None
    ):
        self.loader = loader
        self.transformers = transformers
        self.output_path = output_path

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
        from data.loaders import CSVLoader
        loader = CSVLoader(loader_config.get("params", {}))
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
    pass