"""
Orchestration steps for ML pipeline workflows.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
from scipy import sparse
import numpy as np
from pathlib import Path

from src.features.vectorizer import create_vectorizer
from src.engine.similarity_strategies import create_similarity_strategy


class OrchestrationStep(ABC):
    """Abstract base class for orchestration steps."""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}

    @abstractmethod
    def execute(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Execute the step.

        Args:
            data: Input data for this step
            context: Shared context with registry, metadata, etc.

        Returns:
            Output data to pass to next step
        """
        pass


class LoadDataStep(OrchestrationStep):
    """Load processed data for feature generation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("Load Data", config)

    def execute(
        self, input_path: Optional[str], context: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Load data from file.

        Args:
            input_path: Path to input data file
            context: Shared context

        Returns:
            Loaded DataFrame
        """
        # Use provided path or get from config
        path = input_path or self.config.get("input_path")

        if not path:
            raise ValueError("No input path provided")

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        print(f"  Loading data from: {path}")

        # Load based on file type
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix == ".csv":
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        # Validate required columns
        required_col = self.config.get("text_column", "combined_text")
        if required_col not in df.columns:
            raise ValueError(f"Missing required column: '{required_col}'")

        print(f"  ✓ Loaded {len(df)} documents")

        # Store in context
        context["dataframe"] = df
        context["input_path"] = str(path)
        context["num_documents"] = len(df)
        context["text_column"] = required_col

        return df


class FitVectorizerStep(OrchestrationStep):
    """Fit vectorizer on text data and register as artifact."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("Fit Vectorizer", config)

    def execute(self, df: pd.DataFrame, context: Dict[str, Any]) -> Any:
        """
        Fit vectorizer and register as artifact.

        Args:
            df: Input DataFrame with text column
            context: Shared context with registry

        Returns:
            Fitted vectorizer
        """
        text_column = context.get("text_column", "combined_text")
        documents = df[text_column].tolist()

        # Create and fit vectorizer
        vectorizer_type = self.config.get("type", "tfidf")
        vectorizer_params = self.config.get("params", {})

        print(f"  Creating {vectorizer_type} vectorizer...")
        print(f"  Parameters: {vectorizer_params}")

        vectorizer = create_vectorizer(vectorizer_type, **vectorizer_params)

        print(f"  Fitting on {len(documents)} documents...")
        vectorizer.fit(documents)

        vocab_size = len(vectorizer.vocabulary)
        print(f"  ✓ Fitted vectorizer with vocabulary size: {vocab_size}")

        # Register as artifact
        registry = context["registry"]
        registry.register_artifact(
            artifact_type="vectorizer",
            artifact_name="vectorizer",
            artifact_data=vectorizer,
            metadata={
                "type": vectorizer_type,
                "params": vectorizer_params,
                "vocabulary_size": vocab_size,
            },
        )

        print(f"  ✓ Registered vectorizer artifact")

        # Store in context
        context["vectorizer"] = vectorizer
        context["vectorizer_type"] = vectorizer_type

        return vectorizer


class GenerateFeaturesStep(OrchestrationStep):
    """Generate feature matrix and register as artifact."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("Generate Features", config)

    def execute(self, vectorizer: Any, context: Dict[str, Any]) -> sparse.csr_matrix:
        """
        Transform documents to feature matrix.

        Args:
            vectorizer: Fitted vectorizer
            context: Shared context

        Returns:
            Feature matrix
        """
        df = context["dataframe"]
        text_column = context["text_column"]
        documents = df[text_column].tolist()

        print(f"  Transforming {len(documents)} documents...")
        feature_matrix = vectorizer.transform(documents)

        print(f"  ✓ Generated feature matrix: {feature_matrix.shape}")
        print(f"    Non-zero elements: {feature_matrix.nnz}")
        print(
            f"    Sparsity: {1 - feature_matrix.nnz / np.prod(feature_matrix.shape):.4f}"
        )

        # Register as artifact
        registry = context["registry"]
        registry.register_artifact(
            artifact_type="feature_matrix",
            artifact_name=self.config.get("artifact_name", "features"),
            artifact_data=feature_matrix,
            metadata={
                "shape": list(feature_matrix.shape),
                "nnz": int(feature_matrix.nnz),
                "dtype": str(feature_matrix.dtype),
                "sparsity": float(
                    1 - feature_matrix.nnz / np.prod(feature_matrix.shape)
                ),
            },
        )

        print(f"  ✓ Registered feature_matrix artifact")

        # Register metadata (including movie IDs for recommendations)
        df = context.get("dataframe")
        movie_ids = df['movieId'].tolist() if 'movieId' in df.columns else list(range(len(df)))
        
        metadata = {
            "input_path": context["input_path"],
            "num_documents": context["num_documents"],
            "vectorizer_type": context["vectorizer_type"],
            "text_column": text_column,
            "feature_shape": list(feature_matrix.shape),
            "movie_ids": movie_ids,
        }

        registry.register_artifact(
            artifact_type="metadata", 
            artifact_name="metadata", 
            artifact_data=metadata
        )

        print(f"  ✓ Registered metadata artifact")

        # Store in context
        context["feature_matrix"] = feature_matrix

        return feature_matrix


class GenerateSimilarityStep(OrchestrationStep):
    """Generate similarity matrix and register as artifact."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("Generate Similarity Matrix", config)

    def execute(
        self, feature_matrix: sparse.csr_matrix, context: Dict[str, Any]
    ) -> sparse.csr_matrix:
        """
        Compute similarity matrix from features.

        Args:
            feature_matrix: Input feature matrix
            context: Shared context

        Returns:
            Similarity matrix
        """
        strategy_name = self.config.get("strategy", "cosine")

        print(f"  Computing {strategy_name} similarity...")

        # Create similarity strategy
        strategy = create_similarity_strategy(strategy_name)

        # Compute similarity
        similarity_matrix = strategy.compute(feature_matrix)

        print(f"  ✓ Generated similarity matrix: {similarity_matrix.shape}")

        # Register as artifact
        registry = context["registry"]

        registry.register_artifact(
            artifact_type="similarity_matrix",
            artifact_name=self.config.get("artifact_name", "similarity"),
            artifact_data=similarity_matrix,
            metadata={
                "shape": list(similarity_matrix.shape),
                "strategy": strategy_name,
            },
        )

        print(f"  ✓ Registered similarity_matrix artifact")

        # Store in context
        context["similarity_matrix"] = similarity_matrix

        return similarity_matrix
