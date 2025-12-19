from abc import ABC, abstractmethod
from scipy.sparse import csr_matrix

class Vectorizer(ABC):
    def __init__(self, **kwargs):
        self._is_fitted = False
        self._vocabulary = None
        self._config = kwargs

    @abstractmethod
    def fit(self, corpus) -> 'Vectorizer': pass

    @abstractmethod
    def transform(self, documents) -> csr_matrix: pass
        
    def fit_transform(self, documents) -> csr_matrix:
        return self.fit(documents).transform(documents)

    @property
    def vocabulary(self):
        return self._vocabulary
    
    @property
    def is_fitted(self):
        return self._is_fitted
    
    @property 
    def config(self):
        return self._config.copy()
    
    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}(fitted={self._is_fitted}, config={self._config})"
    
# TF-IDF Vectorizer implementation

from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTfidfVectorizer

class TfidfVectorizer(Vectorizer):
    """TF-IDF Vectorizer implementation"""

    def __init__(self, **kwargs):
        """
        Initialize TF-IDF vectorizer with given or default parameters.

        Configurable parameters include:
            - max_features: Maximum number of features to consider
            - stop_words: Stop words to remove
            - ngram_range: The range of n-grams to consider
            - min_df: Minimum document frequency for terms
            - max_df: Maximum document frequency for terms
            - sublinear_tf: Whether to apply sublinear TF scaling
        """

        params = {
            'max_features': kwargs.get('max_features', None),
            'stop_words': kwargs.get('stop_words', None),
            'ngram_range': tuple(kwargs.get('ngram_range', (1, 2))),
            'min_df': kwargs.get('min_df', 1),
            'max_df': kwargs.get('max_df', 0.7),
            'sublinear_tf': kwargs.get('sublinear_tf', True),
            'norm': kwargs.get('norm', 'l2'),
        }

        super().__init__(**params)
        self._vectorizer = SklearnTfidfVectorizer(**params)

    def fit(self, documents) -> 'TfidfVectorizer':
        """
        Fit the TF-IDF vectorizer to the documents.
        """
        self._vectorizer.fit(documents)
        self._is_fitted = True
        self._vocabulary = self._vectorizer.get_feature_names_out().tolist()
        return self
    
    def transform(self, documents) -> csr_matrix:
        """
        Transform documents to TF-IDF vectors. Raises error if not fitted.
        """
        if not self._is_fitted:
            raise ValueError("Vectorizer must be fitted before transformation")
        return self._vectorizer.transform(documents)
    
"""
All classes defined in this module (current and future implementations of 
the `Vectorizer` abstract base class) are designed to be fully 
pickle-able. This ensures that fitted objects, including internal state 
such as learned vocabulary, configuration, and fitted parameters 
(e.g., IDF values in TF-IDF), can be serialized and deserialized reliably.

It is strongly recommended to persist instances using `joblib.dump()` 
and load them with `joblib.load()`, 
because joblib handles large NumPy arrays, SciPy sparse matrices, 
and scikit-learn estimators efficiently, provides optional compression, 
and supports memory mapping. 
"""

def create_vectorizer(vectorizer_type: str, **kwargs) -> Vectorizer:
    """Factory function to create vectorizer instances based on type."""
    if vectorizer_type == 'tfidf':
        return TfidfVectorizer(**kwargs)
    else:
        raise ValueError(f"Unknown vectorizer type: {vectorizer_type}")