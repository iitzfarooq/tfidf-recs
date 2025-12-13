from abc import ABC, abstractmethod
import pandas as pd
import re
from typing import List
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

try:
    nltk.data.find('corpora/wordnet')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('stopwords')

class BaseTransformer(ABC):
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

# Transformers to clean and preprocess MovieLens dataset

class YearExtractor(BaseTransformer):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['year'] = df['title'].apply(self.extract_year)
        df['clean_title'] = df['title'].apply(self.clean_title)
        return df

    @staticmethod
    def extract_year(title: str) -> int:
        match = re.search(r'\((\d{4})\)', title)
        return int(match.group(1)) if match else None
    
    @staticmethod
    def clean_title(title: str) -> str:
        return re.sub(r'\s*\(\d{4}\)', '', title).strip()
    
class GenreExpander(BaseTransformer):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df['genres_list'] = df['genres'].apply(lambda x: x.split('|') if isinstance(x, str) else [])

        return df

class TextProcessor(BaseTransformer):
    def __init__(self, method: str = 'lemmatization'):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        self.method = method
        self.method_prc = self.lemmatizer if method == 'lemmatization' else self.stemmer
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df['combined_text'] = (
            df['title'].fillna('') + ' ' + 
            df['genres_list'].apply(self.join_tokens)
        )
        
        df['combined_text'] = df['combined_text'].apply(self.pipeline)
        return df

    def clean_text(self, text: str) -> str:
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        return text.split()
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        stop_words = set(stopwords.words('english'))
        return [word for word in tokens if word.lower() not in stop_words]
    
    def apply_method(self, tokens: List[str]) -> List[str]:
        return [
            self.method_prc.lemmatize(word) 
            if self.method == 'lemmatization' 
            else self.method_prc.stem(word) 
            for word in tokens
        ]
    
    def join_tokens(self, tokens: List[str]) -> str:
        return ' '.join(tokens)
    
    def pipeline(self, text: str) -> str:
        text = self.clean_text(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.apply_method(tokens)
        return ' '.join(tokens)
    
# Factory method to create transformers based on config
def get_transformer(transformer_type: str, **kwargs) -> BaseTransformer:
    if transformer_type == 'YearExtractor':
        return YearExtractor()
    elif transformer_type == 'GenreExpander':
        return GenreExpander()
    elif transformer_type == 'TextProcessor':
        return TextProcessor(**kwargs)
    else:
        raise ValueError(f"Unsupported transformer type: {transformer_type}")