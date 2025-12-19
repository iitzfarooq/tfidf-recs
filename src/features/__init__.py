"""
Features module - Core vectorization logic.

For feature generation workflows, use the orchestration module.
"""

from features.vectorizer import (
    Vectorizer,
    TfidfVectorizer,
    create_vectorizer
)

__all__ = [
    'Vectorizer',
    'TfidfVectorizer',
    'create_vectorizer'
]
