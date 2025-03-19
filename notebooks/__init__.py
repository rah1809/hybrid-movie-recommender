"""
Movie Recommendation System notebooks package.
"""

from .collaborative_filtering import UserBasedCF, ItemBasedCF
from .matrix_factorization import SVDRecommender

__all__ = ['UserBasedCF', 'ItemBasedCF', 'SVDRecommender'] 