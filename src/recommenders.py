"""
Recommendation algorithms implementation module.
"""

from abc import ABC, abstractmethod
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrivialRecommender:
    def __init__(self, min_ratings=100):
        self.min_ratings = min_ratings
        self.popular_movies = None

    def fit(self, ratings_df, movies_df):
        movie_counts = ratings_df.groupby('movie_id')['rating'].count()
        popular_movies = movie_counts[movie_counts >= self.min_ratings].index.tolist()
        self.popular_movies = movies_df[movies_df['movie_id'].isin(popular_movies)]

    def get_top_n_recommendations(self, user_id, movies_df, n=10):
        recommendations = self.popular_movies[['movie_id', 'title']].head(n)
        return [(row['movie_id'], row['title'], 5.0) for _, row in recommendations.iterrows()]

class MPIRecommender:
    def __init__(self, min_rating=4.0):
        self.min_rating = min_rating
        self.popular_movies = None

    def fit(self, ratings_df, movies_df):
        avg_ratings = ratings_df.groupby('movie_id')['rating'].mean()
        popular_movies = avg_ratings[avg_ratings >= self.min_rating].index.tolist()
        self.popular_movies = movies_df[movies_df['movie_id'].isin(popular_movies)]

    def get_top_n_recommendations(self, user_id, movies_df, n=10):
        recommendations = self.popular_movies[['movie_id', 'title']].head(n)
        return [(row['movie_id'], row['title'], 5.0) for _, row in recommendations.iterrows()]

class BaseRecommender(ABC):
    """Base abstract class for recommendation algorithms."""
    
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, ratings: pd.DataFrame) -> None:
        """Train the recommender system."""
        pass
    
    @abstractmethod
    def predict(self, user_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """Generate recommendations for a user."""
        pass

class CollaborativeFiltering(BaseRecommender):
    """User-based collaborative filtering implementation."""
    
    def __init__(self, k_neighbors: int = 5):
        super().__init__(name="Collaborative Filtering")
        self.k_neighbors = k_neighbors
        self.user_item_matrix = None
        self.similarity_matrix = None
    
    def fit(self, ratings: pd.DataFrame) -> None:
        """
        Train the collaborative filtering model.
        
        Args:
            ratings (pd.DataFrame): DataFrame with columns [user_id, item_id, rating]
        """
        try:
            logger.info("Training collaborative filtering model...")
            # Create user-item matrix
            self.user_item_matrix = ratings.pivot(
                index='user_id',
                columns='item_id',
                values='rating'
            ).fillna(0)
            
            # Calculate user similarity matrix
            self.similarity_matrix = cosine_similarity(self.user_item_matrix)
            self.is_fitted = True
            logger.info("Collaborative filtering model trained successfully")
            
        except Exception as e:
            logger.error(f"Error training collaborative filtering model: {str(e)}")
            raise

    def predict(self, user_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """
        Generate recommendations for a user.
        
        Args:
            user_id (int): Target user ID
            n_recommendations (int): Number of recommendations to generate
            
        Returns:
            List[Tuple[int, float]]: List of (item_id, predicted_rating) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        try:
            user_idx = self.user_item_matrix.index.get_loc(user_id)
            
            # Get similar users
            similar_users = np.argsort(self.similarity_matrix[user_idx])[-self.k_neighbors-1:-1]
            
            # Get items user hasn't rated
            user_unrated = self.user_item_matrix.columns[
                self.user_item_matrix.iloc[user_idx].values == 0
            ]
            
            # Calculate predicted ratings
            predictions = []
            for item in user_unrated:
                item_idx = self.user_item_matrix.columns.get_loc(item)
                similar_users_ratings = self.user_item_matrix.iloc[similar_users, item_idx]
                if similar_users_ratings.sum() > 0:
                    pred_rating = np.average(
                        similar_users_ratings,
                        weights=self.similarity_matrix[user_idx, similar_users]
                    )
                    predictions.append((item, pred_rating))
            
            # Sort and return top N recommendations
            return sorted(predictions, key=lambda x: x[1], reverse=True)[:n_recommendations]
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            raise

class MatrixFactorization(BaseRecommender):
    """SVD-based matrix factorization implementation."""
    
    def __init__(self, n_factors: int = 100, n_epochs: int = 20, lr: float = 0.005, reg: float = 0.02):
        super().__init__(name="Matrix Factorization")
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.user_factors = None
        self.item_factors = None
        self.global_mean = None
    
    def fit(self, ratings: pd.DataFrame) -> None:
        """
        Train the matrix factorization model using SVD.
        
        Args:
            ratings (pd.DataFrame): DataFrame with columns [user_id, item_id, rating]
        """
        try:
            logger.info("Training matrix factorization model...")
            
            # Create user-item matrix
            user_item_matrix = ratings.pivot(
                index='user_id',
                columns='item_id',
                values='rating'
            ).fillna(0).values
            
            # Perform SVD
            U, sigma, Vt = svds(user_item_matrix, k=self.n_factors)
            
            self.user_factors = U
            self.item_factors = Vt.T
            self.global_mean = ratings['rating'].mean()
            
            self.is_fitted = True
            logger.info("Matrix factorization model trained successfully")
            
        except Exception as e:
            logger.error(f"Error training matrix factorization model: {str(e)}")
            raise
    
    def predict(self, user_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """
        Generate recommendations for a user using matrix factorization.
        
        Args:
            user_id (int): Target user ID
            n_recommendations (int): Number of recommendations to generate
            
        Returns:
            List[Tuple[int, float]]: List of (item_id, predicted_rating) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        try:
            user_idx = user_id  # Assuming user_ids are zero-based indices
            
            # Calculate predicted ratings for all items
            predicted_ratings = np.dot(self.user_factors[user_idx], self.item_factors.T)
            
            # Create (item_id, rating) pairs and sort by rating
            recommendations = [
                (item_id, rating) 
                for item_id, rating in enumerate(predicted_ratings)
            ]
            
            return sorted(recommendations, key=lambda x: x[1], reverse=True)[:n_recommendations]
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            raise 