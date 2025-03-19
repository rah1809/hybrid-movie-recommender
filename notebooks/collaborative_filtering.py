"""
Collaborative Filtering implementation using the Surprise library.
"""

import numpy as np
import pandas as pd
from surprise import KNNBasic, Dataset
from typing import List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserBasedCF:
    """User-based Collaborative Filtering using Surprise library."""
    
    def __init__(self, k: int = 40, min_k: int = 5):
        """
        Initialize User-based CF model.
        
        Args:
            k (int): Number of neighbors to use
            min_k (int): Minimum number of neighbors
        """
        self.k = k
        self.min_k = min_k
        self.model = KNNBasic(
            k=k,
            min_k=min_k,
            sim_options={'name': 'cosine', 'user_based': True}
        )
        self.is_fitted = False
    
    def fit(self, data: Dataset) -> None:
        """
        Train the model using Surprise dataset.
        
        Args:
            data: Surprise Dataset object
        """
        try:
            logger.info("Training User-based CF model...")
            trainset = data.build_full_trainset()
            self.model.fit(trainset)
            self.is_fitted = True
            logger.info("User-based CF model trained successfully")
        except Exception as e:
            logger.error(f"Error training User-based CF model: {str(e)}")
            raise
    
    def get_top_n_recommendations(self, user_id: int, movies_df: pd.DataFrame, n: int = 10) -> List[Tuple[int, str, float]]:
        """
        Generate top-N recommendations for a user.
        
        Args:
            user_id (int): Target user ID
            movies_df (pd.DataFrame): DataFrame containing movie information
            n (int): Number of recommendations to generate
            
        Returns:
            List[Tuple[int, str, float]]: List of (movie_id, title, predicted_rating) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            # Get all movies
            all_movies = movies_df['movie_id'].unique()
            
            # Generate predictions for all movies
            predictions = []
            for movie_id in all_movies:
                pred = self.model.predict(user_id, movie_id)
                predictions.append((movie_id, pred.est))
            
            # Sort by predicted rating and get top N
            top_n = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
            
            # Add movie titles
            recommendations = []
            for movie_id, pred_rating in top_n:
                title = movies_df[movies_df['movie_id'] == movie_id]['title'].iloc[0]
                recommendations.append((movie_id, title, pred_rating))
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            raise

class ItemBasedCF:
    """Item-based Collaborative Filtering using Surprise library."""
    
    def __init__(self, k: int = 40, min_k: int = 5):
        """
        Initialize Item-based CF model.
        
        Args:
            k (int): Number of neighbors to use
            min_k (int): Minimum number of neighbors
        """
        self.k = k
        self.min_k = min_k
        self.model = KNNBasic(
            k=k,
            min_k=min_k,
            sim_options={'name': 'cosine', 'user_based': False}
        )
        self.is_fitted = False
    
    def fit(self, data: Dataset) -> None:
        """
        Train the model using Surprise dataset.
        
        Args:
            data: Surprise Dataset object
        """
        try:
            logger.info("Training Item-based CF model...")
            trainset = data.build_full_trainset()
            self.model.fit(trainset)
            self.is_fitted = True
            logger.info("Item-based CF model trained successfully")
        except Exception as e:
            logger.error(f"Error training Item-based CF model: {str(e)}")
            raise
    
    def get_top_n_recommendations(self, user_id: int, movies_df: pd.DataFrame, n: int = 10) -> List[Tuple[int, str, float]]:
        """
        Generate top-N recommendations for a user.
        
        Args:
            user_id (int): Target user ID
            movies_df (pd.DataFrame): DataFrame containing movie information
            n (int): Number of recommendations to generate
            
        Returns:
            List[Tuple[int, str, float]]: List of (movie_id, title, predicted_rating) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            # Get all movies
            all_movies = movies_df['movie_id'].unique()
            
            # Generate predictions for all movies
            predictions = []
            for movie_id in all_movies:
                pred = self.model.predict(user_id, movie_id)
                predictions.append((movie_id, pred.est))
            
            # Sort by predicted rating and get top N
            top_n = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
            
            # Add movie titles
            recommendations = []
            for movie_id, pred_rating in top_n:
                title = movies_df[movies_df['movie_id'] == movie_id]['title'].iloc[0]
                recommendations.append((movie_id, title, pred_rating))
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            raise 