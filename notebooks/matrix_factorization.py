"""
Matrix Factorization implementation using SVD from Surprise library.
"""

import numpy as np
import pandas as pd
from surprise import SVD, Dataset
from typing import List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SVDRecommender:
    """Matrix Factorization using SVD from Surprise library."""
    
    def __init__(self, n_factors: int = 100, n_epochs: int = 20, 
                 lr_all: float = 0.005, reg_all: float = 0.02):
        """
        Initialize SVD model.
        
        Args:
            n_factors (int): Number of latent factors
            n_epochs (int): Number of iterations for SGD
            lr_all (float): Learning rate for all parameters
            reg_all (float): Regularization term for all parameters
        """
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.model = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all,
            random_state=42
        )
        self.is_fitted = False
    
    def fit(self, data: Dataset) -> None:
        """
        Train the SVD model using Surprise dataset.
        
        Args:
            data: Surprise Dataset object
        """
        try:
            logger.info("Training SVD model...")
            trainset = data.build_full_trainset()
            self.model.fit(trainset)
            self.is_fitted = True
            logger.info("SVD model trained successfully")
        except Exception as e:
            logger.error(f"Error training SVD model: {str(e)}")
            raise
    
    def get_top_n_recommendations(self, user_id: int, movies_df: pd.DataFrame, n: int = 10) -> List[Tuple[int, str, float]]:
        """
        Generate top-N recommendations for a user using SVD.
        
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
    
    def get_model_parameters(self) -> dict:
        """
        Get the current model parameters.
        
        Returns:
            dict: Dictionary containing model parameters
        """
        return {
            'n_factors': self.n_factors,
            'n_epochs': self.n_epochs,
            'learning_rate': self.lr_all,
            'regularization': self.reg_all
        } 