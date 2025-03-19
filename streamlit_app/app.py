"""
Streamlit app for movie recommendations using multiple algorithms.
"""

import streamlit as st

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

import numpy as np
import pandas as pd
import sys
import os
from typing import List, Set, Dict, Optional, Tuple, Any, Union
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
import altair as alt
from datetime import datetime
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root directory to the Python path
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

try:
    # Import recommenders
    from src.recommenders import TrivialRecommender, MPIRecommender
    from notebooks.collaborative_filtering import UserBasedCF, ItemBasedCF
    from notebooks.matrix_factorization import SVDRecommender
except ImportError as e:
    st.error(f"Error importing modules: {str(e)}")
    logger.error(f"Import error: {str(e)}")
    sys.exit(1)

# Add genre icons mapping
GENRE_ICONS = {
    'Action': '🎬',
    'Adventure': '🌄',
    'Animation': '🎨',
    'Children': '👶',
    'Comedy': '😂',
    'Crime': '🚔',
    'Documentary': '📚',
    'Drama': '🎭',
    'Fantasy': '🧙‍♂️',
    'Film-Noir': '🎥',
    'Horror': '👻',
    'Musical': '🎵',
    'Mystery': '🔍',
    'Romance': '❤️',
    'Sci-Fi': '🚀',
    'Thriller': '😱',
    'War': '⚔️',
    'Western': '🤠'
}

class ContentBasedRecommender:
    """Enhanced content-based recommender using TF-IDF and cosine similarity."""
    
    def __init__(self):
        self.title_vectorizer = TfidfVectorizer(stop_words='english')
        self.genre_vectorizer = TfidfVectorizer(stop_words='english')
        self.title_features = None
        self.genre_features = None
        self.movies_df = None
        self.user_profiles = {}
        self.cold_start_cache = {}
        
    def fit(self, movies_df: pd.DataFrame):
        """
        Fit the recommender using movie genres and titles.
        
        Args:
            movies_df: DataFrame with movie information
        """
        self.movies_df = movies_df
        
        # Create title features
        titles = movies_df['title'].str.replace(r'\(\d{4}\)', '', regex=True)
        self.title_features = self.title_vectorizer.fit_transform(titles)
        
        # Create genre features
        genres = movies_df['genres'].str.replace('|', ' ')
        self.genre_features = self.genre_vectorizer.fit_transform(genres)
        
        # Initialize cold start recommendations
        self._initialize_cold_start_recommendations()
        
    def _initialize_cold_start_recommendations(self):
        """Initialize recommendations for cold start cases based on popularity and diversity."""
        try:
            # Get popular movies from different genres
            genres = self.movies_df['genres'].str.split('|').explode().unique()
            
            for genre in genres:
                # Get top rated movies for each genre
                genre_movies = self.movies_df[self.movies_df['genres'].str.contains(genre, na=False)]
                top_genre_movies = genre_movies.head(5)[['movie_id', 'title']]
                self.cold_start_cache[genre] = [
                    (row['movie_id'], row['title'], 1.0) 
                    for _, row in top_genre_movies.iterrows()
                ]
            
            # Get overall popular movies
            self.cold_start_cache['popular'] = [
                (row['movie_id'], row['title'], 1.0)
                for _, row in self.movies_df.head(20).iterrows()
            ]
            
        except Exception as e:
            logger.error(f"Error initializing cold start recommendations: {str(e)}")
    
    def create_user_profile(self, user_ratings: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a user profile based on their rated movies.
        
        Args:
            user_ratings: DataFrame with user's ratings
            
        Returns:
            Tuple of title and genre profile vectors
        """
        try:
            title_profile = np.zeros(self.title_features.shape[1])
            genre_profile = np.zeros(self.genre_features.shape[1])
            total_weight = 0
            
            for _, row in user_ratings.iterrows():
                try:
                    movie_idx = self.movies_df[self.movies_df['movie_id'] == row['movie_id']].index[0]
                    # Scale rating to [-1, 1] and only consider positive ratings
                    weight = (row['rating'] - 2.5) / 2.5
                    if weight > 0:
                        title_profile += weight * self.title_features[movie_idx].toarray()[0]
                        genre_profile += weight * self.genre_features[movie_idx].toarray()[0]
                        total_weight += weight
                except IndexError:
                    continue
            
            if total_weight > 0:
                title_profile /= total_weight
                genre_profile /= total_weight
            
            return title_profile, genre_profile
            
        except Exception as e:
            logger.error(f"Error creating user profile: {str(e)}")
            return np.zeros(self.title_features.shape[1]), np.zeros(self.genre_features.shape[1])
    
    def get_recommendations(self, user_id: int, ratings_df: pd.DataFrame, n: int = 10) -> List[Tuple[int, str, float]]:
        """
        Get personalized recommendations for a user based on their profile.
        
        Args:
            user_id: ID of the user
            ratings_df: DataFrame with ratings information
            n: Number of recommendations
            
        Returns:
            List of (movie_id, title, similarity_score) tuples
        """
        try:
            # Get user's ratings
            user_ratings = ratings_df[ratings_df['user_id'] == user_id]
            
            # Handle cold start
            if len(user_ratings) < 5:
                return self._get_cold_start_recommendations(user_ratings)
            
            # Create or get user profile
            if user_id not in self.user_profiles:
                title_profile, genre_profile = self.create_user_profile(user_ratings)
                self.user_profiles[user_id] = (title_profile, genre_profile)
            else:
                title_profile, genre_profile = self.user_profiles[user_id]
            
            # Calculate similarity scores
            title_similarity = cosine_similarity(
                title_profile.reshape(1, -1),
                self.title_features
            ).flatten()
            
            genre_similarity = cosine_similarity(
                genre_profile.reshape(1, -1),
                self.genre_features
            ).flatten()
            
            # Combine similarities with weights
            similarity_scores = 0.3 * title_similarity + 0.7 * genre_similarity
            
            # Get movies user hasn't rated
            rated_movies = set(user_ratings['movie_id'])
            available_movies = [
                (idx, row['movie_id'], row['title'], similarity_scores[idx])
                for idx, row in self.movies_df.iterrows()
                if row['movie_id'] not in rated_movies
            ]
            
            # Sort by similarity and get top N
            recommendations = sorted(
                available_movies,
                key=lambda x: x[3],
                reverse=True
            )[:n]
            
            return [(mid, title, score) for _, mid, title, score in recommendations]
            
        except Exception as e:
            logger.error(f"Error getting content-based recommendations: {str(e)}")
            return []
    
    def _get_cold_start_recommendations(self, user_ratings: pd.DataFrame) -> List[Tuple[int, str, float]]:
        """Get recommendations for new users based on available information."""
        try:
            if len(user_ratings) == 0:
                # Completely new user: return popular movies
                return self.cold_start_cache['popular']
            
            # Get genres from the few rated movies
            rated_genres = set()
            for _, row in user_ratings.iterrows():
                movie = self.movies_df[self.movies_df['movie_id'] == row['movie_id']]
                if not movie.empty:
                    genres = movie.iloc[0]['genres'].split('|')
                    rated_genres.update(genres)
            
            # Get recommendations from each liked genre
            recommendations = []
            for genre in rated_genres:
                if genre in self.cold_start_cache:
                    recommendations.extend(self.cold_start_cache[genre][:2])
            
            # Fill remaining slots with popular movies
            if len(recommendations) < 10:
                remaining = 10 - len(recommendations)
                popular_recs = [
                    rec for rec in self.cold_start_cache['popular']
                    if rec not in recommendations
                ][:remaining]
                recommendations.extend(popular_recs)
            
            return recommendations[:10]
            
        except Exception as e:
            logger.error(f"Error getting cold start recommendations: {str(e)}")
            return self.cold_start_cache['popular'][:10]

class SwitchedHybridRecommender:
    """Switched hybrid recommender that chooses between different recommendation algorithms."""
    
    def __init__(self, collaborative_models: Dict, content_model: ContentBasedRecommender,
                 rating_threshold: int = 5, similarity_threshold: float = 0.3):
        """
        Initialize the switched hybrid recommender.
        
        Args:
            collaborative_models: Dictionary of collaborative filtering models
            content_model: Content-based recommender model
            rating_threshold: Minimum number of ratings needed for collaborative filtering
            similarity_threshold: Minimum similarity score for recommendations
        """
        self.collaborative_models = collaborative_models
        self.content_model = content_model
        self.rating_threshold = rating_threshold
        self.similarity_threshold = similarity_threshold
        self.model_weights = {
            'User-based CF': 0.4,
            'Item-based CF': 0.3,
            'SVD': 0.3
        }
    
    def fit(self, ratings: pd.DataFrame, movies: pd.DataFrame, surprise_data: Dataset):
        """Fit all component models."""
        try:
            # Initialize and fit collaborative models if not already fitted
            for model_name, model in self.collaborative_models.items():
                if not hasattr(model, 'is_fitted') or not model.is_fitted:
                    model.fit(surprise_data)
                    model.is_fitted = True
            
            # Fit content-based model if not already fitted
            if not hasattr(self.content_model, 'movies_df'):
                self.content_model.fit(movies)
            
            logger.info("Fitted all component models in switched hybrid recommender")
            
        except Exception as e:
            logger.error(f"Error fitting switched hybrid recommender: {str(e)}")
    
    def get_recommendations(self, user_id: int, ratings_df: pd.DataFrame, movies_df: pd.DataFrame,
                          n: int = 10) -> List[Tuple[int, str, float]]:
        """
        Get recommendations using the most appropriate algorithm.
        
        Args:
            user_id: ID of the user
            ratings_df: DataFrame with ratings
            movies_df: DataFrame with movies
            n: Number of recommendations to return
            
        Returns:
            List of (movie_id, title, similarity_score) tuples
        """
        try:
            # Get user's ratings
            user_ratings = ratings_df[ratings_df['user_id'] == user_id]
            
            if len(user_ratings) < self.rating_threshold:
                # Use content-based recommendations for new users
                logger.info(f"Using content-based recommendations for user {user_id} (insufficient ratings)")
                return self.content_model.get_recommendations(user_id, ratings_df, n)
            
            # Get recommendations from each collaborative model
            cf_recommendations = {}
            for model_name, model in self.collaborative_models.items():
                try:
                    recs = model.get_top_n_recommendations(user_id, movies_df, n=n)
                    if recs:
                        cf_recommendations[model_name] = recs
                except Exception as e:
                    logger.error(f"Error getting {model_name} recommendations: {str(e)}")
            
            if not cf_recommendations:
                # Fallback to content-based if collaborative filtering fails
                logger.info(f"Falling back to content-based recommendations for user {user_id}")
                return self.content_model.get_recommendations(user_id, ratings_df, n)
            
            # Combine collaborative filtering recommendations with weights
            movie_scores = {}
            for model_name, recs in cf_recommendations.items():
                weight = self.model_weights.get(model_name, 1.0 / len(cf_recommendations))
                for movie_id, title, score in recs:
                    if movie_id not in movie_scores:
                        movie_scores[movie_id] = {'title': title, 'score': 0}
                    movie_scores[movie_id]['score'] += score * weight
            
            # Get content-based recommendations for diversity
            content_recs = self.content_model.get_recommendations(user_id, ratings_df, n)
            for movie_id, title, score in content_recs:
                if movie_id not in movie_scores:
                    movie_scores[movie_id] = {'title': title, 'score': score * 0.2}  # Lower weight for diversity
            
            # Sort and return top N recommendations
            sorted_recs = sorted(
                [(mid, data['title'], data['score']) 
                 for mid, data in movie_scores.items()],
                key=lambda x: x[2],
                reverse=True
            )
            
            return sorted_recs[:n]
            
        except Exception as e:
            logger.error(f"Error in switched hybrid recommender: {str(e)}")
            # Fallback to content-based recommendations
            return self.content_model.get_recommendations(user_id, ratings_df, n)
    
    def explain_recommendations(self, user_id: int, recommendations: List[Tuple[int, str, float]], 
                              ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> Dict:
        """
        Provide explanations for recommendations.
        
        Args:
            user_id: ID of the user
            recommendations: List of recommendations to explain
            ratings_df: DataFrame with ratings
            movies_df: DataFrame with movies
            
        Returns:
            Dictionary with explanations for each recommendation
        """
        try:
            explanations = {}
            user_ratings = ratings_df[ratings_df['user_id'] == user_id]
            
            for movie_id, title, score in recommendations:
                movie = movies_df[movies_df['movie_id'] == movie_id].iloc[0]
                
                # Get similar movies that the user has rated
                similar_rated = []
                for _, row in user_ratings.iterrows():
                    rated_movie = movies_df[movies_df['movie_id'] == row['movie_id']].iloc[0]
                    if any(g in rated_movie['genres'] for g in movie['genres'].split('|')):
                        similar_rated.append({
                            'title': rated_movie['title'],
                            'rating': row['rating'],
                            'genres': rated_movie['genres']
                        })
                
                explanation = {
                    'score': score,
                    'genres': movie['genres'].split('|'),
                    'similar_rated': similar_rated[:3],  # Top 3 similar movies
                    'reason': self._generate_recommendation_reason(
                        movie, similar_rated, user_ratings, score
                    )
                }
                
                explanations[movie_id] = explanation
            
            return explanations
            
        except Exception as e:
            logger.error(f"Error generating explanations: {str(e)}")
            return {}
    
    def _generate_recommendation_reason(self, movie: pd.Series, similar_rated: List[Dict],
                                     user_ratings: pd.DataFrame, score: float) -> str:
        """Generate a human-readable explanation for a recommendation."""
        try:
            reasons = []
            
            # Genre-based reason
            if similar_rated:
                liked_genres = set()
                for item in similar_rated:
                    if item['rating'] >= 4:
                        liked_genres.update(item['genres'].split('|'))
                common_genres = set(movie['genres'].split('|')) & liked_genres
                if common_genres:
                    reasons.append(
                        f"You've enjoyed {len(similar_rated)} movies in the "
                        f"{', '.join(common_genres)} genre(s)"
                    )
            
            # Rating pattern reason
            if len(user_ratings) >= 5:
                avg_rating = user_ratings['rating'].mean()
                if score > avg_rating:
                    reasons.append(
                        f"This movie's predicted rating ({score:.1f}) is higher than "
                        f"your average rating ({avg_rating:.1f})"
                    )
            
            # Popularity reason
            if not reasons:
                reasons.append("This is a popular movie that matches your viewing history")
            
            return " and ".join(reasons) + "."
            
        except Exception as e:
            logger.error(f"Error generating recommendation reason: {str(e)}")
            return "Based on your viewing history."

# Update the CSS section at the top of the file
st.markdown("""
<style>
    /* Main theme colors and variables */
    :root {
        --primary-color: #1E88E5;
        --secondary-color: #FFA000;
        --background-dark: #121212;
        --text-light: #FFFFFF;
        --text-gray: #9E9E9E;
        --success-color: #4CAF50;
        --warning-color: #FFC107;
        --error-color: #F44336;
    }

    /* Global styles */
    .stApp {
        background-color: var(--background-dark);
        color: var(--text-light);
    }

    /* Enhanced Navigation bar */
    .nav-bar {
        background: linear-gradient(135deg, var(--background-dark) 0%, #1a1a1a 100%);
        padding: 1.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 2rem;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }

    .nav-logo {
        font-size: 1.8rem;
        font-weight: bold;
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }

    /* Enhanced Movie Card Styling */
    .movie-card {
        position: relative;
        width: 100%;
        aspect-ratio: 2/3;
        border-radius: 12px;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        background: linear-gradient(135deg, #1a1a1a 0%, #0a0a0a 100%);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    
    .movie-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 12px 24px rgba(0,0,0,0.3);
        z-index: 2;
    }
    
    .movie-poster-container {
        position: relative;
        width: 100%;
        height: 100%;
        overflow: hidden;
    }
    
    .movie-poster {
        width: 100%;
        height: 100%;
        object-fit: cover;
        border-radius: 12px;
        transition: transform 0.4s ease;
    }
    
    .movie-card:hover .movie-poster {
        transform: scale(1.1);
    }
    
    .movie-overlay {
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 1.5rem;
        background: linear-gradient(transparent, rgba(0,0,0,0.95));
        color: white;
        opacity: 0;
        transform: translateY(20px);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .movie-card:hover .movie-overlay {
        opacity: 1;
        transform: translateY(0);
    }
    
    .movie-title-overlay {
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 0.8rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.4);
    }
    
    .movie-metadata {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.8rem;
        font-size: 1rem;
    }
    
    .movie-year {
        color: var(--text-gray);
        font-weight: 500;
    }
    
    .movie-rating {
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    
    .ml-rating {
        color: var(--success-color);
        font-weight: bold;
    }
    
    .movie-genres {
        font-size: 0.9rem;
        color: var(--text-gray);
        margin-bottom: 1rem;
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    
    .genre-tag {
        background: rgba(255,255,255,0.1);
        padding: 0.3rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        transition: background-color 0.3s ease;
    }
    
    .genre-tag:hover {
        background: rgba(255,255,255,0.2);
    }
    
    .movie-actions {
        display: flex;
        gap: 0.8rem;
        margin-top: 1rem;
    }
    
    .action-button {
        padding: 0.6rem 1.2rem;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        font-size: 0.9rem;
        font-weight: 500;
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    
    .primary-button {
        background-color: var(--primary-color);
        color: white;
    }
    
    .secondary-button {
        background-color: rgba(255,255,255,0.1);
        color: white;
    }
    
    .action-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    
    .primary-button:hover {
        background-color: #1976D2;
    }
    
    .secondary-button:hover {
        background-color: rgba(255,255,255,0.2);
    }

    /* Enhanced Filter Section */
    .filter-section {
        background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }

    .filter-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: var(--text-light);
    }

    .filter-group {
        margin-bottom: 1.5rem;
    }

    .filter-label {
        font-size: 0.9rem;
        color: var(--text-gray);
        margin-bottom: 0.5rem;
    }

    /* Enhanced Pagination */
    .pagination {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 1rem;
        margin: 2rem 0;
    }

    .page-button {
        background: rgba(255,255,255,0.1);
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        color: var(--text-light);
        cursor: pointer;
        transition: all 0.3s ease;
    }

    .page-button:hover {
        background: rgba(255,255,255,0.2);
        transform: translateY(-2px);
    }

    .page-button:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }

    /* Loading Animation */
    @keyframes pulse {
        0% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.1); opacity: 0.7; }
        100% { transform: scale(1); opacity: 1; }
    }

    .loading {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }

    .loading-icon {
        font-size: 2rem;
        animation: pulse 1.5s ease-in-out infinite;
    }

    /* Enhanced Tooltips */
    [data-tooltip] {
        position: relative;
        cursor: help;
    }

    [data-tooltip]:before {
        content: attr(data-tooltip);
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        padding: 0.5rem 1rem;
        background: rgba(0,0,0,0.8);
        color: white;
        border-radius: 4px;
        font-size: 0.8rem;
        white-space: nowrap;
        opacity: 0;
        visibility: hidden;
        transition: all 0.3s ease;
    }

    [data-tooltip]:hover:before {
        opacity: 1;
        visibility: visible;
        transform: translateX(-50%) translateY(-8px);
    }

    /* Smooth Transitions */
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Enhanced Metrics Display */
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-label {
        font-size: 0.9rem;
        color: var(--text-gray);
        margin-top: 0.5rem;
    }

    /* Enhanced Footer */
    .footer {
        margin-top: 4rem;
        padding: 2rem;
        background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
        border-radius: 12px;
        text-align: center;
    }

    .footer-logo {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .footer-text {
        color: var(--text-gray);
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def create_movie_card(movie_id: int, title: str, genres: str, rating: float, year: int = None) -> str:
    """Create an enhanced movie card with hover effects and interaction buttons."""
    return f"""
    <div class="movie-card" 
         data-movie-id="{movie_id}"
         data-title="{title}"
         data-year="{year if year else 'N/A'}"
         data-rating="{rating}"
         data-genres="{genres}"
         onclick="showMovieInfo(this.dataset)">
        <div class="movie-poster-container">
            <div class="movie-poster-placeholder">
                <h3>{title}</h3>
                <p>{year if year else 'N/A'}</p>
            </div>
            <div class="movie-overlay">
                <div class="movie-details">
                    <h3 class="movie-title-overlay">{title}</h3>
                    <div class="movie-metadata">
                        <span class="movie-year">{year if year else 'N/A'}</span>
                        <div class="movie-rating">
                            <span class="ml-rating">{'⭐' * int(round(rating))} {rating:.1f}</span>
                        </div>
                    </div>
                    <div class="movie-genres">
                        {' '.join([f'{GENRE_ICONS.get(g.strip(), "🎬")} {g.strip()}' for g in genres.split('|')])}
                    </div>
                    <div class="movie-actions">
                        <button class="info-button" onclick="event.stopPropagation(); toggleFavorite({movie_id})">
                            <span id="fav-{movie_id}">🤍</span>
                        </button>
                        <button class="info-button" onclick="event.stopPropagation(); showWhyRecommended({movie_id})">
                            Why?
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """

def display_recommendations(recommendations: List[Tuple], movies_df: pd.DataFrame, ratings_df: pd.DataFrame, recommenders: dict, selected_genres: List[str] = None, year_range: Tuple[int, int] = None):
    """Display movie recommendations in a Netflix-style grid layout."""
    # Add JavaScript for movie info modal
    st.markdown("""
    <script>
    function showMovieInfo(movieData) {
        // Create modal container
        const modal = document.createElement('div');
        modal.className = 'movie-modal';
        
        // Create modal content
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h2 class="modal-title">${movieData.title}</h2>
                    <span class="close-button" onclick="closeModal(this)">&times;</span>
                </div>
                <div class="modal-body">
                    <div class="movie-detail-grid">
                        <div class="movie-info-section">
                            <div class="movie-metadata-grid">
                                <div class="metadata-item">
                                    <span class="metadata-label">Year</span>
                                    <div class="metadata-value">${movieData.year}</div>
                                </div>
                                <div class="metadata-item">
                                    <span class="metadata-label">Rating</span>
                                    <div class="metadata-value">${parseFloat(movieData.rating).toFixed(1)}/5.0</div>
                                </div>
                                <div class="metadata-item">
                                    <span class="metadata-label">Genres</span>
                                    <div class="metadata-value">${movieData.genres}</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Add modal to document
        document.body.appendChild(modal);
        
        // Show modal with animation
        requestAnimationFrame(() => {
            modal.style.display = 'block';
            modal.style.opacity = '0';
            requestAnimationFrame(() => {
                modal.style.opacity = '1';
            });
        });
        
        // Add event listeners
        modal.addEventListener('click', (event) => {
            if (event.target === modal) {
                closeModal(modal);
            }
        });
        
        // Add keyboard event listener
        document.addEventListener('keydown', handleKeyPress);
    }
    
    function closeModal(element) {
        const modal = element.closest('.movie-modal');
        modal.style.opacity = '0';
        setTimeout(() => {
            modal.remove();
            // Remove keyboard event listener
            document.removeEventListener('keydown', handleKeyPress);
        }, 300);
    }
    
    function handleKeyPress(event) {
        if (event.key === 'Escape') {
            const modals = document.querySelectorAll('.movie-modal');
            modals.forEach(modal => closeModal(modal));
        }
    }
    </script>
    """, unsafe_allow_html=True)

    if not recommendations:
        st.warning("No recommendations found. Try different filters or another user ID.")
        return

    # Add movie search functionality
    st.markdown("""
        <style>
        .search-container {
            margin: 20px 0;
            padding: 20px;
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="search-container">', unsafe_allow_html=True)
        search_query = st.text_input("🔍 Search Movies", placeholder="Enter movie title, genre, or year...")
        st.markdown('</div>', unsafe_allow_html=True)

    # Filter recommendations based on search
    filtered_recommendations = []
    for movie_id, title, score in recommendations:
        try:
            movie_info = movies_df[movies_df['movie_id'] == movie_id].iloc[0]
            movie_genres = movie_info['genres'].split('|')
            
            # Extract year from title
            year_match = re.search(r'\((\d{4})\)', title)
            year = int(year_match.group(1)) if year_match else None
            
            # Apply search filter
            if search_query:
                search_lower = search_query.lower()
                if not (search_lower in title.lower() or 
                       search_lower in movie_info['genres'].lower() or 
                       (year and str(year) in search_lower)):
                    continue
            
            # Apply genre and year filters
            if ((not selected_genres or any(genre in movie_genres for genre in selected_genres)) and
                (not year_range or (year and year_range[0] <= year <= year_range[1]))):
                filtered_recommendations.append((movie_id, title, score, movie_info['genres'], year))
        except Exception as e:
            logger.error(f"Error processing movie {movie_id}: {str(e)}")
            continue

    # Display recommendations by genre
    st.markdown('<div class="netflix-container">', unsafe_allow_html=True)
    
    # Display trending/top picks first
    top_picks = filtered_recommendations[:6]
    if top_picks:
        st.markdown('<div class="movie-section">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-title">🔥 Top Picks for You</h2>', unsafe_allow_html=True)
        st.markdown('<div class="movie-row">', unsafe_allow_html=True)
        
        for movie_id, title, score, genres, year in top_picks:
            try:
                st.markdown(
                    create_movie_card(
                        movie_id=movie_id,
                        title=title,
                        genres=genres,
                        rating=score,
                        year=year
                    ),
                    unsafe_allow_html=True
                )
            except Exception as e:
                logger.error(f"Error displaying movie {movie_id}: {str(e)}")
                continue
        
        st.markdown('</div></div>', unsafe_allow_html=True)

    # Group recommendations by genre
    genre_groups = {}
    for movie_id, title, score, genres, year in filtered_recommendations:
        for genre in genres.split('|'):
            if genre not in genre_groups:
                genre_groups[genre] = []
            genre_groups[genre].append((movie_id, title, score, genres, year))

    # Display recommendations by genre
    for genre, movies in genre_groups.items():
        if len(movies) >= 4:  # Only show genres with enough movies
            st.markdown('<div class="movie-section">', unsafe_allow_html=True)
            st.markdown(f'<h2 class="section-title">{genre} Movies</h2>', unsafe_allow_html=True)
            st.markdown('<div class="movie-row">', unsafe_allow_html=True)
            
            for movie_id, title, score, genres, year in movies[:6]:  # Show up to 6 movies per row
                try:
                    st.markdown(
                        create_movie_card(
                            movie_id=movie_id,
                            title=title,
                            genres=genres,
                            rating=score,
                            year=year
                        ),
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    logger.error(f"Error displaying movie {movie_id}: {str(e)}")
                    continue
            
            st.markdown('</div></div>', unsafe_allow_html=True)
    
    if not filtered_recommendations:
        st.warning("No movies found matching your search criteria.")
    
    st.markdown('</div>', unsafe_allow_html=True)

def get_most_common_genres(recommendations: List[Tuple], movies_df: pd.DataFrame, top_n: int = 3) -> List[str]:
    """Get the most common genres in the recommendations."""
    all_genres = []
    for movie_id, title, score, genres, year in recommendations:  # Updated tuple unpacking
        all_genres.extend(genres.split('|'))
    genre_counts = pd.Series(all_genres).value_counts()
    return genre_counts.head(top_n).index.tolist()

def create_navigation_bar():
    """Create the navigation bar."""
    st.markdown("""
    <div class="nav-bar">
        <div class="nav-logo">🎬 MovieLens Recommender</div>
    </div>
    """, unsafe_allow_html=True)

def get_user_favorite_genre(user_id: int, ratings: pd.DataFrame, movies: pd.DataFrame) -> str:
    """Get the most frequently rated genre for a user."""
    try:
        # Get user's rated movies
        user_movies = ratings[ratings['user_id'] == user_id].merge(
            movies, on='movie_id', how='inner'
        )
        
        # Split genres and count occurrences
        genre_counts = pd.Series([
            genre for genres in user_movies['genres'].str.split('|')
            for genre in genres
        ]).value_counts()
        
        return genre_counts.index[0] if not genre_counts.empty else "Unknown"
    except Exception as e:
        logger.error(f"Error getting user's favorite genre: {str(e)}")
        return "Unknown"

def get_user_cf_recommendations(user_id: int, n: int = 10) -> List[Tuple[int, str, float]]:
    """Get recommendations using User-based CF."""
    try:
        return user_cf_model.get_recommendations(user_id, n)
    except Exception as e:
        logger.error(f"Error getting User-CF recommendations: {str(e)}")
        return []

def get_item_cf_recommendations(user_id: int, n: int = 10) -> List[Tuple[int, str, float]]:
    """Get recommendations using Item-based CF."""
    try:
        return item_cf_model.get_recommendations(user_id, n)
    except Exception as e:
        logger.error(f"Error getting Item-CF recommendations: {str(e)}")
        return []

def get_svd_recommendations(user_id: int, n: int = 10) -> List[Tuple[int, str, float]]:
    """Get recommendations using SVD."""
    try:
        return svd_model.get_recommendations(user_id, n)
    except Exception as e:
        logger.error(f"Error getting SVD recommendations: {str(e)}")
        return []

def get_hybrid_recommendations(user_id: int, n: int = 10) -> List[Tuple[int, str, float]]:
    """Get recommendations using Hybrid approach."""
    try:
        # Combine recommendations from all models with weights
        user_cf_recs = get_user_cf_recommendations(user_id, n)
        item_cf_recs = get_item_cf_recommendations(user_id, n)
        svd_recs = get_svd_recommendations(user_id, n)
        
        # Create a weighted score for each movie
        movie_scores = {}
        weights = {'user_cf': 0.4, 'item_cf': 0.3, 'svd': 0.3}
        
        for movie_id, title, score in user_cf_recs:
            if movie_id not in movie_scores:
                movie_scores[movie_id] = {'title': title, 'score': 0}
            movie_scores[movie_id]['score'] += score * weights['user_cf']
            
        for movie_id, title, score in item_cf_recs:
            if movie_id not in movie_scores:
                movie_scores[movie_id] = {'title': title, 'score': 0}
            movie_scores[movie_id]['score'] += score * weights['item_cf']
            
        for movie_id, title, score in svd_recs:
            if movie_id not in movie_scores:
                movie_scores[movie_id] = {'title': title, 'score': 0}
            movie_scores[movie_id]['score'] += score * weights['svd']
        
        # Sort by weighted score and return top N
        sorted_movies = sorted(
            [(mid, data['title'], data['score']) 
             for mid, data in movie_scores.items()],
            key=lambda x: x[2],
            reverse=True
        )
        
        return sorted_movies[:n]
        
    except Exception as e:
        logger.error(f"Error getting Hybrid recommendations: {str(e)}")
        return []

@st.cache_data(ttl=3600)
def load_data() -> tuple:
    """Load and preprocess data with caching."""
    try:
        # Load ratings data
        ratings_data = pd.read_csv('data/ml-1m/ratings.dat', 
                                 sep='::', 
                                 names=['user_id', 'movie_id', 'rating', 'timestamp'],
                                 engine='python',
                                 encoding='latin-1')
        
        # Load movies data
        movies_data = pd.read_csv('data/ml-1m/movies.dat',
                                sep='::',
                                names=['movie_id', 'title', 'genres'],
                                engine='python',
                                encoding='latin-1')
        
        # Convert timestamp to datetime
        ratings_data['timestamp'] = pd.to_datetime(ratings_data['timestamp'], unit='s')
        
        # Extract year from title
        movies_data['year'] = movies_data['title'].str.extract(r'\((\d{4})\)').astype('float')
        movies_data['title_no_year'] = movies_data['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True)
        
        # Create Surprise reader object
        reader = Reader(rating_scale=(1, 5))
        
        # Create Surprise dataset
        surprise_data = Dataset.load_from_df(
            ratings_data[['user_id', 'movie_id', 'rating']], 
            reader
        )
        
        return ratings_data, movies_data, surprise_data
        
    except FileNotFoundError:
        logger.error("Data files not found. Please ensure the data files exist in the data/ml-1m directory.")
        st.error("Data files not found. Please ensure you have downloaded the MovieLens 1M dataset and placed the files in the data/ml-1m directory.")
        return None, None, None
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        st.error("Error loading data. Please check if the data files exist and are accessible.")
        return None, None, None

@st.cache_data(ttl=3600)
def initialize_recommenders(ratings: pd.DataFrame, movies: pd.DataFrame, _surprise_data: Dataset) -> Dict:
    """Initialize all recommendation models with caching."""
    try:
        logger.info("Initializing recommendation models...")
        recommenders = {}
        
        # Initialize Trivial Recommender
        trivial_model = TrivialRecommender(min_ratings=100)
        trivial_model.fit(ratings, movies)
        recommenders['Trivial'] = trivial_model
        logger.info("Initialized Trivial Recommender")
        
        # Initialize MPI Recommender
        mpi_model = MPIRecommender()
        mpi_model.fit(ratings, movies)
        recommenders['MPI'] = mpi_model
        logger.info("Initialized MPI Recommender")
        
        # Initialize Content-based Recommender
        content_model = ContentBasedRecommender()
        content_model.fit(movies)
        recommenders['Content-based'] = content_model
        logger.info("Initialized Content-based Recommender")
        
        # Initialize User-based CF
        user_cf_model = UserBasedCF(k=40, min_k=5)
        user_cf_model.fit(_surprise_data)
        recommenders['User-based CF'] = user_cf_model
        logger.info("Initialized User-based CF")
        
        # Initialize Item-based CF
        item_cf_model = ItemBasedCF(k=40, min_k=5)
        item_cf_model.fit(_surprise_data)
        recommenders['Item-based CF'] = item_cf_model
        logger.info("Initialized Item-based CF")
        
        # Initialize SVD
        svd_model = SVDRecommender(n_factors=100, n_epochs=20)
        svd_model.fit(_surprise_data)
        recommenders['SVD'] = svd_model
        logger.info("Initialized SVD")
        
        # Initialize Switched Hybrid Recommender
        collaborative_models = {
            'User-based CF': user_cf_model,
            'Item-based CF': item_cf_model,
            'SVD': svd_model
        }
        switched_hybrid = SwitchedHybridRecommender(
            collaborative_models=collaborative_models,
            content_model=content_model,
            rating_threshold=5,
            similarity_threshold=0.3
        )
        switched_hybrid.fit(ratings, movies, _surprise_data)
        recommenders['Switched Hybrid'] = switched_hybrid
        logger.info("Initialized Switched Hybrid Recommender")
        
        return recommenders
        
    except Exception as e:
        logger.error(f"Error initializing recommenders: {str(e)}")
        st.error(f"Error initializing recommendation models: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def evaluate_all_models(ratings_data: pd.DataFrame, movies_df: pd.DataFrame, _recommenders: Dict, n_recommendations: int = 10) -> Dict[str, Dict[str, float]]:
    """
    Evaluate all recommendation models and cache the results.
    
    Args:
        ratings_data: DataFrame with user ratings
        movies_df: DataFrame with movie information
        _recommenders: Dictionary of recommender models (unhashable)
        n_recommendations: Number of recommendations to generate
        
    Returns:
        Dictionary of evaluation metrics for each model
    """
    try:
        metrics = {
            'User-Based CF': {'precision@k': 0.0, 'recall@k': 0.0, 'ndcg@k': 0.0, 'coverage': 0.0},
            'Item-Based CF': {'precision@k': 0.0, 'recall@k': 0.0, 'ndcg@k': 0.0, 'coverage': 0.0},
            'SVD': {'precision@k': 0.0, 'recall@k': 0.0, 'ndcg@k': 0.0, 'coverage': 0.0},
            'Hybrid': {'precision@k': 0.0, 'recall@k': 0.0, 'ndcg@k': 0.0, 'coverage': 0.0},
            'Content-based': {'precision@k': 0.0, 'recall@k': 0.0, 'ndcg@k': 0.0, 'coverage': 0.0},
            'Switched Hybrid': {'precision@k': 0.0, 'recall@k': 0.0, 'ndcg@k': 0.0, 'coverage': 0.0}
        }
        
        # Sample users for evaluation (use seed for reproducibility)
        np.random.seed(42)
        sample_users = np.random.choice(ratings_data['user_id'].unique(), size=100, replace=False)
        
        for user_id in sample_users:
            # Get user's actual ratings
            user_ratings = ratings_data[ratings_data['user_id'] == user_id]
            actual_items = set(user_ratings[user_ratings['rating'] >= 4]['movie_id'])
            
            if not actual_items:
                continue
                
            # Get recommendations from each model
            for model_name in metrics.keys():
                try:
                    if model_name == 'Hybrid':
                        recs = get_hybrid_recommendations(user_id, n_recommendations)
                    elif model_name in _recommenders:
                        if model_name == 'Content-based':
                            recs = _recommenders[model_name].get_recommendations(
                                user_id=user_id,
                                ratings_df=ratings_data,
                                n=n_recommendations
                            )
                        elif model_name == 'Switched Hybrid':
                            recs = _recommenders[model_name].get_recommendations(
                                user_id=user_id,
                                ratings_df=ratings_data,
                                movies_df=movies_df,
                                n=n_recommendations
                            )
                        else:
                            recs = _recommenders[model_name].get_top_n_recommendations(
                                user_id=user_id,
                                movies_df=movies_df,
                                n=n_recommendations
                            )
                    else:
                        logger.warning(f"Model {model_name} not found in recommenders")
                        continue
                    
                    if not recs:
                        continue
                    
                    predicted_items = set(movie_id for movie_id, _, _ in recs)
                    
                    # Calculate precision and recall
                    intersection = len(actual_items & predicted_items)
                    metrics[model_name]['precision@k'] += intersection / len(predicted_items) if predicted_items else 0
                    metrics[model_name]['recall@k'] += intersection / len(actual_items) if actual_items else 0
                    
                    # Calculate NDCG
                    relevance = np.zeros(len(predicted_items))
                    for i, (movie_id, _, _) in enumerate(recs):
                        if movie_id in actual_items:
                            relevance[i] = 1
                            
                    # Calculate DCG
                    dcg = np.sum(relevance / np.log2(np.arange(2, len(relevance) + 2)))
                    
                    # Calculate IDCG
                    ideal_length = min(len(actual_items), len(predicted_items))
                    ideal_relevance = np.ones(ideal_length)
                    idcg = np.sum(ideal_relevance / np.log2(np.arange(2, ideal_length + 2)))
                    
                    # Calculate final NDCG
                    metrics[model_name]['ndcg@k'] += dcg / idcg if idcg > 0 else 0
                    
                    # Calculate coverage
                    all_movies = set(ratings_data['movie_id'].unique())
                    metrics[model_name]['coverage'] += len(predicted_items) / len(all_movies)
                    
                except Exception as e:
                    logger.error(f"Error evaluating {model_name} for user {user_id}: {str(e)}")
                    continue
        
        # Average metrics
        n_users = len(sample_users)
        for model_name in metrics.keys():
            for metric in metrics[model_name].keys():
                metrics[model_name][metric] /= n_users
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}")
        return {}

def display_evaluation_metrics(metrics: Dict[str, Dict[str, float]]):
    """Display evaluation metrics in a formatted way."""
    if not metrics:
        st.error("No evaluation metrics available.")
        return
    
    # Create tabs for different metric visualizations
    tab1, tab2 = st.tabs(["📊 Metrics Overview", "📈 Detailed Comparison"])
    
    with tab1:
        # Create a summary table
        st.subheader("Model Performance Summary")
        
        # Convert metrics to a DataFrame for easier display
        df = pd.DataFrame.from_dict(metrics, orient='index')
        
        # Format metrics for display
        df = df.round(3) * 100  # Convert to percentages
        
        # Create a styled table
        st.dataframe(
            df.style.background_gradient(cmap='YlOrRd')
                   .format("{:.1f}%"),
            height=200
        )
        
        # Add description of metrics
        with st.expander("ℹ️ Metrics Explanation"):
            st.markdown("""
            - **Precision@K**: Percentage of recommended items that are relevant
            - **Recall@K**: Percentage of relevant items that are recommended
            - **NDCG@K**: Quality of ranking, considering position of relevant items
            - **Coverage**: Percentage of all items that can be recommended
            """)
    
    with tab2:
        # Create comparative bar charts
        st.subheader("Comparative Analysis")
        
        # Prepare data for plotting
        metrics_df = pd.DataFrame([
            {'Model': model, 'Metric': metric, 'Value': value * 100}
            for model, model_metrics in metrics.items()
            for metric, value in model_metrics.items()
        ])
        
        # Create bar chart using Plotly
        fig = px.bar(
            metrics_df,
            x='Model',
            y='Value',
            color='Metric',
            barmode='group',
            title='Model Performance Comparison',
            labels={'Value': 'Percentage (%)'}
        )
        
        fig.update_layout(
            template='plotly_dark',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add insights
        st.subheader("Key Insights")
        
        # Find best model for each metric
        best_models = {}
        for metric in metrics[list(metrics.keys())[0]].keys():
            best_model = max(metrics.items(), key=lambda x: x[1][metric])[0]
            best_value = metrics[best_model][metric] * 100
            best_models[metric] = (best_model, best_value)
        
        # Display insights
        cols = st.columns(2)
        with cols[0]:
            st.markdown("#### Best Performing Models")
            for metric, (model, value) in best_models.items():
                st.markdown(f"- **{metric}**: {model} ({value:.1f}%)")
        
        with cols[1]:
            st.markdown("#### Recommendations")
            st.markdown("""
            - Consider using the Hybrid model for balanced performance
            - User-Based CF shows good precision but lower coverage
            - SVD provides the best scalability for large datasets
            """)

def create_mode_selector() -> str:
    """Create an enhanced mode selection UI component."""
    st.markdown("""
        <style>
        .mode-container {
            display: flex;
            justify-content: space-around;
            padding: 1rem;
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .mode-option {
            display: flex;
            flex-direction: column;
            align-items: center;
            cursor: pointer;
            padding: 1rem;
            border-radius: 8px;
            transition: all 0.3s ease;
        }
        .mode-option:hover {
            background: rgba(255,255,255,0.1);
        }
        .mode-icon {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }
        .mode-label {
            font-size: 1rem;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        rec_selected = st.button("🎯 Get Recommendations", key="rec_mode", help="Get personalized movie recommendations")
    with col2:
        eval_selected = st.button("📊 Evaluation Dashboard", key="eval_mode", help="View algorithm performance metrics")
    
    return "Get Recommendations" if rec_selected else "Evaluation Dashboard" if eval_selected else "Get Recommendations"

def create_user_selector(ratings_df: pd.DataFrame) -> int:
    """Create an enhanced user selection UI component."""
    st.markdown("### 👤 User Profile Selection")
    
    # Create columns for different selection methods
    col1, col2 = st.columns(2)
    
    with col1:
        selection_method = st.radio(
            "Selection Method",
            ["Choose Random User", "Enter User ID"],
            help="Choose how you want to select a user"
        )
    
    if selection_method == "Choose Random User":
        # Get active users (users with more than 20 ratings)
        active_users = ratings_df['user_id'].value_counts()
        active_users = active_users[active_users >= 20].index.tolist()
        
        with col2:
            st.markdown("#### Active User Profile")
            if st.button("🎲 Generate Random User"):
                user_id = random.choice(active_users)
                st.session_state.selected_user_id = user_id
    else:
        with col2:
            user_id = st.number_input(
                "Enter User ID",
                min_value=1,
                max_value=ratings_df['user_id'].max(),
                value=1,
                help="Enter a specific user ID"
            )
            st.session_state.selected_user_id = user_id
    
    return st.session_state.get('selected_user_id', 1)

def create_enhanced_filters(movies_df: pd.DataFrame) -> Tuple[List[str], Tuple[float, float], Tuple[int, int]]:
    """Create enhanced filter UI components."""
    st.markdown("### 🎯 Recommendation Filters")
    
    # Genre filter with icons
    st.markdown("#### 🎭 Genre Selection")
    all_genres = sorted(list(set(
        genre for genres in movies_df['genres'].str.split('|') 
        for genre in genres
    )))
    
    # Create genre buttons with icons
    genre_cols = st.columns(4)
    selected_genres = []
    
    for i, genre in enumerate(all_genres):
        with genre_cols[i % 4]:
            icon = GENRE_ICONS.get(genre, '🎬')
            if st.checkbox(f"{icon} {genre}", key=f"genre_{genre}"):
                selected_genres.append(genre)
    
    # Rating filter with dynamic label
    st.markdown("#### ⭐ Rating Range")
    rating_range = st.slider(
        "Select rating range",
        1.0, 5.0, (3.0, 5.0),
        step=0.5,
        help="Filter movies by rating range"
    )
    st.markdown(f"Current range: **{rating_range[0]:.1f}** - **{rating_range[1]:.1f}** stars")
    
    # Year filter with dynamic label
    st.markdown("#### 📅 Release Year")
    min_year = int(movies_df['year'].min())
    max_year = int(movies_df['year'].max())
    year_range = st.slider(
        "Select release year range",
        min_year, max_year, (min_year, max_year),
        help="Filter movies by release year"
    )
    st.markdown(f"Selected years: **{year_range[0]}** - **{year_range[1]}**")
    
    return selected_genres, rating_range, year_range

def create_pagination_controls(total_items: int, items_per_page: int = 20) -> Tuple[int, int]:
    """Create pagination controls for movie recommendations."""
    total_pages = (total_items + items_per_page - 1) // items_per_page
    
    if total_pages > 1:
        st.markdown("### 📄 Page Navigation")
        col1, col2, col3 = st.columns([2, 3, 2])
        
        with col1:
            if st.button("⬅️ Previous", key="prev_page"):
                st.session_state.current_page = max(1, st.session_state.get('current_page', 1) - 1)
        
        with col2:
            current_page = st.selectbox(
                "Page",
                range(1, total_pages + 1),
                index=st.session_state.get('current_page', 1) - 1,
                key="page_select"
            )
            st.session_state.current_page = current_page
        
        with col3:
            if st.button("Next ➡️", key="next_page"):
                st.session_state.current_page = min(total_pages, st.session_state.get('current_page', 1) + 1)
        
        start_idx = (st.session_state.current_page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, total_items)
        
        return start_idx, end_idx
    
    return 0, min(items_per_page, total_items)

def add_interaction_javascript():
    """Add JavaScript for movie card interactions."""
    st.markdown("""
    <script>
    // Favorite movies storage
    let favoriteMovies = new Set(JSON.parse(localStorage.getItem('favoriteMovies') || '[]'));
    
    function toggleFavorite(movieId) {
        event.stopPropagation();
        const favIcon = document.getElementById(`fav-${movieId}`);
        if (favoriteMovies.has(movieId)) {
            favoriteMovies.delete(movieId);
            favIcon.textContent = '🤍';
        } else {
            favoriteMovies.add(movieId);
            favIcon.textContent = '❤️';
        }
        localStorage.setItem('favoriteMovies', JSON.stringify([...favoriteMovies]));
    }
    
    function showWhyRecommended(movieId) {
        event.stopPropagation();
        // Send message to Streamlit
        window.parent.postMessage({
            type: 'showWhyRecommended',
            movieId: movieId
        }, '*');
    }
    
    // Initialize favorite icons
    document.addEventListener('DOMContentLoaded', () => {
        favoriteMovies.forEach(movieId => {
            const favIcon = document.getElementById(`fav-${movieId}`);
            if (favIcon) favIcon.textContent = '❤️';
        });
    });
    </script>
    """, unsafe_allow_html=True)

def create_new_user_onboarding():
    """Create onboarding experience for new users."""
    st.markdown("### 🎬 Welcome to Movie Recommendations!")
    st.markdown("""
    To help us provide better recommendations, please tell us about your movie preferences:
    """)
    
    # Genre preferences
    st.markdown("#### 🎭 What genres do you enjoy?")
    genre_preferences = {}
    genre_cols = st.columns(3)
    for i, (genre, icon) in enumerate(GENRE_ICONS.items()):
        with genre_cols[i % 3]:
            genre_preferences[genre] = st.slider(
                f"{icon} {genre}",
                0, 5, 3,
                help=f"Rate how much you enjoy {genre} movies"
            )
    
    # Movie watching frequency
    st.markdown("#### 🎟️ How often do you watch movies?")
    frequency = st.radio(
        "",
        ["Rarely (1-2 movies/month)", "Sometimes (1-2 movies/week)", "Often (3+ movies/week)"]
    )
    
    # Preferred eras
    st.markdown("#### 📅 Which movie eras do you prefer?")
    eras = st.multiselect(
        "",
        ["Classic (pre-1970)", "Retro (1970-1990)", "Modern (1990-2010)", "Contemporary (2010-present)"],
        default=["Modern (1990-2010)", "Contemporary (2010-present)"]
    )
    
    return {
        "genre_preferences": genre_preferences,
        "watching_frequency": frequency,
        "preferred_eras": eras
    }

def explain_recommendation(movie_id: int, user_id: int, recommender_name: str, 
                         ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> str:
    """Generate an explanation for why a movie was recommended."""
    try:
        movie = movies_df[movies_df['movie_id'] == movie_id].iloc[0]
        user_ratings = ratings_df[ratings_df['user_id'] == user_id]
        
        explanation_parts = []
        
        # Genre-based explanation
        movie_genres = set(movie['genres'].split('|'))
        user_genre_ratings = {}
        
        for _, row in user_ratings.iterrows():
            rated_movie = movies_df[movies_df['movie_id'] == row['movie_id']].iloc[0]
            for genre in rated_movie['genres'].split('|'):
                if genre not in user_genre_ratings:
                    user_genre_ratings[genre] = []
                user_genre_ratings[genre].append(row['rating'])
        
        liked_genres = [
            genre for genre, ratings in user_genre_ratings.items()
            if np.mean(ratings) >= 3.5 and genre in movie_genres
        ]
        
        if liked_genres:
            explanation_parts.append(
                f"You've enjoyed {', '.join(liked_genres)} movies in the past"
            )
        
        # Rating pattern explanation
        if len(user_ratings) > 0:
            avg_rating = user_ratings['rating'].mean()
            if avg_rating >= 3.5:
                explanation_parts.append(
                    f"This movie matches your rating patterns (your average rating: {avg_rating:.1f})"
                )
        
        # Algorithm-specific explanation
        if recommender_name == "Content-based":
            explanation_parts.append(
                "This movie has similar content features to movies you've enjoyed"
            )
        elif recommender_name == "User-based CF":
            explanation_parts.append(
                "Users with similar taste enjoyed this movie"
            )
        elif recommender_name == "Item-based CF":
            explanation_parts.append(
                "This movie is similar to other movies you've rated highly"
            )
        
        # Combine explanations
        if explanation_parts:
            return " and ".join(explanation_parts) + "."
        else:
            return "This movie matches your general preferences."
            
    except Exception as e:
        logger.error(f"Error generating recommendation explanation: {str(e)}")
        return "Unable to generate explanation."

def main():
    """Main function for the Streamlit app."""
    # Create navigation bar
    create_navigation_bar()

    # Add JavaScript for interactions
    add_interaction_javascript()

    # Sidebar
    with st.sidebar:
        st.title("🎬 Controls")
        
        # Mode selection with enhanced UI
        mode = create_mode_selector()
        
        st.markdown("---")

        if mode == "Get Recommendations":
            # Load data with progress bar
            with st.spinner("📚 Loading MovieLens dataset..."):
                ratings, movies, surprise_data = load_data()
                if ratings is None or movies is None:
                    st.error("Failed to load data. Please check the data files.")
                    return

            # Initialize recommenders with progress
            with st.spinner("🎬 Initializing recommendation models..."):
                recommenders = initialize_recommenders(ratings, movies, surprise_data)
                if recommenders is None:
                    st.error("Failed to initialize recommenders.")
                    return

            # Create filter section container
            with st.container():
                st.subheader("🎯 Filters")
                
                # Enhanced user selection
                user_id = create_user_selector(ratings)
                
                # Show user statistics if available
                if user_id in ratings['user_id'].values:
                    user_ratings = ratings[ratings['user_id'] == user_id]
                    with st.expander("👤 User Profile", expanded=True):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Ratings", len(user_ratings))
                            st.metric("Average Rating", f"{user_ratings['rating'].mean():.2f}")
                        with col2:
                            favorite_genre = get_user_favorite_genre(user_id, ratings, movies)
                            st.metric("Favorite Genre", favorite_genre)
                            st.metric("Rating Range", f"{user_ratings['rating'].min():.1f} - {user_ratings['rating'].max():.1f}")
                
                # Algorithm selection with improved UI
                st.subheader("🤖 Algorithm")
                algorithm = st.selectbox(
                    "Choose Algorithm",
                    ["Switched Hybrid", "User-based CF", "Item-based CF", "SVD", "Content-based"],
                    help="Select the recommendation algorithm"
                )
                
                # Add algorithm description
                algorithm_descriptions = {
                    "Switched Hybrid": "Intelligently combines multiple algorithms for optimal recommendations",
                    "User-based CF": "Recommends movies based on similar users' preferences",
                    "Item-based CF": "Recommends movies similar to ones you've liked",
                    "SVD": "Uses matrix factorization to find hidden patterns",
                    "Content-based": "Recommends movies based on content features"
                }
                st.info(algorithm_descriptions[algorithm])
                
                # Enhanced filters with clear section headers
                st.markdown("### 🎭 Genre Filters")
                # Genre filter with icons
                all_genres = sorted(list(set(
                    genre for genres in movies['genres'].str.split('|') 
                    for genre in genres
                )))
                
                # Create genre buttons with icons in a grid
                cols = st.columns(2)
                selected_genres = []
                for i, genre in enumerate(all_genres):
                    with cols[i % 2]:
                        icon = GENRE_ICONS.get(genre, '🎬')
                        if st.checkbox(f"{icon} {genre}", key=f"genre_{genre}"):
                            selected_genres.append(genre)
                
                # Rating filter with dynamic label
                st.markdown("### ⭐ Rating Range")
                rating_range = st.slider(
                    "Select rating range",
                    1.0, 5.0, (3.0, 5.0),
                    step=0.5,
                    help="Filter movies by rating range"
                )
                st.caption(f"Current range: **{rating_range[0]:.1f}** - **{rating_range[1]:.1f}** stars")
                
                # Year filter with dynamic label
                st.markdown("### 📅 Release Year")
                min_year = int(movies['year'].min())
                max_year = int(movies['year'].max())
                year_range = st.slider(
                    "Select release year range",
                    min_year, max_year, (min_year, max_year),
                    help="Filter movies by release year"
                )
                st.caption(f"Selected years: **{year_range[0]}** - **{year_range[1]}**")
                
                # Number of recommendations
                st.markdown("### 🎯 Results")
                n_recommendations = st.slider(
                    "Number of recommendations",
                    5, 50, 20,
                    help="Number of movies to recommend"
                )

    # Main content area
    if mode == "Get Recommendations":
        try:
            # Generate recommendations
            with st.spinner(f"🎯 Finding the best movies for you using {algorithm}..."):
                if algorithm == "Switched Hybrid":
                    recommendations = recommenders["Switched Hybrid"].get_recommendations(
                        user_id=user_id,
                        ratings_df=ratings,
                        movies_df=movies,
                        n=n_recommendations
                    )
                else:
                    recommendations = recommenders[algorithm].get_top_n_recommendations(
                        user_id=user_id,
                        movies_df=movies,
                        n=n_recommendations
                    )

                if recommendations:
                    # Apply filters
                    filtered_recommendations = []
                    for movie_id, title, score in recommendations:
                        try:
                            movie_info = movies[movies['movie_id'] == movie_id].iloc[0]
                            movie_genres = movie_info['genres'].split('|')
                            
                            # Extract year from title
                            year_match = re.search(r'\((\d{4})\)', title)
                            year = int(year_match.group(1)) if year_match else None

                            # Apply filters
                            if ((not selected_genres or any(genre in movie_genres for genre in selected_genres)) and
                                (rating_range[0] <= score <= rating_range[1]) and
                                (year and year_range[0] <= year <= year_range[1])):
                                filtered_recommendations.append((movie_id, title, score, movie_info['genres'], year))
                        except Exception as e:
                            logger.error(f"Error processing movie {movie_id}: {str(e)}")
                            continue

                    # Display recommendations
                    if filtered_recommendations:
                        st.markdown("## 🎬 Your Personalized Movie Recommendations")
                        st.caption(f"Showing {len(filtered_recommendations)} movies matching your criteria")
                        
                        # Create movie grid
                        cols = st.columns(4)
                        for i, (movie_id, title, score, genres, year) in enumerate(filtered_recommendations):
                            with cols[i % 4]:
                                st.markdown(
                                    create_movie_card(
                                        movie_id=movie_id,
                                        title=title,
                                        genres=genres,
                                        rating=score,
                                        year=year
                                    ),
                                    unsafe_allow_html=True
                                )
                                
                                # Add "Why Recommended?" button
                                if st.button(f"Why? 🤔", key=f"why_{movie_id}"):
                                    explanation = explain_recommendation(
                                        movie_id, user_id, algorithm,
                                        ratings, movies
                                    )
                                    st.info(explanation)
                    else:
                        st.warning("No movies match your current filters. Try adjusting your criteria.")
                else:
                    st.warning("No recommendations found. Try a different algorithm or user.")

        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            st.error(f"❌ Error: {str(e)}")
            st.error("Please try a different user ID or algorithm.")
            
    else:  # Evaluation Dashboard
        st.title("📊 Algorithm Performance Analysis")
        
        # Create tabs for different views
        metrics_tab, comparison_tab = st.tabs(["📊 Metrics Overview", "🔍 Detailed Comparison"])
        
        with metrics_tab:
            # Load data and initialize recommenders if not already done
            with st.spinner("Loading data and initializing models..."):
                if 'ratings' not in locals():
                    ratings, movies, surprise_data = load_data()
                if ratings is not None and 'recommenders' not in locals():
                    recommenders = initialize_recommenders(ratings, movies, surprise_data)
                    
                if ratings is not None and recommenders is not None:
                    # Evaluate models
                    with st.spinner("Evaluating recommendation algorithms..."):
                        evaluation_metrics = evaluate_all_models(ratings, movies, recommenders)
                        display_evaluation_metrics(evaluation_metrics)
                else:
                    st.error("Failed to load data or initialize models.")
        
        with comparison_tab:
            st.markdown("### 🔍 Algorithm Comparison")
            st.info("Coming soon: Detailed algorithm comparison with interactive visualizations!")

    # Footer
    st.markdown("""
    <div class="footer">
        <p class="footer-logo">🎬 MovieLens Recommender</p>
        <p class="footer-text">Developed by Rahmat Khan | 2024</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 