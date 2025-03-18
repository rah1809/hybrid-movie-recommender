"""
Streamlit app for movie recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from typing import List, Tuple
import logging

# Add parent directory to path to import recommenders
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.recommenders import CollaborativeFiltering, MatrixFactorization

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide"
)

@st.cache_data
def load_data():
    """Load and preprocess MovieLens data."""
    try:
        # Load ratings
        ratings = pd.read_csv('../data/ml-1m/ratings.dat', 
                            sep='::', 
                            names=['user_id', 'movie_id', 'rating', 'timestamp'],
                            engine='python')
        
        # Load movies
        movies = pd.read_csv('../data/ml-1m/movies.dat',
                           sep='::',
                           names=['movie_id', 'title', 'genres'],
                           engine='python')
        
        return ratings, movies
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        st.error("Error loading MovieLens dataset. Please make sure the data files are in the correct location.")
        return None, None

def train_recommender(ratings: pd.DataFrame, algorithm: str = "collaborative"):
    """Train the selected recommendation algorithm."""
    try:
        if algorithm == "collaborative":
            model = CollaborativeFiltering(k_neighbors=5)
        else:
            model = MatrixFactorization(n_factors=100)
        
        model.fit(ratings)
        return model
    
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        st.error("Error training the recommendation model.")
        return None

def main():
    # Title and description
    st.title("🎬 Movie Recommendation System")
    st.markdown("""
    This app demonstrates a hybrid movie recommendation system using the MovieLens dataset.
    Choose a recommendation algorithm and rate some movies to get personalized recommendations!
    """)
    
    # Load data
    ratings, movies = load_data()
    if ratings is None or movies is None:
        return
    
    # Sidebar
    st.sidebar.title("Settings")
    algorithm = st.sidebar.selectbox(
        "Choose Algorithm",
        ["Collaborative Filtering", "Matrix Factorization"],
        help="Select the recommendation algorithm to use"
    )
    
    # Main content
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.subheader("Rate Some Movies")
        # Sample 10 random popular movies for rating
        popular_movies = (
            ratings.groupby('movie_id')
            .agg({'rating': ['count', 'mean']})
            .sort_values(('rating', 'count'), ascending=False)
            .head(10)
        )
        
        user_ratings = {}
        for movie_id in popular_movies.index:
            movie_title = movies[movies['movie_id'] == movie_id]['title'].iloc[0]
            rating = st.slider(
                f"{movie_title}",
                min_value=1.0,
                max_value=5.0,
                step=0.5,
                value=3.0,
                key=f"rating_{movie_id}"
            )
            user_ratings[movie_id] = rating
    
    with col2:
        st.subheader("Your Recommendations")
        if st.button("Get Recommendations"):
            with st.spinner("Training model and generating recommendations..."):
                try:
                    # Create temporary user ratings DataFrame
                    temp_ratings = pd.DataFrame({
                        'user_id': [999999] * len(user_ratings),
                        'movie_id': list(user_ratings.keys()),
                        'rating': list(user_ratings.values())
                    })
                    
                    # Combine with existing ratings
                    combined_ratings = pd.concat([ratings, temp_ratings])
                    
                    # Train model
                    model = train_recommender(
                        combined_ratings,
                        "collaborative" if algorithm == "Collaborative Filtering" else "matrix"
                    )
                    
                    if model:
                        # Get recommendations
                        recommendations = model.predict(999999, n_recommendations=5)
                        
                        # Display recommendations
                        st.write("Here are your personalized movie recommendations:")
                        for movie_id, score in recommendations:
                            movie_info = movies[movies['movie_id'] == movie_id].iloc[0]
                            st.write(f"🎥 **{movie_info['title']}** ({movie_info['genres']})")
                            st.write(f"Predicted rating: {score:.2f}")
                            st.write("---")
                
                except Exception as e:
                    logger.error(f"Error generating recommendations: {str(e)}")
                    st.error("Error generating recommendations. Please try again.")

if __name__ == "__main__":
    main() 