import sys
import os
# Add src folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

"""
Streamlit app for movie recommendations using multiple algorithms.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from typing import List, Tuple, Dict
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

# Add parent directory to path to import recommenders
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.recommenders import TrivialRecommender, MPIRecommender
from notebooks.collaborative_filtering import UserBasedCF, ItemBasedCF
from notebooks.matrix_factorization import SVDRecommender

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Hybrid Movie Recommender",
    page_icon="🎥",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .movie-title {
        font-size: 1.2em;
        font-weight: bold;
        color: #1E88E5;
    }
    .movie-rating {
        font-size: 1.1em;
        color: #FFA000;
    }
    .movie-genres {
        color: #4CAF50;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load MovieLens data and create Surprise dataset."""
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
        
        # Create Surprise reader and dataset
        reader = Reader(rating_scale=(1, 5))
        surprise_data = Dataset.load_from_df(
            ratings[['user_id', 'movie_id', 'rating']], 
            reader
        )
        
        return ratings, movies, surprise_data
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        st.error("Error loading MovieLens dataset. Please make sure the data files are in the correct location.")
        return None, None, None

def evaluate_recommender(recommender, ratings_df: pd.DataFrame, k: int = 5) -> Dict[str, float]:
    """Evaluate a recommender using Precision@K and Recall@K.
    
    Args:
        recommender: Recommender model instance
        ratings_df: DataFrame with user ratings
        k: Number of recommendations to consider
        
    Returns:
        Dict with precision and recall scores
    """
    # Sample users for evaluation
    test_users = np.random.choice(ratings_df['user_id'].unique(), size=100)
    
    precision_scores = []
    recall_scores = []
    
    for user_id in test_users:
        # Get actual highly rated movies (rated >= 4)
        actual = ratings_df[
            (ratings_df['user_id'] == user_id) & 
            (ratings_df['rating'] >= 4)
        ]['movie_id'].tolist()
        
        if actual:  # Only evaluate if user has rated some movies highly
            # Get recommendations
            recommendations = recommender.get_top_n_recommendations(user_id, movies, k)
            predicted = [movie_id for movie_id, _, _ in recommendations]
            
            # Calculate metrics
            n_relevant = len(set(actual) & set(predicted))
            precision = n_relevant / len(predicted) if predicted else 0
            recall = n_relevant / len(actual) if actual else 0
            
            precision_scores.append(precision)
            recall_scores.append(recall)
    
    return {
        'precision': np.mean(precision_scores),
        'recall': np.mean(recall_scores)
    }

@st.cache_data
def evaluate_all_models(recommenders: Dict, ratings: pd.DataFrame, surprise_data: Dataset) -> Dict:
    """Evaluate all recommendation models.
    
    Args:
        recommenders: Dictionary of recommender instances
        ratings: Ratings DataFrame
        surprise_data: Surprise Dataset object
        
    Returns:
        Dict containing evaluation metrics for each model
    """
    results = {}
    
    for name, model in recommenders.items():
        # Get RMSE and MAE using cross-validation
        cv_results = cross_validate(
            model.model if hasattr(model, 'model') else model,
            surprise_data,
            measures=['RMSE', 'MAE'],
            cv=5,
            verbose=False
        )
        
        # Get Precision and Recall
        pr_metrics = evaluate_recommender(model, ratings)
        
        results[name] = {
            'RMSE': cv_results['test_rmse'].mean(),
            'MAE': cv_results['test_mae'].mean(),
            'Precision@5': pr_metrics['precision'],
            'Recall@5': pr_metrics['recall']
        }
    
    return results

def plot_error_metrics(results: Dict) -> go.Figure:
    """Create a bar plot comparing RMSE and MAE across models."""
    models = list(results.keys())
    rmse_values = [results[model]['RMSE'] for model in models]
    mae_values = [results[model]['MAE'] for model in models]
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=('RMSE Comparison', 'MAE Comparison'))
    
    # Add RMSE bars
    fig.add_trace(
        go.Bar(name='RMSE', x=models, y=rmse_values, marker_color='#1f77b4'),
        row=1, col=1
    )
    
    # Add MAE bars
    fig.add_trace(
        go.Bar(name='MAE', x=models, y=mae_values, marker_color='#2ca02c'),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=400,
        showlegend=False,
        title_text="Error Metrics Comparison",
        title_x=0.5
    )
    
    return fig

def plot_precision_recall(results: Dict) -> go.Figure:
    """Create a bar plot comparing Precision@K and Recall@K across models."""
    models = list(results.keys())
    precision_values = [results[model]['Precision@5'] for model in models]
    recall_values = [results[model]['Recall@5'] for model in models]
    
    fig = make_subplots(rows=1, cols=2, 
                       subplot_titles=('Precision@5 Comparison', 'Recall@5 Comparison'))
    
    # Add Precision bars
    fig.add_trace(
        go.Bar(name='Precision@5', x=models, y=precision_values, marker_color='#ff7f0e'),
        row=1, col=1
    )
    
    # Add Recall bars
    fig.add_trace(
        go.Bar(name='Recall@5', x=models, y=recall_values, marker_color='#d62728'),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=400,
        showlegend=False,
        title_text="Precision-Recall Metrics Comparison",
        title_x=0.5
    )
    
    return fig

def initialize_recommenders(ratings, movies, surprise_data):
    """Initialize all recommender systems."""
    recommenders = {}
    
    try:
        # Initialize Trivial Recommender
        trivial_rec = TrivialRecommender(min_ratings=100)
        trivial_rec.fit(ratings, movies)
        recommenders['Trivial'] = trivial_rec
        
        # Initialize MPI Recommender
        mpi_rec = MPIRecommender(min_rating=4.0)
        mpi_rec.fit(ratings, movies)
        recommenders['MPI'] = mpi_rec
        
        # Initialize User-based CF
        user_cf = UserBasedCF(k=40)
        user_cf.fit(surprise_data)
        recommenders['User-based CF'] = user_cf
        
        # Initialize Item-based CF
        item_cf = ItemBasedCF(k=40)
        item_cf.fit(surprise_data)
        recommenders['Item-based CF'] = item_cf
        
        # Initialize SVD
        svd_rec = SVDRecommender(n_factors=100, n_epochs=20)
        svd_rec.fit(surprise_data)
        recommenders['SVD'] = svd_rec
        
        return recommenders
    
    except Exception as e:
        logger.error(f"Error initializing recommenders: {str(e)}")
        st.error("Error initializing recommendation models.")
        return None

def main():
    # Title and description
    st.title("🎥 Hybrid Movie Recommendation Engine")
    st.markdown("""
    Welcome to the Hybrid Movie Recommendation System!  
    Choose an algorithm, enter a user ID, and get personalized movie suggestions.  
    You can also evaluate and compare model performance from the sidebar.
    """)
    
    # Load data
    ratings, movies, surprise_data = load_data()
    if ratings is None:
        return
    
    # Initialize recommenders
    recommenders = initialize_recommenders(ratings, movies, surprise_data)
    if recommenders is None:
        return
    
    # Sidebar controls
    st.sidebar.title("🎬 Settings")
    
    # Instructions section in sidebar
    st.sidebar.title("📖 Instructions")
    st.sidebar.info("""
    **How to Use:**
    1️⃣ Choose a recommendation algorithm  
    2️⃣ Enter User ID  
    3️⃣ Select number of recommendations  
    4️⃣ Click **Get Recommendations** 🎯  
    5️⃣ Or switch to **Evaluation Dashboard** for performance comparison 📊
    """)
    
    # Mode selection
    mode = st.sidebar.radio(
        "Select Mode",
        ["Get Recommendations", "Evaluation Dashboard"],
        help="Choose between getting recommendations or viewing model performance"
    )
    
    if mode == "Get Recommendations":
        # Algorithm selection
        algorithm = st.sidebar.selectbox(
            "Choose Recommendation Algorithm",
            ["Trivial", "MPI", "User-based CF", "Item-based CF", "SVD"],
            help="Select the recommendation algorithm to use"
        )
        
        # User ID input
        user_id = st.sidebar.number_input(
            "Enter User ID",
            min_value=1,
            max_value=ratings['user_id'].max(),
            value=1,
            help="Enter the ID of the user to get recommendations for"
        )
        
        # Number of recommendations
        n_recommendations = st.sidebar.slider(
            "Number of Recommendations",
            min_value=1,
            max_value=20,
            value=5,
            help="Select how many movie recommendations to generate"
        )
        
        # Generate recommendations button
        if st.sidebar.button("🔍 Get Recommendations"):
            with st.spinner("🎬 Generating recommendations..."):
                try:
                    # Get recommendations
                    recommendations = recommenders[algorithm].get_top_n_recommendations(
                        user_id=user_id,
                        movies_df=movies,
                        n=n_recommendations
                    )
                    
                    # Display recommendations
                    st.subheader(f"🎯 Top {n_recommendations} Recommendations")
                    
                    # Create two columns
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        for i, (movie_id, title, score) in enumerate(recommendations, 1):
                            # Get movie genres
                            genres = movies[movies['movie_id'] == movie_id]['genres'].iloc[0]
                            
                            # Create expandable card for each movie
                            with st.expander(f"{i}. {title}"):
                                # Convert score to star rating (1-5 stars)
                                stars = "⭐" * int(round(score))
                                st.markdown(f'<p class="movie-title">{title}</p>', unsafe_allow_html=True)
                                st.markdown(f'<p class="movie-rating">Rating: {score:.2f} {stars}</p>', unsafe_allow_html=True)
                                st.markdown(f'<p class="movie-genres">Genres: {genres}</p>', unsafe_allow_html=True)
                    
                    with col2:
                        # Show algorithm info
                        st.info(f"📊 **Algorithm Info**\n\n"
                               f"🎯 Using: {algorithm}\n\n"
                               f"👤 User ID: {user_id}\n\n"
                               f"🎬 Total movies: {len(movies)}")
                
                except Exception as e:
                    logger.error(f"Error generating recommendations: {str(e)}")
                    st.error("❌ Error generating recommendations. Please try again.")
    
    else:  # Evaluation Dashboard
        st.subheader("📊 Model Performance Comparison")
        
        with st.spinner("📈 Calculating evaluation metrics..."):
            # Get evaluation results
            results = evaluate_all_models(recommenders, ratings, surprise_data)
            
            # Display error metrics plot
            st.plotly_chart(plot_error_metrics(results), use_container_width=True)
            
            # Display precision-recall plot
            st.plotly_chart(plot_precision_recall(results), use_container_width=True)
            
            # Display metrics table
            st.subheader("📋 Detailed Metrics")
            metrics_df = pd.DataFrame(results).round(4).T
            st.dataframe(metrics_df, use_container_width=True)
            
            # Add explanations
            st.markdown("""
            ### 📝 Metrics Explanation
            - **RMSE (Root Mean Square Error)**: Lower is better. Measures the average magnitude of prediction errors.
            - **MAE (Mean Absolute Error)**: Lower is better. Measures the average absolute difference between predicted and actual ratings.
            - **Precision@5**: Higher is better. Proportion of recommended items that are relevant.
            - **Recall@5**: Higher is better. Proportion of relevant items that are recommended.
            """)
    
    # Add footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 🎯 About the Algorithms
    - **Trivial:** Recommends top-rated movies
    - **MPI:** Most Popular Items based on rating frequency
    - **User-based CF:** Collaborative Filtering using similar users
    - **Item-based CF:** Collaborative Filtering using similar items
    - **SVD:** Matrix Factorization using Singular Value Decomposition
    """)
    
    # Author footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("📊 **Developed by Rahmat Khan**  \nFor academic demo purposes.")
    
    # Pro tip at bottom of main page
    st.markdown("---")
    st.markdown("💡 **Pro Tip:** Try different algorithms & User IDs to compare results!")

if __name__ == "__main__":
    main() 
