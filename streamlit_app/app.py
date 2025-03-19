import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import plotly.express as px
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import recommenders
from src.recommenders import TrivialRecommender, MPIRecommender

# Page config
st.set_page_config(
    page_title="MovieLens Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .movie-card {
        background-color: #1E1E1E;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #333;
    }
    .metric-card {
        background-color: #2E2E2E;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'user_id' not in st.session_state:
    st.session_state.user_id = 1
if 'recommender' not in st.session_state:
    st.session_state.recommender = None
if 'favorite_movies' not in st.session_state:
    st.session_state.favorite_movies = set()

@st.cache_data
def load_data():
    try:
        movies_df = pd.read_csv("data/ml-1m/movies.dat", 
                               sep="::", 
                               names=["MovieID", "Title", "Genres"], 
                               encoding="ISO-8859-1", 
                               engine="python")
        
        ratings_df = pd.read_csv("data/ml-1m/ratings.dat",
                                sep="::",
                                names=["UserID", "MovieID", "Rating", "Timestamp"],
                                encoding="ISO-8859-1",
                                engine="python")
        
        # Convert timestamp to datetime
        ratings_df['Timestamp'] = pd.to_datetime(ratings_df['Timestamp'], unit='s')
        
        return movies_df, ratings_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

def get_user_stats(ratings_df, user_id):
    user_ratings = ratings_df[ratings_df['UserID'] == user_id]
    if len(user_ratings) > 0:
        return {
            'total_ratings': len(user_ratings),
            'avg_rating': user_ratings['Rating'].mean(),
            'first_rating': user_ratings['Timestamp'].min(),
            'last_rating': user_ratings['Timestamp'].max(),
            'rating_distribution': user_ratings['Rating'].value_counts().sort_index()
        }
    return None

def show_movie_details(movie, ratings_df):
    movie_ratings = ratings_df[ratings_df['MovieID'] == movie['MovieID']]
    
    st.markdown(f"### {movie['Title']}")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Genres:** {movie['Genres']}")
        st.markdown(f"**Average Rating:** {'⭐' * int(movie_ratings['Rating'].mean())}")
        
    with col2:
        st.markdown(f"**Total Ratings:** {len(movie_ratings)}")
        if movie['MovieID'] in st.session_state.favorite_movies:
            if st.button("❤️ Remove from Favorites", key=f"fav_{movie['MovieID']}"):
                st.session_state.favorite_movies.remove(movie['MovieID'])
        else:
            if st.button("🤍 Add to Favorites", key=f"fav_{movie['MovieID']}"):
                st.session_state.favorite_movies.add(movie['MovieID'])

    # Show rating distribution
    rating_dist = movie_ratings['Rating'].value_counts().sort_index()
    fig = px.bar(x=rating_dist.index, y=rating_dist.values,
                 labels={'x': 'Rating', 'y': 'Count'},
                 title='Rating Distribution')
    st.plotly_chart(fig, use_container_width=True)

def main():
    st.title("🎬 MovieLens Recommender")
    
    movies_df, ratings_df = load_data()
    
    if movies_df is not None and ratings_df is not None:
        # Sidebar
        with st.sidebar:
            st.header("🎯 Settings")
            
            # User Profile
            st.subheader("👤 User Profile")
            profile_tab, search_tab = st.tabs(["Profile", "Search"])
            
            with profile_tab:
                st.session_state.user_id = st.number_input("User ID", 
                                                         min_value=1, 
                                                         max_value=6040, 
                                                         value=st.session_state.user_id)
                
                user_stats = get_user_stats(ratings_df, st.session_state.user_id)
                if user_stats:
                    st.metric("Total Ratings", user_stats['total_ratings'])
                    st.metric("Average Rating", f"{user_stats['avg_rating']:.2f} ⭐")
                    
                    # Rating distribution
                    fig = px.bar(x=user_stats['rating_distribution'].index,
                               y=user_stats['rating_distribution'].values,
                               labels={'x': 'Rating', 'y': 'Count'},
                               title='Your Rating Distribution')
                    st.plotly_chart(fig, use_container_width=True)
            
            with search_tab:
                search_query = st.text_input("🔍 Search Movies")
                if search_query:
                    results = movies_df[movies_df['Title'].str.contains(search_query, case=False)]
                    st.write(f"Found {len(results)} matches")
                    for _, movie in results.iterrows():
                        st.markdown(f"- {movie['Title']} ({movie['Genres']})")
            
            # Algorithm Selection
            st.subheader("🤖 Algorithm")
            algorithm = st.selectbox(
                "Choose Algorithm",
                ["Simple Popular", "Matrix Factorization", "Hybrid"]
            )
            
            # Initialize recommender
            if algorithm == "Simple Popular" and not isinstance(st.session_state.recommender, TrivialRecommender):
                st.session_state.recommender = TrivialRecommender()
                st.session_state.recommender.fit(movies_df, ratings_df)
            elif algorithm == "Matrix Factorization" and not isinstance(st.session_state.recommender, MPIRecommender):
                st.session_state.recommender = MPIRecommender()
                st.session_state.recommender.fit(movies_df, ratings_df)
            
            # Filters
            st.subheader("🎭 Filters")
            genres = movies_df['Genres'].str.split('|').explode().unique()
            selected_genres = st.multiselect("Select Genres", genres)
            
            rating_range = st.slider("Rating Range", 1.0, 5.0, (3.0, 5.0))
            year_range = st.slider("Year Range", 1919, 2000, (1919, 2000))
            
            num_recommendations = st.slider("Number of Recommendations", 5, 20, 10)
        
        # Main content
        tab1, tab2, tab3 = st.tabs(["🎯 Recommendations", "❤️ Favorites", "📊 Analytics"])
        
        with tab1:
            if st.session_state.recommender:
                if isinstance(st.session_state.recommender, MPIRecommender):
                    recommendations = st.session_state.recommender.recommend(st.session_state.user_id)
                else:
                    recommendations = st.session_state.recommender.recommend()
                
                # Apply filters
                if selected_genres:
                    recommendations = recommendations[
                        recommendations['Genres'].apply(lambda x: any(g in x for g in selected_genres))
                    ]
                
                st.subheader(f"Found {len(recommendations)} Recommendations")
                
                # Display recommendations in a grid
                cols = st.columns(2)
                for idx, (_, movie) in enumerate(recommendations.iterrows()):
                    with cols[idx % 2]:
                        with st.expander(f"{movie['Title']}", expanded=False):
                            show_movie_details(movie, ratings_df)
        
        with tab2:
            st.subheader("Your Favorite Movies")
            if st.session_state.favorite_movies:
                favorites = movies_df[movies_df['MovieID'].isin(st.session_state.favorite_movies)]
                for _, movie in favorites.iterrows():
                    with st.expander(f"{movie['Title']}", expanded=False):
                        show_movie_details(movie, ratings_df)
            else:
                st.info("Add movies to your favorites by clicking the heart button!")
        
        with tab3:
            st.subheader("Movie Analytics")
            
            # Genre distribution
            genre_dist = movies_df['Genres'].str.split('|').explode().value_counts()
            fig = px.bar(x=genre_dist.index, y=genre_dist.values,
                        labels={'x': 'Genre', 'y': 'Count'},
                        title='Movies by Genre')
            st.plotly_chart(fig, use_container_width=True)
            
            # Rating trends over time
            ratings_over_time = ratings_df.set_index('Timestamp')['Rating'].resample('M').mean()
            fig = px.line(ratings_over_time,
                         labels={'value': 'Average Rating', 'Timestamp': 'Date'},
                         title='Rating Trends Over Time')
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
