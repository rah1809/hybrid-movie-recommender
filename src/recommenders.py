import pandas as pd
import numpy as np

class TrivialRecommender:
    def __init__(self):
        self.movies_df = None
        self.ratings_df = None
        
    def fit(self, movies_df, ratings_df):
        self.movies_df = movies_df
        self.ratings_df = ratings_df
        
    def recommend(self, n=5):
        movie_stats = self.ratings_df.groupby('MovieID').agg({
            'Rating': ['count', 'mean']
        }).reset_index()
        movie_stats.columns = ['MovieID', 'Count', 'Mean']
        top_movies = movie_stats.sort_values(['Count', 'Mean'], 
                                           ascending=[False, False]).head(n)
        return self.movies_df[self.movies_df['MovieID'].isin(top_movies['MovieID'])]

class MPIRecommender:
    def __init__(self):
        self.movies_df = None
        self.ratings_df = None
        
    def fit(self, movies_df, ratings_df):
        self.movies_df = movies_df
        self.ratings_df = ratings_df
        
    def recommend(self, user_id, n=5):
        user_ratings = self.ratings_df[self.ratings_df['UserID'] == user_id]
        movie_stats = self.ratings_df.groupby('MovieID').agg({
            'Rating': ['count', 'mean']
        }).reset_index()
        movie_stats.columns = ['MovieID', 'Count', 'Mean']
        movie_stats = movie_stats[~movie_stats['MovieID'].isin(user_ratings['MovieID'])]
        top_movies = movie_stats.sort_values(['Count', 'Mean'], 
                                           ascending=[False, False]).head(n)
        return self.movies_df[self.movies_df['MovieID'].isin(top_movies['MovieID'])]
