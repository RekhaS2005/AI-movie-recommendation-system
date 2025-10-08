import pandas as pd
import numpy as np
import urllib.request
import zipfile
import os
import io
from typing import Tuple

class DataLoader:
    """Handles MovieLens dataset downloading and preprocessing"""
    
    def __init__(self):
        self.data_dir = "data"
        self.movielens_url = "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    
    def download_movielens_data(self) -> None:
        """Download and extract MovieLens dataset"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        zip_path = os.path.join(self.data_dir, "ml-latest-small.zip")
        
        # Download if not exists
        if not os.path.exists(zip_path):
            print("Downloading MovieLens dataset...")
            urllib.request.urlretrieve(self.movielens_url, zip_path)
        
        # Extract if not already extracted
        extracted_dir = os.path.join(self.data_dir, "ml-latest-small")
        if not os.path.exists(extracted_dir):
            print("Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
    
    def load_ratings_data(self) -> pd.DataFrame:
        """Load and preprocess ratings data"""
        ratings_path = os.path.join(self.data_dir, "ml-latest-small", "ratings.csv")
        
        ratings_df = pd.read_csv(ratings_path)
        
        # Data validation and cleaning
        ratings_df = ratings_df.dropna()
        ratings_df = ratings_df[ratings_df['rating'] > 0]  # Remove invalid ratings
        
        # Convert timestamp to datetime
        ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
        
        return ratings_df
    
    def load_movies_data(self) -> pd.DataFrame:
        """Load and preprocess movies data"""
        movies_path = os.path.join(self.data_dir, "ml-latest-small", "movies.csv")
        
        movies_df = pd.read_csv(movies_path)
        
        # Data cleaning
        movies_df = movies_df.dropna(subset=['title', 'genres'])
        
        # Extract year from title
        movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)').astype(float)
        
        # Clean title (remove year)
        movies_df['clean_title'] = movies_df['title'].str.replace(r'\s*\(\d{4}\)\s*$', '', regex=True)
        
        # Handle genres
        movies_df['genres'] = movies_df['genres'].replace('(no genres listed)', np.nan)
        
        return movies_df
    
    def create_user_item_matrix(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """Create user-item rating matrix"""
        user_item_matrix = ratings_df.pivot_table(
            index='userId', 
            columns='movieId', 
            values='rating',
            fill_value=0
        )
        
        return user_item_matrix
    
    def compute_basic_stats(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> dict:
        """Compute basic statistics about the dataset"""
        stats = {
            'total_ratings': len(ratings_df),
            'total_users': ratings_df['userId'].nunique(),
            'total_movies': ratings_df['movieId'].nunique(),
            'total_movies_catalog': len(movies_df),
            'rating_scale': (ratings_df['rating'].min(), ratings_df['rating'].max()),
            'average_rating': ratings_df['rating'].mean(),
            'rating_std': ratings_df['rating'].std(),
            'sparsity': 1 - (len(ratings_df) / (ratings_df['userId'].nunique() * ratings_df['movieId'].nunique()))
        }
        
        return stats
    
    def filter_data(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame, 
                   min_user_ratings: int = 20, min_movie_ratings: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Filter data to reduce sparsity"""
        print(f"Original data: {len(ratings_df)} ratings, {ratings_df['userId'].nunique()} users, {ratings_df['movieId'].nunique()} movies")
        
        # Filter users with minimum ratings
        user_counts = ratings_df['userId'].value_counts()
        valid_users = user_counts[user_counts >= min_user_ratings].index
        ratings_filtered = ratings_df[ratings_df['userId'].isin(valid_users)]
        
        # Filter movies with minimum ratings
        movie_counts = ratings_filtered['movieId'].value_counts()
        valid_movies = movie_counts[movie_counts >= min_movie_ratings].index
        ratings_filtered = ratings_filtered[ratings_filtered['movieId'].isin(valid_movies)]
        
        # Filter movies dataframe
        movies_filtered = movies_df[movies_df['movieId'].isin(ratings_filtered['movieId'].unique())]
        
        print(f"Filtered data: {len(ratings_filtered)} ratings, {ratings_filtered['userId'].nunique()} users, {ratings_filtered['movieId'].nunique()} movies")
        
        return ratings_filtered, movies_filtered
    
    def load_movielens_data(self, apply_filtering: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Main method to load and prepare MovieLens data"""
        # Download data if needed
        self.download_movielens_data()
        
        # Load datasets
        ratings_df = self.load_ratings_data()
        movies_df = self.load_movies_data()
        
        # Apply filtering to reduce sparsity
        if apply_filtering:
            ratings_df, movies_df = self.filter_data(ratings_df, movies_df)
        
        # Compute and print statistics
        stats = self.compute_basic_stats(ratings_df, movies_df)
        print("Dataset Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        return ratings_df, movies_df
