import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD, NMF
from scipy.sparse import csr_matrix
from surprise import Dataset, Reader, SVD, KNNBasic
from surprise.model_selection import cross_validate
import warnings
warnings.filterwarnings('ignore')

class CollaborativeFiltering:
    """Implements various collaborative filtering approaches"""
    
    def __init__(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
        self.ratings_df = ratings_df
        self.movies_df = movies_df
        self.user_item_matrix = None
        self.user_similarity_matrix = None
        self.item_similarity_matrix = None
        self.svd_model = None
        self.nmf_model = None
        self.surprise_model = None
        
        self._prepare_data()
        self._train_models()
    
    def _prepare_data(self):
        """Prepare data for collaborative filtering"""
        # Create user-item matrix
        self.user_item_matrix = self.ratings_df.pivot_table(
            index='userId', 
            columns='movieId', 
            values='rating',
            fill_value=0
        )
        
        # Create sparse matrix for memory efficiency
        self.user_item_sparse = csr_matrix(self.user_item_matrix.values)
        
        # Prepare data for Surprise library
        reader = Reader(rating_scale=(0.5, 5.0))
        self.surprise_data = Dataset.load_from_df(
            self.ratings_df[['userId', 'movieId', 'rating']], reader)
    
    def _train_models(self):
        """Train collaborative filtering models"""
        print("Training collaborative filtering models...")
        
        # Compute user-based similarity matrix
        self._compute_user_similarity()
        
        # Compute item-based similarity matrix
        self._compute_item_similarity()
        
        # Train SVD model using Surprise
        self._train_surprise_svd()
        
        # Train NMF model
        self._train_nmf()
        
        print("Collaborative filtering models trained successfully!")
    
    def _compute_user_similarity(self):
        """Compute user-user similarity matrix using cosine similarity"""
        # Only compute for non-zero users to save memory
        non_zero_users = (self.user_item_matrix.sum(axis=1) > 0)
        user_matrix = self.user_item_matrix[non_zero_users]
        
        # Compute cosine similarity
        self.user_similarity_matrix = cosine_similarity(user_matrix)
        self.user_similarity_df = pd.DataFrame(
            self.user_similarity_matrix,
            index=user_matrix.index,
            columns=user_matrix.index
        )
    
    def _compute_item_similarity(self):
        """Compute item-item similarity matrix"""
        # Transpose for item-based similarity
        item_matrix = self.user_item_matrix.T
        non_zero_items = (item_matrix.sum(axis=1) > 0)
        item_matrix = item_matrix[non_zero_items]
        
        # Compute cosine similarity
        self.item_similarity_matrix = cosine_similarity(item_matrix)
        self.item_similarity_df = pd.DataFrame(
            self.item_similarity_matrix,
            index=item_matrix.index,
            columns=item_matrix.index
        )
    
    def _train_surprise_svd(self):
        """Train SVD model using Surprise library"""
        self.surprise_model = SVD(
            n_factors=50,
            lr_all=0.005,
            reg_all=0.02,
            n_epochs=100,
            random_state=42
        )
        
        # Train on full dataset
        trainset = self.surprise_data.build_full_trainset()
        self.surprise_model.fit(trainset)
    
    def _train_nmf(self):
        """Train NMF model for non-negative matrix factorization"""
        # NMF requires non-negative values
        user_item_nonneg = self.user_item_matrix.copy()
        user_item_nonneg[user_item_nonneg < 0] = 0
        
        self.nmf_model = NMF(
            n_components=50,
            init='random',
            random_state=42,
            max_iter=200,
            alpha_W=0.01,
            alpha_H=0.01,
            l1_ratio=0.5
        )
        
        # Fit the model
        self.nmf_W = self.nmf_model.fit_transform(user_item_nonneg)
        self.nmf_H = self.nmf_model.components_
    
    def get_user_recommendations(self, user_id: int, n_recommendations: int = 10) -> pd.DataFrame:
        """Get recommendations for a specific user using hybrid of multiple CF approaches"""
        
        if user_id not in self.user_item_matrix.index:
            return pd.DataFrame()  # Return empty if user not found
        
        # Get movies the user hasn't rated
        user_ratings = self.user_item_matrix.loc[user_id]
        unrated_movies = user_ratings[user_ratings == 0].index
        
        if len(unrated_movies) == 0:
            return pd.DataFrame()
        
        # Get predictions from different methods
        predictions = []
        
        # Method 1: User-based collaborative filtering
        user_based_scores = self._predict_user_based(user_id, unrated_movies)
        
        # Method 2: Item-based collaborative filtering
        item_based_scores = self._predict_item_based(user_id, unrated_movies)
        
        # Method 3: SVD predictions
        svd_scores = self._predict_svd(user_id, unrated_movies)
        
        # Method 4: NMF predictions
        nmf_scores = self._predict_nmf(user_id, unrated_movies)
        
        # Combine predictions (weighted average)
        for movie_id in unrated_movies:
            combined_score = (
                0.25 * user_based_scores.get(movie_id, 0) +
                0.25 * item_based_scores.get(movie_id, 0) +
                0.3 * svd_scores.get(movie_id, 0) +
                0.2 * nmf_scores.get(movie_id, 0)
            )
            predictions.append((movie_id, combined_score))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_predictions = predictions[:n_recommendations]
        
        # Create recommendation dataframe
        recommendations = []
        for movie_id, predicted_rating in top_predictions:
            movie_info = self.movies_df[self.movies_df['movieId'] == movie_id]
            if not movie_info.empty:
                movie_data = movie_info.iloc[0]
                # Get movie statistics
                movie_ratings = self.ratings_df[self.ratings_df['movieId'] == movie_id]
                avg_rating = movie_ratings['rating'].mean() if not movie_ratings.empty else 0
                rating_count = len(movie_ratings)
                
                recommendations.append({
                    'movieId': movie_id,
                    'title': movie_data['title'],
                    'genres': movie_data['genres'],
                    'predicted_rating': predicted_rating,
                    'average_rating': avg_rating,
                    'rating_count': rating_count
                })
        
        return pd.DataFrame(recommendations)
    
    def _predict_user_based(self, user_id: int, movie_ids) -> dict:
        """Predict ratings using user-based collaborative filtering"""
        predictions = {}
        
        if user_id not in self.user_similarity_df.index:
            return predictions
        
        user_similarities = self.user_similarity_df.loc[user_id]
        
        for movie_id in movie_ids:
            if movie_id not in self.user_item_matrix.columns:
                predictions[movie_id] = 0
                continue
                
            # Find users who rated this movie
            movie_ratings = self.user_item_matrix[movie_id]
            rated_users = movie_ratings[movie_ratings > 0]
            
            if len(rated_users) == 0:
                predictions[movie_id] = 0
                continue
            
            # Calculate weighted average of similar users' ratings
            similarities = user_similarities[rated_users.index]
            similarities = similarities[similarities > 0]  # Only positive similarities
            
            if len(similarities) == 0:
                predictions[movie_id] = 0
                continue
            
            weighted_ratings = (similarities * rated_users[similarities.index]).sum()
            similarity_sum = similarities.sum()
            
            if similarity_sum > 0:
                predictions[movie_id] = weighted_ratings / similarity_sum
            else:
                predictions[movie_id] = 0
        
        return predictions
    
    def _predict_item_based(self, user_id: int, movie_ids) -> dict:
        """Predict ratings using item-based collaborative filtering"""
        predictions = {}
        
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_movies = user_ratings[user_ratings > 0]
        
        for movie_id in movie_ids:
            if movie_id not in self.item_similarity_df.index:
                predictions[movie_id] = 0
                continue
            
            # Find similarities with movies the user has rated
            item_similarities = self.item_similarity_df.loc[movie_id]
            relevant_similarities = item_similarities[rated_movies.index]
            relevant_similarities = relevant_similarities[relevant_similarities > 0]
            
            if len(relevant_similarities) == 0:
                predictions[movie_id] = 0
                continue
            
            # Calculate weighted average
            weighted_ratings = (relevant_similarities * rated_movies[relevant_similarities.index]).sum()
            similarity_sum = relevant_similarities.sum()
            
            if similarity_sum > 0:
                predictions[movie_id] = weighted_ratings / similarity_sum
            else:
                predictions[movie_id] = 0
        
        return predictions
    
    def _predict_svd(self, user_id: int, movie_ids) -> dict:
        """Predict ratings using SVD model"""
        predictions = {}
        
        for movie_id in movie_ids:
            try:
                prediction = self.surprise_model.predict(user_id, movie_id)
                predictions[movie_id] = prediction.est
            except:
                predictions[movie_id] = 0
        
        return predictions
    
    def _predict_nmf(self, user_id: int, movie_ids) -> dict:
        """Predict ratings using NMF model"""
        predictions = {}
        
        if user_id not in self.user_item_matrix.index:
            return predictions
        
        # Get user index
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        
        for movie_id in movie_ids:
            if movie_id not in self.user_item_matrix.columns:
                predictions[movie_id] = 0
                continue
            
            # Get movie index
            movie_idx = self.user_item_matrix.columns.get_loc(movie_id)
            
            # Predict rating using NMF factorization
            predicted_rating = np.dot(self.nmf_W[user_idx], self.nmf_H[:, movie_idx])
            predictions[movie_id] = predicted_rating
        
        return predictions
    
    def get_similar_users(self, user_id: int, n_users: int = 10) -> pd.DataFrame:
        """Find similar users to a given user"""
        if user_id not in self.user_similarity_df.index:
            return pd.DataFrame()
        
        similarities = self.user_similarity_df.loc[user_id].sort_values(ascending=False)
        similar_users = similarities.head(n_users + 1)[1:]  # Exclude the user itself
        
        result = []
        for similar_user, similarity in similar_users.items():
            user_stats = self.ratings_df[self.ratings_df['userId'] == similar_user]
            result.append({
                'userId': similar_user,
                'similarity': similarity,
                'rating_count': len(user_stats),
                'avg_rating': user_stats['rating'].mean()
            })
        
        return pd.DataFrame(result)
    
    def get_similar_items(self, movie_id: int, n_items: int = 10) -> pd.DataFrame:
        """Find similar movies to a given movie"""
        if movie_id not in self.item_similarity_df.index:
            return pd.DataFrame()
        
        similarities = self.item_similarity_df.loc[movie_id].sort_values(ascending=False)
        similar_movies = similarities.head(n_items + 1)[1:]  # Exclude the movie itself
        
        result = []
        for similar_movie, similarity in similar_movies.items():
            movie_info = self.movies_df[self.movies_df['movieId'] == similar_movie]
            if not movie_info.empty:
                movie_data = movie_info.iloc[0]
                movie_stats = self.ratings_df[self.ratings_df['movieId'] == similar_movie]
                
                result.append({
                    'movieId': similar_movie,
                    'title': movie_data['title'],
                    'genres': movie_data['genres'],
                    'similarity': similarity,
                    'rating_count': len(movie_stats),
                    'avg_rating': movie_stats['rating'].mean() if not movie_stats.empty else 0
                })
        
        return pd.DataFrame(result)
