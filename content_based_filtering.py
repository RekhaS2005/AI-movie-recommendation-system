import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.preprocessing import MultiLabelBinarizer
import re
from typing import List, Dict

class ContentBasedFiltering:
    """Implements content-based filtering using TF-IDF on movie metadata"""
    
    def __init__(self, movies_df: pd.DataFrame):
        self.movies_df = movies_df.copy()
        self.tfidf_matrix = None
        self.genre_matrix = None
        self.content_similarity_matrix = None
        self.tfidf_vectorizer = None
        self.genre_binarizer = None
        
        self._prepare_content_features()
        self._compute_similarity_matrices()
    
    def _prepare_content_features(self):
        """Prepare content features for TF-IDF analysis"""
        print("Preparing content features...")
        
        # Clean and prepare text features
        self.movies_df['content_features'] = self._create_content_features()
        
        # Prepare genre features
        self._prepare_genre_features()
        
        # Create TF-IDF matrix
        self._create_tfidf_matrix()
    
    def _create_content_features(self) -> pd.Series:
        """Create combined content features from title and genres"""
        content_features = []
        
        for _, row in self.movies_df.iterrows():
            features = []
            
            # Process title (remove year and special characters)
            if pd.notna(row['clean_title']):
                title_words = re.findall(r'\b\w+\b', row['clean_title'].lower())
                features.extend(title_words)
            
            # Process genres
            if pd.notna(row['genres']):
                genres = row['genres'].split('|')
                # Convert to lowercase and add multiple times for emphasis
                genre_features = [genre.lower().replace('-', '').replace(' ', '') for genre in genres]
                features.extend(genre_features * 3)  # Give genres more weight
            
            content_features.append(' '.join(features))
        
        return pd.Series(content_features, index=self.movies_df.index)
    
    def _prepare_genre_features(self):
        """Prepare binary genre features"""
        # Extract all genres
        all_genres = []
        for genres in self.movies_df['genres'].dropna():
            all_genres.extend(genres.split('|'))
        
        unique_genres = list(set(all_genres))
        
        # Create binary genre matrix
        genre_lists = []
        for genres in self.movies_df['genres']:
            if pd.notna(genres):
                genre_lists.append(genres.split('|'))
            else:
                genre_lists.append([])
        
        self.genre_binarizer = MultiLabelBinarizer()
        self.genre_matrix = self.genre_binarizer.fit_transform(genre_lists)
        self.genre_features_df = pd.DataFrame(
            self.genre_matrix,
            columns=self.genre_binarizer.classes_,
            index=self.movies_df.index
        )
    
    def _create_tfidf_matrix(self):
        """Create TF-IDF matrix from content features"""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            self.movies_df['content_features']
        )
    
    def _compute_similarity_matrices(self):
        """Compute content similarity matrices"""
        print("Computing content similarity matrices...")
        
        # Content similarity using TF-IDF
        self.content_similarity_matrix = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)
        
        # Genre similarity using cosine similarity
        self.genre_similarity_matrix = cosine_similarity(self.genre_matrix)
        
        print("Content similarity matrices computed!")
    
    def get_movie_recommendations(self, movie_id: int, n_recommendations: int = 10) -> pd.DataFrame:
        """Get recommendations based on a specific movie"""
        
        movie_idx = self.movies_df[self.movies_df['movieId'] == movie_id].index
        if len(movie_idx) == 0:
            return pd.DataFrame()
        
        movie_idx = movie_idx[0]
        
        # Get content similarity scores
        content_scores = list(enumerate(self.content_similarity_matrix[movie_idx]))
        genre_scores = list(enumerate(self.genre_similarity_matrix[movie_idx]))
        
        # Combine content and genre similarities
        combined_scores = []
        for i, (content_score, genre_score) in enumerate(zip(content_scores, genre_scores)):
            combined_score = 0.6 * content_score[1] + 0.4 * genre_score[1]
            combined_scores.append((i, combined_score))
        
        # Sort by similarity and get top recommendations
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get top N recommendations (excluding the input movie itself)
        recommendations = []
        for idx, score in combined_scores[1:n_recommendations+1]:
            movie_data = self.movies_df.iloc[idx]
            recommendations.append({
                'movieId': movie_data['movieId'],
                'title': movie_data['title'],
                'genres': movie_data['genres'],
                'similarity_score': score,
                'year': movie_data.get('year', 'Unknown')
            })
        
        return pd.DataFrame(recommendations)
    
    def get_genre_recommendations(self, preferred_genres: List[str], n_recommendations: int = 10) -> pd.DataFrame:
        """Get recommendations based on preferred genres"""
        
        if not preferred_genres:
            return pd.DataFrame()
        
        # Create user profile based on preferred genres
        user_genre_profile = np.zeros(len(self.genre_binarizer.classes_))
        
        for genre in preferred_genres:
            if genre in self.genre_binarizer.classes_:
                genre_idx = list(self.genre_binarizer.classes_).index(genre)
                user_genre_profile[genre_idx] = 1
        
        # Calculate similarities with all movies
        genre_similarities = cosine_similarity([user_genre_profile], self.genre_matrix)[0]
        
        # Get content similarities for movies with matching genres
        matching_movies = np.where(genre_similarities > 0)[0]
        
        if len(matching_movies) == 0:
            return pd.DataFrame()
        
        # Combine genre and content similarities
        recommendations = []
        for movie_idx in matching_movies:
            movie_data = self.movies_df.iloc[movie_idx]
            
            # Calculate combined score
            genre_score = genre_similarities[movie_idx]
            
            # Get average content similarity with similar genre movies
            content_scores = self.content_similarity_matrix[movie_idx][matching_movies]
            avg_content_score = np.mean(content_scores)
            
            combined_score = 0.7 * genre_score + 0.3 * avg_content_score
            
            recommendations.append({
                'movieId': movie_data['movieId'],
                'title': movie_data['title'],
                'genres': movie_data['genres'],
                'similarity_score': combined_score,
                'genre_match_score': genre_score,
                'content_score': avg_content_score,
                'year': movie_data.get('year', 'Unknown')
            })
        
        # Sort by combined score and return top N
        recommendations_df = pd.DataFrame(recommendations)
        recommendations_df = recommendations_df.sort_values('similarity_score', ascending=False)
        
        return recommendations_df.head(n_recommendations)
    
    def get_similar_movies_by_content(self, movie_title: str, n_recommendations: int = 10) -> pd.DataFrame:
        """Get similar movies by searching with movie title"""
        
        # Find movie by title (fuzzy matching)
        movie_matches = self.movies_df[
            self.movies_df['title'].str.contains(movie_title, case=False, na=False)
        ]
        
        if movie_matches.empty:
            return pd.DataFrame()
        
        # Use the first match
        movie_id = movie_matches.iloc[0]['movieId']
        return self.get_movie_recommendations(movie_id, n_recommendations)
    
    def analyze_content_features(self, movie_id: int) -> Dict:
        """Analyze content features of a specific movie"""
        
        movie_data = self.movies_df[self.movies_df['movieId'] == movie_id]
        if movie_data.empty:
            return {}
        
        movie_idx = movie_data.index[0]
        movie_info = movie_data.iloc[0]
        
        # Get TF-IDF feature names and scores
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        tfidf_scores = self.tfidf_matrix[movie_idx].toarray()[0]
        
        # Get top TF-IDF features
        top_features_idx = np.argsort(tfidf_scores)[-10:][::-1]
        top_features = [(feature_names[i], tfidf_scores[i]) for i in top_features_idx if tfidf_scores[i] > 0]
        
        # Get genre information
        genre_info = {}
        if pd.notna(movie_info['genres']):
            genres = movie_info['genres'].split('|')
            for genre in genres:
                if genre in self.genre_features_df.columns:
                    genre_info[genre] = self.genre_features_df.loc[movie_idx, genre]
        
        return {
            'movie_title': movie_info['title'],
            'genres': movie_info['genres'],
            'top_content_features': top_features,
            'genre_features': genre_info,
            'content_feature_text': self.movies_df.loc[movie_idx, 'content_features']
        }
    
    def get_genre_statistics(self) -> pd.DataFrame:
        """Get statistics about genres in the dataset"""
        genre_stats = []
        
        for genre in self.genre_binarizer.classes_:
            genre_count = self.genre_features_df[genre].sum()
            genre_percentage = (genre_count / len(self.movies_df)) * 100
            
            genre_stats.append({
                'genre': genre,
                'movie_count': genre_count,
                'percentage': genre_percentage
            })
        
        stats_df = pd.DataFrame(genre_stats)
        return stats_df.sort_values('movie_count', ascending=False)
    
    def find_diverse_recommendations(self, preferred_genres: List[str], 
                                   n_recommendations: int = 10, 
                                   diversity_weight: float = 0.3) -> pd.DataFrame:
        """Get diverse recommendations that balance similarity and diversity"""
        
        # Get initial recommendations
        initial_recs = self.get_genre_recommendations(preferred_genres, n_recommendations * 2)
        
        if initial_recs.empty:
            return pd.DataFrame()
        
        # Calculate diversity scores
        selected_recommendations = []
        selected_indices = []
        
        for _ in range(min(n_recommendations, len(initial_recs))):
            best_score = -1
            best_idx = -1
            
            for idx, row in initial_recs.iterrows():
                if idx in selected_indices:
                    continue
                
                similarity_score = row['similarity_score']
                
                # Calculate diversity score (average distance from already selected items)
                if not selected_recommendations:
                    diversity_score = 1.0
                else:
                    movie_idx = self.movies_df[self.movies_df['movieId'] == row['movieId']].index[0]
                    diversity_scores = []
                    
                    for selected_rec in selected_recommendations:
                        selected_movie_idx = self.movies_df[self.movies_df['movieId'] == selected_rec['movieId']].index[0]
                        content_sim = self.content_similarity_matrix[movie_idx][selected_movie_idx]
                        diversity_scores.append(1 - content_sim)
                    
                    diversity_score = np.mean(diversity_scores)
                
                # Combined score
                combined_score = (1 - diversity_weight) * similarity_score + diversity_weight * diversity_score
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_idx = idx
            
            if best_idx != -1:
                selected_recommendations.append(initial_recs.loc[best_idx].to_dict())
                selected_indices.append(best_idx)
        
        return pd.DataFrame(selected_recommendations)
