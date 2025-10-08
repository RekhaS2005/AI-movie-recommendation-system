import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from collaborative_filtering import CollaborativeFiltering
from content_based_filtering import ContentBasedFiltering

class HybridRecommender:
    """Combines collaborative and content-based filtering for hybrid recommendations"""
    
    def __init__(self, cf_model: CollaborativeFiltering, cb_model: ContentBasedFiltering):
        self.cf_model = cf_model
        self.cb_model = cb_model
        self.ratings_df = cf_model.ratings_df
        self.movies_df = cf_model.movies_df
        
        # Default weights for combining recommendations
        self.default_cf_weight = 0.6
        self.default_cb_weight = 0.4
    
    def get_hybrid_recommendations(self, user_id: int, preferred_genres: List[str] = None, 
                                  n_recommendations: int = 10, 
                                  cf_weight: float = None, cb_weight: float = None) -> pd.DataFrame:
        """Get hybrid recommendations combining collaborative and content-based filtering"""
        
        # Use default weights if not provided
        if cf_weight is None:
            cf_weight = self.default_cf_weight
        if cb_weight is None:
            cb_weight = self.default_cb_weight
        
        # Normalize weights
        total_weight = cf_weight + cb_weight
        cf_weight = cf_weight / total_weight
        cb_weight = cb_weight / total_weight
        
        # Adjust weights based on user profile completeness
        adjusted_weights = self._adjust_weights_by_user_profile(user_id, cf_weight, cb_weight)
        cf_weight, cb_weight = adjusted_weights
        
        # Get collaborative filtering recommendations
        cf_recommendations = self.cf_model.get_user_recommendations(user_id, n_recommendations * 2)
        
        # Get content-based recommendations
        cb_recommendations = pd.DataFrame()
        if preferred_genres:
            cb_recommendations = self.cb_model.get_genre_recommendations(preferred_genres, n_recommendations * 2)
        else:
            # Use user's historical preferences to infer genres
            user_genres = self._infer_user_genre_preferences(user_id)
            if user_genres:
                cb_recommendations = self.cb_model.get_genre_recommendations(user_genres, n_recommendations * 2)
        
        # Combine recommendations
        hybrid_recommendations = self._combine_recommendations(
            cf_recommendations, cb_recommendations, cf_weight, cb_weight, n_recommendations
        )
        
        return hybrid_recommendations
    
    def _adjust_weights_by_user_profile(self, user_id: int, cf_weight: float, cb_weight: float) -> Tuple[float, float]:
        """Adjust weights based on user profile completeness"""
        
        # Get user rating history
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        num_ratings = len(user_ratings)
        
        # Adjust weights based on rating count
        if num_ratings < 5:  # New user - favor content-based
            cf_weight *= 0.3
            cb_weight *= 1.7
        elif num_ratings < 20:  # Some history - balanced approach
            cf_weight *= 0.7
            cb_weight *= 1.3
        # For users with many ratings, keep original weights
        
        # Normalize weights
        total_weight = cf_weight + cb_weight
        return cf_weight / total_weight, cb_weight / total_weight
    
    def _infer_user_genre_preferences(self, user_id: int, top_n_genres: int = 3) -> List[str]:
        """Infer user's preferred genres from rating history"""
        
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        
        if user_ratings.empty:
            return []
        
        # Get highly rated movies (rating >= 4.0)
        high_rated = user_ratings[user_ratings['rating'] >= 4.0]
        
        if high_rated.empty:
            high_rated = user_ratings[user_ratings['rating'] >= user_ratings['rating'].median()]
        
        # Get movies information
        user_movies = high_rated.merge(self.movies_df, on='movieId')
        
        # Count genre preferences weighted by rating
        genre_scores = {}
        for _, row in user_movies.iterrows():
            if pd.notna(row['genres']):
                genres = row['genres'].split('|')
                weight = row['rating'] / 5.0  # Normalize rating to [0,1]
                
                for genre in genres:
                    if genre != '(no genres listed)':
                        genre_scores[genre] = genre_scores.get(genre, 0) + weight
        
        # Get top preferred genres
        if genre_scores:
            sorted_genres = sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)
            return [genre for genre, _ in sorted_genres[:top_n_genres]]
        
        return []
    
    def _combine_recommendations(self, cf_recs: pd.DataFrame, cb_recs: pd.DataFrame, 
                               cf_weight: float, cb_weight: float, n_recommendations: int) -> pd.DataFrame:
        """Combine collaborative and content-based recommendations"""
        
        # Create a comprehensive movie score dictionary
        movie_scores = {}
        
        # Process collaborative filtering recommendations
        if not cf_recs.empty:
            for _, row in cf_recs.iterrows():
                movie_id = row['movieId']
                cf_score = row['predicted_rating'] / 5.0  # Normalize to [0,1]
                
                movie_scores[movie_id] = {
                    'cf_score': cf_score,
                    'cb_score': 0,
                    'movie_data': row
                }
        
        # Process content-based recommendations
        if not cb_recs.empty:
            for _, row in cb_recs.iterrows():
                movie_id = row['movieId']
                cb_score = row['similarity_score']
                
                if movie_id in movie_scores:
                    movie_scores[movie_id]['cb_score'] = cb_score
                else:
                    movie_scores[movie_id] = {
                        'cf_score': 0,
                        'cb_score': cb_score,
                        'movie_data': row
                    }
        
        # Calculate hybrid scores
        hybrid_recommendations = []
        for movie_id, scores in movie_scores.items():
            hybrid_score = cf_weight * scores['cf_score'] + cb_weight * scores['cb_score']
            
            movie_data = scores['movie_data']
            
            # Get additional movie statistics
            movie_ratings = self.ratings_df[self.ratings_df['movieId'] == movie_id]
            avg_rating = movie_ratings['rating'].mean() if not movie_ratings.empty else 0
            rating_count = len(movie_ratings)
            
            hybrid_recommendations.append({
                'movieId': movie_id,
                'title': movie_data['title'],
                'genres': movie_data['genres'],
                'hybrid_score': hybrid_score,
                'cf_score': scores['cf_score'],
                'cb_score': scores['cb_score'],
                'predicted_rating': scores['cf_score'] * 5,  # Convert back to 5-point scale
                'similarity_score': scores['cb_score'],
                'average_rating': avg_rating,
                'rating_count': rating_count
            })
        
        # Sort by hybrid score and return top N
        hybrid_df = pd.DataFrame(hybrid_recommendations)
        hybrid_df = hybrid_df.sort_values('hybrid_score', ascending=False)
        
        return hybrid_df.head(n_recommendations)
    
    def get_explanation(self, user_id: int, movie_id: int) -> Dict:
        """Provide explanation for why a movie was recommended"""
        
        explanation = {
            'movie_id': movie_id,
            'collaborative_factors': [],
            'content_factors': [],
            'overall_reasoning': ''
        }
        
        # Get movie information
        movie_info = self.movies_df[self.movies_df['movieId'] == movie_id]
        if movie_info.empty:
            return explanation
        
        movie_data = movie_info.iloc[0]
        explanation['movie_title'] = movie_data['title']
        explanation['genres'] = movie_data['genres']
        
        # Analyze collaborative factors
        try:
            # Get similar users who liked this movie
            similar_users = self.cf_model.get_similar_users(user_id, 5)
            if not similar_users.empty:
                explanation['collaborative_factors'].append(
                    f"Users similar to you (similarity: {similar_users['similarity'].mean():.3f}) liked this movie"
                )
            
            # Get user's rating prediction
            cf_recs = self.cf_model.get_user_recommendations(user_id, 50)
            movie_rec = cf_recs[cf_recs['movieId'] == movie_id]
            if not movie_rec.empty:
                predicted_rating = movie_rec.iloc[0]['predicted_rating']
                explanation['collaborative_factors'].append(
                    f"Predicted rating based on your preferences: {predicted_rating:.2f}/5.0"
                )
        except:
            pass
        
        # Analyze content factors
        try:
            # Get user's preferred genres
            user_genres = self._infer_user_genre_preferences(user_id)
            movie_genres = movie_data['genres'].split('|') if pd.notna(movie_data['genres']) else []
            
            matching_genres = set(user_genres).intersection(set(movie_genres))
            if matching_genres:
                explanation['content_factors'].append(
                    f"Matches your preferred genres: {', '.join(matching_genres)}"
                )
            
            # Get similar movies user has rated highly
            user_high_rated = self.ratings_df[
                (self.ratings_df['userId'] == user_id) & 
                (self.ratings_df['rating'] >= 4.0)
            ]['movieId'].tolist()
            
            if user_high_rated:
                # Check content similarity with user's highly rated movies
                similar_movies = []
                for rated_movie_id in user_high_rated[:5]:  # Check top 5
                    similar_recs = self.cb_model.get_movie_recommendations(rated_movie_id, 20)
                    if not similar_recs.empty and movie_id in similar_recs['movieId'].values:
                        rated_movie_info = self.movies_df[self.movies_df['movieId'] == rated_movie_id]
                        if not rated_movie_info.empty:
                            similar_movies.append(rated_movie_info.iloc[0]['title'])
                
                if similar_movies:
                    explanation['content_factors'].append(
                        f"Similar to movies you liked: {', '.join(similar_movies[:3])}"
                    )
        except:
            pass
        
        # Generate overall reasoning
        if explanation['collaborative_factors'] and explanation['content_factors']:
            explanation['overall_reasoning'] = "This movie is recommended based on both similar users' preferences and content similarity to your liked movies."
        elif explanation['collaborative_factors']:
            explanation['overall_reasoning'] = "This movie is recommended primarily based on collaborative filtering from similar users."
        elif explanation['content_factors']:
            explanation['overall_reasoning'] = "This movie is recommended primarily based on content similarity to your preferences."
        else:
            explanation['overall_reasoning'] = "This movie is recommended as part of our hybrid approach."
        
        return explanation
    
    def get_diversified_recommendations(self, user_id: int, preferred_genres: List[str] = None,
                                     n_recommendations: int = 10, diversity_factor: float = 0.3) -> pd.DataFrame:
        """Get diversified hybrid recommendations to avoid filter bubbles"""
        
        # Get more recommendations initially
        initial_recs = self.get_hybrid_recommendations(
            user_id, preferred_genres, n_recommendations * 3
        )
        
        if initial_recs.empty:
            return initial_recs
        
        # Apply diversification algorithm
        diversified_recs = []
        remaining_recs = initial_recs.copy()
        
        # Select first recommendation (highest score)
        first_rec = remaining_recs.iloc[0]
        diversified_recs.append(first_rec)
        remaining_recs = remaining_recs.iloc[1:]
        
        # Select remaining recommendations with diversity consideration
        for _ in range(min(n_recommendations - 1, len(remaining_recs))):
            best_score = -1
            best_idx = -1
            
            for idx, candidate in remaining_recs.iterrows():
                # Calculate diversity from already selected movies
                diversity_score = self._calculate_diversity_score(
                    candidate, diversified_recs
                )
                
                # Combine original score with diversity
                combined_score = (
                    (1 - diversity_factor) * candidate['hybrid_score'] +
                    diversity_factor * diversity_score
                )
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_idx = idx
            
            if best_idx != -1:
                selected_rec = remaining_recs.loc[best_idx]
                diversified_recs.append(selected_rec)
                remaining_recs = remaining_recs.drop(best_idx)
        
        return pd.DataFrame(diversified_recs)
    
    def _calculate_diversity_score(self, candidate_movie, selected_movies) -> float:
        """Calculate diversity score for a candidate movie"""
        
        if not selected_movies:
            return 1.0
        
        candidate_genres = set(candidate_movie['genres'].split('|')) if pd.notna(candidate_movie['genres']) else set()
        
        diversity_scores = []
        for selected_movie in selected_movies:
            selected_genres = set(selected_movie['genres'].split('|')) if pd.notna(selected_movie['genres']) else set()
            
            # Calculate genre diversity (Jaccard distance)
            if candidate_genres or selected_genres:
                genre_similarity = len(candidate_genres.intersection(selected_genres)) / len(candidate_genres.union(selected_genres))
                genre_diversity = 1 - genre_similarity
            else:
                genre_diversity = 0.5
            
            diversity_scores.append(genre_diversity)
        
        return np.mean(diversity_scores)
    
    def compare_approaches(self, user_id: int, preferred_genres: List[str] = None, 
                          n_recommendations: int = 10) -> Dict:
        """Compare different recommendation approaches for analysis"""
        
        results = {
            'collaborative_filtering': pd.DataFrame(),
            'content_based': pd.DataFrame(),
            'hybrid': pd.DataFrame(),
            'comparison_metrics': {}
        }
        
        # Get recommendations from each approach
        try:
            results['collaborative_filtering'] = self.cf_model.get_user_recommendations(user_id, n_recommendations)
        except:
            pass
        
        try:
            if preferred_genres:
                results['content_based'] = self.cb_model.get_genre_recommendations(preferred_genres, n_recommendations)
            else:
                user_genres = self._infer_user_genre_preferences(user_id)
                if user_genres:
                    results['content_based'] = self.cb_model.get_genre_recommendations(user_genres, n_recommendations)
        except:
            pass
        
        try:
            results['hybrid'] = self.get_hybrid_recommendations(user_id, preferred_genres, n_recommendations)
        except:
            pass
        
        # Calculate comparison metrics
        results['comparison_metrics'] = self._calculate_comparison_metrics(results)
        
        return results
    
    def _calculate_comparison_metrics(self, results: Dict) -> Dict:
        """Calculate metrics to compare different approaches"""
        
        metrics = {
            'recommendation_counts': {},
            'genre_diversity': {},
            'overlap_analysis': {}
        }
        
        # Count recommendations from each approach
        for approach, recs in results.items():
            if approach != 'comparison_metrics' and isinstance(recs, pd.DataFrame):
                metrics['recommendation_counts'][approach] = len(recs)
        
        # Calculate genre diversity for each approach
        for approach, recs in results.items():
            if approach != 'comparison_metrics' and isinstance(recs, pd.DataFrame) and not recs.empty:
                all_genres = []
                for genres in recs['genres']:
                    if pd.notna(genres):
                        all_genres.extend(genres.split('|'))
                metrics['genre_diversity'][approach] = len(set(all_genres))
        
        # Calculate overlap between approaches
        cf_movies = set(results['collaborative_filtering']['movieId']) if not results['collaborative_filtering'].empty else set()
        cb_movies = set(results['content_based']['movieId']) if not results['content_based'].empty else set()
        hybrid_movies = set(results['hybrid']['movieId']) if not results['hybrid'].empty else set()
        
        metrics['overlap_analysis'] = {
            'cf_cb_overlap': len(cf_movies.intersection(cb_movies)),
            'cf_hybrid_overlap': len(cf_movies.intersection(hybrid_movies)),
            'cb_hybrid_overlap': len(cb_movies.intersection(hybrid_movies)),
            'all_three_overlap': len(cf_movies.intersection(cb_movies).intersection(hybrid_movies))
        }
        
        return metrics
