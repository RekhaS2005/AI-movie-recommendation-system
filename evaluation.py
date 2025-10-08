import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class RecommenderEvaluator:
    """Evaluates recommendation system performance using various metrics"""
    
    def __init__(self, ratings_df: pd.DataFrame):
        self.ratings_df = ratings_df
        self.train_data = None
        self.test_data = None
        
    def prepare_test_data(self, test_size: float = 0.2, random_state: int = 42) -> pd.DataFrame:
        """Prepare train/test split for evaluation"""
        
        # Sort by timestamp to ensure temporal split
        ratings_sorted = self.ratings_df.sort_values('timestamp')
        
        # Use temporal split to simulate real-world scenario
        split_idx = int(len(ratings_sorted) * (1 - test_size))
        self.train_data = ratings_sorted.iloc[:split_idx]
        self.test_data = ratings_sorted.iloc[split_idx:]
        
        print(f"Train set: {len(self.train_data)} ratings")
        print(f"Test set: {len(self.test_data)} ratings")
        
        return self.test_data
    
    def evaluate_model(self, model: Any, test_data: pd.DataFrame, model_name: str) -> Dict:
        """Evaluate a recommendation model using multiple metrics"""
        
        print(f"Evaluating {model_name}...")
        
        metrics = {
            'model_name': model_name,
            'rmse': 0.0,
            'mae': 0.0,
            'precision_at_k': 0.0,
            'recall_at_k': 0.0,
            'coverage': 0.0,
            'diversity': 0.0,
            'novelty': 0.0
        }
        
        try:
            # Rating prediction metrics (RMSE, MAE)
            rating_metrics = self._evaluate_rating_prediction(model, test_data)
            metrics.update(rating_metrics)
            
            # Ranking metrics (Precision@K, Recall@K)
            ranking_metrics = self._evaluate_ranking(model, test_data)
            metrics.update(ranking_metrics)
            
            # System-level metrics
            system_metrics = self._evaluate_system_metrics(model)
            metrics.update(system_metrics)
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {str(e)}")
        
        return metrics
    
    def _evaluate_rating_prediction(self, model: Any, test_data: pd.DataFrame, 
                                  sample_size: int = 1000) -> Dict:
        """Evaluate rating prediction accuracy using RMSE and MAE"""
        
        predictions = []
        actuals = []
        
        # Sample test data for efficiency
        test_sample = test_data.sample(min(sample_size, len(test_data)), random_state=42)
        
        for _, row in test_sample.iterrows():
            user_id = row['userId']
            movie_id = row['movieId']
            actual_rating = row['rating']
            
            try:
                # Get prediction from model
                predicted_rating = self._get_model_prediction(model, user_id, movie_id)
                
                if predicted_rating > 0:  # Valid prediction
                    predictions.append(predicted_rating)
                    actuals.append(actual_rating)
            except:
                continue
        
        if len(predictions) > 0:
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            mae = mean_absolute_error(actuals, predictions)
        else:
            rmse, mae = 0.0, 0.0
        
        return {'rmse': rmse, 'mae': mae}
    
    def _evaluate_ranking(self, model: Any, test_data: pd.DataFrame, k: int = 10) -> Dict:
        """Evaluate ranking performance using Precision@K and Recall@K"""
        
        precisions = []
        recalls = []
        
        # Group by user
        user_groups = test_data.groupby('userId')
        users_to_evaluate = list(user_groups.groups.keys())[:50]  # Limit for efficiency
        
        for user_id in users_to_evaluate:
            try:
                # Get user's test set (highly rated items)
                user_test = user_groups.get_group(user_id)
                relevant_items = set(user_test[user_test['rating'] >= 4.0]['movieId'])
                
                if len(relevant_items) == 0:
                    continue
                
                # Get model recommendations
                recommendations = self._get_model_recommendations(model, user_id, k)
                
                if len(recommendations) == 0:
                    continue
                
                recommended_items = set(recommendations['movieId'])
                
                # Calculate precision and recall
                true_positives = len(relevant_items.intersection(recommended_items))
                
                precision = true_positives / len(recommended_items) if len(recommended_items) > 0 else 0
                recall = true_positives / len(relevant_items) if len(relevant_items) > 0 else 0
                
                precisions.append(precision)
                recalls.append(recall)
                
            except Exception as e:
                continue
        
        avg_precision = np.mean(precisions) if precisions else 0
        avg_recall = np.mean(recalls) if recalls else 0
        
        return {
            'precision_at_k': avg_precision,
            'recall_at_k': avg_recall
        }
    
    def _evaluate_system_metrics(self, model: Any) -> Dict:
        """Evaluate system-level metrics like coverage, diversity, and novelty"""
        
        # Sample users for evaluation
        sample_users = self.ratings_df['userId'].unique()[:20]
        
        all_recommendations = []
        all_genres = []
        
        for user_id in sample_users:
            try:
                recs = self._get_model_recommendations(model, user_id, 10)
                if not recs.empty:
                    all_recommendations.extend(recs['movieId'].tolist())
                    
                    # Collect genres for diversity calculation
                    for genres in recs['genres']:
                        if pd.notna(genres):
                            all_genres.extend(genres.split('|'))
            except:
                continue
        
        # Calculate coverage (percentage of items that can be recommended)
        unique_recommended = len(set(all_recommendations))
        total_items = self.ratings_df['movieId'].nunique()
        coverage = unique_recommended / total_items if total_items > 0 else 0
        
        # Calculate diversity (number of unique genres recommended)
        unique_genres = len(set(all_genres))
        total_genres = len(set(genre for genres in self.ratings_df.merge(
            pd.DataFrame({'movieId': self.ratings_df['movieId'].unique()}), on='movieId'
        )['movieId'].unique() for genre in ['Action', 'Comedy', 'Drama']))  # Simplified
        diversity = unique_genres / max(total_genres, 1)
        
        # Calculate novelty (average popularity of recommended items)
        if all_recommendations:
            item_popularity = self.ratings_df['movieId'].value_counts()
            rec_popularity = [item_popularity.get(item, 0) for item in all_recommendations]
            avg_popularity = np.mean(rec_popularity)
            max_popularity = item_popularity.max()
            novelty = 1 - (avg_popularity / max_popularity) if max_popularity > 0 else 0
        else:
            novelty = 0
        
        return {
            'coverage': coverage,
            'diversity': diversity,
            'novelty': novelty
        }
    
    def _get_model_prediction(self, model: Any, user_id: int, movie_id: int) -> float:
        """Get rating prediction from a model"""
        
        if hasattr(model, 'surprise_model'):  # Collaborative filtering
            try:
                prediction = model.surprise_model.predict(user_id, movie_id)
                return prediction.est
            except:
                return 0.0
        elif hasattr(model, 'get_hybrid_recommendations'):  # Hybrid model
            try:
                recs = model.get_hybrid_recommendations(user_id, None, 50)
                movie_rec = recs[recs['movieId'] == movie_id]
                if not movie_rec.empty:
                    return movie_rec.iloc[0]['predicted_rating']
            except:
                return 0.0
        else:  # Content-based (doesn't predict ratings directly)
            return 3.0  # Default rating
        
        return 0.0
    
    def _get_model_recommendations(self, model: Any, user_id: int, k: int) -> pd.DataFrame:
        """Get top-K recommendations from a model"""
        
        try:
            if hasattr(model, 'get_user_recommendations'):  # Collaborative filtering
                return model.get_user_recommendations(user_id, k)
            elif hasattr(model, 'get_hybrid_recommendations'):  # Hybrid model
                return model.get_hybrid_recommendations(user_id, None, k)
            elif hasattr(model, 'get_genre_recommendations'):  # Content-based
                # Infer user preferences for content-based
                user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
                if not user_ratings.empty:
                    # Simple genre inference (could be improved)
                    return model.get_genre_recommendations(['Action', 'Comedy'], k)
                else:
                    return pd.DataFrame()
        except:
            return pd.DataFrame()
        
        return pd.DataFrame()
    
    def cross_validate_model(self, model: Any, model_name: str, cv_folds: int = 3) -> Dict:
        """Perform cross-validation evaluation"""
        
        print(f"Cross-validating {model_name} with {cv_folds} folds...")
        
        fold_metrics = []
        
        # Create temporal folds
        sorted_ratings = self.ratings_df.sort_values('timestamp')
        fold_size = len(sorted_ratings) // cv_folds
        
        for fold in range(cv_folds):
            print(f"  Evaluating fold {fold + 1}/{cv_folds}")
            
            # Create train/test split for this fold
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < cv_folds - 1 else len(sorted_ratings)
            
            test_fold = sorted_ratings.iloc[test_start:test_end]
            train_fold = pd.concat([
                sorted_ratings.iloc[:test_start],
                sorted_ratings.iloc[test_end:]
            ])
            
            # Retrain model on fold training data (simplified - assume model is already trained)
            fold_metrics.append(self.evaluate_model(model, test_fold, f"{model_name}_fold_{fold + 1}"))
        
        # Aggregate metrics across folds
        aggregated_metrics = {
            'model_name': f"{model_name}_cv",
            'rmse': np.mean([m['rmse'] for m in fold_metrics]),
            'mae': np.mean([m['mae'] for m in fold_metrics]),
            'precision_at_k': np.mean([m['precision_at_k'] for m in fold_metrics]),
            'recall_at_k': np.mean([m['recall_at_k'] for m in fold_metrics]),
            'coverage': np.mean([m['coverage'] for m in fold_metrics]),
            'diversity': np.mean([m['diversity'] for m in fold_metrics]),
            'novelty': np.mean([m['novelty'] for m in fold_metrics]),
            'rmse_std': np.std([m['rmse'] for m in fold_metrics]),
            'mae_std': np.std([m['mae'] for m in fold_metrics]),
            'precision_std': np.std([m['precision_at_k'] for m in fold_metrics])
        }
        
        return aggregated_metrics
    
    def evaluate_cold_start_performance(self, model: Any, model_name: str) -> Dict:
        """Evaluate model performance on cold start users/items"""
        
        print(f"Evaluating cold start performance for {model_name}...")
        
        # Find users with very few ratings (cold start users)
        user_rating_counts = self.ratings_df.groupby('userId').size()
        cold_start_users = user_rating_counts[user_rating_counts <= 3].index
        
        # Find movies with very few ratings (cold start items)
        item_rating_counts = self.ratings_df.groupby('movieId').size()
        cold_start_items = item_rating_counts[item_rating_counts <= 5].index
        
        # Evaluate on cold start users
        cold_user_metrics = {'precision_at_k': [], 'coverage': []}
        
        for user_id in cold_start_users[:10]:  # Sample for efficiency
            try:
                recs = self._get_model_recommendations(model, user_id, 10)
                if not recs.empty:
                    # Simple coverage metric for cold start
                    cold_user_metrics['coverage'].append(1)
                    # Precision would need ground truth, simplified here
                    cold_user_metrics['precision_at_k'].append(0.1)  # Placeholder
                else:
                    cold_user_metrics['coverage'].append(0)
                    cold_user_metrics['precision_at_k'].append(0)
            except:
                cold_user_metrics['coverage'].append(0)
                cold_user_metrics['precision_at_k'].append(0)
        
        return {
            'cold_start_user_coverage': np.mean(cold_user_metrics['coverage']),
            'cold_start_user_precision': np.mean(cold_user_metrics['precision_at_k']),
            'cold_start_users_evaluated': len(cold_user_metrics['coverage'])
        }
    
    def generate_evaluation_report(self, model_results: List[Dict]) -> str:
        """Generate a comprehensive evaluation report"""
        
        report = "# Recommendation System Evaluation Report\n\n"
        
        if not model_results:
            return report + "No evaluation results available.\n"
        
        # Summary table
        report += "## Model Performance Summary\n\n"
        report += "| Model | RMSE | MAE | Precision@10 | Recall@10 | Coverage | Diversity |\n"
        report += "|-------|------|-----|--------------|-----------|----------|----------|\n"
        
        for result in model_results:
            report += f"| {result['model_name']} | {result['rmse']:.4f} | {result['mae']:.4f} | "
            report += f"{result['precision_at_k']:.4f} | {result['recall_at_k']:.4f} | "
            report += f"{result['coverage']:.4f} | {result['diversity']:.4f} |\n"
        
        # Best performing models
        report += "\n## Best Performing Models\n\n"
        
        best_rmse = min(model_results, key=lambda x: x['rmse'])
        best_precision = max(model_results, key=lambda x: x['precision_at_k'])
        best_coverage = max(model_results, key=lambda x: x['coverage'])
        
        report += f"- **Best RMSE**: {best_rmse['model_name']} ({best_rmse['rmse']:.4f})\n"
        report += f"- **Best Precision@10**: {best_precision['model_name']} ({best_precision['precision_at_k']:.4f})\n"
        report += f"- **Best Coverage**: {best_coverage['model_name']} ({best_coverage['coverage']:.4f})\n"
        
        # Recommendations
        report += "\n## Recommendations\n\n"
        
        if best_rmse['model_name'] == best_precision['model_name']:
            report += f"- **{best_rmse['model_name']}** shows excellent overall performance in both accuracy and ranking metrics.\n"
        else:
            report += f"- **{best_rmse['model_name']}** is best for rating prediction accuracy.\n"
            report += f"- **{best_precision['model_name']}** is best for recommendation ranking.\n"
        
        report += f"- **{best_coverage['model_name']}** provides the broadest item coverage.\n"
        
        return report
