import pandas as pd
import numpy as np
import os
import pickle
from typing import Any, Dict, List, Tuple
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def save_model(model: Any, filename: str, model_dir: str = "models") -> None:
    """Save a trained model to disk"""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    filepath = os.path.join(model_dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")

def load_model(filename: str, model_dir: str = "models") -> Any:
    """Load a trained model from disk"""
    filepath = os.path.join(model_dir, filename)
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filepath}")
        return model
    else:
        print(f"Model file {filepath} not found")
        return None

def create_user_item_heatmap(ratings_df: pd.DataFrame, sample_users: int = 50, 
                           sample_movies: int = 50) -> go.Figure:
    """Create a heatmap visualization of user-item rating matrix"""
    
    # Sample users and movies for visualization
    top_users = ratings_df['userId'].value_counts().head(sample_users).index
    top_movies = ratings_df['movieId'].value_counts().head(sample_movies).index
    
    # Create user-item matrix
    sample_ratings = ratings_df[
        (ratings_df['userId'].isin(top_users)) & 
        (ratings_df['movieId'].isin(top_movies))
    ]
    
    pivot_matrix = sample_ratings.pivot_table(
        index='userId', columns='movieId', values='rating', fill_value=0
    )
    
    # Create heatmap
    fig = px.imshow(
        pivot_matrix.values,
        labels=dict(x="Movie ID", y="User ID", color="Rating"),
        x=pivot_matrix.columns,
        y=pivot_matrix.index,
        color_continuous_scale="Viridis",
        title=f"User-Item Rating Matrix (Top {sample_users} users, {sample_movies} movies)"
    )
    
    fig.update_layout(height=600, width=800)
    return fig

def create_rating_distribution_plot(ratings_df: pd.DataFrame) -> go.Figure:
    """Create rating distribution visualization"""
    
    rating_counts = ratings_df['rating'].value_counts().sort_index()
    
    fig = go.Figure(data=[
        go.Bar(x=rating_counts.index, y=rating_counts.values,
               marker_color='skyblue', text=rating_counts.values,
               textposition='auto')
    ])
    
    fig.update_layout(
        title="Distribution of Ratings",
        xaxis_title="Rating",
        yaxis_title="Count",
        showlegend=False
    )
    
    return fig

def create_genre_analysis_plot(movies_df: pd.DataFrame) -> go.Figure:
    """Create genre analysis visualization"""
    
    # Extract all genres
    all_genres = []
    for genres in movies_df['genres'].dropna():
        if genres != '(no genres listed)':
            all_genres.extend(genres.split('|'))
    
    genre_counts = pd.Series(all_genres).value_counts().head(15)
    
    fig = px.bar(
        x=genre_counts.values,
        y=genre_counts.index,
        orientation='h',
        title="Top 15 Movie Genres",
        labels={'x': 'Number of Movies', 'y': 'Genre'}
    )
    
    fig.update_layout(height=500)
    return fig

def create_model_comparison_plot(evaluation_results: List[Dict]) -> go.Figure:
    """Create model comparison visualization"""
    
    if not evaluation_results:
        return go.Figure()
    
    models = [result['model_name'] for result in evaluation_results]
    rmse_values = [result['rmse'] for result in evaluation_results]
    precision_values = [result['precision_at_k'] for result in evaluation_results]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('RMSE (Lower is Better)', 'Precision@10 (Higher is Better)'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # RMSE plot
    fig.add_trace(
        go.Bar(x=models, y=rmse_values, name='RMSE', marker_color='lightcoral'),
        row=1, col=1
    )
    
    # Precision plot
    fig.add_trace(
        go.Bar(x=models, y=precision_values, name='Precision@10', marker_color='lightblue'),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        title_text="Model Performance Comparison"
    )
    
    return fig

def create_recommendation_explanation_plot(explanation: Dict) -> go.Figure:
    """Create visualization for recommendation explanation"""
    
    if not explanation:
        return go.Figure()
    
    factors = []
    scores = []
    colors = []
    
    # Add collaborative factors
    for factor in explanation.get('collaborative_factors', []):
        factors.append(f"CF: {factor[:50]}...")
        scores.append(0.8)  # Placeholder score
        colors.append('lightblue')
    
    # Add content factors
    for factor in explanation.get('content_factors', []):
        factors.append(f"CB: {factor[:50]}...")
        scores.append(0.6)  # Placeholder score
        colors.append('lightgreen')
    
    if not factors:
        return go.Figure()
    
    fig = go.Figure(data=[
        go.Bar(y=factors, x=scores, orientation='h',
               marker_color=colors, text=scores,
               textposition='auto')
    ])
    
    fig.update_layout(
        title=f"Recommendation Explanation: {explanation.get('movie_title', 'Unknown')}",
        xaxis_title="Relevance Score",
        yaxis_title="Factors",
        height=max(300, len(factors) * 40)
    )
    
    return fig

def calculate_sparsity(ratings_df: pd.DataFrame) -> float:
    """Calculate sparsity of the rating matrix"""
    total_possible_ratings = ratings_df['userId'].nunique() * ratings_df['movieId'].nunique()
    actual_ratings = len(ratings_df)
    sparsity = 1 - (actual_ratings / total_possible_ratings)
    return sparsity

def get_user_statistics(ratings_df: pd.DataFrame, user_id: int) -> Dict:
    """Get statistics for a specific user"""
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    
    if user_ratings.empty:
        return {}
    
    stats = {
        'user_id': user_id,
        'total_ratings': len(user_ratings),
        'average_rating': user_ratings['rating'].mean(),
        'rating_std': user_ratings['rating'].std(),
        'min_rating': user_ratings['rating'].min(),
        'max_rating': user_ratings['rating'].max(),
        'rating_range': user_ratings['rating'].max() - user_ratings['rating'].min(),
        'first_rating_date': user_ratings['timestamp'].min(),
        'last_rating_date': user_ratings['timestamp'].max()
    }
    
    return stats

def get_movie_statistics(ratings_df: pd.DataFrame, movies_df: pd.DataFrame, movie_id: int) -> Dict:
    """Get statistics for a specific movie"""
    movie_ratings = ratings_df[ratings_df['movieId'] == movie_id]
    movie_info = movies_df[movies_df['movieId'] == movie_id]
    
    if movie_ratings.empty or movie_info.empty:
        return {}
    
    movie_data = movie_info.iloc[0]
    
    stats = {
        'movie_id': movie_id,
        'title': movie_data['title'],
        'genres': movie_data['genres'],
        'year': movie_data.get('year', 'Unknown'),
        'total_ratings': len(movie_ratings),
        'average_rating': movie_ratings['rating'].mean(),
        'rating_std': movie_ratings['rating'].std(),
        'min_rating': movie_ratings['rating'].min(),
        'max_rating': movie_ratings['rating'].max(),
        'rating_distribution': movie_ratings['rating'].value_counts().to_dict()
    }
    
    return stats

def format_large_number(number: int) -> str:
    """Format large numbers for display"""
    if number >= 1_000_000:
        return f"{number/1_000_000:.1f}M"
    elif number >= 1_000:
        return f"{number/1_000:.1f}K"
    else:
        return str(number)

def create_diversity_metrics(recommendations: pd.DataFrame) -> Dict:
    """Calculate diversity metrics for a set of recommendations"""
    
    if recommendations.empty:
        return {}
    
    # Genre diversity
    all_genres = []
    for genres in recommendations['genres'].dropna():
        all_genres.extend(genres.split('|'))
    
    unique_genres = len(set(all_genres))
    total_genre_mentions = len(all_genres)
    genre_diversity = unique_genres / max(total_genre_mentions, 1)
    
    # Rating diversity (if available)
    rating_diversity = 0
    if 'average_rating' in recommendations.columns:
        ratings = recommendations['average_rating'].dropna()
        if len(ratings) > 1:
            rating_diversity = ratings.std() / ratings.mean() if ratings.mean() > 0 else 0
    
    # Year diversity (if available)
    year_diversity = 0
    if 'year' in recommendations.columns:
        years = recommendations['year'].dropna()
        if len(years) > 1:
            year_diversity = len(set(years)) / len(years)
    
    return {
        'genre_diversity': genre_diversity,
        'unique_genres': unique_genres,
        'rating_diversity': rating_diversity,
        'year_diversity': year_diversity,
        'total_recommendations': len(recommendations)
    }

def export_recommendations_to_csv(recommendations: pd.DataFrame, filename: str = "recommendations.csv") -> str:
    """Export recommendations to CSV file"""
    
    if recommendations.empty:
        return "No recommendations to export"
    
    try:
        recommendations.to_csv(filename, index=False)
        return f"Recommendations exported to {filename}"
    except Exception as e:
        return f"Error exporting recommendations: {str(e)}"

def create_interactive_recommendation_table(recommendations: pd.DataFrame) -> None:
    """Create an interactive recommendation table in Streamlit"""
    
    if recommendations.empty:
        st.warning("No recommendations available")
        return
    
    # Add selection functionality
    selected_movies = st.multiselect(
        "Select movies to compare:",
        options=recommendations['title'].tolist(),
        default=recommendations['title'].tolist()[:3] if len(recommendations) >= 3 else recommendations['title'].tolist()
    )
    
    if selected_movies:
        filtered_recs = recommendations[recommendations['title'].isin(selected_movies)]
        
        # Display selected recommendations
        st.dataframe(filtered_recs, use_container_width=True)
        
        # Export functionality
        if st.button("Export Selected to CSV"):
            export_message = export_recommendations_to_csv(filtered_recs, "selected_recommendations.csv")
            st.success(export_message)

def validate_input_data(ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> List[str]:
    """Validate input data and return list of issues"""
    
    issues = []
    
    # Check for required columns
    required_rating_cols = ['userId', 'movieId', 'rating', 'timestamp']
    for col in required_rating_cols:
        if col not in ratings_df.columns:
            issues.append(f"Missing required column in ratings: {col}")
    
    required_movie_cols = ['movieId', 'title', 'genres']
    for col in required_movie_cols:
        if col not in movies_df.columns:
            issues.append(f"Missing required column in movies: {col}")
    
    # Check for data quality issues
    if ratings_df['rating'].isnull().any():
        issues.append("Null ratings found in data")
    
    if (ratings_df['rating'] < 0).any() or (ratings_df['rating'] > 5).any():
        issues.append("Invalid rating values found (should be between 0 and 5)")
    
    if movies_df['title'].isnull().any():
        issues.append("Movies with missing titles found")
    
    # Check data consistency
    rating_movies = set(ratings_df['movieId'].unique())
    catalog_movies = set(movies_df['movieId'].unique())
    
    missing_in_catalog = rating_movies - catalog_movies
    if missing_in_catalog:
        issues.append(f"{len(missing_in_catalog)} movies in ratings not found in movies catalog")
    
    return issues

def display_data_quality_report(ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> None:
    """Display data quality report in Streamlit"""
    
    st.subheader("Data Quality Report")
    
    issues = validate_input_data(ratings_df, movies_df)
    
    if issues:
        st.warning("Data quality issues found:")
        for issue in issues:
            st.write(f"• {issue}")
    else:
        st.success("No data quality issues found!")
    
    # Basic statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Data Sparsity", f"{calculate_sparsity(ratings_df):.4f}")
    
    with col2:
        avg_ratings_per_user = ratings_df.groupby('userId').size().mean()
        st.metric("Avg Ratings/User", f"{avg_ratings_per_user:.1f}")
    
    with col3:
        avg_ratings_per_movie = ratings_df.groupby('movieId').size().mean()
        st.metric("Avg Ratings/Movie", f"{avg_ratings_per_movie:.1f}")
