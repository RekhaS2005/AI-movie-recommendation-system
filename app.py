import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import pickle
from data_loader import DataLoader
from collaborative_filtering import CollaborativeFiltering
from content_based_filtering import ContentBasedFiltering
from hybrid_recommender import HybridRecommender
from evaluation import RecommenderEvaluator
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

@st.cache_data
def load_and_prepare_data():
    """Load and prepare MovieLens data"""
    data_loader = DataLoader()
    return data_loader.load_movielens_data()

@st.cache_resource
def initialize_models(_ratings_df, _movies_df):
    """Initialize and train recommendation models"""
    # Collaborative filtering
    cf_model = CollaborativeFiltering(_ratings_df, _movies_df)
    
    # Content-based filtering
    cb_model = ContentBasedFiltering(_movies_df)
    
    # Hybrid recommender
    hybrid_model = HybridRecommender(cf_model, cb_model)
    
    return cf_model, cb_model, hybrid_model

def main():
    st.title("🎬 Movie Recommendation System")
    st.markdown("### A comprehensive recommendation engine using collaborative, content-based, and hybrid filtering")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["Dataset Overview", "Model Performance", "Get Recommendations", "Model Comparison"]
    )
    
    try:
        # Load data
        with st.spinner("Loading MovieLens dataset..."):
            ratings_df, movies_df = load_and_prepare_data()
        
        # Initialize models
        with st.spinner("Initializing recommendation models..."):
            cf_model, cb_model, hybrid_model = initialize_models(ratings_df, movies_df)
        
        if page == "Dataset Overview":
            show_dataset_overview(ratings_df, movies_df)
        elif page == "Model Performance":
            show_model_performance(cf_model, cb_model, hybrid_model, ratings_df)
        elif page == "Get Recommendations":
            show_recommendations(cf_model, cb_model, hybrid_model, ratings_df, movies_df)
        elif page == "Model Comparison":
            show_model_comparison(cf_model, cb_model, hybrid_model, ratings_df)
            
    except Exception as e:
        st.error(f"Error loading data or models: {str(e)}")
        st.info("Please ensure you have a stable internet connection to download the MovieLens dataset.")

def show_dataset_overview(ratings_df, movies_df):
    """Display dataset overview and statistics"""
    st.header("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Ratings", f"{len(ratings_df):,}")
    with col2:
        st.metric("Total Movies", f"{len(movies_df):,}")
    with col3:
        st.metric("Total Users", f"{ratings_df['userId'].nunique():,}")
    with col4:
        st.metric("Average Rating", f"{ratings_df['rating'].mean():.2f}")
    
    # Rating distribution
    st.subheader("Rating Distribution")
    rating_counts = ratings_df['rating'].value_counts().sort_index()
    fig = px.bar(x=rating_counts.index, y=rating_counts.values,
                 labels={'x': 'Rating', 'y': 'Count'},
                 title="Distribution of Ratings")
    st.plotly_chart(fig, use_container_width=True)
    
    # Top rated movies
    st.subheader("Top 20 Most Rated Movies")
    movie_rating_counts = ratings_df.groupby('movieId').size().reset_index(name='rating_count')
    top_movies = movie_rating_counts.merge(movies_df, on='movieId').nlargest(20, 'rating_count')
    
    fig = px.bar(top_movies, x='rating_count', y='title', orientation='h',
                 title="Most Rated Movies", labels={'rating_count': 'Number of Ratings'})
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Genre distribution
    st.subheader("Genre Distribution")
    all_genres = []
    for genres in movies_df['genres']:
        if pd.notna(genres):
            all_genres.extend(genres.split('|'))
    
    genre_counts = pd.Series(all_genres).value_counts().head(15)
    fig = px.bar(x=genre_counts.values, y=genre_counts.index, orientation='h',
                 title="Top 15 Genres", labels={'x': 'Count', 'y': 'Genre'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Sample data preview
    st.subheader("Sample Data")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Ratings Sample:**")
        st.dataframe(ratings_df.head(10))
    
    with col2:
        st.write("**Movies Sample:**")
        st.dataframe(movies_df.head(10))

def show_model_performance(cf_model, cb_model, hybrid_model, ratings_df):
    """Display model performance metrics"""
    st.header("📈 Model Performance Evaluation")
    
    # Evaluate models
    evaluator = RecommenderEvaluator(ratings_df)
    
    with st.spinner("Evaluating model performance..."):
        # Prepare test data
        test_data = evaluator.prepare_test_data()
        
        # Evaluate each model
        cf_metrics = evaluator.evaluate_model(cf_model, test_data, "Collaborative Filtering")
        cb_metrics = evaluator.evaluate_model(cb_model, test_data, "Content-Based")
        hybrid_metrics = evaluator.evaluate_model(hybrid_model, test_data, "Hybrid")
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Collaborative Filtering")
        st.metric("RMSE", f"{cf_metrics['rmse']:.4f}")
        st.metric("MAE", f"{cf_metrics['mae']:.4f}")
        st.metric("Precision@10", f"{cf_metrics['precision_at_k']:.4f}")
    
    with col2:
        st.subheader("Content-Based")
        st.metric("RMSE", f"{cb_metrics['rmse']:.4f}")
        st.metric("MAE", f"{cb_metrics['mae']:.4f}")
        st.metric("Precision@10", f"{cb_metrics['precision_at_k']:.4f}")
    
    with col3:
        st.subheader("Hybrid")
        st.metric("RMSE", f"{hybrid_metrics['rmse']:.4f}")
        st.metric("MAE", f"{hybrid_metrics['mae']:.4f}")
        st.metric("Precision@10", f"{hybrid_metrics['precision_at_k']:.4f}")
    
    # Performance comparison chart
    st.subheader("Performance Comparison")
    
    metrics_data = {
        'Model': ['Collaborative Filtering', 'Content-Based', 'Hybrid'],
        'RMSE': [cf_metrics['rmse'], cb_metrics['rmse'], hybrid_metrics['rmse']],
        'MAE': [cf_metrics['mae'], cb_metrics['mae'], hybrid_metrics['mae']],
        'Precision@10': [cf_metrics['precision_at_k'], cb_metrics['precision_at_k'], hybrid_metrics['precision_at_k']]
    }
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('RMSE (Lower is Better)', 'MAE (Lower is Better)', 'Precision@10 (Higher is Better)')
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    fig.add_trace(go.Bar(x=metrics_data['Model'], y=metrics_data['RMSE'], 
                        marker_color=colors, name='RMSE'), row=1, col=1)
    fig.add_trace(go.Bar(x=metrics_data['Model'], y=metrics_data['MAE'], 
                        marker_color=colors, name='MAE', showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(x=metrics_data['Model'], y=metrics_data['Precision@10'], 
                        marker_color=colors, name='Precision@10', showlegend=False), row=1, col=3)
    
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Export model performance metrics
    st.subheader("📥 Export Performance Metrics")
    metrics_export_df = pd.DataFrame(metrics_data)
    
    col1, col2 = st.columns(2)
    with col1:
        csv_metrics = metrics_export_df.to_csv(index=False)
        st.download_button(
            label="Download Metrics as CSV",
            data=csv_metrics,
            file_name="model_performance_metrics.csv",
            mime="text/csv",
            key="metrics_csv"
        )
    
    with col2:
        json_metrics = metrics_export_df.to_json(orient='records', indent=2)
        st.download_button(
            label="Download Metrics as JSON",
            data=json_metrics,
            file_name="model_performance_metrics.json",
            mime="application/json",
            key="metrics_json"
        )

def show_recommendations(cf_model, cb_model, hybrid_model, ratings_df, movies_df):
    """Show personalized recommendations"""
    st.header("🎯 Get Personalized Recommendations")
    
    # User input section
    st.subheader("Select Your Preferences")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Select user ID (for collaborative filtering)
        user_ids = sorted(ratings_df['userId'].unique())
        selected_user = st.selectbox("Select User ID (for collaborative filtering):", 
                                   options=user_ids[:100])  # Limit to first 100 users for performance
        
        # Select favorite genres
        all_genres = set()
        for genres in movies_df['genres'].dropna():
            all_genres.update(genres.split('|'))
        all_genres = sorted(list(all_genres))
        
        selected_genres = st.multiselect("Select Favorite Genres:", 
                                       options=all_genres,
                                       default=['Action', 'Comedy'])
    
    with col2:
        # Select number of recommendations
        num_recommendations = st.slider("Number of Recommendations:", 
                                       min_value=5, max_value=20, value=10)
        
        # Select recommendation method
        method = st.selectbox("Recommendation Method:", 
                            options=['Hybrid', 'Collaborative Filtering', 'Content-Based'])
    
    if st.button("Get Recommendations", type="primary"):
        with st.spinner("Generating recommendations..."):
            try:
                if method == 'Collaborative Filtering':
                    recommendations = cf_model.get_user_recommendations(selected_user, num_recommendations)
                elif method == 'Content-Based':
                    # For content-based, use genres as user profile
                    recommendations = cb_model.get_genre_recommendations(selected_genres, num_recommendations)
                else:  # Hybrid
                    recommendations = hybrid_model.get_hybrid_recommendations(
                        selected_user, selected_genres, num_recommendations)
                
                if not recommendations.empty:
                    st.subheader(f"Top {num_recommendations} Recommendations using {method}")
                    
                    # Display recommendations in a nice format
                    for idx, row in recommendations.iterrows():
                        with st.container():
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.write(f"**{row['title']}**")
                                st.write(f"Genres: {row['genres']}")
                                if 'predicted_rating' in row:
                                    st.write(f"Predicted Rating: ⭐ {row['predicted_rating']:.2f}")
                                if 'similarity_score' in row:
                                    st.write(f"Similarity Score: {row['similarity_score']:.3f}")
                            
                            with col2:
                                if 'average_rating' in row:
                                    st.metric("Avg Rating", f"{row['average_rating']:.2f}")
                                if 'rating_count' in row:
                                    st.metric("# Ratings", f"{row['rating_count']:,}")
                            
                            st.divider()
                    
                    # Export functionality
                    st.subheader("📥 Export Recommendations")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        csv_data = recommendations.to_csv(index=False)
                        st.download_button(
                            label="Download as CSV",
                            data=csv_data,
                            file_name=f"recommendations_{method.lower().replace(' ', '_')}.csv",
                            mime="text/csv",
                            key=f"csv_{method}"
                        )
                    
                    with col2:
                        json_data = recommendations.to_json(orient='records', indent=2)
                        st.download_button(
                            label="Download as JSON",
                            data=json_data,
                            file_name=f"recommendations_{method.lower().replace(' ', '_')}.json",
                            mime="application/json",
                            key=f"json_{method}"
                        )
                    
                    with col3:
                        # Excel export
                        import io
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                            recommendations.to_excel(writer, index=False, sheet_name='Recommendations')
                        excel_data = buffer.getvalue()
                        st.download_button(
                            label="Download as Excel",
                            data=excel_data,
                            file_name=f"recommendations_{method.lower().replace(' ', '_')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key=f"excel_{method}"
                        )
                else:
                    st.warning("No recommendations found. Please try different parameters.")
                    
            except Exception as e:
                st.error(f"Error generating recommendations: {str(e)}")
    
    # Show user's rating history (for collaborative filtering)
    if selected_user:
        st.subheader(f"User {selected_user}'s Recent Ratings")
        user_ratings = ratings_df[ratings_df['userId'] == selected_user].merge(
            movies_df, on='movieId').sort_values('timestamp', ascending=False).head(10)
        
        if not user_ratings.empty:
            for _, row in user_ratings.iterrows():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"**{row['title']}** ({row['genres']})")
                with col2:
                    st.write(f"⭐ {row['rating']}")
        else:
            st.info("No rating history found for this user.")

def show_model_comparison(cf_model, cb_model, hybrid_model, ratings_df):
    """Show detailed model comparison and analysis"""
    st.header("🔍 Model Comparison & Analysis")
    
    # Model characteristics
    st.subheader("Model Characteristics")
    
    characteristics = {
        'Aspect': [
            'Data Dependency',
            'Cold Start Problem',
            'Diversity',
            'Scalability',
            'Interpretability'
        ],
        'Collaborative Filtering': [
            'User-Item interactions',
            'High (needs user history)',
            'Medium',
            'Medium',
            'Low'
        ],
        'Content-Based': [
            'Item features/metadata',
            'Low (works with item info)',
            'Low (similar items)',
            'High',
            'High'
        ],
        'Hybrid': [
            'Both interactions & features',
            'Medium (combines both)',
            'High',
            'Medium',
            'Medium'
        ]
    }
    
    comparison_df = pd.DataFrame(characteristics)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Algorithm details
    st.subheader("Algorithm Implementation Details")
    
    with st.expander("Collaborative Filtering Details"):
        st.write("""
        **User-Based Collaborative Filtering:**
        - Uses cosine similarity to find similar users
        - Recommends items liked by similar users
        - Handles the cold start problem for new users poorly
        
        **Item-Based Collaborative Filtering:**
        - Uses co-occurrence matrix for item similarities
        - More stable than user-based for sparse data
        - Better performance with large item catalogs
        
        **Matrix Factorization (SVD):**
        - Reduces dimensionality of user-item matrix
        - Captures latent factors in user preferences
        - More robust to data sparsity
        """)
    
    with st.expander("Content-Based Filtering Details"):
        st.write("""
        **TF-IDF on Movie Metadata:**
        - Analyzes movie genres, titles, and descriptions
        - Creates feature vectors for each movie
        - Uses cosine similarity for content matching
        
        **Advantages:**
        - No cold start problem for new items
        - Provides explanations for recommendations
        - Works well with rich item metadata
        
        **Limitations:**
        - Limited diversity in recommendations
        - Requires good feature engineering
        - May create filter bubbles
        """)
    
    with st.expander("Hybrid Approach Details"):
        st.write("""
        **Combination Strategy:**
        - Weighted combination of collaborative and content-based scores
        - Dynamic weighting based on user profile completeness
        - Fallback mechanisms for cold start scenarios
        
        **Benefits:**
        - Combines strengths of both approaches
        - Better handling of various user scenarios
        - Improved recommendation diversity and accuracy
        """)
    
    # Performance insights
    st.subheader("Performance Insights")
    
    insights = [
        "**Collaborative Filtering** works best for users with substantial rating history",
        "**Content-Based** excels at recommending niche or new items with good metadata",
        "**Hybrid approach** provides the most balanced and robust recommendations",
        "**Matrix factorization** helps with data sparsity but may lose interpretability",
        "**Evaluation metrics** show trade-offs between accuracy and diversity"
    ]
    
    for insight in insights:
        st.write(f"• {insight}")

if __name__ == "__main__":
    main()
