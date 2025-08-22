"""
Visualization Module for Recommendation System
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# Set style for visualizations
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('default')
    
try:
    sns.set_palette("husl")
except:
    pass


class RecommenderVisualizer:
    """
    A comprehensive visualization toolkit for recommendation systems.
    
    This class provides various plotting methods to analyze rating distributions,
    similarity matrices, recommendation results, and system performance metrics.
    """
    
    def __init__(self, recommender):
        """
        Initialize the visualizer with a recommender system instance.
        
        Args:
            recommender: An instance of CollaborativeFilteringRecommender
                        that contains the data and computed similarities.
        """
        self.recommender = recommender
    
    def plot_data_analysis(self, save_path=None):
        """
        Create comprehensive visualizations of the dataset analysis.
        
        Generates a 2x3 subplot layout showing rating distribution, user activity,
        movie popularity, and similarity matrices for thorough data exploration.
        
        Args:
            save_path (str, optional): Path to save the plot. If None, displays interactively.
            
        Returns:
            matplotlib.figure.Figure: The generated figure object.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Rating distribution
        all_ratings = self.recommender.ratings_matrix.values[self.recommender.ratings_matrix.values > 0]
        axes[0, 0].hist(all_ratings, bins=5, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Distribution of Ratings', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Rating')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Number of ratings per user
        user_rating_counts = (self.recommender.ratings_matrix > 0).sum(axis=1)
        axes[0, 1].bar(range(len(user_rating_counts)), user_rating_counts.values, color='lightcoral')
        axes[0, 1].set_title('Number of Ratings per User', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('User ID')
        axes[0, 1].set_ylabel('Number of Ratings')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Number of ratings per movie (top 10)
        movie_rating_counts = (self.recommender.ratings_matrix > 0).sum(axis=0)
        top_movies = movie_rating_counts.nlargest(10)
        axes[0, 2].bar(range(len(top_movies)), top_movies.values, color='lightgreen')
        axes[0, 2].set_title('Top 10 Most Rated Movies', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Movie Rank')
        axes[0, 2].set_ylabel('Number of Ratings')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. User similarity heatmap
        if self.recommender.user_similarity is not None:
            im1 = axes[1, 0].imshow(self.recommender.user_similarity, cmap='coolwarm', aspect='auto')
            axes[1, 0].set_title('User Similarity Matrix', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('User ID')
            axes[1, 0].set_ylabel('User ID')
            plt.colorbar(im1, ax=axes[1, 0])
        else:
            axes[1, 0].text(0.5, 0.5, 'User Similarity\nNot Calculated', 
                           ha='center', va='center', fontsize=12)
            axes[1, 0].set_title('User Similarity Matrix', fontsize=14, fontweight='bold')
        
        # 5. Item similarity heatmap (subset)
        if self.recommender.item_similarity is not None:
            # Show similarity for top 20 movies
            top_20_movies = movie_rating_counts.nlargest(20).index
            top_20_indices = [self.recommender.ratings_matrix.columns.get_loc(movie) for movie in top_20_movies]
            subset_similarity = self.recommender.item_similarity[np.ix_(top_20_indices, top_20_indices)]
            
            im2 = axes[1, 1].imshow(subset_similarity, cmap='coolwarm', aspect='auto')
            axes[1, 1].set_title('Item Similarity Matrix (Top 20 Movies)', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Movie Index')
            axes[1, 1].set_ylabel('Movie Index')
            plt.colorbar(im2, ax=axes[1, 1])
        else:
            axes[1, 1].text(0.5, 0.5, 'Item Similarity\nNot Calculated', 
                           ha='center', va='center', fontsize=12)
            axes[1, 1].set_title('Item Similarity Matrix', fontsize=14, fontweight='bold')
        
        # 6. Average rating per user
        user_avg_ratings = self.recommender.user_mean_ratings
        axes[1, 2].plot(user_avg_ratings.index, user_avg_ratings.values, 'o-', color='purple', markersize=4)
        axes[1, 2].set_title('Average Rating per User', fontsize=14, fontweight='bold')
        axes[1, 2].set_xlabel('User ID')
        axes[1, 2].set_ylabel('Average Rating')
        axes[1, 2].axhline(y=self.recommender.global_mean, color='red', linestyle='--', 
                          label=f'Global Mean: {self.recommender.global_mean:.2f}')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Analysis plot saved to {save_path}")
        else:
            plt.show()
            
        return fig
    
    def plot_recommendations(self, user_id, recommendations, save_path=None):
        """
        Visualize recommendations for a specific user.
        
        Creates a side-by-side comparison showing predicted ratings for recommended
        movies and the user's existing rating history.
        
        Args:
            user_id (int): Target user ID for the recommendations
            recommendations (list): List of (movie_id, predicted_rating) tuples
            save_path (str, optional): Path to save the plot. If None, displays interactively.
            
        Returns:
            matplotlib.figure.Figure: The generated figure object, or None if no recommendations.
        """
        if not recommendations:
            print(f"No recommendations available for user {user_id}")
            return None
        
        movie_ids = [rec[0] for rec in recommendations]
        pred_ratings = [rec[1] for rec in recommendations]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Recommended movies with predicted ratings
        bars1 = ax1.barh(range(len(movie_ids)), pred_ratings, color='lightblue', edgecolor='navy', alpha=0.7)
        ax1.set_yticks(range(len(movie_ids)))
        ax1.set_yticklabels([f'Movie {mid}' for mid in movie_ids])
        ax1.set_xlabel('Predicted Rating')
        ax1.set_title(f'Top {len(recommendations)} Recommendations for User {user_id}', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlim(0, 5)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, (bar, rating) in enumerate(zip(bars1, pred_ratings)):
            ax1.text(rating + 0.05, i, f'{rating:.2f}', va='center', fontweight='bold')
        
        # User's existing ratings
        user_ratings = self.recommender.ratings_matrix.loc[user_id]
        rated_movies = user_ratings[user_ratings > 0]
        
        if len(rated_movies) > 0:
            bars2 = ax2.bar(range(len(rated_movies)), rated_movies.values, 
                           color='lightcoral', edgecolor='darkred', alpha=0.7)
            ax2.set_xticks(range(len(rated_movies)))
            ax2.set_xticklabels([f'M{mid}' for mid in rated_movies.index], rotation=45)
            ax2.set_ylabel('Rating')
            ax2.set_title(f'Existing Ratings by User {user_id}', fontsize=14, fontweight='bold')
            ax2.set_ylim(0, 5)
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, rating in zip(bars2, rated_movies.values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{rating:.1f}', ha='center', va='bottom', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, f'User {user_id}\nhas no existing ratings', 
                    ha='center', va='center', fontsize=12, 
                    transform=ax2.transAxes)
            ax2.set_title(f'Existing Ratings by User {user_id}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Recommendations plot saved to {save_path}")
        else:
            plt.show()
            
        return fig
    
    def plot_similarity_comparison(self, method1='cosine', method2='pearson', save_path=None):
        """
        Compare different similarity calculation methods side by side.
        
        Computes and visualizes similarity matrices using two different methods
        to help understand how similarity metric choice affects recommendations.
        
        Args:
            method1 (str): First similarity method to compare
            method2 (str): Second similarity method to compare
            save_path (str, optional): Path to save the plot. If None, displays interactively.
            
        Returns:
            matplotlib.figure.Figure: The generated figure object.
        """
        # Store original similarities
        original_user_sim = self.recommender.user_similarity.copy() if self.recommender.user_similarity is not None else None
        original_item_sim = self.recommender.item_similarity.copy() if self.recommender.item_similarity is not None else None
        
        # Calculate similarities with first method
        user_sim1 = self.recommender.calculate_user_similarity(method1)
        item_sim1 = self.recommender.calculate_item_similarity(method1)
        
        # Calculate similarities with second method
        user_sim2 = self.recommender.calculate_user_similarity(method2)
        item_sim2 = self.recommender.calculate_item_similarity(method2)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # User similarity comparison
        im1 = axes[0, 0].imshow(user_sim1, cmap='coolwarm', aspect='auto')
        axes[0, 0].set_title(f'User Similarity - {method1.title()}', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('User ID')
        axes[0, 0].set_ylabel('User ID')
        plt.colorbar(im1, ax=axes[0, 0])
        
        im2 = axes[0, 1].imshow(user_sim2, cmap='coolwarm', aspect='auto')
        axes[0, 1].set_title(f'User Similarity - {method2.title()}', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('User ID')
        axes[0, 1].set_ylabel('User ID')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Item similarity comparison (subset)
        movie_rating_counts = (self.recommender.ratings_matrix > 0).sum(axis=0)
        top_15_movies = movie_rating_counts.nlargest(15).index
        top_15_indices = [self.recommender.ratings_matrix.columns.get_loc(movie) for movie in top_15_movies]
        
        subset_sim1 = item_sim1[np.ix_(top_15_indices, top_15_indices)]
        subset_sim2 = item_sim2[np.ix_(top_15_indices, top_15_indices)]
        
        im3 = axes[1, 0].imshow(subset_sim1, cmap='coolwarm', aspect='auto')
        axes[1, 0].set_title(f'Item Similarity - {method1.title()} (Top 15)', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Movie Index')
        axes[1, 0].set_ylabel('Movie Index')
        plt.colorbar(im3, ax=axes[1, 0])
        
        im4 = axes[1, 1].imshow(subset_sim2, cmap='coolwarm', aspect='auto')
        axes[1, 1].set_title(f'Item Similarity - {method2.title()} (Top 15)', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Movie Index')
        axes[1, 1].set_ylabel('Movie Index')
        plt.colorbar(im4, ax=axes[1, 1])
        
        plt.tight_layout()
        
        # Restore original similarities
        self.recommender.user_similarity = original_user_sim
        self.recommender.item_similarity = original_item_sim
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Similarity comparison plot saved to {save_path}")
        else:
            plt.show()
            
        return fig
    
    def plot_evaluation_metrics(self, evaluation_results, save_path=None):
        """
        Visualize evaluation metrics for different recommendation methods.
        
        Creates bar charts comparing RMSE and MAE values across different
        recommendation approaches or parameter settings.
        
        Args:
            evaluation_results (dict): Dictionary with method names as keys and
                                     evaluation dictionaries as values containing 'RMSE' and 'MAE'
            save_path (str, optional): Path to save the plot. If None, displays interactively.
            
        Returns:
            matplotlib.figure.Figure: The generated figure object.
        """
        methods = list(evaluation_results.keys())
        rmse_values = [evaluation_results[method]['RMSE'] for method in methods]
        mae_values = [evaluation_results[method]['MAE'] for method in methods]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # RMSE comparison
        bars1 = ax1.bar(methods, rmse_values, color='skyblue', edgecolor='navy', alpha=0.7)
        ax1.set_title('Root Mean Square Error (RMSE)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('RMSE')
        ax1.set_ylim(0, max(rmse_values) * 1.1)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars1, rmse_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(rmse_values) * 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # MAE comparison
        bars2 = ax2.bar(methods, mae_values, color='lightcoral', edgecolor='darkred', alpha=0.7)
        ax2.set_title('Mean Absolute Error (MAE)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('MAE')
        ax2.set_ylim(0, max(mae_values) * 1.1)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars2, mae_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(mae_values) * 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Evaluation metrics plot saved to {save_path}")
        else:
            plt.show()
            
        return fig
