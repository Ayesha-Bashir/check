"""
Collaborative Filtering Movie Recommendation System
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
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


class CollaborativeFilteringRecommender:
    """
    A comprehensive collaborative filtering recommendation system.
    
    This class implements both user-based and item-based collaborative filtering
    algorithms with various similarity metrics and evaluation capabilities.
    
    Attributes:
        ratings_matrix (pd.DataFrame): User-item ratings matrix
        user_similarity (np.ndarray): User-user similarity matrix
        item_similarity (np.ndarray): Item-item similarity matrix
        user_mean_ratings (pd.Series): Mean rating for each user
        item_mean_ratings (pd.Series): Mean rating for each item
    """
    
    def __init__(self):
        """Initialize the recommender system."""
        self.ratings_matrix = None
        self.user_similarity = None
        self.item_similarity = None
        self.user_mean_ratings = None
        self.item_mean_ratings = None
        self.global_mean = None
        
    def load_data(self, ratings_data):
        """
        Load and preprocess the ratings data.
        
        Args:
            ratings_data (list): List of dictionaries containing user ratings
            
        Returns:
            pd.DataFrame: Processed ratings matrix
        """
        # Convert to DataFrame
        rows = []
        for user_data in ratings_data:
            user_id = int(user_data['user_id'])
            for rating in user_data['ratings']:
                rows.append({
                    'user_id': user_id,
                    'movie_id': rating['movie_id'],
                    'rating': rating['rating']
                })
        
        df = pd.DataFrame(rows)
        
        # Create user-item matrix
        self.ratings_matrix = df.pivot_table(
            index='user_id', 
            columns='movie_id', 
            values='rating', 
            fill_value=0
        )
        
        # Calculate mean ratings
        self.user_mean_ratings = self.ratings_matrix.replace(0, np.nan).mean(axis=1)
        self.item_mean_ratings = self.ratings_matrix.replace(0, np.nan).mean(axis=0)
        self.global_mean = df['rating'].mean()
        
        print(f"✓ Data loaded successfully!")
        print(f"  - Users: {len(self.ratings_matrix)}")
        print(f"  - Movies: {len(self.ratings_matrix.columns)}")
        print(f"  - Total ratings: {len(df)}")
        print(f"  - Sparsity: {(1 - len(df) / (len(self.ratings_matrix) * len(self.ratings_matrix.columns))) * 100:.2f}%")
        
        return self.ratings_matrix
    
    def calculate_user_similarity(self, method='cosine'):
        """
        Calculate user-user similarity matrix.
        
        Args:
            method (str): Similarity metric ('cosine', 'pearson', 'jaccard')
            
        Returns:
            np.ndarray: User similarity matrix
        """
        if method == 'cosine':
            # Replace 0s with NaN for proper cosine calculation
            matrix = self.ratings_matrix.replace(0, np.nan)
            matrix_filled = matrix.fillna(0)
            self.user_similarity = cosine_similarity(matrix_filled)
        
        elif method == 'pearson':
            # Calculate Pearson correlation
            matrix = self.ratings_matrix.replace(0, np.nan)
            self.user_similarity = matrix.T.corr().fillna(0).values
        
        elif method == 'jaccard':
            # Binary Jaccard similarity
            binary_matrix = (self.ratings_matrix > 0).astype(int)
            intersection = np.dot(binary_matrix, binary_matrix.T)
            union = np.sum(binary_matrix, axis=1).reshape(-1, 1) + np.sum(binary_matrix, axis=1) - intersection
            self.user_similarity = intersection / union
            self.user_similarity = np.nan_to_num(self.user_similarity)
        
        print(f"✓ User similarity calculated using {method} method")
        return self.user_similarity
    
    def calculate_item_similarity(self, method='cosine'):
        """
        Calculate item-item similarity matrix.
        
        Args:
            method (str): Similarity metric ('cosine', 'pearson', 'jaccard')
            
        Returns:
            np.ndarray: Item similarity matrix
        """
        if method == 'cosine':
            matrix = self.ratings_matrix.replace(0, np.nan)
            matrix_filled = matrix.fillna(0)
            self.item_similarity = cosine_similarity(matrix_filled.T)
        
        elif method == 'pearson':
            matrix = self.ratings_matrix.replace(0, np.nan)
            self.item_similarity = matrix.corr().fillna(0).values
        
        elif method == 'jaccard':
            binary_matrix = (self.ratings_matrix > 0).astype(int)
            intersection = np.dot(binary_matrix.T, binary_matrix)
            union = np.sum(binary_matrix, axis=0).reshape(-1, 1) + np.sum(binary_matrix, axis=0) - intersection
            self.item_similarity = intersection / union
            self.item_similarity = np.nan_to_num(self.item_similarity)
        
        print(f"✓ Item similarity calculated using {method} method")
        return self.item_similarity
    
    def predict_user_based(self, user_id, movie_id, k=5):
        """
        Predict rating using user-based collaborative filtering.
        
        Args:
            user_id (int): Target user ID
            movie_id (int): Target movie ID
            k (int): Number of similar users to consider
            
        Returns:
            float: Predicted rating
        """
        if user_id not in self.ratings_matrix.index:
            return self.global_mean
        
        if movie_id not in self.ratings_matrix.columns:
            return self.user_mean_ratings[user_id]
        
        user_idx = self.ratings_matrix.index.get_loc(user_id)
        movie_idx = self.ratings_matrix.columns.get_loc(movie_id)
        
        # Get users who have rated this movie
        movie_ratings = self.ratings_matrix.iloc[:, movie_idx]
        rated_users = movie_ratings[movie_ratings > 0].index
        
        if len(rated_users) == 0:
            return self.user_mean_ratings[user_id]
        
        # Get similarities and ratings for users who rated this movie
        similarities = []
        ratings = []
        
        for other_user in rated_users:
            if other_user != user_id:
                other_idx = self.ratings_matrix.index.get_loc(other_user)
                sim = self.user_similarity[user_idx, other_idx]
                if sim > 0:
                    similarities.append(sim)
                    ratings.append(movie_ratings[other_user])
        
        if len(similarities) == 0:
            return self.user_mean_ratings[user_id]
        
        # Get top-k similar users
        sim_ratings = list(zip(similarities, ratings))
        sim_ratings.sort(reverse=True)
        top_k = sim_ratings[:k]
        
        # Calculate weighted average
        numerator = sum(sim * rating for sim, rating in top_k)
        denominator = sum(sim for sim, rating in top_k)
        
        if denominator == 0:
            return self.user_mean_ratings[user_id]
        
        return numerator / denominator
    
    def predict_item_based(self, user_id, movie_id, k=5):
        """
        Predict rating using item-based collaborative filtering.
        
        Args:
            user_id (int): Target user ID
            movie_id (int): Target movie ID
            k (int): Number of similar items to consider
            
        Returns:
            float: Predicted rating
        """
        if user_id not in self.ratings_matrix.index:
            return self.global_mean
        
        if movie_id not in self.ratings_matrix.columns:
            return self.user_mean_ratings[user_id]
        
        user_ratings = self.ratings_matrix.loc[user_id]
        rated_movies = user_ratings[user_ratings > 0].index
        
        if len(rated_movies) == 0:
            return self.global_mean
        
        movie_idx = self.ratings_matrix.columns.get_loc(movie_id)
        
        # Get similarities and ratings for movies rated by this user
        similarities = []
        ratings = []
        
        for other_movie in rated_movies:
            if other_movie != movie_id:
                other_idx = self.ratings_matrix.columns.get_loc(other_movie)
                sim = self.item_similarity[movie_idx, other_idx]
                if sim > 0:
                    similarities.append(sim)
                    ratings.append(user_ratings[other_movie])
        
        if len(similarities) == 0:
            return self.item_mean_ratings[movie_id]
        
        # Get top-k similar items
        sim_ratings = list(zip(similarities, ratings))
        sim_ratings.sort(reverse=True)
        top_k = sim_ratings[:k]
        
        # Calculate weighted average
        numerator = sum(sim * rating for sim, rating in top_k)
        denominator = sum(sim for sim, rating in top_k)
        
        if denominator == 0:
            return self.item_mean_ratings[movie_id]
        
        return numerator / denominator
    
    def get_recommendations(self, user_id, method='user_based', n_recommendations=5, k=5):
        """
        Get movie recommendations for a user.
        
        Args:
            user_id (int): Target user ID
            method (str): Recommendation method ('user_based' or 'item_based')
            n_recommendations (int): Number of recommendations to return
            k (int): Number of neighbors to consider
            
        Returns:
            list: List of tuples (movie_id, predicted_rating)
        """
        if user_id not in self.ratings_matrix.index:
            print(f"User {user_id} not found in the dataset")
            return []
        
        user_ratings = self.ratings_matrix.loc[user_id]
        unrated_movies = user_ratings[user_ratings == 0].index
        
        predictions = []
        for movie_id in unrated_movies:
            if method == 'user_based':
                pred = self.predict_user_based(user_id, movie_id, k)
            else:
                pred = self.predict_item_based(user_id, movie_id, k)
            predictions.append((movie_id, pred))
        
        # Sort by predicted rating and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]
    
    def evaluate_predictions(self, test_set, method='user_based', k=5):
        """
        Evaluate the recommendation system using test set.
        
        Args:
            test_set (list): List of (user_id, movie_id, actual_rating) tuples
            method (str): Prediction method to use
            k (int): Number of neighbors to consider
            
        Returns:
            dict: Evaluation metrics (RMSE, MAE)
        """
        predictions = []
        actuals = []
        
        for user_id, movie_id, actual_rating in test_set:
            if method == 'user_based':
                pred = self.predict_user_based(user_id, movie_id, k)
            else:
                pred = self.predict_item_based(user_id, movie_id, k)
            
            predictions.append(pred)
            actuals.append(actual_rating)
        
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        
        return {'RMSE': rmse, 'MAE': mae}
    
    def visualize_data_analysis(self):
        """Create comprehensive visualizations of the dataset."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Rating distribution
        all_ratings = self.ratings_matrix.values[self.ratings_matrix.values > 0]
        axes[0, 0].hist(all_ratings, bins=5, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Distribution of Ratings')
        axes[0, 0].set_xlabel('Rating')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. Number of ratings per user
        user_rating_counts = (self.ratings_matrix > 0).sum(axis=1)
        axes[0, 1].bar(range(len(user_rating_counts)), user_rating_counts.values, color='lightcoral')
        axes[0, 1].set_title('Number of Ratings per User')
        axes[0, 1].set_xlabel('User ID')
        axes[0, 1].set_ylabel('Number of Ratings')
        
        # 3. Number of ratings per movie
        movie_rating_counts = (self.ratings_matrix > 0).sum(axis=0)
        top_movies = movie_rating_counts.nlargest(10)
        axes[0, 2].bar(range(len(top_movies)), top_movies.values, color='lightgreen')
        axes[0, 2].set_title('Top 10 Most Rated Movies')
        axes[0, 2].set_xlabel('Movie Rank')
        axes[0, 2].set_ylabel('Number of Ratings')
        
        # 4. User similarity heatmap
        if self.user_similarity is not None:
            im1 = axes[1, 0].imshow(self.user_similarity, cmap='coolwarm', aspect='auto')
            axes[1, 0].set_title('User Similarity Matrix')
            axes[1, 0].set_xlabel('User ID')
            axes[1, 0].set_ylabel('User ID')
            plt.colorbar(im1, ax=axes[1, 0])
        
        # 5. Item similarity heatmap (subset)
        if self.item_similarity is not None:
            # Show similarity for top 20 movies
            top_20_movies = movie_rating_counts.nlargest(20).index
            top_20_indices = [self.ratings_matrix.columns.get_loc(movie) for movie in top_20_movies]
            subset_similarity = self.item_similarity[np.ix_(top_20_indices, top_20_indices)]
            
            im2 = axes[1, 1].imshow(subset_similarity, cmap='coolwarm', aspect='auto')
            axes[1, 1].set_title('Item Similarity Matrix (Top 20 Movies)')
            axes[1, 1].set_xlabel('Movie Index')
            axes[1, 1].set_ylabel('Movie Index')
            plt.colorbar(im2, ax=axes[1, 1])
        
        # 6. Average rating per user
        user_avg_ratings = self.user_mean_ratings
        axes[1, 2].plot(user_avg_ratings.index, user_avg_ratings.values, 'o-', color='purple')
        axes[1, 2].set_title('Average Rating per User')
        axes[1, 2].set_xlabel('User ID')
        axes[1, 2].set_ylabel('Average Rating')
        axes[1, 2].axhline(y=self.global_mean, color='red', linestyle='--', label=f'Global Mean: {self.global_mean:.2f}')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.show()
    
    def visualize_recommendations(self, user_id, recommendations):
        """
        Visualize recommendations for a specific user.
        
        Args:
            user_id (int): Target user ID
            recommendations (list): List of (movie_id, predicted_rating) tuples
        """
        if not recommendations:
            print(f"No recommendations available for user {user_id}")
            return
        
        movie_ids = [rec[0] for rec in recommendations]
        pred_ratings = [rec[1] for rec in recommendations]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Recommended movies with predicted ratings
        ax1.barh(range(len(movie_ids)), pred_ratings, color='lightblue', edgecolor='navy')
        ax1.set_yticks(range(len(movie_ids)))
        ax1.set_yticklabels([f'Movie {mid}' for mid in movie_ids])
        ax1.set_xlabel('Predicted Rating')
        ax1.set_title(f'Top {len(recommendations)} Recommendations for User {user_id}')
        ax1.set_xlim(0, 5)
        
        # User's existing ratings
        user_ratings = self.ratings_matrix.loc[user_id]
        rated_movies = user_ratings[user_ratings > 0]
        
        ax2.bar(range(len(rated_movies)), rated_movies.values, color='lightcoral', edgecolor='darkred')
        ax2.set_xticks(range(len(rated_movies)))
        ax2.set_xticklabels([f'M{mid}' for mid in rated_movies.index], rotation=45)
        ax2.set_ylabel('Rating')
        ax2.set_title(f'Existing Ratings by User {user_id}')
        ax2.set_ylim(0, 5)
        
        plt.tight_layout()
        plt.show()
    
    def extended_algorithm(self, user_id, **kwargs):
        """
        Extended algorithm placeholder for custom implementations.
        
        This function is left empty for you to implement your own
        advanced recommendation algorithms, such as:
        - Matrix Factorization (SVD, NMF)
        - Deep Learning approaches
        - Hybrid methods
        - Content-based filtering integration
        - Time-aware recommendations
        - Social network integration
        
        Args:
            user_id (int): Target user ID
            **kwargs: Additional parameters for your custom algorithm
            
        Returns:
            list: Your custom recommendations
        """
        # TODO: Implement your extended algorithm here
        print("🚀 Extended algorithm placeholder - implement your custom logic here!")
        print(f"   Target user: {user_id}")
        print(f"   Additional parameters: {kwargs}")
        
        # Example structure for your implementation:
        # 1. Implement matrix factorization
        # 2. Add content-based features
        # 3. Incorporate temporal dynamics
        # 4. Use ensemble methods
        # 5. Apply deep learning models
        
        return []


def create_sample_data():
    """Create sample dataset for testing."""
    return [
        {
            "user_id": "0",
            "ratings": [
                {"movie_id": 1, "rating": 4},
                {"movie_id": 15, "rating": 3},
                {"movie_id": 23, "rating": 5}
            ]
        },
        {
            "user_id": "1", 
            "ratings": [
                {"movie_id": 5, "rating": 2},
                {"movie_id": 1, "rating": 4},
                {"movie_id": 18, "rating": 3}
            ]
        },
        {
            "user_id": "2",
            "ratings": [
                {"movie_id": 12, "rating": 5},
                {"movie_id": 7, "rating": 3},
                {"movie_id": 23, "rating": 4}
            ]
        },
        {
            "user_id": "3",
            "ratings": [
                {"movie_id": 9, "rating": 3},
                {"movie_id": 15, "rating": 2},
                {"movie_id": 31, "rating": 4}
            ]
        },
        {
            "user_id": "4",
            "ratings": [
                {"movie_id": 2, "rating": 5},
                {"movie_id": 12, "rating": 3},
                {"movie_id": 45, "rating": 4}
            ]
        },
        {
            "user_id": "5",
            "ratings": [
                {"movie_id": 18, "rating": 4},
                {"movie_id": 7, "rating": 5},
                {"movie_id": 29, "rating": 2}
            ]
        },
        {
            "user_id": "6",
            "ratings": [
                {"movie_id": 1, "rating": 3},
                {"movie_id": 33, "rating": 4},
                {"movie_id": 56, "rating": 5}
            ]
        },
        {
            "user_id": "7",
            "ratings": [
                {"movie_id": 23, "rating": 2},
                {"movie_id": 41, "rating": 4},
                {"movie_id": 9, "rating": 3}
            ]
        },
        {
            "user_id": "8",
            "ratings": [
                {"movie_id": 15, "rating": 5},
                {"movie_id": 67, "rating": 3},
                {"movie_id": 2, "rating": 4}
            ]
        },
        {
            "user_id": "9",
            "ratings": [
                {"movie_id": 12, "rating": 4},
                {"movie_id": 78, "rating": 2},
                {"movie_id": 5, "rating": 5}
            ]
        },
        {
            "user_id": "10",
            "ratings": [
                {"movie_id": 34, "rating": 3},
                {"movie_id": 18, "rating": 4},
                {"movie_id": 91, "rating": 2}
            ]
        },
        {
            "user_id": "11",
            "ratings": [
                {"movie_id": 7, "rating": 5},
                {"movie_id": 52, "rating": 3},
                {"movie_id": 1, "rating": 4}
            ]
        },
        {
            "user_id": "12",
            "ratings": [
                {"movie_id": 89, "rating": 2},
                {"movie_id": 23, "rating": 5},
                {"movie_id": 14, "rating": 3}
            ]
        },
        {
            "user_id": "13",
            "ratings": [
                {"movie_id": 15, "rating": 4},
                {"movie_id": 27, "rating": 3},
                {"movie_id": 62, "rating": 5}
            ]
        },
        {
            "user_id": "14",
            "ratings": [
                {"movie_id": 9, "rating": 3},
                {"movie_id": 76, "rating": 4},
                {"movie_id": 12, "rating": 2}
            ]
        },
        {
            "user_id": "15",
            "ratings": [
                {"movie_id": 44, "rating": 5},
                {"movie_id": 2, "rating": 3},
                {"movie_id": 18, "rating": 4}
            ]
        },
        {
            "user_id": "16",
            "ratings": [
                {"movie_id": 7, "rating": 2},
                {"movie_id": 83, "rating": 4},
                {"movie_id": 35, "rating": 5}
            ]
        },
        {
            "user_id": "17",
            "ratings": [
                {"movie_id": 1, "rating": 5},
                {"movie_id": 58, "rating": 3},
                {"movie_id": 23, "rating": 4}
            ]
        },
        {
            "user_id": "18",
            "ratings": [
                {"movie_id": 12, "rating": 3},
                {"movie_id": 71, "rating": 2},
                {"movie_id": 15, "rating": 5}
            ]
        },
        {
            "user_id": "19",
            "ratings": [
                {"movie_id": 26, "rating": 4},
                {"movie_id": 9, "rating": 3},
                {"movie_id": 47, "rating": 5}
            ]
        }
    ]


def main():
    """Main function demonstrating the collaborative filtering system."""
    print("🎬 Collaborative Filtering Movie Recommendation System")
    print("=" * 60)
    
    # Create recommender instance
    recommender = CollaborativeFilteringRecommender()
    
    # Load data
    ratings_data = create_sample_data()
    ratings_matrix = recommender.load_data(ratings_data)
    
    print("\n📊 Ratings Matrix (first 5 users and 10 movies):")
    print(ratings_matrix.iloc[:5, :10])
    
    # Calculate similarities
    print("\n🔍 Calculating similarities...")
    user_sim = recommender.calculate_user_similarity(method='cosine')
    item_sim = recommender.calculate_item_similarity(method='cosine')
    
    # Generate recommendations
    print("\n🎯 Generating recommendations...")
    target_user = 0
    user_based_recs = recommender.get_recommendations(
        target_user, method='user_based', n_recommendations=5
    )
    item_based_recs = recommender.get_recommendations(
        target_user, method='item_based', n_recommendations=5
    )
    
    print(f"\nUser-based recommendations for User {target_user}:")
    for movie_id, rating in user_based_recs:
        print(f"  Movie {movie_id}: {rating:.2f}")
    
    print(f"\nItem-based recommendations for User {target_user}:")
    for movie_id, rating in item_based_recs:
        print(f"  Movie {movie_id}: {rating:.2f}")
    
    # Visualizations
    print("\n📈 Creating visualizations...")
    recommender.visualize_data_analysis()
    recommender.visualize_recommendations(target_user, user_based_recs)
    
    # Test extended algorithm
    print("\n🚀 Testing extended algorithm placeholder:")
    recommender.extended_algorithm(target_user, algorithm='custom', threshold=0.8)
    
    print("\n✅ Demo completed! Check the generated visualizations.")


if __name__ == "__main__":
    main()