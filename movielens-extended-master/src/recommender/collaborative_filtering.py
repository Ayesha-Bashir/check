"""
Collaborative Filtering Movie Recommendation System
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
        global_mean (float): Global mean rating across all users
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
            matrix = self.ratings_matrix.replace(0, np.nan)
            matrix_filled = matrix.fillna(0)
            self.user_similarity = cosine_similarity(matrix_filled)
        
        elif method == 'pearson':
            matrix = self.ratings_matrix.replace(0, np.nan)
            self.user_similarity = matrix.T.corr().fillna(0).values
        
        elif method == 'jaccard':
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
        
        movie_ratings = self.ratings_matrix.iloc[:, movie_idx]
        rated_users = movie_ratings[movie_ratings > 0].index
        
        if len(rated_users) == 0:
            return self.user_mean_ratings[user_id]
        
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
        
        sim_ratings = list(zip(similarities, ratings))
        sim_ratings.sort(reverse=True)
        top_k = sim_ratings[:k]
        
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
        
        sim_ratings = list(zip(similarities, ratings))
        sim_ratings.sort(reverse=True)
        top_k = sim_ratings[:k]
        
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
