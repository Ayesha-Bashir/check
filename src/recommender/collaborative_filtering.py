"""
Core Collaborative Filtering Recommender System
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import TruncatedSVD
import warnings

warnings.filterwarnings('ignore')


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
        global_mean (float): Global mean rating across all users and items
    """

    def __init__(self):
        """
        Initialize the recommender system.

        Sets all matrices and statistics to None, which will be populated
        when data is loaded using the load_data method.
        """
        self.ratings_matrix = None
        self.user_similarity = None
        self.item_similarity = None
        self.user_mean_ratings = None
        self.item_mean_ratings = None
        self.global_mean = None

    def load_data(self, dataset):
        """
        Loads ratings data into ratings_matrix with columns as actual movie_ids.
        """
        import pandas as pd
        user_ids = []
        data = {}
        for user in dataset:
            user_id = user["user_id"]
            user_ids.append(user_id)
            for rating in user["ratings"]:
                movie_id = int(rating["movie_id"])
                if user_id not in data:
                    data[user_id] = {}
                data[user_id][movie_id] = rating["rating"]
        self.ratings_matrix = pd.DataFrame.from_dict(data, orient="index")
        # Normalize item IDs to int columns
        self.ratings_matrix.columns = pd.Index(
            [int(c) for c in self.ratings_matrix.columns])
        # Optional: normalize user index to int if applicable (defensive)
        try:
            self.ratings_matrix.index = pd.Index(
                [int(i) for i in self.ratings_matrix.index])
        except Exception:
            pass
        # Build alias map for robust resolution later
        self._item_alias = {int(c): c for c in self.ratings_matrix.columns}
        self.user_mean_ratings = self.ratings_matrix.mean(axis=1, skipna=True)
        self.item_mean_ratings = self.ratings_matrix.mean(axis=0, skipna=True)
        self.global_mean = float(self.ratings_matrix.stack(
        ).mean()) if self.ratings_matrix.size else 0.0
        self.user_similarity = None
        self.item_similarity = None

    def calculate_user_similarity(self, method='cosine'):
        """
        Calculate user-user similarity matrix using specified method.

        Computes similarity between all pairs of users based on their rating patterns.
        The similarity matrix is stored in self.user_similarity for later use.

        Args:
            method (str): Similarity metric to use. Options:
                         - 'cosine': Cosine similarity (default)
                         - 'pearson': Pearson correlation coefficient
                         - 'jaccard': Jaccard similarity (binary overlap)

        Returns:
            np.ndarray: Square matrix where element [i,j] represents similarity 
                       between user i and user j.

        Raises:
            ValueError: If an unsupported similarity method is specified.
        """
        if self.ratings_matrix is None:
            raise ValueError(
                "Data must be loaded before calculating similarity")

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
            union = np.sum(binary_matrix, axis=1).reshape(-1, 1) + \
                np.sum(binary_matrix, axis=1) - intersection
            self.user_similarity = intersection / union
            self.user_similarity = np.nan_to_num(self.user_similarity)

        else:
            raise ValueError(f"Unsupported similarity method: {method}")

        print(f"✓ User similarity calculated using {method} method")
        return self.user_similarity

    def calculate_item_similarity(self, method='cosine'):
        """
        Calculate item-item similarity matrix using specified method.

        Computes similarity between all pairs of items based on user rating patterns.
        The similarity matrix is stored in self.item_similarity for later use.

        Args:
            method (str): Similarity metric to use. Options:
                         - 'cosine': Cosine similarity (default)
                         - 'pearson': Pearson correlation coefficient
                         - 'jaccard': Jaccard similarity (binary overlap)

        Returns:
            np.ndarray: Square matrix where element [i,j] represents similarity 
                       between item i and item j.

        Raises:
            ValueError: If an unsupported similarity method is specified.
        """
        if self.ratings_matrix is None:
            raise ValueError(
                "Data must be loaded before calculating similarity")

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
            union = np.sum(binary_matrix, axis=0).reshape(-1, 1) + \
                np.sum(binary_matrix, axis=0) - intersection
            self.item_similarity = intersection / union
            self.item_similarity = np.nan_to_num(self.item_similarity)

        else:
            raise ValueError(f"Unsupported similarity method: {method}")

        print(f"✓ Item similarity calculated using {method} method")
        return self.item_similarity

    def predict_user_based(self, user_id, movie_id, k=5):
        """
        Predict rating using user-based collaborative filtering.

        Finds similar users who have rated the target movie and computes
        a weighted average of their ratings based on user similarity scores.

        Args:
            user_id (int): ID of the target user for prediction
            movie_id (int): ID of the target movie to predict rating for
            k (int): Number of most similar users to consider (default: 5)

        Returns:
            float: Predicted rating value between 1-5, or fallback values:
                  - Global mean if user not found
                  - User mean if movie not found or no similar users available
        """
        if self.user_similarity is None:
            raise ValueError(
                "User similarity must be calculated before prediction")

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

        Finds similar movies that the user has rated and computes
        a weighted average of their ratings based on item similarity scores.

        Args:
            user_id (int): ID of the target user for prediction
            movie_id (int): ID of the target movie to predict rating for
            k (int): Number of most similar items to consider (default: 5)

        Returns:
            float: Predicted rating value between 1-5, or fallback values:
                  - Global mean if user not found
                  - User mean if movie not found
                  - Item mean if no similar items available
        """
        if self.item_similarity is None:
            raise ValueError(
                "Item similarity must be calculated before prediction")

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
        Generate movie recommendations for a specific user.

        Predicts ratings for all unrated movies and returns the top-N
        recommendations sorted by predicted rating score.

        Args:
            user_id (int): ID of the target user to generate recommendations for
            method (str): Recommendation approach to use:
                         - 'user_based': Use user-based collaborative filtering
                         - 'item_based': Use item-based collaborative filtering
            n_recommendations (int): Number of top recommendations to return (default: 5)
            k (int): Number of neighbors to consider in similarity calculations (default: 5)

        Returns:
            list: List of tuples (movie_id, predicted_rating) sorted by predicted rating
                 in descending order. Empty list if user not found.
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
            elif method == 'item_based':
                pred = self.predict_item_based(user_id, movie_id, k)
            else:
                raise ValueError(f"Unsupported method: {method}")
            predictions.append((movie_id, pred))

        # Sort by predicted rating and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]

    def evaluate_predictions(self, test_set, method='user_based', k=5):
        """
        Evaluate the recommendation system using a test set.

        Computes prediction accuracy metrics by comparing predicted ratings
        with actual ratings from a held-out test set.

        Args:
            test_set (list): List of (user_id, movie_id, actual_rating) tuples
                           representing known ratings to evaluate against
            method (str): Prediction method to use ('user_based' or 'item_based')
            k (int): Number of neighbors to consider in predictions (default: 5)

        Returns:
            dict: Dictionary containing evaluation metrics:
                 - 'RMSE': Root Mean Square Error
                 - 'MAE': Mean Absolute Error
        """
        predictions = []
        actuals = []

        for user_id, movie_id, actual_rating in test_set:
            if method == 'user_based':
                pred = self.predict_user_based(user_id, movie_id, k)
            elif method == 'item_based':
                pred = self.predict_item_based(user_id, movie_id, k)
            else:
                raise ValueError(f"Unsupported method: {method}")

            predictions.append(pred)
            actuals.append(actual_rating)

        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)

        return {'RMSE': rmse, 'MAE': mae}

    def extended_algorithm(self, user_id, n_recommendations=5, n_factors=20, **kwargs):
        """
        SVD with baseline bias (global + user + item) plus residual SVD.
        Returns top-N recommendations for unrated items for user_id.
        """
        import numpy as np
        import pandas as pd
        from sklearn.decomposition import TruncatedSVD

        R = self.ratings_matrix.copy().apply(pd.to_numeric, errors="coerce")

        # Means
        mu = float(np.nanmean(R.values)) if R.size else 0.0
        bu = R.mean(axis=1) - mu
        bi = R.mean(axis=0) - mu

        # Baseline for all users/items
        B = (pd.DataFrame(np.add.outer(bu.values, bi.values),
             index=R.index, columns=R.columns) + mu)

        # Residuals (NaN where unrated)
        E = R - B
        E_filled = E.fillna(0.0)

        X = E_filled.to_numpy(dtype=float)
        # Cap factors
        max_k = max(1, min(X.shape) - 1)
        k = max(1, min(int(n_factors), max_k))

        svd = TruncatedSVD(n_components=k, random_state=42)
        U = svd.fit_transform(X)        # shape: (n_users, k)
        S = svd.singular_values_        # shape: (k,)
        VT = svd.components_            # shape: (k, n_items)

        # Reconstruct user residual row
        try:
            user_idx = list(R.index).index(user_id)
        except ValueError:
            raise ValueError(f"user_id {user_id} not found in ratings_matrix")

        Su = U[user_idx, :] * S
        ehat_u = Su @ VT   # shape: (n_items,)

        # Baseline row for user
        Bu_row = B.loc[user_id].to_numpy(dtype=float)

        # Predicted full row
        yhat = Bu_row + ehat_u
        yhat = np.clip(yhat, 1.0, 5.0)

        # Recommend only for items the user has not rated in R (NaN in R row)
        unrated_mask = R.loc[user_id].isna().to_numpy()
        movie_ids = list(R.columns)
        recs = [(int(mid), float(score)) for mid, score,
                mask in zip(movie_ids, yhat, unrated_mask) if mask]

        # Sort high→low; tie-break randomized later in route
        recs.sort(key=lambda t: t[1], reverse=True)

        # Return top-N
        return recs[:n_recommendations]

        # When computing cosine similarity:
        similarity_matrix = sklearn.metrics.pairwise.cosine_similarity(
            R.fillna(0.0))

    def add_ephemeral_user(self, liked_movie_ids, like_rating=5.0):
        """
        Adds a temporary user with fixed ratings for given movie IDs.
        Returns the new user_id.
        Updates ratings_matrix, mean ratings, and invalidates similarity caches.
        """
        import numpy as np
        import pandas as pd
        if self.ratings_matrix is None:
            raise ValueError(
                "ratings_matrix is None; load data before adding users.")
        try:
            new_user_id = int(max(self.ratings_matrix.index)) + \
                1 if len(self.ratings_matrix.index) else 0
        except Exception:
            new_user_id = len(self.ratings_matrix.index) + 1
        new_row = pd.Series(index=self.ratings_matrix.columns, dtype=float)
        new_row[:] = np.nan
        available_cols = set(int(c) for c in self.ratings_matrix.columns)
        liked_ids = [int(x) for x in liked_movie_ids]
        overlap = [mid for mid in liked_ids if mid in available_cols]
        matched = 0
        for mid in overlap:
            new_row[mid] = float(like_rating)
            matched += 1
        if matched == 0:
            raise ValueError(
                "No overlap between selected movieIds and ratings_matrix columns; check generator IDs.")
        self.ratings_matrix.loc[new_user_id] = new_row
        self.user_mean_ratings = self.ratings_matrix.mean(axis=1, skipna=True)
        self.item_mean_ratings = self.ratings_matrix.mean(axis=0, skipna=True)
        self.global_mean = float(self.ratings_matrix.stack(
        ).mean()) if self.ratings_matrix.size else 0.0
        if hasattr(self, "user_similarity"):
            self.user_similarity = None
        if hasattr(self, "item_similarity"):
            self.item_similarity = None
        return new_user_id
