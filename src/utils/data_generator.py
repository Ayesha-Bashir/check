"""
Data Generation Utilities for Recommendation System
"""

import random
import numpy as np
from typing import List, Dict, Tuple, Optional


class DataGenerator:
    """
    A utility class for generating synthetic movie rating datasets.

    This class provides methods to create realistic synthetic datasets with
    configurable parameters such as number of users, movies, sparsity levels,
    and rating distributions.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the data generator with optional random seed.

        Args:
            seed (int, optional): Random seed for reproducible results.
                                 If None, uses current system time.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.seed = seed

    def generate_realistic_dataset(self,
                                   n_users: int = 100,
                                   n_movies: int = 200,
                                   sparsity: float = 0.95) -> List[Dict]:
        """
        Generate a realistic synthetic movie rating dataset.

        Args:
            n_users (int): Number of users to generate
            n_movies (int): Number of movies to generate  
            sparsity (float): Fraction of missing ratings (0.0 to 1.0)

        Returns:
            list: List of dictionaries containing user rating data
        """
        # Calculate number of ratings to generate
        total_possible_ratings = n_users * n_movies
        n_ratings = int(total_possible_ratings * (1 - sparsity))

        # Generate random user-movie pairs
        user_movie_pairs = set()
        while len(user_movie_pairs) < n_ratings:
            user_id = random.randint(0, n_users - 1)
            movie_id = random.randint(1, n_movies)
            user_movie_pairs.add((user_id, movie_id))

        # Initialize user ratings
        ratings_data = {user_id: [] for user_id in range(n_users)}

        # Generate ratings
        for user_id, movie_id in user_movie_pairs:
            # Simple rating generation around 3.5 with some randomness
            rating = np.random.normal(3.5, 1.0)
            rating = max(1, min(5, rating))  # Clamp to 1-5 range
            rating = round(rating * 2) / 2   # Round to nearest 0.5

            ratings_data[user_id].append({
                'movie_id': movie_id,
                'rating': rating
            })

        # Convert to expected format
        result = []
        for user_id, ratings in ratings_data.items():
            if ratings:  # Only include users with ratings
                result.append({
                    'user_id': str(user_id),
                    'ratings': ratings
                })

        return result

    def generate_simple_dataset(self,
                                n_users: int = 20,
                                n_movies: int = 50,
                                min_ratings_per_user: int = 2,
                                max_ratings_per_user: int = 8) -> List[Dict]:
        """
        Generate a simple synthetic dataset for quick testing.

        Creates a straightforward dataset with random ratings, useful for
        testing basic functionality without complex user/movie patterns.

        Args:
            n_users (int): Number of users to generate (default: 20)
            n_movies (int): Number of movies to generate (default: 50)
            min_ratings_per_user (int): Minimum ratings per user (default: 2)
            max_ratings_per_user (int): Maximum ratings per user (default: 8)

        Returns:
            list: List of dictionaries containing user rating data.
        """
        result = []

        for user_id in range(n_users):
            n_ratings = random.randint(
                min_ratings_per_user, max_ratings_per_user)

            # Select random movies for this user
            user_movies = random.sample(
                range(1, n_movies + 1), min(n_ratings, n_movies))

            ratings = []
            for movie_id in user_movies:
                # Generate random rating (1-5 scale)
                rating = random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
                ratings.append({
                    'movie_id': movie_id,
                    'rating': rating
                })

            result.append({
                'user_id': str(user_id),
                'ratings': ratings
            })

        return result

    def generate_clustered_dataset(self,
                                   n_users: int = 80,
                                   n_movies: int = 120,
                                   n_clusters: int = 4) -> List[Dict]:
        """
        Generate a dataset with user clusters having similar preferences.

        Args:
            n_users (int): Total number of users to generate
            n_movies (int): Total number of movies to generate
            n_clusters (int): Number of user clusters to create

        Returns:
            list: List of dictionaries containing user rating data with cluster patterns
        """
        # Assign users to clusters
        users_per_cluster = n_users // n_clusters
        cluster_assignments = []
        for cluster_id in range(n_clusters):
            cluster_assignments.extend([cluster_id] * users_per_cluster)

        # Handle remaining users
        remaining_users = n_users - len(cluster_assignments)
        for i in range(remaining_users):
            cluster_assignments.append(i % n_clusters)

        random.shuffle(cluster_assignments)

        # Each cluster prefers different movies
        movies_per_cluster = n_movies // n_clusters
        cluster_preferences = {}
        for cluster_id in range(n_clusters):
            start_movie = cluster_id * movies_per_cluster + 1
            end_movie = min((cluster_id + 1) *
                            movies_per_cluster, n_movies) + 1
            cluster_preferences[cluster_id] = list(
                range(start_movie, end_movie))

        result = []

        for user_id in range(n_users):
            cluster_id = cluster_assignments[user_id]
            preferred_movies = cluster_preferences[cluster_id]

            # Generate 3-12 ratings per user
            n_ratings = random.randint(3, 12)

            # 70% preferred movies, 30% random movies
            n_preferred = int(n_ratings * 0.7)
            n_random = n_ratings - n_preferred

            user_movies = []

            # Add preferred movies
            if preferred_movies:
                user_movies.extend(random.sample(preferred_movies,
                                                 min(n_preferred, len(preferred_movies))))

            # Add random movies
            all_other_movies = [m for m in range(
                1, n_movies + 1) if m not in user_movies]
            if all_other_movies and n_random > 0:
                user_movies.extend(random.sample(all_other_movies,
                                                 min(n_random, len(all_other_movies))))

            # Generate ratings
            ratings = []
            for movie_id in user_movies:
                if movie_id in preferred_movies:
                    # Higher ratings for preferred movies
                    rating = random.choice([3.5, 4, 4.5, 5])
                else:
                    # Lower ratings for non-preferred movies
                    rating = random.choice([1, 2, 2.5, 3, 3.5])

                ratings.append({
                    'movie_id': movie_id,
                    'rating': rating
                })

            if ratings:
                result.append({
                    'user_id': str(user_id),
                    'ratings': ratings
                })

        return result

    def generate_dataset_with_parameters(self, **params) -> Tuple[List[Dict], Dict]:
        """
        Generate a dataset based on provided parameters with automatic method selection.

        Automatically selects the most appropriate generation method based on
        the provided parameters and returns both the dataset and metadata.

        Args:
            **params: Keyword arguments for dataset generation. Common parameters:
                     - dataset_type: 'simple', 'realistic', or 'clustered'
                     - n_users: Number of users
                     - n_movies: Number of movies
                     - Other method-specific parameters

        Returns:
            tuple: (dataset, metadata) where dataset is the generated data and
                  metadata contains information about the generation process.
        """
        # Extract common parameters
        dataset_type = params.get('dataset_type', 'simple')
        n_users = params.get('n_users', 20)
        n_movies = params.get('n_movies', 50)
        movie_ids = params.get('movie_ids', None)

        # Generate metadata
        metadata = {
            'generation_method': dataset_type,
            'parameters': params.copy(),
            'seed': self.seed
        }

        # Generate dataset based on type
        if dataset_type == 'simple':
            dataset = self.generate_simple_dataset(
                n_users=n_users,
                n_movies=n_movies,
                min_ratings_per_user=params.get('min_ratings_per_user', 2),
                max_ratings_per_user=params.get('max_ratings_per_user', 8)
            )

        elif dataset_type == 'realistic':
            dataset = self.generate_realistic_dataset(
                n_users=n_users,
                n_movies=n_movies,
                sparsity=params.get('sparsity', 0.95)
            )

        elif dataset_type == 'clustered':
            dataset = self.generate_clustered_dataset(
                n_users=n_users,
                n_movies=n_movies,
                n_clusters=params.get('n_clusters', 4)
            )

        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        # Add statistics to metadata
        total_ratings = sum(len(user['ratings']) for user in dataset)
        actual_sparsity = 1 - (total_ratings / (len(dataset) * n_movies))

        metadata['statistics'] = {
            'total_users': len(dataset),
            'total_movies': n_movies,
            'total_ratings': total_ratings,
            'actual_sparsity': actual_sparsity,
            'avg_ratings_per_user': total_ratings / len(dataset) if dataset else 0
        }

        return dataset, metadata


def generate_dataset_with_parameters(
    dataset_type="simple",
    n_users=20,
    n_movies=50,
    seed=None,
    movie_ids: list[int] | None = None,
    # ...other params...
):
    import numpy as np
    rng = np.random.default_rng(seed)
    if movie_ids is not None:
        items = [int(x)
                 for x in (movie_ids[:n_movies] if n_movies else movie_ids)]
    else:
        items = list(range(n_movies))
    dataset = []
    for user_id in range(n_users):
        ratings = []
        for item_id in items:
            rating = float(
                np.clip(np.random.normal(loc=3.5, scale=1.0), 1.0, 5.0))
            ratings.append({"movie_id": int(item_id), "rating": float(rating)})
        dataset.append({"user_id": user_id, "ratings": ratings})
    # ...return dataset and metadata...
