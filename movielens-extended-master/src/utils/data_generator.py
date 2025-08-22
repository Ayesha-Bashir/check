class DataGenerator:
    """
    A class to generate synthetic datasets for testing and evaluation purposes.
    """

    def __init__(self, seed=None):
        """
        Initialize the DataGenerator with an optional random seed.
        
        Args:
            seed (int): Random seed for reproducibility.
        """
        self.seed = seed

    def generate_dataset_with_parameters(self, dataset_type="simple", **kwargs):
        """
        Generate a synthetic dataset based on specified parameters.
        
        Args:
            dataset_type (str): Type of dataset to generate ('simple', 'realistic', 'clustered').
            **kwargs: Additional parameters for dataset generation.
        
        Returns:
            list: Generated dataset.
            dict: Metadata about the generated dataset.
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        if dataset_type == "simple":
            return self._generate_simple_dataset(**kwargs)
        elif dataset_type == "realistic":
            return self._generate_realistic_dataset(**kwargs)
        elif dataset_type == "clustered":
            return self._generate_clustered_dataset(**kwargs)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    def _generate_simple_dataset(self, n_users=20, n_movies=50, min_ratings_per_user=2, max_ratings_per_user=8):
        """
        Generate a simple synthetic dataset.
        
        Args:
            n_users (int): Number of users.
            n_movies (int): Number of movies.
            min_ratings_per_user (int): Minimum ratings per user.
            max_ratings_per_user (int): Maximum ratings per user.
        
        Returns:
            list: Generated dataset.
            dict: Metadata about the dataset.
        """
        dataset = []
        for user_id in range(n_users):
            n_ratings = np.random.randint(min_ratings_per_user, max_ratings_per_user + 1)
            rated_movies = np.random.choice(range(n_movies), size=n_ratings, replace=False)
            ratings = np.random.randint(1, 6, size=n_ratings)  # Ratings between 1 and 5
            user_data = {
                "user_id": user_id,
                "ratings": [{"movie_id": int(movie_id), "rating": int(rating)} for movie_id, rating in zip(rated_movies, ratings)]
            }
            dataset.append(user_data)

        metadata = {
            "n_users": n_users,
            "n_movies": n_movies,
            "statistics": {
                "average_ratings_per_user": np.mean([len(user["ratings"]) for user in dataset]),
                "total_ratings": sum(len(user["ratings"]) for user in dataset)
            }
        }
        return dataset, metadata

    def _generate_realistic_dataset(self, n_users=20, n_movies=50, sparsity=0.95):
        """
        Generate a realistic synthetic dataset.
        
        Args:
            n_users (int): Number of users.
            n_movies (int): Number of movies.
            sparsity (float): Sparsity level of the dataset.
        
        Returns:
            list: Generated dataset.
            dict: Metadata about the dataset.
        """
        dataset = []
        for user_id in range(n_users):
            n_ratings = np.random.binomial(n_movies, 1 - sparsity)
            rated_movies = np.random.choice(range(n_movies), size=n_ratings, replace=False)
            ratings = np.random.randint(1, 6, size=n_ratings)  # Ratings between 1 and 5
            user_data = {
                "user_id": user_id,
                "ratings": [{"movie_id": int(movie_id), "rating": int(rating)} for movie_id, rating in zip(rated_movies, ratings)]
            }
            dataset.append(user_data)

        metadata = {
            "n_users": n_users,
            "n_movies": n_movies,
            "statistics": {
                "average_ratings_per_user": np.mean([len(user["ratings"]) for user in dataset]),
                "total_ratings": sum(len(user["ratings"]) for user in dataset)
            }
        }
        return dataset, metadata

    def _generate_clustered_dataset(self, n_users=20, n_movies=50, n_clusters=4):
        """
        Generate a clustered synthetic dataset.
        
        Args:
            n_users (int): Number of users.
            n_movies (int): Number of movies.
            n_clusters (int): Number of clusters to create.
        
        Returns:
            list: Generated dataset.
            dict: Metadata about the dataset.
        """
        dataset = []
        cluster_size = n_users // n_clusters
        for cluster_id in range(n_clusters):
            for user_id in range(cluster_size):
                user_id = cluster_id * cluster_size + user_id
                n_ratings = np.random.randint(5, 15)  # More ratings for clustered users
                rated_movies = np.random.choice(range(n_movies), size=n_ratings, replace=False)
                ratings = np.random.randint(1, 6, size=n_ratings)  # Ratings between 1 and 5
                user_data = {
                    "user_id": user_id,
                    "ratings": [{"movie_id": int(movie_id), "rating": int(rating)} for movie_id, rating in zip(rated_movies, ratings)]
                }
                dataset.append(user_data)

        metadata = {
            "n_users": n_users,
            "n_movies": n_movies,
            "statistics": {
                "average_ratings_per_user": np.mean([len(user["ratings"]) for user in dataset]),
                "total_ratings": sum(len(user["ratings"]) for user in dataset)
            }
        }
        return dataset, metadata