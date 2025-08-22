"""
Movie Recommendation System Package
===================================

A comprehensive collaborative filtering recommendation system with visualization
and web interface capabilities.

Author: Taban Abdollahi
Date: August 15, 2025
"""

from .recommender import CollaborativeFilteringRecommender
from .visualization import RecommenderVisualizer
from .utils import DataGenerator, FileManager

__version__ = "1.0.0"
__author__ = "Taban Abdollahi"

__all__ = [
    'CollaborativeFilteringRecommender',
    'RecommenderVisualizer', 
    'DataGenerator',
    'FileManager'
]
