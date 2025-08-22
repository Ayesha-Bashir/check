"""
File I/O Utilities for Recommendation System
"""

import json
import pickle
import csv
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import zipfile
import io
import base64


class FileManager:
    """
    A utility class for handling file I/O operations for recommendation system data.
    
    Provides methods to save and load datasets, models, results, and visualizations
    in various formats with proper error handling and validation.
    """
    
    def __init__(self, base_path: str = "outputs"):
        """
        Initialize the file manager with a base directory.
        
        Args:
            base_path (str): Base directory for all file operations (default: "outputs")
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
    
    def save_dataset(self, data: List[Dict], filename: str, format_type: str = 'json') -> str:
        """
        Save a dataset to file in the specified format.
        
        Args:
            data (list): The dataset to save (list of user rating dictionaries)
            filename (str): Name of the file (without extension)
            format_type (str): File format ('json', 'csv', 'pickle')
            
        Returns:
            str: Full path to the saved file
            
        Raises:
            ValueError: If format_type is not supported
            IOError: If file cannot be written
        """
        if format_type not in ['json', 'csv', 'pickle']:
            raise ValueError(f"Unsupported format: {format_type}")
        
        filepath = self.base_path / f"{filename}.{format_type}"
        
        try:
            if format_type == 'json':
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            
            elif format_type == 'csv':
                # Convert to flat format for CSV
                rows = []
                for user_data in data:
                    user_id = user_data['user_id']
                    for rating in user_data['ratings']:
                        rows.append({
                            'user_id': user_id,
                            'movie_id': rating['movie_id'],
                            'rating': rating['rating']
                        })
                
                with open(filepath, 'w', newline='', encoding='utf-8') as f:
                    if rows:
                        writer = csv.DictWriter(f, fieldnames=['user_id', 'movie_id', 'rating'])
                        writer.writeheader()
                        writer.writerows(rows)
            
            elif format_type == 'pickle':
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f)
            
            print(f"✓ Dataset saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            raise IOError(f"Failed to save dataset: {str(e)}")
    
    def load_dataset(self, filename: str, format_type: str = 'json') -> List[Dict]:
        """
        Load a dataset from file.
        
        Args:
            filename (str): Name of the file (with or without extension)
            format_type (str): File format ('json', 'csv', 'pickle')
            
        Returns:
            list: The loaded dataset
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If format_type is not supported
        """
        if format_type not in ['json', 'csv', 'pickle']:
            raise ValueError(f"Unsupported format: {format_type}")
        
        # Handle filename with or without extension
        if not filename.endswith(f'.{format_type}'):
            filename = f"{filename}.{format_type}"
        
        filepath = self.base_path / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        try:
            if format_type == 'json':
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            elif format_type == 'csv':
                # Convert from flat CSV format back to nested format
                df = pd.read_csv(filepath)
                data = {}
                
                for _, row in df.iterrows():
                    user_id = str(row['user_id'])
                    if user_id not in data:
                        data[user_id] = {'user_id': user_id, 'ratings': []}
                    
                    data[user_id]['ratings'].append({
                        'movie_id': int(row['movie_id']),
                        'rating': float(row['rating'])
                    })
                
                return list(data.values())
            
            elif format_type == 'pickle':
                with open(filepath, 'rb') as f:
                    return pickle.load(f)
                    
        except Exception as e:
            raise IOError(f"Failed to load dataset: {str(e)}")
    
    def save_recommendations(self, user_id: int, recommendations: List[tuple], 
                           method: str, metadata: Optional[Dict] = None) -> str:
        """
        Save recommendation results to a JSON file.
        
        Args:
            user_id (int): ID of the user for whom recommendations were generated
            recommendations (list): List of (movie_id, predicted_rating) tuples
            method (str): Method used for recommendations ('user_based' or 'item_based')
            metadata (dict, optional): Additional metadata about the recommendations
            
        Returns:
            str: Path to the saved file
        """
        filename = f"recommendations_user_{user_id}_{method}.json"
        filepath = self.base_path / filename
        
        result = {
            'user_id': user_id,
            'method': method,
            'recommendations': [
                {'movie_id': movie_id, 'predicted_rating': rating}
                for movie_id, rating in recommendations
            ],
            'metadata': metadata or {}
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Recommendations saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            raise IOError(f"Failed to save recommendations: {str(e)}")
    
    def save_evaluation_results(self, results: Dict, filename: str = "evaluation_results") -> str:
        """
        Save evaluation metrics to a JSON file.
        
        Args:
            results (dict): Dictionary containing evaluation results
            filename (str): Name of the file (without extension)
            
        Returns:
            str: Path to the saved file
        """
        filepath = self.base_path / f"{filename}.json"
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Evaluation results saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            raise IOError(f"Failed to save evaluation results: {str(e)}")
    
    def create_download_package(self, dataset: List[Dict], recommendations: Dict, 
                              metadata: Dict, package_name: str = "recommendation_package") -> str:
        """
        Create a downloadable ZIP package containing dataset, recommendations, and metadata.
        
        Args:
            dataset (list): The dataset used
            recommendations (dict): Dictionary of recommendations for different users/methods
            metadata (dict): Metadata about the generation and analysis
            package_name (str): Name of the ZIP package
            
        Returns:
            str: Path to the created ZIP file
        """
        zip_filepath = self.base_path / f"{package_name}.zip"
        
        try:
            with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add dataset
                dataset_json = json.dumps(dataset, indent=2, ensure_ascii=False)
                zipf.writestr("dataset.json", dataset_json)
                
                # Add dataset in CSV format
                csv_buffer = io.StringIO()
                rows = []
                for user_data in dataset:
                    user_id = user_data['user_id']
                    for rating in user_data['ratings']:
                        rows.append({
                            'user_id': user_id,
                            'movie_id': rating['movie_id'],
                            'rating': rating['rating']
                        })
                
                if rows:
                    writer = csv.DictWriter(csv_buffer, fieldnames=['user_id', 'movie_id', 'rating'])
                    writer.writeheader()
                    writer.writerows(rows)
                    zipf.writestr("dataset.csv", csv_buffer.getvalue())
                
                # Add recommendations
                rec_json = json.dumps(recommendations, indent=2, ensure_ascii=False)
                zipf.writestr("recommendations.json", rec_json)
                
                # Add metadata
                meta_json = json.dumps(metadata, indent=2, ensure_ascii=False)
                zipf.writestr("metadata.json", meta_json)
                
                # Add README
                readme_content = self._create_readme_content(metadata)
                zipf.writestr("README.txt", readme_content)
            
            print(f"✓ Download package created: {zip_filepath}")
            return str(zip_filepath)
            
        except Exception as e:
            raise IOError(f"Failed to create download package: {str(e)}")
    
    def _create_readme_content(self, metadata: Dict) -> str:
        """
        Create README content for download packages.
        
        Args:
            metadata (dict): Metadata about the analysis
            
        Returns:
            str: README content
        """
        content = """Movie Recommendation System - Analysis Package
================================================

This package contains the results of a collaborative filtering analysis.

Files included:
- dataset.json: The dataset used in JSON format
- dataset.csv: The dataset in CSV format
- recommendations.json: Generated recommendations for users
- metadata.json: Technical details about the analysis
- README.txt: This file

Analysis Details:
"""
        
        if 'generation_method' in metadata:
            content += f"- Dataset Type: {metadata['generation_method']}\n"
        
        if 'statistics' in metadata:
            stats = metadata['statistics']
            content += f"- Total Users: {stats.get('total_users', 'N/A')}\n"
            content += f"- Total Movies: {stats.get('total_movies', 'N/A')}\n"
            content += f"- Total Ratings: {stats.get('total_ratings', 'N/A')}\n"
            content += f"- Dataset Sparsity: {stats.get('actual_sparsity', 'N/A'):.2%}\n"
        
        if 'similarity_methods' in metadata:
            content += f"- Similarity Methods: {', '.join(metadata['similarity_methods'])}\n"
        
        content += """
Usage:
- Load dataset.json or dataset.csv into your recommendation system
- Use recommendations.json to see example recommendations
- Check metadata.json for technical parameters

For questions about this analysis, please refer to the original system documentation.
"""
        
        return content
    
    def get_file_as_base64(self, filepath: Union[str, Path]) -> str:
        """
        Convert a file to base64 encoding for web download.
        
        Args:
            filepath (str or Path): Path to the file to encode
            
        Returns:
            str: Base64 encoded file content
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        try:
            with open(filepath, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            raise IOError(f"Failed to encode file: {str(e)}")
    
    def list_saved_files(self, extension: Optional[str] = None) -> List[str]:
        """
        List all saved files in the base directory.
        
        Args:
            extension (str, optional): Filter by file extension (e.g., 'json', 'zip')
            
        Returns:
            list: List of filenames
        """
        if extension:
            pattern = f"*.{extension}"
        else:
            pattern = "*"
        
        return [f.name for f in self.base_path.glob(pattern) if f.is_file()]
    
    def cleanup_old_files(self, max_files: int = 50) -> int:
        """
        Clean up old files to prevent disk space issues.
        
        Args:
            max_files (int): Maximum number of files to keep
            
        Returns:
            int: Number of files deleted
        """
        all_files = [(f, f.stat().st_mtime) for f in self.base_path.iterdir() if f.is_file()]
        all_files.sort(key=lambda x: x[1], reverse=True)  # Sort by modification time, newest first
        
        if len(all_files) <= max_files:
            return 0
        
        files_to_delete = all_files[max_files:]
        deleted_count = 0
        
        for file_path, _ in files_to_delete:
            try:
                file_path.unlink()
                deleted_count += 1
            except Exception as e:
                print(f"Warning: Could not delete {file_path}: {e}")
        
        if deleted_count > 0:
            print(f"✓ Cleaned up {deleted_count} old files")
        
        return deleted_count
