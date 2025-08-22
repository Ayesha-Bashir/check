class FileManager:
    """Handles file operations for the movie recommendation system."""

    def __init__(self, base_path):
        self.base_path = base_path

    def create_download_package(self, dataset, recommendations, metadata, package_name):
        """Create a downloadable package of results."""
        package_path = os.path.join(self.base_path, f"{package_name}.zip")
        
        with zipfile.ZipFile(package_path, 'w') as zipf:
            # Save dataset
            dataset_path = os.path.join(self.base_path, f"{package_name}_dataset.csv")
            pd.DataFrame(dataset).to_csv(dataset_path, index=False)
            zipf.write(dataset_path, arcname=f"{package_name}_dataset.csv")
            
            # Save recommendations
            recommendations_path = os.path.join(self.base_path, f"{package_name}_recommendations.json")
            with open(recommendations_path, 'w') as f:
                json.dump(recommendations, f)
            zipf.write(recommendations_path, arcname=f"{package_name}_recommendations.json")
            
            # Save metadata
            metadata_path = os.path.join(self.base_path, f"{package_name}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            zipf.write(metadata_path, arcname=f"{package_name}_metadata.json")

        return package_path

    def clean_up(self):
        """Remove temporary files created during the download package creation."""
        for file in os.listdir(self.base_path):
            file_path = os.path.join(self.base_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)