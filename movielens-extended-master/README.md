# MovieLens Extended - Movie Recommendation System

## Overview
This project is a movie recommendation system built using Flask. It utilizes collaborative filtering algorithms to provide personalized movie recommendations based on user preferences. The system includes data visualization features to analyze the dataset and the recommendations generated.

## Project Structure
```
movielens-extended-master
├── app.py
├── requirements.txt
├── README.md
├── data
│   ├── movies.csv
│   └── links.csv
├── outputs
├── src
│   ├── recommender
│   │   └── collaborative_filtering.py
│   ├── visualization
│   │   └── plots.py
│   ├── utils
│   │   ├── data_generator.py
│   │   ├── file_manager.py
│   │   └── poster_cache.py
│   └── data_loader.py
├── templates
│   ├── index.html
│   ├── view.html
│   ├── recommend.html
│   ├── 404.html
│   └── 500.html
├── static
│   └── style.css
```

## Setup Instructions
1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd movielens-extended-master
   ```

2. **Install Dependencies**
   It is recommended to use a virtual environment. You can create one using `venv` or `conda`.
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Prepare Data**
   Ensure that the `data/movies.csv` and `data/links.csv` files are present in the `data` directory. These files contain the necessary movie data and links.

4. **Run the Application**
   Start the Flask application by running:
   ```bash
   python app.py
   ```
   The application will be accessible at `http://127.0.0.1:5001`.

## Usage
- **Home Page**: The main page provides an overview of how the recommendation system works.
- **View Recommendations**: Navigate to the view page to see recommended movies based on user preferences.
- **Error Pages**: Custom error pages are provided for 404 and 500 errors.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.