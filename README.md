# MovieLens Extended: Advanced Movie Recommendation System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**GitHub Repository:** [https://github.com/taban3010/movielens-extended](https://github.com/taban3010/movielens-extended)  
**GitHub Username:** taban3010

---

## Table of Contents
- [Background](#background)
- [Algorithm](#algorithm)
- [Extension](#extension)
- [Deployment Guide](#deployment-guide)
- [Usage Instructions](#usage-instructions)
- [Solution Documentation](#solution-documentation)
- [Contribution](#contribution)
- [References](#references)

---

## Background

MovieLens is a web-based recommender system and virtual community that recommends movies for its users to watch, based on their film preferences. The original platform, developed by the GroupLens Research group at the University of Minnesota, generates personalized predictions for movies users haven't seen yet using collaborative filtering algorithms.

### How the Original Platform Functions

The original MovieLens system operates through a multi-step process:

1. **Data Collection**: Users rate movies on a 1-5 scale based on their preferences and viewing experiences
2. **User-Item Matrix Construction**: The system creates a sparse matrix where rows represent users and columns represent movies, with ratings as values
3. **Similarity Computation**: The platform uses correlation analysis to identify:
   - Users with similar rating patterns (user-based approach)
   - Movies with similar rating distributions (item-based approach)
4. **Prediction Generation**: Recommendations are made by computing weighted averages of ratings from similar users or for similar items
5. **Recommendation Delivery**: The system presents personalized movie suggestions ranked by predicted ratings

This collaborative filtering approach addresses the "information overload" problem by helping users discover relevant content from vast movie catalogs containing thousands of films. MovieLens has been instrumental as a research platform for recommendation systems, contributing to advances in collaborative filtering, machine learning, and personalization technologies (Harper & Konstan, 2015).

The platform's significance extends beyond entertainment, serving as a benchmark dataset for recommendation system research and helping establish fundamental principles in collaborative filtering that are now used across various domains including e-commerce, music streaming, and social media platforms.

---

## Algorithm

### Original MovieLens Collaborative Filtering Implementation

This project implements the core **Collaborative Filtering (CF)** algorithms that power traditional recommendation systems. The implementation includes both neighborhood-based approaches with multiple similarity metrics.

#### 1. User-Based Collaborative Filtering

The user-based approach identifies users with similar preferences and uses their ratings to predict preferences for the target user.

```python
def predict_user_based(self, user_id, movie_id, k=5, similarity_method='cosine'):
    """
    Predict rating using user-based collaborative filtering
    """
    user_idx = self.ratings_matrix.index.get_loc(user_id)
    movie_idx = self.ratings_matrix.columns.get_loc(movie_id)
    
    # Find users who rated the target movie
    movie_ratings = self.ratings_matrix.iloc[:, movie_idx]
    rated_users = movie_ratings[movie_ratings > 0]
    
    if len(rated_users) == 0:
        return self.global_mean
    
    # Calculate similarities with target user
    similarities = []
    for other_user_id in rated_users.index:
        other_user_idx = self.ratings_matrix.index.get_loc(other_user_id)
        sim = self.user_similarity[user_idx, other_user_idx]
        
        if sim > 0:  # Only consider positive similarities
            similarities.append((sim, rated_users[other_user_id]))
    
    # Weighted average of top-k similar users
    similarities.sort(key=lambda x: x[0], reverse=True)
    top_k = similarities[:k]
    
    if not top_k:
        return self.global_mean
    
    numerator = sum(sim * rating for sim, rating in top_k)
    denominator = sum(sim for sim, _ in top_k)
    
    return numerator / denominator if denominator > 0 else self.global_mean
```

**How it works:**
- Computes user similarities using cosine similarity, Pearson correlation, or Jaccard index
- Identifies the k most similar users who have rated the target movie
- Predicts ratings as weighted averages from similar users
- Best suited for datasets with many users and established user preferences

#### 2. Item-Based Collaborative Filtering

The item-based approach finds movies with similar rating patterns and uses the user's ratings for similar movies to make predictions.

```python
def predict_item_based(self, user_id, movie_id, k=5):
    """
    Predict rating using item-based collaborative filtering
    """
    user_ratings = self.ratings_matrix.loc[user_id]
    rated_movies = user_ratings[user_ratings > 0]
    
    if len(rated_movies) == 0:
        return self.global_mean
    
    movie_idx = self.ratings_matrix.columns.get_loc(movie_id)
    
    # Calculate similarities with target movie
    similarities = []
    for other_movie_id in rated_movies.index:
        other_movie_idx = self.ratings_matrix.columns.get_loc(other_movie_id)
        sim = self.item_similarity[movie_idx, other_movie_idx]
        
        if sim > 0:
            similarities.append((sim, rated_movies[other_movie_id]))
    
    # Weighted average of top-k similar items
    similarities.sort(key=lambda x: x[0], reverse=True)
    top_k = similarities[:k]
    
    if not top_k:
        return self.global_mean
    
    numerator = sum(sim * rating for sim, rating in top_k)
    denominator = sum(sim for sim, _ in top_k)
    
    return numerator / denominator if denominator > 0 else self.global_mean
```

**How it works:**
- Calculates item-item similarities across all user ratings
- Finds movies most similar to the target movie that the user has already rated
- Predicts ratings based on the user's preferences for similar movies
- More stable than user-based CF and performs well with large item catalogs

#### 3. Similarity Metrics Implementation

The system supports multiple similarity metrics to accommodate different data characteristics:

```python
def calculate_similarity(self, method='cosine'):
    """
    Calculate similarity matrices using specified method
    """
    if method == 'cosine':
        # Fill NaN with 0 for cosine similarity
        filled_matrix = self.ratings_matrix.fillna(0)
        self.user_similarity = cosine_similarity(filled_matrix)
        self.item_similarity = cosine_similarity(filled_matrix.T)
        
    elif method == 'pearson':
        # Pearson correlation with centered ratings
        self.user_similarity = self.ratings_matrix.T.corr().fillna(0).values
        self.item_similarity = self.ratings_matrix.corr().fillna(0).values
        
    elif method == 'jaccard':
        # Jaccard similarity for binary interactions
        binary_matrix = (self.ratings_matrix > 0).astype(int)
        self.user_similarity = self._jaccard_similarity(binary_matrix)
        self.item_similarity = self._jaccard_similarity(binary_matrix.T)
```

---

## Extension

### Advanced Matrix Factorization with SVD

Our extended algorithm implements **Singular Value Decomposition (SVD) Matrix Factorization** with bias terms, providing significant improvements over traditional collaborative filtering approaches.

#### Mathematical Foundation

The algorithm decomposes the user-item rating matrix **R** into latent factor matrices:

```
R̂ᵤᵢ = μ + bᵤ + bᵢ + pᵤᵀqᵢ
```

Where:
- **μ**: Global average rating across all users and items
- **bᵤ**: User bias term (user's tendency to rate above/below average)
- **bᵢ**: Item bias term (item's tendency to be rated above/below average)
- **pᵤ**: User latent factor vector (k dimensions)
- **qᵢ**: Item latent factor vector (k dimensions)

#### Core Implementation

```python
def extended_algorithm(self, user_id, n_recommendations=5, **kwargs):
    """
    Advanced matrix factorization using SVD with bias terms
    """
    n_factors = kwargs.get('n_factors', 20)
    
    # Prepare data for SVD
    R_filled = self.ratings_matrix.fillna(self.global_mean)
    
    # Apply Truncated SVD
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=n_factors, random_state=42)
    user_factors = svd.fit_transform(R_filled)
    item_factors = svd.components_.T
    
    # Get user vector
    user_idx = self.ratings_matrix.index.get_loc(user_id)
    user_vector = user_factors[user_idx]
    
    # Find unrated movies
    user_ratings = self.ratings_matrix.loc[user_id]
    unrated_movies = user_ratings[user_ratings.isna()].index
    
    predictions = []
    for movie_id in unrated_movies:
        movie_idx = self.ratings_matrix.columns.get_loc(movie_id)
        movie_vector = item_factors[movie_idx]
        
        # Calculate bias terms
        user_bias = self.user_mean_ratings[user_id] - self.global_mean
        item_bias = self.item_mean_ratings[movie_id] - self.global_mean
        
        # Matrix factorization prediction
        latent_prediction = np.dot(user_vector, movie_vector)
        
        # Combine bias terms with latent factors
        predicted_rating = (self.global_mean + user_bias + 
                          item_bias + latent_prediction)
        
        # Clip to valid rating range
        predicted_rating = np.clip(predicted_rating, 1, 5)
        
        predictions.append({
            'movie_id': movie_id,
            'predicted_rating': round(predicted_rating, 2)
        })
    
    # Sort and return top recommendations
    predictions.sort(key=lambda x: x['predicted_rating'], reverse=True)
    return predictions[:n_recommendations]
```

#### Advanced Features and Improvements

**1. Bias Term Integration**
```python
def calculate_bias_terms(self):
    """
    Calculate user and item bias terms for improved predictions
    """
    # Global mean
    self.global_mean = self.ratings_matrix.stack().mean()
    
    # User bias: how much each user deviates from global average
    self.user_bias = self.user_mean_ratings - self.global_mean
    
    # Item bias: how much each item deviates from global average  
    self.item_bias = self.item_mean_ratings - self.global_mean
```

**2. Latent Factor Discovery**
The algorithm automatically discovers hidden patterns in user preferences and movie characteristics:
- **User Factors**: Capture personality traits, genre preferences, rating tendencies
- **Movie Factors**: Capture genre, director style, production quality, popularity patterns

**3. Dimensionality Reduction Benefits**
- Reduces a sparse 1000×500 matrix to dense 20×500 and 1000×20 factor matrices
- Eliminates noise while preserving essential rating patterns
- Enables efficient computation even with millions of users and items

#### Why the Extension is Superior

**Performance Improvements:**
1. **Better Accuracy**: Typically achieves 10-15% lower RMSE compared to traditional collaborative filtering
2. **Sparsity Handling**: Effectively handles datasets with >95% missing ratings
3. **Scalability**: O(k) complexity per prediction vs O(n) for neighborhood methods
4. **Cold Start**: Better performance with new items that have few ratings

**User Experience Benefits:**
1. **Serendipitous Discovery**: Uncovers hidden preference patterns users aren't aware of
2. **Personalized Bias Correction**: Accounts for individual rating behaviors (harsh vs lenient critics)
3. **Diverse Recommendations**: Latent factors capture nuanced preferences beyond obvious similarities
4. **Consistent Quality**: More stable predictions with varying data density

**Real-World Impact:**
This approach mirrors the techniques that won the Netflix Prize competition, where matrix factorization methods achieved breakthrough improvements in recommendation accuracy. Netflix reported that better recommendations directly translated to increased user engagement and reduced churn rates.

---

## Deployment Guide

### Prerequisites

Before installing this system, ensure you have the following installed on your machine:

#### Installing Python (if not already installed)

**Windows:**
1. Download Python from [python.org](https://www.python.org/downloads/)
2. Run the installer and check "Add Python to PATH"
3. Verify installation: `python --version`

**macOS:**
```bash
# Using Homebrew (recommended)
brew install python

# Or download from python.org
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

#### Additional Requirements
- **Git**: For cloning the repository
- **Web Browser**: Chrome, Firefox, Safari, or Edge for accessing the web interface

### Installation Steps

#### 1. Clone the Repository
```bash
git clone https://github.com/taban3010/movielens-extended.git
cd movielens-extended
```

#### 2. Create and Activate Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate

# macOS/Linux:
source .venv/bin/activate
```

#### 3. Install Dependencies
```bash
# Upgrade pip (recommended)
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

#### 4. Verify Installation
```bash
# Run tests to verify everything is working
python -m pytest tests/ -v

# If tests pass, the system is ready to use
```

#### 5. Start the Application
```bash
# Navigate to web directory and start Flask server
cd web
python app.py
```

#### 6. Access the Web Interface
Open your web browser and navigate to:
```
http://localhost:5000
```

You should see the MovieLens Extended interface with four main steps:
1. Generate Dataset
2. Run Analysis  
3. Visualizations
4. Download Results

### Troubleshooting Common Issues

**Issue: "Python not found"**
- Solution: Ensure Python is added to your system PATH
- Windows: Reinstall Python with "Add to PATH" checked
- macOS/Linux: Use `python3` instead of `python`

**Issue: "Permission denied" errors**
- Solution: Use virtual environment and avoid system-wide installation
- Ensure you've activated the virtual environment before installing packages

**Issue: "Module not found" errors**
- Solution: Ensure all dependencies are installed in the correct environment
- Run `pip list` to verify installed packages
- Re-run `pip install -r requirements.txt`

**Issue: "Port 5000 already in use"**
- Solution: Either stop other services using port 5000 or modify `web/app.py`:
```python
if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Use different port
```

### Production Deployment

For production deployment on cloud platforms:

**Heroku Deployment:**
```bash
# Create Procfile
echo "web: gunicorn --chdir web app:app" > Procfile

# Install gunicorn
pip install gunicorn
pip freeze > requirements.txt

# Deploy to Heroku
heroku create your-app-name
git push heroku main
```

**AWS/Digital Ocean:**
1. Set up server with Python 3.8+
2. Install nginx as reverse proxy
3. Use gunicorn as WSGI server
4. Set environment variables for production
5. Configure SSL certificates for HTTPS

---

## Usage Instructions

### Step-by-Step Workflow

The system implements a controlled workflow where each step must be completed before the next becomes available:

#### Step 1: Generate Dataset
1. **Choose Dataset Type:**
   - **Simple**: Random ratings with basic user-item interactions
   - **Realistic**: Incorporates rating biases and sparsity patterns
   - **Clustered**: Creates user groups with distinct preferences

2. **Configure Parameters:**
   - Number of users (10-500)
   - Number of movies (20-1000)
   - Sparsity level (for realistic datasets)
   - Random seed (for reproducible results)

3. **Generate**: Click "Generate Dataset" (can only be done once per session)

#### Step 2: Run Analysis
1. **Select Algorithm:**
   - **MovieLens Original**: Traditional collaborative filtering
   - **MovieLens Extended**: Advanced matrix factorization

2. **Configure Analysis:**
   - Target user ID for recommendations
   - Number of recommendations (1-20)
   - Algorithm-specific parameters (similarity method, k-neighbors, latent factors)

3. **Execute**: Click "Run Analysis" (can only be done once per session)

#### Step 3: Generate Visualizations
1. **Data Analysis Plot**: Overview of rating distributions and patterns
2. **Recommendations Plot**: Visualization of recommendations for the target user
3. **Note**: Visualization step can only be completed once per session

#### Step 4: Download Results
1. **Download Package**: Contains complete analysis results in ZIP format
2. **Includes**: Dataset files, recommendations, metadata, and usage instructions
3. **Note**: Download can only be performed once per session

#### Step 5: Restart Process
After completing all steps, use the "Restart" button to begin a new analysis with different parameters.

### Understanding the Results

**Recommendation Output:**
- Movie IDs with predicted ratings (1-5 scale)
- Confidence scores based on algorithm certainty
- Method descriptions explaining how recommendations were generated

**Evaluation Metrics:**
- RMSE (Root Mean Square Error): Lower values indicate better accuracy
- MAE (Mean Absolute Error): Average prediction error magnitude
- Coverage: Percentage of items the system can recommend

**Visualization Insights:**
- Rating distribution patterns reveal user behavior
- Similarity heatmaps show user/item relationships
- Recommendation scatter plots display prediction confidence

---

## Solution Documentation

For a detailed explanation of the development journey, algorithm implementation challenges, and technical solutions, please refer to the comprehensive [SOLUTION.md](SOLUTION.md) file. This document provides:

- **Development Journey**: Step-by-step progression from basic collaborative filtering to advanced matrix factorization
- **Technical Challenges**: Detailed analysis of scalability, sparsity, and performance issues encountered
- **Implementation Details**: Code explanations, optimization strategies, and algorithm comparisons
- **Research Foundation**: Background research that informed algorithm selection and design decisions
- **Performance Analysis**: Detailed evaluation metrics and comparison results

The SOLUTION.md serves as a technical companion to this README, offering deeper insights into the reasoning behind implementation choices and the complete development process.

---

## Contribution

**Taban Abdollahi (taban3010)**

This project was completed as an individual effort with comprehensive implementation across all system components:

**Core Algorithm Development:**
- Designed and implemented the complete collaborative filtering system architecture
- Developed user-based and item-based collaborative filtering algorithms with multiple similarity metrics (cosine, Pearson correlation, Jaccard index)
- Implemented advanced matrix factorization extension using Singular Value Decomposition with bias terms
- Created sophisticated evaluation framework with RMSE, MAE, and coverage metrics

**Data Engineering:**
- Built comprehensive data generation modules supporting three distinct dataset types (realistic, clustered, simple)
- Implemented robust data preprocessing pipelines with missing value handling and normalization
- Created flexible data structures supporting various sparsity levels and user/item configurations
- Developed data validation and consistency checking mechanisms

**Web Application Development:**
- Designed and implemented complete Flask-based web interface with modern Bootstrap frontend
- Created intuitive step-by-step workflow system with progress tracking and one-time execution controls
- Developed comprehensive visualization components using matplotlib and seaborn for similarity matrices, recommendation analysis, and performance metrics
- Implemented file management system for session handling and result packaging

**Software Engineering:**
- Established complete project structure with modular design and separation of concerns
- Created comprehensive testing suite with unit tests and integration tests achieving >90% code coverage
- Implemented robust error handling and user feedback mechanisms throughout the system
- Developed deployment documentation and automated setup procedures

**Research and Documentation:**
- Conducted extensive literature review of collaborative filtering and matrix factorization techniques
- Created comprehensive technical documentation with mathematical foundations and algorithmic explanations
- Developed user guides and deployment instructions for multiple platforms
- Implemented code commenting and docstring standards for maintainability

*Project Timeline: 6 weeks of individual development, encompassing research, design, implementation, testing, and documentation phases.*

---

## References

1. Harper, F. M., & Konstan, J. A. (2015). The MovieLens datasets: History and context. *ACM Transactions on Interactive Intelligent Systems*, 5(4), 1-19. doi:10.1145/2827872

2. Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. *Computer*, 42(8), 30-37. doi:10.1109/MC.2009.263

3. Sarwar, B., Karypis, G., Konstan, J., & Riedl, J. (2001). Item-based collaborative filtering recommendation algorithms. *Proceedings of the 10th International Conference on World Wide Web*, 285-295. doi:10.1145/371920.372071

4. Su, X., & Khoshgoftaar, T. M. (2009). A survey of collaborative filtering techniques. *Advances in Artificial Intelligence*, 2009, 1-19. doi:10.1155/2009/421425

5. Koren, Y. (2008). Factorization meets the neighborhood: A multifaceted collaborative filtering model. *Proceedings of the 14th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 426-434. doi:10.1145/1401890.1401944

6. Ricci, F., Rokach, L., Shapira, B., & Kantor, P. B. (Eds.). (2010). *Recommender systems handbook*. Springer Science & Business Media. doi:10.1007/978-0-387-85820-3

7. Breese, J. S., Heckerman, D., & Kadie, C. (1998). Empirical analysis of predictive algorithms for collaborative filtering. *Proceedings of the Fourteenth Conference on Uncertainty in Artificial Intelligence*, 43-52.

8. GroupLens Research. (2023). MovieLens. Retrieved from https://movielens.org/info/about

9. Netflix Prize. (2009). *The Netflix Prize Rules*. Retrieved from https://www.netflixprize.com/rules.html

10. Herlocker, J. L., Konstan, J. A., Terveen, L. G., & Riedl, J. T. (2004). Evaluating collaborative filtering recommender systems. *ACM Transactions on Information Systems*, 22(1), 5-53. doi:10.1145/963770.963772

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- GroupLens Research at the University of Minnesota for the original MovieLens platform inspiration
- The collaborative filtering and recommendation systems research community
- Open source libraries that made this implementation possible (NumPy, pandas, scikit-learn, Flask)

Metadata Enrichment API (TMDb/OMDb)
-----------------------------------
Setup:
- Install dependencies: pip install -r requirements.txt
- Required environment variables in .env:
    TMDB_BEARER=<your TMDb v4 token>
    OMDB_KEY=<your OMDb API key>
- Run the enrichment server: python server.py

Attribution:
This product uses the TMDB API but is not endorsed or certified by TMDB.

Note:
SVD logic unchanged; UI now enriches and opens Netflix-style modal for recommendations.
