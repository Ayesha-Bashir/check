"""
Flask Web Application for Movie Recommendation System
"""
import os
import io
import base64
from datetime import datetime
import random
import copy
import collections
import requests

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, make_response

from src.recommender.collaborative_filtering import CollaborativeFilteringRecommender
from src.visualization.plots import RecommenderVisualizer
from src.utils.data_generator import DataGenerator
from src.utils.file_manager import FileManager
from src.data_loader import load_movie_catalog, map_recommendations, load_links
from src.utils.poster_cache import PosterCache

# Use non-interactive backend for matplotlib inside Flask
matplotlib.use('Agg')

app = Flask(__name__)
app.secret_key = 'movie_recommender_secret_key_2025'

# Global storage for session data (in production, use proper session storage)
session_data = {}

# Load movie catalog and build lookup
MOVIES, MOVIE_LOOKUP = load_movie_catalog("data/movies.csv")
N_MOVIES_USED = 500  # adjust as you like
MOVIE_IDS = [int(m["movie_id"]) for m in MOVIES]
USED_MOVIE_IDS = set(MOVIE_IDS[:min(len(MOVIE_IDS), N_MOVIES_USED)])
MOVIES_USED = [m for m in MOVIES if int(m["movie_id"]) in USED_MOVIE_IDS]

# Load links and initialize poster cache
LINKS_MAP = load_links("data/links.csv")
POSTERS = PosterCache(links_map=LINKS_MAP)

# Compute filter metadata


def _get_year_bounds(movies):
    years = [m.get("year") for m in movies if m.get("year")]
    # Ensure max year is at least 2025
    min_year = min(years) if years else 1900
    max_year = max(max(years), 2025) if years else 2025
    return (min_year, max_year)


def _get_top_genres(movies, top_n=10):
    genre_counter = collections.Counter()
    for m in movies:
        genres = str(m.get("genre", "")).split("|")
        genre_counter.update([g for g in genres if g])
    return [g for g, _ in genre_counter.most_common(top_n)]


YEAR_MIN, YEAR_MAX = _get_year_bounds(MOVIES_USED if MOVIES_USED else MOVIES)
GENRES_TOP10 = _get_top_genres(
    MOVIES_USED if MOVIES_USED else MOVIES, top_n=10)

# Initialize model with ONLY these IDs
cf_recommender = CollaborativeFilteringRecommender()
_data_gen = DataGenerator(seed=42)
_initial_dataset, _meta = _data_gen.generate_dataset_with_parameters(
    dataset_type="simple",
    n_users=20,
    n_movies=len(USED_MOVIE_IDS),
    movie_ids=sorted(USED_MOVIE_IDS)
)
cf_recommender.load_data(_initial_dataset)

OMDB_API_KEY = "96846dfe"


@app.route('/')
def how_it_works():
    """Main page of the application."""
    return render_template('index.html')


def _ensure_int_ids(seq):
    return [int(x) for x in seq if str(x).strip() != ""]


@app.route('/view')
def view():
    import random
    pool = MOVIES_USED if MOVIES_USED else MOVIES
    sample = random.sample(pool, k=min(20, len(pool)))
    return render_template('view.html', movies=sample, year_min=YEAR_MIN, year_max=YEAR_MAX, genres_top10=GENRES_TOP10)


@app.route('/generate_dataset', methods=['POST'])
def generate_dataset():
    """Generate a synthetic dataset based on user parameters."""
    try:
        # Get parameters from form
        dataset_type = request.form.get('dataset_type', 'simple')
        n_users = int(request.form.get('n_users', 20))
        n_movies = int(request.form.get('n_movies', 50))
        seed = request.form.get('seed')

        # Convert empty seed to None
        if seed:
            seed = int(seed)
        else:
            seed = None

        # Type-specific parameters
        params = {
            'dataset_type': dataset_type,
            'n_users': n_users,
            'n_movies': n_movies
        }

        if dataset_type == 'realistic':
            params.update({
                'sparsity': float(request.form.get('sparsity', 0.95))
            })
        elif dataset_type == 'clustered':
            params.update({
                'n_clusters': int(request.form.get('n_clusters', 4))
            })
        elif dataset_type == 'simple':
            params.update({
                'min_ratings_per_user': int(request.form.get('min_ratings_per_user', 2)),
                'max_ratings_per_user': int(request.form.get('max_ratings_per_user', 8))
            })

        # Generate dataset
        generator = DataGenerator(seed=seed)
        dataset, metadata = generator.generate_dataset_with_parameters(
            **params)

        # Store in session data
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_data[session_id] = {
            'dataset': dataset,
            'metadata': metadata,
            'recommender': None,
            'recommendations': {},
            'visualizations': {}
        }

        return jsonify({
            'success': True,
            'session_id': session_id,
            'statistics': metadata['statistics'],
            'message': f'Dataset generated successfully with {len(dataset)} users'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/run_analysis', methods=['POST'])
def run_analysis():
    """Run collaborative filtering analysis on the generated dataset."""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        algorithm_type = data.get('algorithm_type', 'original')
        target_user = int(data.get('target_user', 0))
        n_recommendations = int(data.get('n_recommendations', 5))

        # Original algorithm parameters
        similarity_method = data.get('similarity_method', 'cosine')
        recommendation_method = data.get('recommendation_method', 'user_based')
        k_neighbors = int(data.get('k_neighbors', 5))

        # Extended algorithm parameters
        n_factors = int(data.get('n_factors', 20))

        if session_id not in session_data:
            return jsonify({'success': False, 'error': 'Session not found'}), 404

        session = session_data[session_id]
        dataset = session['dataset']

        # Initialize recommender if not exists
        if session['recommender'] is None:
            recommender = CollaborativeFilteringRecommender()
            recommender.load_data(dataset)
            session['recommender'] = recommender
        else:
            recommender = session['recommender']

        # Generate recommendations based on algorithm type
        if algorithm_type == 'original':
            # Calculate similarities for original algorithm
            recommender.calculate_user_similarity(method=similarity_method)
            recommender.calculate_item_similarity(method=similarity_method)

            # Generate recommendations using collaborative filtering
            recommendations = recommender.get_recommendations(
                target_user,
                method=recommendation_method,
                n_recommendations=n_recommendations,
                k=k_neighbors
            )

            method_description = f"{recommendation_method.replace('_', '-').title()} Collaborative Filtering"

        else:  # extended algorithm
            # Generate recommendations using matrix factorization
            recommendations = recommender.extended_algorithm(
                target_user,
                n_recommendations=n_recommendations,
                n_factors=n_factors
            )

            method_description = f"Matrix Factorization (SVD with {n_factors} factors)"

        # Store recommendations
        session['recommendations'][f'{target_user}_{algorithm_type}'] = recommendations
        session['metadata']['algorithm_type'] = algorithm_type
        session['metadata']['last_analysis'] = {
            'target_user': target_user,
            'algorithm_type': algorithm_type,
            'method_description': method_description,
            'n_recommendations': n_recommendations
        }

        # Format recommendations for response
        rec_list = [
            {'movie_id': movie_id, 'predicted_rating': round(rating, 2)}
            for movie_id, rating in recommendations
        ]

        return jsonify({
            'success': True,
            'recommendations': rec_list,
            'user_id': target_user,
            'algorithm_type': algorithm_type,
            'method_description': method_description,
            'message': f'Generated {len(recommendations)} recommendations using {method_description}'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/generate_visualization', methods=['POST'])
def generate_visualization():
    """Generate visualization plots for the analysis."""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        plot_type = data.get('plot_type', 'analysis')

        if session_id not in session_data:
            return jsonify({'success': False, 'error': 'Session not found'}), 404

        session = session_data[session_id]
        recommender = session['recommender']

        if recommender is None:
            return jsonify({'success': False, 'error': 'Analysis not run yet'}), 400

        visualizer = RecommenderVisualizer(recommender)

        if plot_type == 'analysis':
            # Generate data analysis plot
            fig = visualizer.plot_data_analysis()

        elif plot_type == 'recommendations':
            # Generate recommendations plot
            target_user = data.get('target_user', 0)
            method = data.get('method', 'user_based')
            rec_key = f'{target_user}_{method}'

            if rec_key not in session['recommendations']:
                return jsonify({'success': False, 'error': 'Recommendations not found'}), 404

            recommendations = session['recommendations'][rec_key]
            fig = visualizer.plot_recommendations(target_user, recommendations)

        else:
            return jsonify({'success': False, 'error': 'Unknown plot type'}), 400

        # Convert plot to base64
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()

        plt.close(fig)  # Clean up

        return jsonify({
            'success': True,
            'image': img_base64,
            'plot_type': plot_type
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/download_results', methods=['POST'])
def download_results():
    """Create and provide a downloadable package of all results."""
    try:
        data = request.get_json()
        session_id = data.get('session_id')

        if session_id not in session_data:
            return jsonify({'success': False, 'error': 'Session not found'}), 404

        session = session_data[session_id]

        # Create file manager
        file_manager = FileManager(base_path=f"outputs/{session_id}")

        # Create download package
        package_path = file_manager.create_download_package(
            dataset=session['dataset'],
            recommendations=session['recommendations'],
            metadata=session['metadata'],
            package_name=f"movie_recommendations_{session_id}"
        )

        # Convert to absolute path for Flask's send_file
        absolute_path = os.path.abspath(package_path)

        return send_file(
            absolute_path,
            as_attachment=True,
            download_name=f"movie_recommendations_{session_id}.zip",
            mimetype='application/zip'
        )

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/get_user_info/<session_id>/<int:user_id>')
def get_user_info(session_id, user_id):
    """Get information about a specific user's ratings."""
    try:
        if session_id not in session_data:
            return jsonify({'success': False, 'error': 'Session not found'}), 404

        session = session_data[session_id]
        dataset = session['dataset']

        # Find user in dataset
        user_data = None
        for user in dataset:
            if int(user['user_id']) == user_id:
                user_data = user
                break

        if user_data is None:
            return jsonify({'success': False, 'error': f'User {user_id} not found'}), 404

        ratings = user_data['ratings']
        avg_rating = sum(r['rating'] for r in ratings) / \
            len(ratings) if ratings else 0

        return jsonify({
            'success': True,
            'user_id': user_id,
            'total_ratings': len(ratings),
            'average_rating': round(avg_rating, 2),
            'ratings': ratings
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/sessions')
def list_sessions():
    """List all active sessions."""
    sessions = []
    for session_id, data in session_data.items():
        sessions.append({
            'session_id': session_id,
            'statistics': data['metadata']['statistics'],
            'has_analysis': data['recommender'] is not None,
            'num_recommendations': len(data['recommendations'])
        })

    return jsonify({
        'success': True,
        'sessions': sessions
    })


@app.route('/recommend', methods=['POST'])
def recommend():
    import copy

    selected_ids = [int(x) for x in request.form.getlist(
        'selected_movies') if str(x).isdigit()]
    if not selected_ids:
        return render_template("recommend.html", recommendations=[], error="Please select at least one movie.")

    local_rec = copy.deepcopy(cf_recommender)
    try:
        temp_user_id = local_rec.add_ephemeral_user(
            liked_movie_ids=selected_ids, like_rating=5.0
        )
    except ValueError as e:
        return render_template("recommend.html", recommendations=[], error=str(e))

    n_factors = int(request.form.get("n_factors", 20))
    n_recs = int(request.form.get("n_recommendations", 10))
    recs = local_rec.extended_algorithm(
        temp_user_id, n_recommendations=100, n_factors=n_factors
    )

    liked_set = set(selected_ids)
    recs = [(mid, score) for (mid, score) in recs if int(mid) not in liked_set]

    import hashlib
    import random
    seed_str = ",".join(str(x) for x in sorted(selected_ids))
    seed = int(hashlib.sha256(seed_str.encode(
        "utf-8")).hexdigest(), 16) % (2**32)
    rnd = random.Random(seed)
    from itertools import groupby
    grouped = []
    for score, group in groupby(recs, key=lambda t: round(t[1], 4)):
        g = list(group)
        rnd.shuffle(g)
        grouped.extend(g)
    recs = grouped[:n_recs]

    recommendations_list = []
    for mid, score in recs:
        movie = MOVIE_LOOKUP.get(int(mid))
        if movie:
            recommendations_list.append({
                "title": movie.get("title"),
                "genre": movie.get("genre") or movie.get("genres"),
                "year": movie.get("year"),
                "pred": round(score, 2)
            })
    if not recommendations_list:
        return render_template("recommend.html", recommendations=[], error="No recommendations found. Try selecting different movies.")

    return render_template("recommend.html", recommendations=recommendations_list)


@app.route("/api/poster/<int:movie_id>")
def api_poster(movie_id):
    url = POSTERS.get(movie_id)
    resp = make_response(jsonify({"movie_id": movie_id, "url": url}))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.route("/api/search")
def api_search():
    q = (request.args.get("q") or "").strip().lower()
    year_min = request.args.get("year_min", type=int)
    year_max = request.args.get("year_max", type=int)
    genres_raw = (request.args.get("genres") or "").strip()
    selected_genres = [g for g in genres_raw.split(",") if g]
    limit = request.args.get("limit", default=20, type=int)
    do_random = request.args.get("random", default=0, type=int) == 1

    pool = MOVIES_USED if 'MOVIES_USED' in globals() and MOVIES_USED else MOVIES

    # Global bounds for empty-filter detection
    global_min = min(m.get("year", 0)
                     for m in pool if m.get("year") is not None) if pool else 1900
    global_max = max(m.get("year", 0)
                     for m in pool if m.get("year") is not None) if pool else 2025
    if year_min is None:
        year_min = global_min
    if year_max is None:
        year_max = global_max

    def matches(movie):
        if q and q not in str(movie.get("title", "")).lower():
            return False
        y = movie.get("year", None)
        if y is not None and (y < year_min or y > year_max):
            return False
        if selected_genres:
            mg = str(movie.get("genre", ""))
            if not any(g in mg for g in selected_genres):
                return False
        return True

    no_filters = (q == "" and not selected_genres and year_min <=
                  global_min and year_max >= global_max)
    if do_random and no_filters:
        candidates = list(pool)
        random.shuffle(candidates)
    else:
        candidates = [m for m in pool if matches(m)]

    results = []
    for m in candidates[:max(0, min(limit, len(candidates)))]:
        results.append({
            "movie_id": int(m["movie_id"]),
            "title": str(m.get("title", "")),
            "genre": str(m.get("genre", "")),
            "year": m.get("year", None),
            "poster_url": m.get("poster_url", "")
        })

    resp = make_response(jsonify(results))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.route('/api/movie_info/<int:movie_id>')
def api_movie_info(movie_id):
    movie = MOVIE_LOOKUP.get(movie_id)
    if not movie:
        return jsonify({'error': 'Movie not found'}), 404
    title = movie.get('title')
    year = movie.get('year')
    print(f"OMDb lookup: title={title}, year={year}")  # <-- Add this line
    params = {
        't': title,
        'apikey': OMDB_API_KEY
    }
    resp = requests.get('http://www.omdbapi.com/', params=params)
    if resp.status_code == 200:
        data = resp.json()
        if data.get('Response') == 'True':
            return jsonify(data)
        # Try without year if not found
        params.pop('y', None)
        resp2 = requests.get('http://www.omdbapi.com/', params=params)
        data2 = resp2.json()
        if data2.get('Response') == 'True':
            return jsonify(data2)
        else:
            return jsonify({'error': 'Movie not found in OMDb'}), 404
    return jsonify({'error': 'OMDb lookup failed'}), 500


@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors."""
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return render_template('500.html'), 500


if __name__ == '__main__':
    # Create output directory
    os.makedirs('outputs', exist_ok=True)

    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5001)
