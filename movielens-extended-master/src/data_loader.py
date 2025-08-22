def load_movie_catalog(file_path):
    import pandas as pd
    movies = pd.read_csv(file_path)
    return movies.to_dict(orient='records'), {row['movie_id']: row for row in movies.to_dict(orient='records')}

def load_links(file_path):
    import pandas as pd
    links = pd.read_csv(file_path)
    return {row['movie_id']: row['url'] for row in links.to_dict(orient='records')}