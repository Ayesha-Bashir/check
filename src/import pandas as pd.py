import pandas as pd
movies = pd.read_csv('data/movies.csv')
id_to_title = dict(zip(movies['movieId'], movies['title']))
