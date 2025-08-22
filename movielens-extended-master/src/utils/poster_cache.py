class PosterCache:
    def __init__(self, links_map):
        self.links_map = links_map
        self.cache = {}

    def get(self, movie_id):
        if movie_id in self.cache:
            return self.cache[movie_id]
        
        url = self.links_map.get(movie_id)
        self.cache[movie_id] = url
        return url

    def clear_cache(self):
        self.cache.clear()