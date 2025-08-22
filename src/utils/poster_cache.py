import os
import json
import time
import requests

CACHE_PATH = "data/posters_cache.json"
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
# choose size: w185, w342, w500, original
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w342"


class PosterCache:
    def __init__(self, links_map: dict[int, int], ttl_seconds: int = 30*24*3600):
        """
        links_map: {movieId:int -> tmdbId:int or None}
        """
        self.links_map = links_map
        self.ttl = ttl_seconds
        self._cache = {}
        self._load()

    def _load(self):
        try:
            with open(CACHE_PATH, "r") as f:
                self._cache = json.load(f)
        except Exception:
            self._cache = {}

    def _save(self):
        try:
            os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
            with open(CACHE_PATH, "w") as f:
                json.dump(self._cache, f)
        except Exception:
            pass

    def get(self, movie_id: int) -> str | None:
        """
        Returns a poster URL or None. Uses cache if fresh; otherwise queries TMDB.
        """
        key = str(movie_id)
        now = int(time.time())
        # fresh cache?
        if key in self._cache:
            rec = self._cache[key]
            if isinstance(rec, dict) and "url" in rec and "ts" in rec and (now - rec["ts"] < self.ttl):
                return rec["url"]

        # no TMDB key or no tmdbId → no poster
        if not TMDB_API_KEY:
            self._cache[key] = {"url": None, "ts": now}
            self._save()
            return None

        tmdb_id = self.links_map.get(int(movie_id))
        if not tmdb_id:
            self._cache[key] = {"url": None, "ts": now}
            self._save()
            return None

        # fetch details from TMDB: /movie/{tmdb_id}
        try:
            url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
            r = requests.get(url, params={"api_key": TMDB_API_KEY})
            if r.status_code == 200:
                data = r.json()
                poster_path = data.get("poster_path")
                poster_url = f"{TMDB_IMAGE_BASE}{poster_path}" if poster_path else None
                self._cache[key] = {"url": poster_url, "ts": now}
            else:
                self._cache[key] = {"url": None, "ts": now}
        except Exception:
            self._cache[key] = {"url": None, "ts": now}

        self._save()
        return self._cache[key]["url"]
