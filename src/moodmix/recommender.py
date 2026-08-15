"""Content, mood, and diversity-aware recommendation logic."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MOOD_PRESETS
from .features import AudioFeatureTransformer


class ContentRecommender:
    def __init__(self, transformer: AudioFeatureTransformer | None = None):
        self.transformer = transformer or AudioFeatureTransformer()
        self.tracks: pd.DataFrame | None = None
        self.embeddings: np.ndarray | None = None
        self._id_to_index: dict[str, int] = {}

    def fit(self, tracks: pd.DataFrame) -> "ContentRecommender":
        self.tracks = tracks.reset_index(drop=True).copy()
        self.embeddings = self.transformer.fit_transform(self.tracks)
        self._id_to_index = {
            track_id: index for index, track_id in enumerate(self.tracks["track_id"])
        }
        return self

    def _check_fitted(self) -> None:
        if self.tracks is None or self.embeddings is None:
            raise RuntimeError("Fit the recommender before requesting recommendations.")

    @staticmethod
    def _unit(vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def profile(
        self,
        seed_ids: list[str] | None = None,
        mood: str | None = None,
        mood_strength: float = 0.35,
    ) -> np.ndarray:
        self._check_fitted()
        if not 0 <= mood_strength <= 1:
            raise ValueError("mood_strength must be between 0 and 1.")

        seed_ids = seed_ids or []
        valid_indices = [self._id_to_index[item] for item in seed_ids if item in self._id_to_index]
        seed_vector = None
        if valid_indices:
            seed_vector = self._unit(self.embeddings[valid_indices].mean(axis=0))

        mood_vector = None
        if mood:
            if mood not in MOOD_PRESETS:
                raise ValueError(f"Unknown mood: {mood}")
            row = self.transformer.mood_row(MOOD_PRESETS[mood])
            mood_vector = self.transformer.transform(row)[0]

        if seed_vector is None and mood_vector is None:
            raise ValueError("Choose at least one seed track or a mood.")
        if seed_vector is None:
            return mood_vector
        if mood_vector is None:
            return seed_vector
        return self._unit((1 - mood_strength) * seed_vector + mood_strength * mood_vector)

    def recommend_from_vector(
        self,
        profile: np.ndarray,
        *,
        top_k: int = 10,
        exclude_ids: set[str] | None = None,
        diversity: float = 0.15,
        min_popularity: int = 0,
        candidate_pool: int = 250,
    ) -> pd.DataFrame:
        self._check_fitted()
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        if not 0 <= diversity <= 1:
            raise ValueError("diversity must be between 0 and 1.")

        scores = self.embeddings @ self._unit(profile)
        eligible = self.tracks["popularity"].to_numpy() >= min_popularity
        for track_id in exclude_ids or set():
            index = self._id_to_index.get(track_id)
            if index is not None:
                eligible[index] = False

        candidate_indices = np.flatnonzero(eligible)
        pool_size = min(max(candidate_pool, top_k), len(candidate_indices))
        if pool_size == 0:
            return self.tracks.iloc[[]].copy()
        pool = candidate_indices[np.argpartition(scores[candidate_indices], -pool_size)[-pool_size:]]
        pool = pool[np.argsort(scores[pool])[::-1]]

        selected: list[int] = []
        remaining = pool.tolist()
        while remaining and len(selected) < top_k:
            best_index = remaining[0]
            best_objective = -np.inf
            for index in remaining:
                similarity_penalty = 0.0
                artist_penalty = 0.0
                if selected:
                    similarity_penalty = float(
                        np.max(self.embeddings[selected] @ self.embeddings[index])
                    )
                    prior_artists = set(self.tracks.iloc[selected]["artists"])
                    artist_penalty = 0.10 if self.tracks.iloc[index]["artists"] in prior_artists else 0.0
                objective = float(scores[index]) - diversity * similarity_penalty - artist_penalty
                if objective > best_objective:
                    best_objective = objective
                    best_index = index
            selected.append(best_index)
            remaining.remove(best_index)

        result = self.tracks.iloc[selected].copy()
        result["similarity"] = scores[selected]
        result["spotify_url"] = "https://open.spotify.com/track/" + result["track_id"]
        return result.reset_index(drop=True)

    def recommend(
        self,
        seed_ids: list[str] | None = None,
        mood: str | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        mood_strength = float(kwargs.pop("mood_strength", 0.35))
        vector = self.profile(seed_ids, mood, mood_strength)
        result = self.recommend_from_vector(
            vector,
            exclude_ids=set(seed_ids or []),
            **kwargs,
        )
        result["why"] = self._explain(result, seed_ids or [], mood)
        return result

    def _explain(
        self,
        recommendations: pd.DataFrame,
        seed_ids: list[str],
        mood: str | None,
    ) -> list[str]:
        """Generate compact explanations from the closest original audio fields."""
        labels = {
            "danceability": "danceability",
            "energy": "energy",
            "valence": "valence",
            "acousticness": "acoustic character",
            "instrumentalness": "instrumental character",
            "speechiness": "vocal style",
            "tempo": "tempo",
        }
        scales = {feature: 1.0 for feature in labels}
        scales["tempo"] = 100.0

        seed_reference = None
        valid_indices = [self._id_to_index[item] for item in seed_ids if item in self._id_to_index]
        if valid_indices:
            seed_reference = self.tracks.iloc[valid_indices][list(labels)].mean()

        explanations: list[str] = []
        for _, track in recommendations.iterrows():
            parts: list[str] = []
            if seed_reference is not None:
                distances = {
                    feature: abs(float(track[feature]) - float(seed_reference[feature])) / scales[feature]
                    for feature in labels
                }
                closest = sorted(distances, key=distances.get)[:2]
                parts.append("Similar " + " and ".join(labels[item] for item in closest))
            if mood:
                targets = MOOD_PRESETS[mood]
                mood_distances = {
                    feature: abs(float(track[feature]) - target)
                    for feature, target in targets.items()
                    if feature in track
                }
                closest_mood = min(mood_distances, key=mood_distances.get)
                parts.append(f"matches {mood.lower()} {labels.get(closest_mood, closest_mood)}")
            explanations.append("; ".join(parts).capitalize())
        return explanations

    def transform_queries(self, queries: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self.transformer.transform(queries)
