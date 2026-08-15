import numpy as np
import pandas as pd

from moodmix.features import AudioFeatureTransformer
from moodmix.recommender import ContentRecommender


def make_tracks() -> pd.DataFrame:
    rows = []
    for index, (energy, valence, artist) in enumerate(
        [(0.9, 0.8, "A"), (0.85, 0.75, "B"), (0.2, 0.2, "C"), (0.3, 0.25, "D")]
    ):
        rows.append(
            {
                "track_id": str(index),
                "artists": artist,
                "track_name": f"Track {index}",
                "track_genre": "pop" if index < 2 else "ambient",
                "popularity": 50,
                "duration_ms": 180000 + index * 1000,
                "danceability": energy,
                "energy": energy,
                "key": index,
                "loudness": -5 - index,
                "mode": 1,
                "speechiness": 0.05,
                "acousticness": 1 - energy,
                "instrumentalness": 0.1,
                "liveness": 0.1,
                "valence": valence,
                "tempo": 80 + energy * 80,
                "time_signature": 4,
            }
        )
    return pd.DataFrame(rows)


def test_key_encoding_wraps_around():
    tracks = make_tracks().iloc[[0, 1]].copy()
    tracks.loc[tracks.index[0], "key"] = 0
    tracks.loc[tracks.index[1], "key"] = 11
    engineered = AudioFeatureTransformer.engineer(tracks)
    distance = np.linalg.norm(
        engineered.iloc[0][["key_sin", "key_cos"]]
        - engineered.iloc[1][["key_sin", "key_cos"]]
    )
    assert distance < 0.6


def test_recommendation_excludes_seed_and_prefers_similar_track():
    recommender = ContentRecommender().fit(make_tracks())
    result = recommender.recommend(seed_ids=["0"], top_k=1, diversity=0)

    assert result.iloc[0]["track_id"] == "1"
    assert "0" not in result["track_id"].tolist()


def test_mood_only_profile_returns_results():
    recommender = ContentRecommender().fit(make_tracks())
    result = recommender.recommend(mood="Calm", top_k=2)
    assert len(result) == 2
    assert result["why"].str.contains("calm").all()


def test_seed_recommendations_include_explanations():
    recommender = ContentRecommender().fit(make_tracks())
    result = recommender.recommend(seed_ids=["0"], mood="Happy", top_k=1)
    assert "Similar" in result.iloc[0]["why"]
    assert "happy" in result.iloc[0]["why"]
