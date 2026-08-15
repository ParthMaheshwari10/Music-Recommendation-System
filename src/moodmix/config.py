"""Shared feature definitions and interpretable mood targets."""

REQUIRED_COLUMNS = {
    "track_id",
    "artists",
    "track_name",
    "popularity",
    "duration_ms",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
    "track_genre",
}

BOUNDED_FEATURES = [
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
]

# Weights are applied in standardized feature space. Genre and popularity are
# deliberately absent: genre is evaluation-only and popularity is a baseline.
FEATURE_WEIGHTS = {
    "danceability": 1.25,
    "energy": 1.50,
    "speechiness": 0.65,
    "acousticness": 1.00,
    "instrumentalness": 0.80,
    "liveness": 0.55,
    "valence": 1.50,
    "loudness": 0.80,
    "tempo_log": 0.75,
    "duration_log": 0.30,
    "key_sin": 0.25,
    "key_cos": 0.25,
    "key_known": 0.10,
    "mode": 0.20,
    "time_signature": 0.10,
}

MOOD_PRESETS = {
    "Happy": {
        "valence": 0.85,
        "energy": 0.70,
        "danceability": 0.72,
        "acousticness": 0.25,
    },
    "Energetic": {
        "valence": 0.65,
        "energy": 0.92,
        "danceability": 0.78,
        "acousticness": 0.10,
    },
    "Calm": {
        "valence": 0.55,
        "energy": 0.25,
        "danceability": 0.40,
        "acousticness": 0.70,
    },
    "Melancholic": {
        "valence": 0.18,
        "energy": 0.35,
        "danceability": 0.35,
        "acousticness": 0.60,
    },
    "Focused": {
        "valence": 0.50,
        "energy": 0.42,
        "danceability": 0.38,
        "acousticness": 0.48,
        "instrumentalness": 0.70,
        "speechiness": 0.05,
    },
}

