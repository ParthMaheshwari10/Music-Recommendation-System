import pandas as pd

from moodmix.data import load_tracks


def test_load_tracks_deduplicates_and_validates(tmp_path):
    row = {
        "track_id": "abc",
        "artists": "Artist",
        "track_name": "Track",
        "popularity": 50,
        "duration_ms": 180000,
        "danceability": 0.5,
        "energy": 0.6,
        "key": 11,
        "loudness": -7,
        "mode": 1,
        "speechiness": 0.05,
        "acousticness": 0.2,
        "instrumentalness": 0.0,
        "liveness": 0.1,
        "valence": 0.7,
        "tempo": 120,
        "time_signature": 4,
        "track_genre": "Pop",
    }
    path = tmp_path / "tracks.csv"
    pd.DataFrame(
        [
            row,
            {**row, "popularity": 40},
            {**row, "track_id": "different-id", "track_name": " TRACK!!! ", "popularity": 45},
        ]
    ).to_csv(path, index=False)

    tracks = load_tracks(path)

    assert len(tracks) == 1
    assert tracks.iloc[0]["popularity"] == 50
    assert tracks.iloc[0]["track_genre"] == "pop"
