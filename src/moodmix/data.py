"""Loading and validation for the Spotify tracks dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import BOUNDED_FEATURES, REQUIRED_COLUMNS


NUMERIC_COLUMNS = [
    "popularity",
    "duration_ms",
    *BOUNDED_FEATURES,
    "key",
    "loudness",
    "mode",
    "tempo",
    "time_signature",
]


def load_tracks(path: str | Path) -> pd.DataFrame:
    """Load, validate, and deduplicate Spotify track features.

    Duplicate Spotify IDs occur in the source under multiple genre buckets, and
    some recordings also have several Spotify IDs. We keep the most-popular row
    for each normalized artist/title pair so a recommendation list cannot return
    another catalog copy of the seed song.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python scripts/download_data.py` "
            "and then `python scripts/prepare_data.py`."
        )

    tracks = pd.read_csv(path, low_memory=False)
    tracks = tracks.loc[:, ~tracks.columns.str.startswith("Unnamed:")].copy()

    missing = sorted(REQUIRED_COLUMNS.difference(tracks.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    for column in NUMERIC_COLUMNS:
        tracks[column] = pd.to_numeric(tracks[column], errors="coerce")

    tracks["track_id"] = tracks["track_id"].astype("string").str.strip()
    tracks["track_name"] = tracks["track_name"].astype("string").str.strip()
    tracks["artists"] = tracks["artists"].astype("string").str.strip()
    tracks["track_genre"] = tracks["track_genre"].astype("string").str.strip().str.lower()

    tracks = tracks.dropna(subset=list(REQUIRED_COLUMNS))
    for feature in BOUNDED_FEATURES:
        tracks = tracks[tracks[feature].between(0.0, 1.0)]
    tracks = tracks[
        tracks["popularity"].between(0, 100)
        & tracks["duration_ms"].between(30_000, 3_600_000)
        & tracks["tempo"].between(20, 300)
        & tracks["key"].between(-1, 11)
        & tracks["mode"].isin([0, 1])
    ]

    tracks["_canonical_title"] = (
        tracks["track_name"].str.casefold().str.replace(r"\W+", " ", regex=True).str.strip()
    )
    tracks["_canonical_artist"] = (
        tracks["artists"].str.casefold().str.replace(r"\W+", " ", regex=True).str.strip()
    )

    tracks = tracks.sort_values(
        ["track_id", "popularity", "track_genre"],
        ascending=[True, False, True],
        kind="stable",
    ).drop_duplicates("track_id", keep="first")
    tracks = tracks.sort_values("popularity", ascending=False, kind="stable").drop_duplicates(
        ["_canonical_title", "_canonical_artist"], keep="first"
    )
    tracks = tracks.drop(columns=["_canonical_title", "_canonical_artist"])

    return tracks.reset_index(drop=True)


def dataset_summary(tracks: pd.DataFrame) -> dict[str, int]:
    return {
        "tracks": len(tracks),
        "artists": tracks["artists"].nunique(),
        "genres": tracks["track_genre"].nunique(),
    }
