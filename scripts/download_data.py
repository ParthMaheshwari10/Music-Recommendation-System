"""Download a reproducible mirror of the 114k Spotify tracks dataset."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve


URL = "https://huggingface.co/datasets/sfiore/spotify-tracks-dataset/resolve/main/dataset.csv"
DESTINATION = Path("data/raw/spotify_tracks.csv")


def main() -> None:
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}")
    urlretrieve(URL, DESTINATION)
    print(f"Saved {DESTINATION} ({DESTINATION.stat().st_size / 1_000_000:.1f} MB)")


if __name__ == "__main__":
    main()

