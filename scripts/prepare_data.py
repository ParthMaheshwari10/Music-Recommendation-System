"""Validate and deduplicate the downloaded track catalog."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moodmix.data import dataset_summary, load_tracks


SOURCE = Path("data/raw/spotify_tracks.csv")
DESTINATION = Path("data/processed/tracks.csv")


def main() -> None:
    tracks = load_tracks(SOURCE)
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    tracks.to_csv(DESTINATION, index=False)
    print({**dataset_summary(tracks), "output": str(DESTINATION)})


if __name__ == "__main__":
    main()

