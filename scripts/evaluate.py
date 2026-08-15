"""Run the reproducible cold-track retrieval benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moodmix.data import load_tracks
from moodmix.evaluation import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/tracks.csv")
    parser.add_argument("--queries", type=int, default=500)
    parser.add_argument("--validation-queries", type=int, default=250)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    tracks = load_tracks(args.data)
    results, best_weights, tuning_results = run_benchmark(
        tracks,
        k=args.k,
        max_queries=args.queries,
        validation_queries=args.validation_queries,
        tuning_trials=args.trials,
    )
    payload = [result.to_dict() for result in results]
    output = Path("artifacts/evaluation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    weights_output = Path("artifacts/tuned_weights.json")
    weights_output.write_text(json.dumps(best_weights, indent=2), encoding="utf-8")
    tuning_output = Path("artifacts/tuning_results.json")
    tuning_output.write_text(
        json.dumps([result.to_dict() for result in tuning_results], indent=2),
        encoding="utf-8",
    )

    print(f"{'Model':<24} {'P@K':>8} {'R@K':>8} {'NDCG':>8} {'Coverage':>10} {'Artist div.':>12}")
    for result in results:
        print(
            f"{result.model:<24} {result.precision:>8.4f} {result.recall:>8.4f} "
            f"{result.ndcg:>8.4f} {result.catalog_coverage:>10.4f} "
            f"{result.artist_diversity:>12.4f}"
        )
    print(f"Saved {output}")
    print(f"Saved {weights_output}")
    print(f"Saved {tuning_output}")


if __name__ == "__main__":
    main()
