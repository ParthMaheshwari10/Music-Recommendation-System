"""Offline evaluation for cold-track content retrieval."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import FEATURE_WEIGHTS
from .features import AudioFeatureTransformer
from .recommender import ContentRecommender


@dataclass
class RankingResult:
    model: str
    queries: int
    k: int
    precision: float
    recall: float
    ndcg: float
    catalog_coverage: float
    artist_diversity: float

    def to_dict(self) -> dict:
        return asdict(self)


def _metrics(relevance: np.ndarray, total_relevant: int, k: int) -> tuple[float, float, float]:
    relevance = np.asarray(relevance, dtype=float)
    hits = int(relevance.sum())
    precision = hits / k
    recall = hits / total_relevant if total_relevant else 0.0
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(relevance * discounts))
    ideal = min(total_relevant, k)
    idcg = float(np.sum(discounts[:ideal]))
    return precision, recall, dcg / idcg if idcg else 0.0


def cold_query_split(
    tracks: pd.DataFrame,
    *,
    query_fraction: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold query tracks out completely, stratified by evaluation-only genre."""
    counts = tracks["track_genre"].value_counts()
    eligible = tracks[tracks["track_genre"].isin(counts[counts >= 10].index)]
    catalog, queries = train_test_split(
        eligible,
        test_size=query_fraction,
        random_state=random_state,
        stratify=eligible["track_genre"],
    )
    return catalog.reset_index(drop=True), queries.reset_index(drop=True)


def cold_validation_test_split(
    tracks: pd.DataFrame,
    *,
    holdout_fraction: float = 0.30,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create disjoint catalog, validation-query, and test-query partitions."""
    counts = tracks["track_genre"].value_counts()
    eligible = tracks[tracks["track_genre"].isin(counts[counts >= 20].index)]
    catalog, holdout = train_test_split(
        eligible,
        test_size=holdout_fraction,
        random_state=random_state,
        stratify=eligible["track_genre"],
    )
    validation, test = train_test_split(
        holdout,
        test_size=0.5,
        random_state=random_state,
        stratify=holdout["track_genre"],
    )
    return (
        catalog.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def evaluate_model(
    recommender: ContentRecommender,
    queries: pd.DataFrame,
    *,
    model_name: str,
    k: int = 10,
    max_queries: int = 500,
    random_state: int = 42,
    popularity_baseline: bool = False,
) -> RankingResult:
    rng = np.random.default_rng(random_state)
    if len(queries) > max_queries:
        queries = queries.iloc[rng.choice(len(queries), max_queries, replace=False)]

    catalog = recommender.tracks
    query_vectors = recommender.transform_queries(queries)
    popularity_order = np.argsort(catalog["popularity"].to_numpy())[::-1]
    precisions: list[float] = []
    recalls: list[float] = []
    ndcgs: list[float] = []
    unique_recommendations: set[str] = set()
    artist_diversities: list[float] = []

    genre_values = catalog["track_genre"].to_numpy()
    for position, (_, query) in enumerate(queries.iterrows()):
        genre = query["track_genre"]
        total_relevant = int(np.sum(genre_values == genre))
        if popularity_baseline:
            indices = popularity_order[:k]
        else:
            scores = recommender.embeddings @ query_vectors[position]
            indices = np.argpartition(scores, -k)[-k:]
            indices = indices[np.argsort(scores[indices])[::-1]]

        relevance = (genre_values[indices] == genre).astype(float)
        precision, recall, ndcg = _metrics(relevance, total_relevant, k)
        precisions.append(precision)
        recalls.append(recall)
        ndcgs.append(ndcg)
        chosen = catalog.iloc[indices]
        unique_recommendations.update(chosen["track_id"])
        artist_diversities.append(chosen["artists"].nunique() / k)

    return RankingResult(
        model=model_name,
        queries=len(queries),
        k=k,
        precision=float(np.mean(precisions)),
        recall=float(np.mean(recalls)),
        ndcg=float(np.mean(ndcgs)),
        catalog_coverage=len(unique_recommendations) / len(catalog),
        artist_diversity=float(np.mean(artist_diversities)),
    )


FEATURE_GROUPS = {
    "mood": ["valence", "energy"],
    "rhythm": ["danceability", "tempo_log", "loudness"],
    "texture": ["acousticness", "instrumentalness", "speechiness"],
    "context": ["liveness", "duration_log"],
    "tonal": ["key_sin", "key_cos", "key_known", "mode", "time_signature"],
}


def _trial_weights(multipliers: dict[str, float]) -> dict[str, float]:
    weights = FEATURE_WEIGHTS.copy()
    for group, multiplier in multipliers.items():
        for feature in FEATURE_GROUPS[group]:
            weights[feature] *= multiplier
    return weights


def tune_feature_weights(
    catalog: pd.DataFrame,
    validation_queries: pd.DataFrame,
    *,
    k: int = 10,
    max_queries: int = 250,
    n_trials: int = 12,
    random_state: int = 42,
) -> tuple[dict[str, float], list[RankingResult]]:
    """Tune interpretable feature-group multipliers using validation NDCG only."""
    rng = np.random.default_rng(random_state)
    candidates: list[tuple[str, dict[str, float]]] = [("Hand-weighted", FEATURE_WEIGHTS.copy())]
    levels = np.array([0.65, 0.80, 1.0, 1.25, 1.55])
    seen: set[tuple[float, ...]] = set()
    while len(candidates) < max(1, n_trials):
        values = tuple(float(value) for value in rng.choice(levels, len(FEATURE_GROUPS)))
        if values in seen:
            continue
        seen.add(values)
        multipliers = dict(zip(FEATURE_GROUPS, values))
        candidates.append((f"Trial {len(candidates):02d}", _trial_weights(multipliers)))

    results: list[RankingResult] = []
    best_weights = FEATURE_WEIGHTS.copy()
    best_ndcg = -np.inf
    for name, weights in candidates:
        recommender = ContentRecommender(AudioFeatureTransformer(weights)).fit(catalog)
        result = evaluate_model(
            recommender,
            validation_queries,
            model_name=name,
            k=k,
            max_queries=max_queries,
            random_state=random_state,
        )
        results.append(result)
        if result.ndcg > best_ndcg:
            best_ndcg = result.ndcg
            best_weights = weights
    return best_weights, results


def run_benchmark(
    tracks: pd.DataFrame,
    *,
    k: int = 10,
    max_queries: int = 500,
    random_state: int = 42,
    tuning_trials: int = 12,
    validation_queries: int = 250,
) -> tuple[list[RankingResult], dict[str, float], list[RankingResult]]:
    catalog, validation, test = cold_validation_test_split(
        tracks, random_state=random_state
    )

    best_weights, tuning_results = tune_feature_weights(
        catalog,
        validation,
        k=k,
        max_queries=validation_queries,
        n_trials=tuning_trials,
        random_state=random_state,
    )

    weighted = ContentRecommender().fit(catalog)
    tuned = ContentRecommender(AudioFeatureTransformer(best_weights)).fit(catalog)
    unweighted_transformer = AudioFeatureTransformer(
        {feature: 1.0 for feature in FEATURE_WEIGHTS}
    )
    unweighted = ContentRecommender(unweighted_transformer).fit(catalog)

    results = [
        evaluate_model(
            weighted,
            test,
            model_name="Popularity",
            k=k,
            max_queries=max_queries,
            random_state=random_state,
            popularity_baseline=True,
        ),
        evaluate_model(
            unweighted,
            test,
            model_name="Unweighted cosine",
            k=k,
            max_queries=max_queries,
            random_state=random_state,
        ),
        evaluate_model(
            weighted,
            test,
            model_name="Hand-weighted cosine",
            k=k,
            max_queries=max_queries,
            random_state=random_state,
        ),
        evaluate_model(
            tuned,
            test,
            model_name="Validation-tuned cosine",
            k=k,
            max_queries=max_queries,
            random_state=random_state,
        ),
    ]
    return results, best_weights, tuning_results
