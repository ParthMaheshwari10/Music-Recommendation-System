from moodmix.evaluation import _metrics


def test_ranking_metrics_reward_early_relevant_items():
    precision, recall, ndcg = _metrics(
        relevance=[1.0, 0.0, 1.0],
        total_relevant=4,
        k=3,
    )
    assert precision == 2 / 3
    assert recall == 0.5
    assert 0 < ndcg < 1
