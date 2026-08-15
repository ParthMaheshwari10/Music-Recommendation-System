"""Streamlit interface for MoodMix."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moodmix.config import MOOD_PRESETS
from moodmix.data import dataset_summary, load_tracks
from moodmix.features import AudioFeatureTransformer
from moodmix.recommender import ContentRecommender


DATA_PATH = ROOT / "data/processed/tracks.csv"
EVALUATION_PATH = ROOT / "artifacts/evaluation.json"
WEIGHTS_PATH = ROOT / "artifacts/tuned_weights.json"


@st.cache_data
def get_tracks() -> pd.DataFrame:
    return load_tracks(DATA_PATH)


@st.cache_resource
def get_recommender() -> ContentRecommender:
    transformer = None
    if WEIGHTS_PATH.exists():
        weights = json.loads(WEIGHTS_PATH.read_text(encoding="utf-8"))
        transformer = AudioFeatureTransformer(weights)
    return ContentRecommender(transformer).fit(get_tracks())


st.set_page_config(page_title="MoodMix", page_icon="🎧", layout="wide")
st.title("MoodMix")
st.caption("Cold-start music discovery using real Spotify audio features")

try:
    tracks = get_tracks()
    recommender = get_recommender()
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.code("python scripts/download_data.py\npython scripts/prepare_data.py")
    st.stop()

summary = dataset_summary(tracks)
with st.sidebar:
    st.subheader("Your listening context")
    mood = st.selectbox("Mood", list(MOOD_PRESETS), index=0)
    mood_strength = st.slider("Mood influence", 0.0, 1.0, 0.35, 0.05)
    diversity = st.slider("Exploration", 0.0, 0.5, 0.15, 0.05)
    min_popularity = st.slider("Minimum popularity", 0, 80, 10, 5)
    result_count = st.slider("Recommendations", 5, 25, 10)
    st.divider()
    st.write(f"{summary['tracks']:,} tracks")
    st.write(f"{summary['artists']:,} artists")
    st.write(f"{summary['genres']:,} evaluation genres")

tab_recommend, tab_evaluate, tab_method = st.tabs(
    ["Discover", "Evaluation", "How it works"]
)

with tab_recommend:
    choices = tracks.nlargest(20_000, "popularity").copy()
    choices["label"] = choices["track_name"] + " — " + choices["artists"]
    label_to_id = dict(zip(choices["label"], choices["track_id"]))
    selected_labels = st.multiselect(
        "Choose up to five seed tracks",
        options=choices["label"].tolist(),
        max_selections=5,
        placeholder="Type a song or artist…",
    )

    if st.button("Create my mix", type="primary", use_container_width=True):
        seed_ids = [label_to_id[label] for label in selected_labels]
        recommendations = recommender.recommend(
            seed_ids=seed_ids,
            mood=mood,
            mood_strength=mood_strength,
            top_k=result_count,
            diversity=diversity,
            min_popularity=min_popularity,
        )
        display = recommendations[
            ["track_name", "artists", "track_genre", "similarity", "why", "spotify_url"]
        ].rename(
            columns={
                "track_name": "Track",
                "artists": "Artist",
                "track_genre": "Genre",
                "similarity": "Match",
                "why": "Why this track",
                "spotify_url": "Spotify",
            }
        )
        display["Match"] = display["Match"].map(lambda value: f"{value:.1%}")
        st.dataframe(
            display,
            hide_index=True,
            use_container_width=True,
            column_config={"Spotify": st.column_config.LinkColumn("Spotify")},
        )
        chart = recommendations.set_index("track_name")[["energy", "valence"]]
        st.caption("Recommended tracks in valence–energy mood space")
        st.scatter_chart(chart, x="valence", y="energy")

with tab_evaluate:
    st.subheader("Cold-track retrieval benchmark")
    if EVALUATION_PATH.exists():
        evaluation = pd.DataFrame(json.loads(EVALUATION_PATH.read_text(encoding="utf-8")))
        numeric = ["precision", "recall", "ndcg", "catalog_coverage", "artist_diversity"]
        st.dataframe(evaluation.style.format({column: "{:.4f}" for column in numeric}), hide_index=True)
        st.markdown("#### Ranking quality")
        st.bar_chart(evaluation.set_index("model")[["precision", "ndcg"]])
        st.markdown("#### Reach and variety")
        st.bar_chart(evaluation.set_index("model")[["catalog_coverage", "artist_diversity"]])
    else:
        st.info("Run `python scripts/evaluate.py` to generate benchmark results.")
    st.caption(
        "Query tracks are held out from the catalog. Genre is used only as a relevance label, "
        "never as an input feature. These metrics measure audio–genre coherence, not personal taste."
    )

with tab_method:
    st.markdown(
        """
        1. Audio fields are cleaned, robustly scaled, and weighted.
        2. Musical key is encoded cyclically so B is close to C.
        3. Seed tracks form a listening-profile vector.
        4. The selected mood contributes an interpretable target vector.
        5. Cosine similarity retrieves candidates; reranking reduces repetition.

        Popularity and genre are excluded from the similarity embedding. This allows popularity
        to remain an honest baseline and genre to remain an external evaluation label.
        """
    )
