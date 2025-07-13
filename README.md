# Music Recommendation System

A comprehensive content-based music recommendation system built with Python that provides personalized song suggestions using advanced machine learning techniques.

## 🎵 Features

- **Content-Based Filtering**: Recommends songs based on title, artist, genre, and lyrical content
- **Multiple Recommendation Types**: 
  - Similar songs based on content
  - Genre-based recommendations
  - Artist-based suggestions
  - Trending songs
  - Search functionality
- **Fuzzy String Matching**: Handles typos and partial song name matches
- **Diversity Filtering**: Prevents repetitive recommendations from the same artist/genre
- **Interactive Mode**: Command-line interface for real-time queries
- **Rich Dataset**: 255 songs across 15+ genres from 1980s to 2024

## 📊 Dataset Overview

- **Total Songs**: 255 tracks
- **Unique Artists**: 85+ artists
- **Genres**: Pop, Hip Hop, Rock, EDM, Alternative Rock, Grunge, Synthpop, R&B, Reggaeton, Britpop, and more
- **Era Coverage**: Songs spanning from 1980s to 2024
- **Features**: Title, Artist, Genre, plus synthetic popularity, year, and duration data

## 🚀 Quick Start

### Prerequisites
```
pip install -r requirements.txt
```

### Installation

1. **Clone or download the project files**
2. **Run the recommendation system**:
```
python music_recommender.py
```


### Project Structure

```
music-recommendation-system/
├── music_recommender.py # Main recommendation system
├── expanded_songs_dataset.csv # Generated dataset file
├── .env # Environment variables (optional)
└── README.md # This file
```


## 💻 Usage

### Demo Mode (Default)

Run the system to see all features demonstrated

This will show:
- System statistics
- Content-based recommendations
- Genre-based suggestions
- Artist-based recommendations
- Trending songs
- Search functionality
- Fuzzy matching examples

### Interactive Mode

Enable interactive mode by modifying the bottom of `music_recommender.py`:

```
if name == "main":
# main() # Comment this out
interactive_mode() # Uncomment this
```

Then run and use these commands:
- `recommend <song_title>` - Get similar songs
- `genre <genre_name>` - Get top songs from a genre
- `artist <artist_name>` - Get songs from an artist
- `search <query>` - Search for songs
- `trending` - Get trending songs
- `stats` - Show system statistics
- `quit` - Exit the system

### Programmatic Usage

```
from music_recommender import MusicRecommendationSystem

Initialize the system
recommender = MusicRecommendationSystem()
recommender.load_data("expanded_songs_dataset.csv")
recommender.create_features()
recommender.build_recommendation_model()

Get recommendations
recommendations = recommender.get_recommendations("Blinding Lights", num_recommendations=5)
recommender.display_recommendations(recommendations)

Get genre recommendations
hip_hop_songs = recommender.get_genre_recommendations("Hip Hop", num_recommendations=10)

Get artist recommendations
taylor_swift_songs = recommender.get_artist_recommendations("Taylor Swift", num_recommendations=5)

Search for songs
search_results = recommender.search_songs("love", num_results=10)
```

## 🔧 Technical Details

### Algorithm Components

1. **TF-IDF Vectorization**: Converts text features into numerical representations
2. **Cosine Similarity**: Measures similarity between songs
3. **Multi-layered Similarity**: Combines content, genre, and artist similarity
4. **Weighted Features**: Genre (3x), Title/Artist (2x), Lyrics (1x)
5. **Diversity Filtering**: Ensures varied recommendations

### Similarity Matrix Weights

- **Content Similarity**: 40%
- **Genre Similarity**: 35%
- **Artist Similarity**: 25%

### Key Classes and Methods

- `MusicRecommendationSystem`: Main recommendation engine
- `load_data()`: Load dataset from CSV
- `create_features()`: Process and weight text features
- `build_recommendation_model()`: Create similarity matrices
- `get_recommendations()`: Content-based recommendations
- `get_genre_recommendations()`: Genre-based suggestions
- `get_artist_recommendations()`: Artist-based recommendations
- `search_songs()`: Search functionality

## 📈 Example Output

```
🎵 Similar to 'Blinding Lights'
🎵 Starboy - The Weeknd
🎭 Genre: R&B | 📊 Similarity: 0.892 | 📅 Year: 2016

🎵 Can't Feel My Face - The Weeknd
🎭 Genre: R&B | 📊 Similarity: 0.847 | 📅 Year: 2015

🎵 Levitating - Dua Lipa
🎭 Genre: Pop | 📊 Similarity: 0.734 | 📅 Year: 2020

```

## 🚀 Deployment

### Environment Variables

Create a `.env` file for configuration (optional):

```
Dataset Configuration
DATASET_PATH=expanded_songs_dataset.csv
MAX_FEATURES=5000
SIMILARITY_THRESHOLD=0.7

Recommendation Settings
DEFAULT_NUM_RECOMMENDATIONS=5
DIVERSITY_FACTOR=0.3
FUZZY_MATCH_THRESHOLD=70
```

## 📋 Available Genres

- Pop
- Hip Hop
- Rock
- Alternative Rock
- EDM
- R&B
- Grunge
- Synthpop
- Britpop
- Reggaeton
- Electronic
- Post-punk
- New Wave
- Indie

## 🎤 Featured Artists

The dataset includes songs from popular artists across different eras:
- **Modern Pop**: The Weeknd, Harry Styles, Dua Lipa, Taylor Swift
- **Hip-Hop**: Drake, Post Malone, Travis Scott, Kendrick Lamar
- **Classic Rock**: Queen, Led Zeppelin, Guns N' Roses
- **Electronic**: Daft Punk, Avicii, Martin Garrix
- **80s/90s Icons**: Depeche Mode, The Cure, New Order


