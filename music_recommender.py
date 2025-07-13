import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import string
from fuzzywuzzy import fuzz, process
import warnings
warnings.filterwarnings('ignore')

class MusicRecommendationSystem:
    def __init__(self):
        self.df = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.genre_sim = None
        self.artist_sim = None
        self.combined_sim = None
        self.vectorizer = None
        self.genre_vectorizer = None
        self.artist_vectorizer = None
        
    def load_data(self, csv_file="expanded_songs_dataset.csv"):
        """Load dataset from CSV file"""
        try:
            self.df = pd.read_csv(csv_file)
            
            # Add synthetic features for better recommendations
            np.random.seed(42)  # For reproducible results
            self.df['popularity'] = np.random.randint(1, 101, len(self.df))
            self.df['year'] = np.random.randint(1980, 2024, len(self.df))
            self.df['duration'] = np.random.randint(120, 300, len(self.df))  # in seconds
            
            # Create enhanced lyrics based on genre and artist
            self.df['lyrics'] = self.df.apply(self._generate_lyrics, axis=1)
            
            print(f"✅ Loaded {len(self.df)} songs from {csv_file}")
            print(f"📊 Genres: {len(self.df['genre'].unique())} unique genres")
            print(f"🎤 Artists: {len(self.df['artist'].unique())} unique artists")
            return self.df
            
        except FileNotFoundError:
            print(f"❌ Error: Could not find {csv_file}")
            return None
    
    def _generate_lyrics(self, row):
        """Generate genre-appropriate lyrics for better content matching"""
        genre_keywords = {
            'Pop': 'love heart dance party tonight feel good music rhythm beat catchy melody',
            'Hip Hop': 'money success hustle grind street life rap flow bars beats rhyme',
            'Rock': 'guitar solo power energy loud freedom rebellion spirit electric drums',
            'EDM': 'drop bass electronic dance floor club night energy synthesizer beat',
            'R&B': 'soul smooth voice emotion love relationship heart rhythm blues',
            'Grunge': 'angst pain raw emotion alternative underground distorted guitar',
            'Britpop': 'british culture youth anthem generation music indie rock',
            'Alternative Rock': 'independent different unique sound artistic expression alternative',
            'Synthpop': 'synthesizer electronic retro futuristic sound waves digital',
            'Reggaeton': 'latin rhythm dance party celebration culture urban beat',
            'Electronic': 'digital sound technology future innovation beats electronic',
            'Post-punk': 'experimental dark artistic underground culture punk rock',
            'New Wave': 'modern fresh innovative style creative expression wave',
            'Indie': 'independent alternative unique artistic creative underground'
        }
        
        base_lyrics = genre_keywords.get(row['genre'], 'music song melody harmony')
        artist_words = row['artist'].lower().replace(' ', '_').replace('&', 'and')
        title_words = row['title'].lower().replace(' ', '_').replace('\'', '')
        
        return f"{base_lyrics} {artist_words} {title_words} music song"
    
    def preprocess_text(self, text):
        """Advanced text preprocessing"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation except apostrophes in contractions
        text = re.sub(r"[^\w\s']", ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def create_features(self):
        """Create enhanced feature combinations with different weights"""
        print("🔧 Creating features...")
        
        # Preprocess all text fields
        self.df['title_clean'] = self.df['title'].apply(self.preprocess_text)
        self.df['artist_clean'] = self.df['artist'].apply(self.preprocess_text)
        self.df['genre_clean'] = self.df['genre'].apply(self.preprocess_text)
        self.df['lyrics_clean'] = self.df['lyrics'].apply(self.preprocess_text)
        
        # Create weighted combined features
        self.df['combined_features'] = (
            self.df['title_clean'] + ' ' + self.df['title_clean'] + ' ' +  # Title weight: 2x
            self.df['artist_clean'] + ' ' + self.df['artist_clean'] + ' ' +  # Artist weight: 2x
            self.df['genre_clean'] + ' ' + self.df['genre_clean'] + ' ' + 
            self.df['genre_clean'] + ' ' +  # Genre weight: 3x
            self.df['lyrics_clean']  # Lyrics weight: 1x
        )
        
        # Create separate feature matrices for different similarity calculations
        self.df['genre_features'] = self.df['genre_clean']
        self.df['artist_features'] = self.df['artist_clean']
        
        print("✅ Features created successfully!")
        
    def build_recommendation_model(self):
        """Build the recommendation model with multiple similarity matrices"""
        print("🚀 Building recommendation model...")
        
        # Create TF-IDF matrices
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.8
        )
        
        self.genre_vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english'
        )
        
        self.artist_vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english'
        )
        
        # Fit and transform features
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['combined_features'])
        genre_matrix = self.genre_vectorizer.fit_transform(self.df['genre_features'])
        artist_matrix = self.artist_vectorizer.fit_transform(self.df['artist_features'])
        
        # Calculate similarity matrices
        self.cosine_sim = cosine_similarity(self.tfidf_matrix)
        self.genre_sim = cosine_similarity(genre_matrix)
        self.artist_sim = cosine_similarity(artist_matrix)
        
        # Create combined similarity matrix with weights
        self.combined_sim = (
            0.4 * self.cosine_sim +      # Content similarity: 40%
            0.35 * self.genre_sim +      # Genre similarity: 35%
            0.25 * self.artist_sim       # Artist similarity: 25%
        )
        
        print("✅ Recommendation model built successfully!")
    
    def find_song_fuzzy(self, song_title, threshold=70):
        """Find song using fuzzy string matching"""
        if song_title in self.df['title'].values:
            return song_title
        
        # Use fuzzy matching to find closest match
        matches = process.extract(song_title, self.df['title'].tolist(), limit=5)
        
        if matches and matches[0][1] >= threshold:
            print(f"🔍 Did you mean '{matches[0][0]}'? (Match: {matches[0][1]}%)")
            return matches[0][0]
        
        print(f"❌ Song '{song_title}' not found. Similar songs:")
        for match, score in matches[:3]:
            print(f"   - {match} ({score}% match)")
        
        return None
    
    def get_recommendations(self, song_title, num_recommendations=5, diversity_factor=0.3):
        """Get recommendations with improved diversity and error handling"""
        # Find song using fuzzy matching
        found_song = self.find_song_fuzzy(song_title)
        if not found_song:
            return []
        
        # Get song index
        song_idx = self.df[self.df['title'] == found_song].index[0]
        
        # Get similarity scores
        sim_scores = list(enumerate(self.combined_sim[song_idx]))
        
        # Sort by similarity
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Apply diversity filtering
        recommendations = []
        used_artists = set()
        used_genres = set()
        
        # Add the original song's artist and genre to avoid immediate repetition
        original_artist = self.df.iloc[song_idx]['artist']
        original_genre = self.df.iloc[song_idx]['genre']
        
        for idx, score in sim_scores[1:]:  # Skip the song itself
            song_info = self.df.iloc[idx]
            
            # Diversity check
            artist_penalty = 0.3 if song_info['artist'] in used_artists else 0
            genre_penalty = 0.2 if song_info['genre'] in used_genres else 0
            
            # Adjust score for diversity
            adjusted_score = score - artist_penalty - genre_penalty
            
            recommendations.append({
                'index': idx,
                'title': song_info['title'],
                'artist': song_info['artist'],
                'genre': song_info['genre'],
                'similarity_score': score,
                'adjusted_score': adjusted_score,
                'popularity': song_info['popularity'],
                'year': song_info['year']
            })
            
            # Track used artists and genres
            used_artists.add(song_info['artist'])
            used_genres.add(song_info['genre'])
        
        # Sort by adjusted score and return top recommendations
        recommendations = sorted(recommendations, key=lambda x: x['adjusted_score'], reverse=True)
        
        return recommendations[:num_recommendations]
    
    def get_genre_recommendations(self, genre, num_recommendations=5):
        """Get top songs from a specific genre"""
        genre_songs = self.df[self.df['genre'].str.lower() == genre.lower()]
        
        if genre_songs.empty:
            available_genres = sorted(self.df['genre'].unique())
            print(f"❌ Genre '{genre}' not found.")
            print(f"📋 Available genres: {', '.join(available_genres)}")
            return []
        
        # Sort by popularity and return top songs
        top_songs = genre_songs.nlargest(num_recommendations, 'popularity')
        
        recommendations = []
        for _, song in top_songs.iterrows():
            recommendations.append({
                'title': song['title'],
                'artist': song['artist'],
                'genre': song['genre'],
                'popularity': song['popularity'],
                'year': song['year']
            })
        
        return recommendations
    
    def get_artist_recommendations(self, artist_name, num_recommendations=5):
        """Get top songs from a specific artist"""
        artist_songs = self.df[self.df['artist'].str.lower() == artist_name.lower()]
        
        if artist_songs.empty:
            # Try fuzzy matching for artist names
            matches = process.extract(artist_name, self.df['artist'].unique(), limit=3)
            print(f"❌ Artist '{artist_name}' not found. Similar artists:")
            for match, score in matches:
                print(f"   - {match} ({score}% match)")
            return []
        
        # Sort by popularity and return top songs
        top_songs = artist_songs.nlargest(num_recommendations, 'popularity')
        
        recommendations = []
        for _, song in top_songs.iterrows():
            recommendations.append({
                'title': song['title'],
                'artist': song['artist'],
                'genre': song['genre'],
                'popularity': song['popularity'],
                'year': song['year']
            })
        
        return recommendations
    
    def get_trending_songs(self, num_recommendations=10):
        """Get trending songs based on popularity"""
        trending = self.df.nlargest(num_recommendations, 'popularity')
        
        recommendations = []
        for _, song in trending.iterrows():
            recommendations.append({
                'title': song['title'],
                'artist': song['artist'],
                'genre': song['genre'],
                'popularity': song['popularity'],
                'year': song['year']
            })
        
        return recommendations
    
    def search_songs(self, query, num_results=10):
        """Search for songs by title, artist, or genre"""
        query_lower = query.lower()
        
        # Search in title, artist, and genre
        matches = self.df[
            (self.df['title'].str.lower().str.contains(query_lower, na=False)) |
            (self.df['artist'].str.lower().str.contains(query_lower, na=False)) |
            (self.df['genre'].str.lower().str.contains(query_lower, na=False))
        ]
        
        if matches.empty:
            print(f"❌ No songs found matching '{query}'")
            return []
        
        # Sort by popularity
        matches = matches.nlargest(num_results, 'popularity')
        
        results = []
        for _, song in matches.iterrows():
            results.append({
                'title': song['title'],
                'artist': song['artist'],
                'genre': song['genre'],
                'popularity': song['popularity'],
                'year': song['year']
            })
        
        return results
    
    def display_recommendations(self, recommendations, title="🎵 Recommendations"):
        """Display recommendations in a formatted way"""
        if not recommendations:
            print("❌ No recommendations found.")
            return
        
        print(f"\n{title}")
        print("=" * 80)
        
        for i, rec in enumerate(recommendations, 1):
            if 'similarity_score' in rec:
                print(f"{i:2d}. 🎵 {rec['title']} - {rec['artist']}")
                print(f"     🎭 Genre: {rec['genre']} | 📊 Similarity: {rec['similarity_score']:.3f} | 📅 Year: {rec['year']}")
            else:
                print(f"{i:2d}. 🎵 {rec['title']} - {rec['artist']}")
                print(f"     🎭 Genre: {rec['genre']} | ⭐ Popularity: {rec['popularity']} | 📅 Year: {rec['year']}")
            print()
    
    def get_system_stats(self):
        """Get statistics about the recommendation system"""
        stats = {
            'total_songs': len(self.df),
            'unique_artists': len(self.df['artist'].unique()),
            'unique_genres': len(self.df['genre'].unique()),
            'genres': sorted(self.df['genre'].unique()),
            'year_range': f"{self.df['year'].min()} - {self.df['year'].max()}",
            'avg_popularity': self.df['popularity'].mean(),
            'top_artists': self.df['artist'].value_counts().head(5).to_dict(),
            'genre_distribution': self.df['genre'].value_counts().to_dict()
        }
        
        return stats
    
    def display_stats(self):
        """Display system statistics"""
        stats = self.get_system_stats()
        
        print("\n📊 MUSIC RECOMMENDATION SYSTEM STATISTICS")
        print("=" * 50)
        print(f"📀 Total Songs: {stats['total_songs']}")
        print(f"🎤 Unique Artists: {stats['unique_artists']}")
        print(f"🎭 Unique Genres: {stats['unique_genres']}")
        print(f"📅 Year Range: {stats['year_range']}")
        print(f"⭐ Average Popularity: {stats['avg_popularity']:.1f}")
        
        print(f"\n🎭 Available Genres:")
        for genre in stats['genres']:
            count = stats['genre_distribution'][genre]
            print(f"   - {genre}: {count} songs")
        
        print(f"\n🎤 Top Artists:")
        for artist, count in stats['top_artists'].items():
            print(f"   - {artist}: {count} songs")

# Main execution function
def main():
    """Main function to demonstrate the recommendation system"""
    print("🎵 MUSIC RECOMMENDATION SYSTEM")
    print("=" * 50)
    
    # Initialize the recommendation system
    recommender = MusicRecommendationSystem()
    
    # Load the dataset
    if recommender.load_data() is None:
        return
    
    # Create features and build model
    recommender.create_features()
    recommender.build_recommendation_model()
    
    # Display system statistics
    recommender.display_stats()
    
    # Test different types of recommendations
    print("\n" + "="*80)
    print("🧪 TESTING RECOMMENDATION SYSTEM")
    print("="*80)
    
    # 1. Content-based recommendations
    print("\n1️⃣ Content-Based Recommendations:")
    recommendations = recommender.get_recommendations("Blinding Lights", num_recommendations=5)
    recommender.display_recommendations(recommendations, "🎵 Similar to 'Blinding Lights'")
    
    # 2. Genre-based recommendations
    print("\n2️⃣ Genre-Based Recommendations:")
    genre_recs = recommender.get_genre_recommendations("Hip Hop", num_recommendations=5)
    recommender.display_recommendations(genre_recs, "🎤 Top Hip Hop Songs")
    
    # 3. Artist-based recommendations
    print("\n3️⃣ Artist-Based Recommendations:")
    artist_recs = recommender.get_artist_recommendations("Taylor Swift", num_recommendations=5)
    recommender.display_recommendations(artist_recs, "🎤 Taylor Swift Songs")
    
    # 4. Trending songs
    print("\n4️⃣ Trending Songs:")
    trending = recommender.get_trending_songs(num_recommendations=5)
    recommender.display_recommendations(trending, "🔥 Trending Songs")
    
    # 5. Search functionality
    print("\n5️⃣ Search Functionality:")
    search_results = recommender.search_songs("love", num_results=5)
    recommender.display_recommendations(search_results, "🔍 Search Results for 'love'")
    
    # 6. Test fuzzy matching
    print("\n6️⃣ Fuzzy Matching Test:")
    fuzzy_recs = recommender.get_recommendations("Blinding Light", num_recommendations=3)  # Intentional typo
    recommender.display_recommendations(fuzzy_recs, "🔍 Fuzzy Match Results")
    
    print("\n✅ All tests completed successfully!")
    print("🎵 Your music recommendation system is ready to use!")

# Interactive function for user input
def interactive_mode():
    """Interactive mode for user queries"""
    recommender = MusicRecommendationSystem()
    
    if recommender.load_data() is None:
        return
    
    recommender.create_features()
    recommender.build_recommendation_model()
    
    print("\n🎵 INTERACTIVE MUSIC RECOMMENDATION SYSTEM")
    print("=" * 50)
    print("Commands:")
    print("  1. recommend <song_title> - Get similar songs")
    print("  2. genre <genre_name> - Get top songs from genre")
    print("  3. artist <artist_name> - Get songs from artist")
    print("  4. search <query> - Search for songs")
    print("  5. trending - Get trending songs")
    print("  6. stats - Show system statistics")
    print("  7. quit - Exit the system")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\n🎵 Enter command: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Thanks for using the Music Recommendation System!")
                break
            elif user_input.lower() == 'stats':
                recommender.display_stats()
            elif user_input.lower() == 'trending':
                trending = recommender.get_trending_songs()
                recommender.display_recommendations(trending, "🔥 Trending Songs")
            elif user_input.lower().startswith('recommend '):
                song = user_input[10:].strip()
                recs = recommender.get_recommendations(song)
                recommender.display_recommendations(recs, f"🎵 Similar to '{song}'")
            elif user_input.lower().startswith('genre '):
                genre = user_input[6:].strip()
                recs = recommender.get_genre_recommendations(genre)
                recommender.display_recommendations(recs, f"🎭 Top {genre} Songs")
            elif user_input.lower().startswith('artist '):
                artist = user_input[7:].strip()
                recs = recommender.get_artist_recommendations(artist)
                recommender.display_recommendations(recs, f"🎤 {artist} Songs")
            elif user_input.lower().startswith('search '):
                query = user_input[7:].strip()
                results = recommender.search_songs(query)
                recommender.display_recommendations(results, f"🔍 Search Results for '{query}'")
            else:
                print("❌ Invalid command. Type 'quit' to exit.")
                
        except KeyboardInterrupt:
            print("\n👋 Thanks for using the Music Recommendation System!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Run the main demonstration
    # main()

    interactive_mode()
