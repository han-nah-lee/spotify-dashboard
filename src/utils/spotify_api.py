import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# Load environment variables
load_dotenv()

class SpotifyAPI:
    def __init__(self, auth_code=None):
        """Initialize the Spotify API with credentials from .env file
        
        Args:
            auth_code (str, optional): Authorization code from Spotify callback
        """
        # Get credentials from environment variables
        # Or read from our .streamlit/secrets.toml file
        # This is a fallback if the environment variables are not set
        # Reading from .streamlit/secrets.toml which is used in Streamlit sharing
        # Read from toml file if available
        if os.path.exists('.streamlit/secrets.toml'):
            import toml
            secrets = toml.load('.streamlit/secrets.toml')
            self.client_secret = secrets['spotify']['client_secret']
            self.client_id = secrets['spotify']['client_id']
            self.redirect_uri = secrets['spotify']['redirect_uri']
        else:
            # Fallback to environment variables
            self.client_id = os.getenv('client_id')
            self.client_secret = os.getenv('client_secret')
            self.redirect_uri = os.getenv('redirect_uri') or "http://127.0.0.1:8501/callback"
        
        # Use scopes from environment or default to these essential ones
        env_scope = os.getenv('scope')
        self.scope = env_scope if env_scope else "user-read-recently-played user-top-read user-library-read user-read-playback-state user-modify-playback-state user-read-currently-playing"
        
        # Use a more reliable cache path in the user's data directory
        cache_path = os.path.join(os.path.expanduser("~"), ".spotify_cache")
        
        # Create auth manager with authorization code if provided
        self.auth_manager = SpotifyOAuth(
            client_id=self.client_id,
            client_secret=self.client_secret,
            redirect_uri=self.redirect_uri,
            scope=self.scope,
            open_browser=True,
            cache_path=cache_path
        )
        
        try:
            # If an auth code is provided, use it to get access token
            if auth_code:
                token_info = self.auth_manager.get_access_token(auth_code, as_dict=True)
                print(f"Successfully obtained token with auth code")
                
            # Initialize the Spotify client with our auth manager
            self.sp = spotipy.Spotify(auth_manager=self.auth_manager)
            profile = self.sp.current_user()
            self.user_id = profile['id']
            self.user_name = profile['display_name']
            print(f"Authenticated as {self.user_name} ({self.user_id})")
            print("Spotify API initialized successfully.")
        except Exception as e:
            print(f"Authentication error: {str(e)}")
            raise Exception(f"Failed to authenticate with Spotify: {str(e)}")
    
    def get_recently_played(self, limit=50):
        """Get user's recently played tracks (single request, max 50)
        
        Args:
            limit (int): Number of tracks to retrieve (max 50)
            
        Returns:
            list: List of Track objects
        """
        results = self.sp.current_user_recently_played(limit=min(50, limit))
        tracks = []
        
        for item in results['items']:
            track = Track(
                id=item['track']['id'],
                name=item['track']['name'],
                artist=[artist['name'] for artist in item['track']['artists']],
                album=item['track']['album']['name'],
                duration_ms=item['track']['duration_ms'],
                popularity=item['track']['popularity'],
                played_at=item['played_at']
            )
            tracks.append(track)
        
        return tracks
        
    def get_recently_played_extended(self, limit=1000):
        """Get user's recently played tracks with multiple API requests to bypass the 50 limit
        
        Args:
            limit (int): Total number of tracks to retrieve (will make multiple API calls if > 50)
            
        Returns:
            list: List of Track objects
        """
        all_tracks = []
        api_limit = 50  # Spotify API limit per request
        
        # Calculate how many API calls we need to make
        num_requests = min(20, (limit + api_limit - 1) // api_limit)  # Ceiling division, max 20 requests (1000 tracks)
        
        print(f"Fetching up to {num_requests * api_limit} recently played tracks (making {num_requests} API requests)")
        
        # First batch doesn't need an after parameter
        history = []

        try:
            results = self.sp.current_user_recently_played(limit=api_limit)
            history.append(results)
            # Process results
            for item in results['items']:
                track = Track(
                    id=item['track']['id'],
                    name=item['track']['name'],
                    artist=[artist['name'] for artist in item['track']['artists']],
                    album=item['track']['album']['name'],
                    duration_ms=item['track']['duration_ms'],
                    popularity=item['track']['popularity'],
                    played_at=item['played_at']
                )
                all_tracks.append(track)
                
            print(f"Fetched batch 1/{num_requests}: {len(results['items'])} tracks")
            # For subsequent batches, we need the cursor/timestamp of the last item
            for i in range(1, num_requests):
                # If we have enough tracks or no more results, stop
                if len(all_tracks) >= limit or len(results['items']) < api_limit:
                    break
                    
                # Get the timestamp from the last item for pagination
                if 'cursors' in results and results['cursors'] and 'before' in results['cursors']:
                    before_timestamp = results['cursors']['before']
                    
                    # Make the next API call with the cursor
                    results = self.sp.current_user_recently_played(limit=api_limit)
                    # Since pagination isn't working, lets make a request directly to the endpoint
                    # Check if we have results  
                    history.append(results)                    
                    # If no more results, break the loop
                    if not results['items']:
                        print(f"No more results after {len(all_tracks)} tracks")
                        break
                        
                    # Process results
                    for item in results['items']:
                        track = Track(
                            id=item['track']['id'],
                            name=item['track']['name'],
                            artist=[artist['name'] for artist in item['track']['artists']],
                            album=item['track']['album']['name'],
                            duration_ms=item['track']['duration_ms'],
                            popularity=item['track']['popularity'],
                            played_at=item['played_at']
                        )
                        all_tracks.append(track)
                    
                    print(f"Fetched batch {i+1}/{num_requests}: {len(results['items'])} tracks (total: {len(all_tracks)})")
                else:
                    # If there's no cursor, we've reached the end
                    print("No more pagination cursor available")
                    break
                    
        except Exception as e:
            print(f"Error fetching recently played tracks: {str(e)}")
        
        print(f"Total recently played tracks fetched: {len(all_tracks)}")
        return all_tracks[:limit]  # Ensure we don't exceed the requested limit
    
    def get_top_tracks(self, time_range='medium_term', limit=50):
        """Get user's top tracks (single request, max 50)
        
        Args:
            time_range (str): 'short_term' (4 weeks), 'medium_term' (6 months), 'long_term' (years)
            limit (int): Number of tracks to retrieve (max 50)
            
        Returns:
            list: List of Track objects
        """
        results = self.sp.current_user_top_tracks(time_range=time_range, limit=min(50, limit))
        tracks = []
        
        for item in results['items']:
            track = Track(
                id=item['id'],
                name=item['name'],
                artist=[artist['name'] for artist in item['artists']],
                album=item['album']['name'],
                duration_ms=item['duration_ms'],
                popularity=item['popularity'],
                played_at=None  # Top tracks don't have played_at time
            )
            tracks.append(track)
        
        return tracks
        
    def get_top_tracks_extended(self, time_range='medium_term', limit=10000):
        """Get user's top tracks with multiple API requests to bypass the 50 limit
        
        Args:
            time_range (str): 'short_term' (4 weeks), 'medium_term' (6 months), 'long_term' (years)
            limit (int): Total number of tracks to retrieve (will make multiple API calls if > 50)
            
        Returns:
            list: List of Track objects
        """
        all_tracks = []
        api_limit = 50  # Spotify API limit per request
        
        # Calculate how many API calls we need to make
        num_requests = min(20, (limit + api_limit - 1) // api_limit)  # Ceiling division, max 20 requests (1000 tracks)
        
        print(f"Fetching up to {num_requests * api_limit} top tracks (making {num_requests} API requests)")
        
        # Make multiple API calls with different offsets
        for i in range(num_requests):
            offset = i * api_limit
            try:
                results = self.sp.current_user_top_tracks(
                    time_range=time_range, 
                    limit=api_limit,
                    offset=offset
                )
                
                # If no more results, break the loop
                if not results['items']:
                    print(f"No more results after {len(all_tracks)} tracks")
                    break
                    
                # Process results
                for item in results['items']:
                    track = Track(
                        id=item['id'],
                        name=item['name'],
                        artist=[artist['name'] for artist in item['artists']],
                        album=item['album']['name'],
                        duration_ms=item['duration_ms'],
                        popularity=item['popularity'],
                        played_at=None  # Top tracks don't have played_at time
                    )
                    all_tracks.append(track)
                
                print(f"Fetched batch {i+1}/{num_requests}: {len(results['items'])} tracks (total: {len(all_tracks)})")
                
            except Exception as e:
                print(f"Error fetching batch {i+1}: {str(e)}")
                break
        
        print(f"Total top tracks fetched: {len(all_tracks)}")
        return all_tracks[:limit]  # Ensure we don't exceed the requested limit
    
    def get_saved_tracks(self, limit=50):
        """Get user's saved tracks (single request)
        
        Args:
            limit (int): Number of tracks to retrieve
            
        Returns:
            list: List of Track objects
        """
        results = self.sp.current_user_saved_tracks(limit=min(50, limit))
        tracks = []
        
        for item in results['items']:
            track = Track(
                id=item['track']['id'],
                name=item['track']['name'],
                artist=[artist['name'] for artist in item['track']['artists']],
                album=item['track']['album']['name'],
                duration_ms=item['track']['duration_ms'],
                popularity=item['track']['popularity'],
                played_at=None,  # Saved tracks don't have played_at time
                added_at=item['added_at']
            )
            tracks.append(track)
        
        return tracks
        
    def get_saved_tracks_extended(self, limit=1000):
        """Get user's saved tracks with multiple API requests to bypass the 50 limit
        
        Args:
            limit (int): Total number of tracks to retrieve (will make multiple API calls if > 50)
            
        Returns:
            list: List of Track objects
        """
        all_tracks = []
        api_limit = 50  # Spotify API limit per request
        
        # Calculate how many API calls we need to make
        num_requests = min(20, (limit + api_limit - 1) // api_limit)  # Ceiling division, max 20 requests (1000 tracks)
        
        print(f"Fetching up to {num_requests * api_limit} saved tracks (making {num_requests} API requests)")
        
        # Make multiple API calls with different offsets
        for i in range(num_requests):
            offset = i * api_limit
            try:
                results = self.sp.current_user_saved_tracks(
                    limit=api_limit,
                    offset=offset
                )
                
                # If no more results, break the loop
                if not results['items']:
                    print(f"No more results after {len(all_tracks)} tracks")
                    break
                    
                # Process results
                for item in results['items']:
                    track = Track(
                        id=item['track']['id'],
                        name=item['track']['name'],
                        artist=[artist['name'] for artist in item['track']['artists']],
                        album=item['track']['album']['name'],
                        duration_ms=item['track']['duration_ms'],
                        popularity=item['track']['popularity'],
                        played_at=None,  # Saved tracks don't have played_at time
                        added_at=item['added_at']
                    )
                    all_tracks.append(track)
                
                print(f"Fetched batch {i+1}/{num_requests}: {len(results['items'])} saved tracks (total: {len(all_tracks)})")
                
            except Exception as e:
                print(f"Error fetching batch {i+1}: {str(e)}")
                break
        
        print(f"Total saved tracks fetched: {len(all_tracks)}")
        return all_tracks[:limit]  # Ensure we don't exceed the requested limit
    
    def get_auth_url(self):
        """Generate the Spotify authorization URL."""
        return self.auth_manager.get_authorize_url()

    def get_token_from_code(self, code):
        """Exchange the authorization code for an access token.

        Args:
            code (str): Authorization code from Spotify callback.

        Returns:
            dict: Token information including access and refresh tokens.
        """
        return self.auth_manager.get_access_token(code, as_dict=True)

    def refresh_token(self):
        """Refresh the access token if expired."""
        return self.auth_manager.refresh_access_token(self.auth_manager.cache_handler.get_cached_token()['refresh_token'])


class Track:
    def __init__(self, id, name, artist, album, duration_ms, popularity, played_at=None, added_at=None):
        """Initialize a Track object
        
        Args:
            id (str): Spotify track ID
            name (str): Track name
            artist (list): List of artist names
            album (str): Album name
            duration_ms (int): Track duration in milliseconds
            popularity (int): Track popularity (0-100)
            played_at (str, optional): ISO timestamp when track was played
            added_at (str, optional): ISO timestamp when track was added to library
        """
        self.id = id
        self.name = name
        self.artist = artist
        self.album = album
        self.duration_ms = duration_ms
        self.popularity = popularity
        
        # Convert timestamps to datetime objects if they exist
        if played_at:
            self.played_at = datetime.fromisoformat(played_at.replace('Z', '+00:00'))
        else:
            self.played_at = None
        
        if added_at:
            self.added_at = datetime.fromisoformat(added_at.replace('Z', '+00:00'))
        else:
            self.added_at = None
    
    def __str__(self):
        return f"{self.name} by {', '.join(self.artist)}"
    
    def to_dict(self):
        """Convert track to dictionary for DataFrame creation"""
        return {
            'id': self.id,
            'name': self.name,
            'artist': self.artist[0] if self.artist else None,  # Primary artist
            'all_artists': ', '.join(self.artist) if self.artist else None,  # All artists
            'album': self.album,
            'duration_ms': self.duration_ms,
            'duration_min': round(self.duration_ms / 60000, 2),  # Convert to minutes
            'popularity': self.popularity,
            'played_at': self.played_at,
            'added_at': self.added_at
        }


class SpotifyAnalysis:
    def __init__(self, spotify_api):
        """Initialize SpotifyAnalysis with SpotifyAPI instance
        
        Args:
            spotify_api (SpotifyAPI): Instance of SpotifyAPI
        """
        self.spotify_api = spotify_api
    
    def get_listening_history_df(self, limit=50):
        """Get user's listening history as DataFrame
        
        Args:
            limit (int): Number of tracks to retrieve
            
        Returns:
            pandas.DataFrame: DataFrame of recent tracks
        """
        # Use extended method if limit is > 50
        if limit > 50:
            tracks = self.spotify_api.get_recently_played_extended(limit=limit)
            df = self._tracks_to_dataframe(tracks)
            
            # Save the dataframe to CSV
            if not df.empty:
                self.save_to_csv(df, "listening_history.csv")
        else:
            tracks = self.spotify_api.get_recently_played(limit=limit)
            df = self._tracks_to_dataframe(tracks)
            
        return df
    
    def get_top_tracks_df(self, time_range='medium_term', limit=10000):
        """Get user's top tracks as DataFrame
        
        Args:
            time_range (str): 'short_term', 'medium_term', or 'long_term'
            limit (int): Number of tracks to retrieve
            
        Returns:
            pandas.DataFrame: DataFrame of top tracks
        """
        # Use the extended method for retrieving more than 50 tracks
        tracks = self.spotify_api.get_top_tracks_extended(time_range=time_range, limit=limit)
        df = self._tracks_to_dataframe(tracks)
        
        # Save the dataframe to CSV
        if not df.empty:
            self.save_to_csv(df, f"top_tracks_{time_range}.csv")
            
        return df
    
    def get_saved_tracks_df(self, limit=50):
        """Get user's saved tracks as DataFrame
        
        Args:
            limit (int): Number of tracks to retrieve
            
        Returns:
            pandas.DataFrame: DataFrame of saved tracks
        """
        # Use extended method if limit is > 50
        if limit > 50:
            tracks = self.spotify_api.get_saved_tracks_extended(limit=limit)
            df = self._tracks_to_dataframe(tracks)
            
            # Save the dataframe to CSV
            if not df.empty:
                self.save_to_csv(df, "saved_tracks.csv")
        else:
            tracks = self.spotify_api.get_saved_tracks(limit=limit)
            df = self._tracks_to_dataframe(tracks)
            
        return df
    
    def _tracks_to_dataframe(self, tracks):
        """Convert list of Track objects to DataFrame
        
        Args:
            tracks (list): List of Track objects
            
        Returns:
            pandas.DataFrame: DataFrame of tracks
        """
        track_dicts = [track.to_dict() for track in tracks]
        df = pd.DataFrame(track_dicts)
        return df
    
    def analyze_listening_patterns(self, df):
        """Analyze listening patterns from DataFrame
        
        Args:
            df (pandas.DataFrame): DataFrame of tracks
            
        Returns:
            dict: Dictionary with analysis results
        """
        results = {}
        
        # Most listened artists
        if 'artist' in df.columns and df['artist'].notna().any():
            results['top_artists'] = df['artist'].value_counts().head(10).to_dict()
        
        # Most listened albums
        if 'album' in df.columns and df['album'].notna().any():
            results['top_albums'] = df['album'].value_counts().head(10).to_dict()
        
        # Time of day patterns (for recently played tracks)
        if 'played_at' in df.columns and df['played_at'].notna().any():
            df['hour_of_day'] = df['played_at'].dt.hour
            results['hour_distribution'] = df['hour_of_day'].value_counts().sort_index().to_dict()
            
            # Day of week patterns
            df['day_of_week'] = df['played_at'].dt.day_name()
            results['day_distribution'] = df['day_of_week'].value_counts().to_dict()
        
        # Popularity stats
        if 'popularity' in df.columns and df['popularity'].notna().any():
            results['avg_popularity'] = df['popularity'].mean()
            results['popularity_distribution'] = df['popularity'].value_counts().sort_index().to_dict()
            
        # Track listening time by artist
        if 'artist' in df.columns and 'duration_ms' in df.columns:
            artist_time = df.groupby('artist')['duration_ms'].sum().sort_values(ascending=False)
            # Convert to minutes for better readability
            artist_time_minutes = (artist_time / 60000).round(2)
            results['artist_listening_time'] = artist_time_minutes.head(10).to_dict()
            results['total_listening_time_minutes'] = (df['duration_ms'].sum() / 60000).round(2)
            
        # Track listening time by song and play count
        if 'name' in df.columns and 'artist' in df.columns and 'duration_ms' in df.columns:
            # Create a combined song identifier (name + artist)
            df['song_id'] = df['name'] + ' - ' + df['artist']
            
            # Count plays per song
            song_plays = df['song_id'].value_counts().head(20)
            results['song_play_count'] = song_plays.to_dict()
            
            # Calculate listening time per song
            song_time = df.groupby('song_id')['duration_ms'].sum().sort_values(ascending=False)
            song_time_minutes = (song_time / 60000).round(2)
            results['song_listening_time'] = song_time_minutes.head(20).to_dict()
            
            # Find frequently repeated songs (more than 3 plays)
            repeated_songs = song_plays[song_plays >= 3]
            results['repeated_songs'] = repeated_songs.to_dict() if not repeated_songs.empty else {}
        
        return results
        
    def save_to_csv(self, df, filename):
        """Save DataFrame to CSV file
        
        Args:
            df (pandas.DataFrame): DataFrame to save
            filename (str): Filename to save to (will be saved in 'data' directory)
            
        Returns:
            str: Full path to the saved file
        """
        # Create data directory if it doesn't exist
        import os
        # Get root directory of the project
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        # Save to CSV
        filepath = os.path.join(data_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Saved data to {filepath}")
        return filepath


if __name__ == "__main__":
    spotify_api = SpotifyAPI()
    spotify_analysis = SpotifyAnalysis(spotify_api)
    # Call the get_listening_history_df method to get the DataFrame
    df = spotify_analysis.get_listening_history_df(limit=100)
    # Print the DataFrame
    print(df.head())
    # Call the analyze_listening_patterns method to get the analysis results
    analysis_results = spotify_analysis.analyze_listening_patterns(df)
