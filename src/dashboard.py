#!/usr/bin/env python3
"""
Spotify: Spotify Listening Analytics Dashboard (Local CSV Version)
This Streamlit app visualizes a user's Spotify listening history from local CSV files.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import altair as alt
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import os
from utils.artist_analysis import get_top_artists_from_tracks

# Set page configuration
st.set_page_config(
    page_title="Spotify - Spotify Analytics",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1DB954;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1DB954;
        margin-top: 2rem;
    }
    .stat-box {
        background-color: #191414;
        padding: 1rem;
        border-radius: 5px;
        color: white;
        text-align: center;
    }
    .stat-number {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1DB954;
    }
    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for data persistence between reruns
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "history_df" not in st.session_state:
    st.session_state.history_df = None
if "top_df" not in st.session_state:
    st.session_state.top_df = None
if "saved_df" not in st.session_state:
    st.session_state.saved_df = None


def load_data_from_csv():
    """Load data from top tracks CSV file only"""
    with st.spinner("Loading data from top tracks CSV file..."):
        try:
            # Define file path - only using top_tracks_medium_term.csv
            base_path = os.path.dirname(os.path.abspath(__file__))
            top_tracks_path = os.path.join(base_path, "data", "top_tracks_medium_term.csv")
            
            # Load the data file
            top_df = pd.read_csv(top_tracks_path)
            
            # Set empty DataFrames for history and saved tracks since we're not using them
            history_df = pd.DataFrame()
            saved_df = pd.DataFrame()
            
            # Convert datetime columns if they exist
            if 'added_at' in top_df.columns and not top_df['added_at'].empty:
                top_df['added_at'] = pd.to_datetime(top_df['added_at'])
                
            if 'release_date' in top_df.columns and not top_df['release_date'].empty:
                try:
                    top_df['release_date'] = pd.to_datetime(top_df['release_date'])
                except:
                    pass
                    
            st.session_state.history_df = history_df
            st.session_state.top_df = top_df
            st.session_state.saved_df = saved_df
            st.session_state.data_loaded = True
            
            return history_df, top_df, saved_df
        
        except Exception as e:
            st.error(f"Error loading CSV data: {str(e)}")
            st.error("Make sure your CSV files are in the correct location and format.")
            return None, None, None


def plot_top_artists(df, n=10):
    """Plot top N artists with Altair"""
    if df.empty or 'artist' not in df.columns:
        return None
    
    top_artists = df['artist'].value_counts().head(n).reset_index()
    top_artists.columns = ['Artist', 'Count']
    
    chart = alt.Chart(top_artists).mark_bar().encode(
        x=alt.X('Count:Q', title='Number of Tracks'),
        y=alt.Y('Artist:N', sort='-x', title=None),
        color=alt.Color('Count:Q', scale=alt.Scale(scheme='greenblue'), legend=None),
        tooltip=['Artist', 'Count']
    ).properties(
        title=f'Top {n} Artists in Your Spotify History',
        height=n*30 + 50
    ).interactive()
    
    return chart


def plot_hourly_distribution(df):
    """Plot listening patterns by hour of day with Plotly"""
    if df.empty or 'played_at' not in df.columns or df['played_at'].isna().all():
        return None
    
    # Extract hour from played_at
    hours_df = pd.DataFrame({'hour': df['played_at'].dt.hour})
    hourly_counts = hours_df['hour'].value_counts().sort_index()
    
    # Fill missing hours with 0
    all_hours = pd.Series(range(24))
    hourly_counts = hourly_counts.reindex(all_hours, fill_value=0)
    hourly_counts = hourly_counts.reset_index()
    hourly_counts.columns = ['Hour', 'Count']
    
    # Add time labels
    hourly_counts['TimeLabel'] = hourly_counts['Hour'].apply(lambda x: f"{x}:00")
    
    # Create color gradient
    colors = px.colors.sequential.Viridis
    
    fig = px.bar(
        hourly_counts, 
        x='Hour', 
        y='Count',
        labels={'Count': 'Number of Tracks', 'Hour': 'Hour of Day'},
        hover_data={'TimeLabel': True, 'Hour': False},
        color='Count',
        color_continuous_scale=colors
    )
    
    fig.update_layout(
        title='Listening Pattern by Hour of Day',
        xaxis=dict(tickmode='array', tickvals=list(range(0, 24, 2))),
        plot_bgcolor='rgba(0,0,0,0)',
        height=400
    )
    
    return fig


def plot_popularity_distribution(df):
    """Plot track popularity distribution with Plotly"""
    if df.empty or 'popularity' not in df.columns or df['popularity'].isna().all():
        return None
    
    # Create bins for popularity
    bins = list(range(0, 101, 10))  # 0, 10, 20, ..., 100
    labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)]
    
    # Create a new column with binned popularity
    pop_binned = pd.cut(df['popularity'], bins=bins, labels=labels, right=False)
    pop_counts = pop_binned.value_counts().sort_index()
    
    # Convert to dataframe for plotting
    pop_df = pd.DataFrame({'Range': pop_counts.index, 'Count': pop_counts.values})
    
    fig = px.bar(
        pop_df,
        x='Range',
        y='Count',
        labels={'Count': 'Number of Tracks', 'Range': 'Popularity Range'},
        color='Count',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        title='Track Popularity Distribution',
        xaxis=dict(categoryorder='array', categoryarray=labels),
        plot_bgcolor='rgba(0,0,0,0)',
        height=400
    )
    
    return fig


def plot_artists_vs_popularity(df):
    """Plot artists vs. average popularity"""
    if df.empty or 'artist' not in df.columns or 'popularity' not in df.columns:
        return None
    
    # Group by artist and calculate mean popularity
    artist_pop = df.groupby('artist')['popularity'].mean().sort_values(ascending=False).head(15)
    artist_pop = artist_pop.reset_index()
    artist_pop.columns = ['Artist', 'Average Popularity']
    
    fig = px.bar(
        artist_pop,
        x='Average Popularity',
        y='Artist',
        orientation='h',
        labels={'Average Popularity': 'Average Popularity (0-100)'},
        color='Average Popularity',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        title='Top Artists by Average Popularity',
        yaxis=dict(categoryorder='total ascending'),
        plot_bgcolor='rgba(0,0,0,0)',
        height=500
    )
    
    return fig


def plot_top_artists_by_tracks(df, n=15):
    """Plot top N artists based on number of tracks in top tracks"""
    if df.empty or 'artist' not in df.columns:
        return None
    
    # Count tracks by artist
    top_artists = df['artist'].value_counts().head(n).reset_index()
    top_artists.columns = ['Artist', 'Track Count']
    
    # Calculate percentage of total
    total_tracks = len(df)
    top_artists['Percentage'] = (top_artists['Track Count'] / total_tracks * 100).round(1)
    
    # Add average popularity for each artist if available
    if 'popularity' in df.columns:
        artist_popularity = df.groupby('artist')['popularity'].mean().round().astype(int)
        top_artists['Avg Popularity'] = top_artists['Artist'].map(artist_popularity)
    
    # Sort by track count for better visualization
    top_artists = top_artists.sort_values('Track Count', ascending=True)
    
    # Create the visualization
    fig = px.bar(
        top_artists,
        x='Track Count',
        y='Artist',
        orientation='h',
        text='Percentage',
        hover_data=['Avg Popularity'] if 'Avg Popularity' in top_artists.columns else None,
        labels={'Track Count': 'Number of Tracks in Your Top Tracks', 'Artist': ''},
        title=f'Your Top {n} Artists (Based on Number of Top Tracks)',
        color='Avg Popularity' if 'Avg Popularity' in top_artists.columns else None,
        color_continuous_scale='Viridis'
    )
    
    # Style the figure
    fig.update_traces(texttemplate='%{text}%', textposition='outside')
    fig.update_layout(
        yaxis=dict(categoryorder='total ascending'),
        plot_bgcolor='rgba(0,0,0,0)',
        height=n*30 + 120
    )
    
    return fig


def plot_weekly_patterns(df):
    """Plot listening patterns by day of week"""
    if df.empty or 'played_at' not in df.columns or df['played_at'].isna().all():
        return None
    
    # Extract day of week
    days_df = pd.DataFrame({'day': df['played_at'].dt.day_name()})
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_counts = days_df['day'].value_counts().reindex(day_order).fillna(0)
    day_counts = day_counts.reset_index()
    day_counts.columns = ['Day', 'Count']
    
    fig = px.bar(
        day_counts,
        x='Day',
        y='Count',
        labels={'Count': 'Number of Tracks', 'Day': 'Day of Week'},
        color='Count',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        title='Listening Pattern by Day of Week',
        xaxis=dict(categoryorder='array', categoryarray=day_order),
        plot_bgcolor='rgba(0,0,0,0)',
        height=400
    )
    
    return fig


def show_track_table(df, title, max_rows=10):
    """Display a table of tracks"""
    if df.empty:
        return
    
    # Select relevant columns and rename them for display
    if 'played_at' in df.columns and df['played_at'].notna().any():
        display_df = df[['name', 'artist', 'album', 'popularity', 'played_at']].head(max_rows)
        display_df = display_df.rename(columns={
            'name': 'Track',
            'artist': 'Artist',
            'album': 'Album',
            'popularity': 'Popularity',
            'played_at': 'Played At'
        })
    else:
        display_df = df[['name', 'artist', 'album', 'popularity']].head(max_rows)
        display_df = display_df.rename(columns={
            'name': 'Track',
            'artist': 'Artist',
            'album': 'Album',
            'popularity': 'Popularity'
        })
    
    st.subheader(title)
    st.dataframe(display_df, use_container_width=True)


def analyze_audio_features(df):
    """Analyze audio features in the top tracks dataframe if they exist"""
    if df.empty:
        return None
    
    # Check if audio features columns exist
    audio_features = ['danceability', 'energy', 'tempo', 'valence', 'acousticness', 'instrumentalness']
    existing_features = [feature for feature in audio_features if feature in df.columns]
    
    if not existing_features:
        return None
    
    # Create a radar chart of audio features
    feature_avgs = {}
    for feature in existing_features:
        if not df[feature].isna().all():
            feature_avgs[feature] = df[feature].mean()
    
    if not feature_avgs:
        return None
    
    # Create the radar chart using plotly
    categories = list(feature_avgs.keys())
    values = list(feature_avgs.values())
    
    # Scale tempo to 0-1 range if it exists (typical tempo range is 50-150)
    if 'tempo' in feature_avgs:
        values[categories.index('tempo')] = values[categories.index('tempo')] / 200
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Your Music Profile',
        line_color='#1DB954'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title="Audio Features Analysis",
        showlegend=False
    )
    
    return fig


def analyze_genres(df):
    """Analyze genre distribution if genre information is available"""
    if df.empty or 'genres' not in df.columns or df['genres'].isna().all():
        return None
    
    # Process genres - they might be stored as a string representation of a list
    try:
        if isinstance(df['genres'].iloc[0], str):
            # Try to parse the string as a list
            df['genres_list'] = df['genres'].apply(lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else [])
        else:
            df['genres_list'] = df['genres']
    except:
        return None
    
    # Flatten the list of genres
    all_genres = []
    for genres in df['genres_list']:
        if isinstance(genres, list):
            all_genres.extend(genres)
    
    if not all_genres:
        return None
    
    # Count genre occurrences
    genre_counts = pd.Series(all_genres).value_counts().head(15)
    genre_df = pd.DataFrame({'Genre': genre_counts.index, 'Count': genre_counts.values})
    
    # Create horizontal bar chart
    fig = px.bar(
        genre_df,
        x='Count',
        y='Genre',
        orientation='h',
        labels={'Count': 'Number of Occurrences', 'Genre': ''},
        color='Count',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        title='Top Genres in Your Music',
        yaxis=dict(categoryorder='total ascending'),
        plot_bgcolor='rgba(0,0,0,0)',
        height=min(100 + len(genre_counts) * 25, 500)  # Dynamic height based on number of genres
    )
    
    return fig


def track_release_analysis(df):
    """Analyze track release dates for insights"""
    if df.empty or 'release_date' not in df.columns or df['release_date'].isna().all():
        return None
    
    # Extract year from release_date
    df['year'] = pd.to_datetime(df['release_date']).dt.year
    
    # Group by year and count tracks
    yearly_counts = df['year'].value_counts().sort_index()
    
    # Create the year distribution chart
    year_df = pd.DataFrame({'Year': yearly_counts.index, 'Count': yearly_counts.values})
    
    fig = px.bar(
        year_df,
        x='Year',
        y='Count',
        labels={'Count': 'Number of Tracks', 'Year': 'Release Year'},
        color='Count',
        color_continuous_scale='Viridis',
        title='Distribution of Your Top Tracks by Release Year'
    )
    
    fig.update_layout(
        xaxis=dict(tickmode='linear', dtick=5),  # Show every 5 years on x-axis
        plot_bgcolor='rgba(0,0,0,0)',
        height=400
    )
    
    # Calculate decade distribution
    df['decade'] = (df['year'] // 10) * 10
    decade_counts = df['decade'].value_counts().sort_index()
    decade_df = pd.DataFrame({'Decade': [f"{d}s" for d in decade_counts.index], 'Count': decade_counts.values})
    
    decade_fig = px.pie(
        decade_df,
        values='Count',
        names='Decade',
        title='Top Tracks by Decade',
        color_discrete_sequence=px.colors.sequential.Viridis
    )
    
    decade_fig.update_traces(textposition='inside', textinfo='percent+label')
    decade_fig.update_layout(height=400)
    
    return fig, decade_fig


def plot_artist_distribution(df):
    """Create a histogram showing the distribution of tracks per artist"""
    if df.empty or 'artist' not in df.columns:
        return None
    
    # Count the number of tracks per artist
    artist_counts = df['artist'].value_counts()
    
    # Create a dataframe with the distribution counts
    # This will show how many artists have 1 song, how many have 2 songs, etc.
    # Convert to dataframe first and then use value_counts to avoid column name conflict
    track_count_df = pd.DataFrame({'track_count': artist_counts.values})
    distribution = track_count_df['track_count'].value_counts().sort_index().reset_index()
    distribution.columns = ['Tracks per Artist', 'Number of Artists']
    
    # Create the bar chart
    fig = px.bar(
        distribution,
        x='Tracks per Artist',
        y='Number of Artists',
        labels={
            'Tracks per Artist': 'Number of Tracks from an Artist',
            'Number of Artists': 'Count of Artists'
        },
        title='Distribution of Tracks per Artist',
        color='Number of Artists',
        color_continuous_scale='Viridis',
        text_auto=True
    )
    
    fig.update_layout(
        xaxis=dict(tickmode='linear'),
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        bargap=0.1,
        yaxis=dict(title='Number of Artists')
    )
    
    # Add a more detailed hover template
    fig.update_traces(
        hovertemplate='<b>%{x} tracks</b>: %{y} artists<extra></extra>'
    )
    
    return fig


def main():
    """Main function for Streamlit app"""
    # Header with centered title and contact info
    
    st.markdown("<h1 class='main-header' style='text-align: center;'>🎵 Hannah's Spotify Dashboard 🎵</h1>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header' style='text-align: center;'>Contact Info : Hannah Lee | hsl68@cornell.edu</h2>", unsafe_allow_html=True)

    
    
    # Check if data is loaded or attempt to load it automatically
    if not st.session_state.data_loaded:
        # Try to automatically load data on first run
        load_data_from_csv()
        
        if not st.session_state.data_loaded:
            st.info("Click 'Load CSV Data' in the sidebar to load your saved listening history.")
            
            # Display info message
            st.markdown("---")
            st.markdown("### 🔍 About This Dashboard")
            st.write("""
            This dashboard displays your Spotify listening history from saved CSV files.
            No Spotify authentication is required, as the data is read directly from local files.
                     
            Created by Hannah Lee, @hannahstein
            """)
            return
    
    # Data is loaded, display dashboard
    top_df = st.session_state.top_df
    
    # Key metrics - Expanded with more insights from top tracks
    st.markdown("<h2 class='sub-header'>📊 Key Metrics</h2>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"<div class='stat-number'>{len(top_df)}</div>", unsafe_allow_html=True)
        st.markdown("Top Tracks", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        unique_artists = len(top_df['artist'].unique()) if not top_df.empty else 0
        st.markdown(f"<div class='stat-number'>{unique_artists}</div>", unsafe_allow_html=True)
        st.markdown("Different Artists", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        avg_pop = round(top_df['popularity'].mean()) if not top_df.empty else 0
        st.markdown(f"<div class='stat-number'>{avg_pop}/100</div>", unsafe_allow_html=True)
        st.markdown("Avg. Popularity", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col4:
        # Calculate artist diversity score (ratio of unique artists to total tracks)
        diversity_score = round((unique_artists / len(top_df)) * 100) if not top_df.empty and len(top_df) > 0 else 0
        st.markdown(f"<div class='stat-number'>{diversity_score}%</div>", unsafe_allow_html=True)
        st.markdown("Artist Diversity", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Detailed Analysis as a scrollable page
    st.markdown("<h2 class='sub-header'>📈 Detailed Analysis</h2>", unsafe_allow_html=True)
    
    # Section 1: Top Tracks
    st.markdown("---")
    st.header("🎵 Your Top Tracks")
    if not top_df.empty:
        # Display top tracks
        show_track_table(top_df, "Top Tracks from the Past 6 Months", 500)
        
        # Popularity distribution
        st.subheader("Popularity Distribution")
        pop_fig = plot_popularity_distribution(top_df)
        if pop_fig:
            st.plotly_chart(pop_fig, use_container_width=True)
        # Show a table of songs, filted by popularity range
        st.subheader("Top Tracks by Popularity")
        # Get a slider for popularity range
        min_pop, max_pop = st.slider(
            "Select Popularity Range",
            min_value=0,
            max_value=100,
            value=(0, 100),
            step=1
        )
        # Filter the DataFrame based on the selected range
        filtered_df = top_df[(top_df['popularity'] >= min_pop) & (top_df['popularity'] <= max_pop)]
        # Show the filtered DataFrame
        show_track_table(filtered_df, f"Top Tracks with Popularity between {min_pop} and {max_pop}", 500)
    else:
        st.info("No top tracks data available.")
    
    # Section 2.5: Artist Distribution Analysis
    st.markdown("---")
    st.header("🎸 Artist Distribution Analysis")
    if not top_df.empty:
        st.subheader("How Many Songs Do You Listen to from Each Artist?")
        artist_dist_fig = plot_artist_distribution(top_df)
        if artist_dist_fig:
            st.plotly_chart(artist_dist_fig, use_container_width=True)
            
            # Add explanation text
            total_artists = len(top_df['artist'].unique())
            artists_with_one_track = sum(top_df['artist'].value_counts() == 1)
            percentage_one_track = round((artists_with_one_track / total_artists) * 100)
            
            st.markdown(f"""
            This graph shows how your listening is distributed across artists. Each bar represents the number of artists 
            who appear in your top tracks with that specific number of songs.
            
            **Insights:**
            - **{artists_with_one_track}** artists ({percentage_one_track}% of your total artists) appear with just one track
            - The shape of this distribution helps identify if you're a "diverse listener" (many artists with few songs each) 
              or a "deep diver" (fewer artists with many songs each)
            """)
            
            # Add slider and filtered dataframe by track count
            st.subheader("Find Artists by Number of Songs")
            
            # Calculate the range for the slider
            min_tracks = 1
            max_tracks = top_df['artist'].value_counts().max()
            
            # Create a slider for selecting the track count range
            min_tracks_slider, max_tracks_slider = st.slider(
                "Select Number of Songs per Artist",
                min_value=min_tracks,
                max_value=max_tracks,
                value=(1, max_tracks),
                step=1
            )
            
            # Get artists with song counts in the selected range
            artist_counts = top_df['artist'].value_counts()
            selected_artists = artist_counts[(artist_counts >= min_tracks_slider) & (artist_counts <= max_tracks_slider)].index.tolist()
            
            # Filter tracks for the selected artists
            filtered_artists_df = top_df[top_df['artist'].isin(selected_artists)]
            
            # Sort by artist and popularity
            filtered_artists_df = filtered_artists_df.sort_values(by=['artist', 'popularity'], ascending=[True, False])
            
            # Create a new column with the count of songs per artist for display
            artist_song_counts = top_df['artist'].value_counts().to_dict()
            filtered_artists_df['Songs Count'] = filtered_artists_df['artist'].map(artist_song_counts)
            
            # Show the filtered DataFrame with artist song counts
            if not filtered_artists_df.empty:
                st.write(f"Showing tracks from artists with {min_tracks_slider} to {max_tracks_slider} songs in your top tracks:")
                
                # Select and rename columns for display
                display_df = filtered_artists_df.rename(columns={
                    'artist': 'Artist',
                    'name': 'Track',
                    'album': 'Album',
                    'popularity': 'Popularity'
                })
                # lets group by artist and count the number of songs
                display_df = display_df.groupby(['Artist', 'Songs Count']).size().reset_index(name='Count')
                st.dataframe(display_df[['Artist', 'Songs Count']], use_container_width=True)
            else:
                st.info(f"No artists with {min_tracks_slider} to {max_tracks_slider} songs in your collection.")
    else:
        st.info("No artist distribution data available.")
    
    # Section 3: Enhanced Track Analysis
    st.markdown("---")
    st.header("📊 Enhanced Track Analysis")
    
    # Music taste insights from top_df
    if not top_df.empty:
        st.subheader("🎧 Music Taste Insights")
        
        # Create two columns for insights
        col1, col2 = st.columns(2)
        
        with col1:
            # Mainstream vs. Obscure analysis based on top tracks
            avg_pop = round(top_df['popularity'].mean()) if not top_df.empty else 0
            if avg_pop < 40:
                st.markdown(f"### 🔍 Obscure Taste")
                st.write(f"Your average track popularity is {avg_pop}/100")
                st.write("You tend to listen to less mainstream music compared to the average Spotify user.")
            elif avg_pop > 70:
                st.markdown(f"### 🌟 Mainstream Taste")
                st.write(f"Your average track popularity is {avg_pop}/100")
                st.write("You tend to listen to very popular tracks that are widely played.")
            else:
                st.markdown(f"### ⚖️ Balanced Taste")
                st.write(f"Your average track popularity is {avg_pop}/100")
                st.write("You have a good balance between popular tracks and less mainstream music.")
        
        with col2:
            # Artist diversity
            unique_artists = len(top_df['artist'].unique())
            total_tracks = len(top_df)
            diversity_ratio = unique_artists / total_tracks
            
            st.markdown(f"### 👥 Artist Diversity")
            st.write(f"{unique_artists} different artists across {total_tracks} tracks")
            
            if diversity_ratio > 0.8:
                st.write("You have very diverse listening habits, with few repeat artists.")
            elif diversity_ratio > 0.5:
                st.write("You have a healthy balance of favorite artists and musical exploration.")
            else:
                st.write("You tend to focus on a core set of favorite artists.")
        
        # Most common decade if release_date is available
        if 'release_date' in top_df.columns:
            try:
                # Extract year and find most common decade
                top_df['year'] = pd.to_datetime(top_df['release_date']).dt.year
                top_df['decade'] = (top_df['year'] // 10) * 10
                decade_counts = top_df['decade'].value_counts()
                top_decade = decade_counts.index[0]
                decade_percentage = (decade_counts.iloc[0] / len(top_df)) * 100
                
                st.subheader("🗓️ Favorite Era")
                st.write(f"{top_decade}s ({decade_percentage:.1f}% of your top tracks)")
                
                # Additional decade insights
                if top_decade >= 2020:
                    st.write("You're all about the latest hits and current music.")
                elif top_decade >= 2010:
                    st.write("The 2010s defined your music taste - relatively recent but not brand new.")
                elif top_decade >= 2000:
                    st.write("You have a strong connection to the music of the 2000s.")
                elif top_decade >= 1990:
                    st.write("90s music features prominently in your listening habits.")
                else:
                    st.write(f"You have an appreciation for classic tracks from the {top_decade}s.")
            except:
                pass
    else:
        st.info("No listening pattern data available.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center">
        <p><a href="mailto:hsl68@cornell.edu">hsl68@cornell.edu</a> 
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
