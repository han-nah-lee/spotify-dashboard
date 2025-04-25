"""
Utility functions for artist analysis in Spotify data.
"""
import pandas as pd
import plotly.express as px

def get_top_artists_from_tracks(df, n=15):
    """
    Get top artists based on the number of tracks in the user's top tracks.
    
    Args:
        df: DataFrame containing track data with 'artist' column
        n: Number of top artists to return
        
    Returns:
        Plotly figure with top artists visualization
    """
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
    
    # Sort by track count
    top_artists = top_artists.sort_values('Track Count', ascending=True)
    
    # Create the visualization
    fig = px.bar(
        top_artists,
        x='Track Count',
        y='Artist',
        orientation='h',
        text='Percentage',
        hover_data=['Avg Popularity'] if 'Avg Popularity' in top_artists.columns else None,
        labels={'Track Count': 'Number of Tracks in Your Top List', 'Artist': ''},
        title=f'Your Top {n} Artists (Based on Top Tracks)',
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
