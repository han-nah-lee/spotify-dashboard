# Spotty: Spotify Analytics Dashboard

Spotty is a Streamlit-based dashboard application that provides insightful analytics on a user's Spotify listening habits. It visualizes data such as top played songs, artist diversity, track popularity, and listening patterns, offering users a deeper understanding of their music preferences.

## Features

- **Top Tracks Analysis**: View your most played tracks over the past 6 months, including detailed insights into track popularity and artist diversity.
- **Artist Insights**: Discover your top artists based on track count and average popularity.
- **Listening Patterns**: Analyze your listening habits by hour of the day and day of the week.
- **Music Taste Insights**: Understand your music preferences, including mainstream vs. obscure taste and favorite music eras.
- **Customizable Filters**: Filter tracks by popularity range or number of songs per artist.

## How It Works

Spotty uses local CSV files or Spotify's API to fetch and analyze your listening data. The dashboard is built with Streamlit, making it interactive and easy to use.

### Data Sources
- **Local CSV Files**: Spotty can load data from pre-exported CSV files, such as `top_tracks_medium_term.csv`.
- **Spotify API**: For advanced users, Spotty can connect to Spotify's API to fetch real-time data (requires authentication).

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd spotify-dashboard
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run src/main.py
   ```

## Usage

- **Local CSV Mode**: Place your CSV files in the `src/data/` directory and run the app to visualize your data.
- **Spotify API Mode**: Authenticate with Spotify to fetch your top tracks and listening history directly from your account.

## File Structure

- `src/main.py`: Main entry point for the dashboard.
- `src/utils/`: Utility functions for data analysis and Spotify API integration.
- `src/data/`: Directory for storing local CSV files.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve Spotty.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For questions or feedback, contact:
- **Hannah Lee**: hsl68@cornell.edu