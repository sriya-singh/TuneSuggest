# TuneSuggest

TuneSuggest is a web application powered by Streamlit that provides personalized music recommendations based on user-selected songs. It allows users to manually select a list of their favorite tracks and generates tailored recommendations using K-Means clustering.

## Features

- **Song Selection**: Add your favorite songs by specifying their name and release year.
- **Recommendations**: Get tailored song recommendations based on your selected songs using advanced clustering techniques.
- **Interactive UI**: Explore recommended songs with interactive visuals and hover effects.

## Usage

- **Add Songs**: Enter the name and release year of a song in the sidebar, then click "Add Song".
- **View Recommendations**: Specify the number of recommendations and click "Get Recommendations" to see suggested songs.
- **Clear Selection**: Use the "Clear All" button to reset the selected songs list.

## Technologies Used

- **Streamlit**: For creating the interactive web application.
- **Python Libraries**: Pandas, NumPy, Spotipy, Plotly, Scikit-learn, and others for data manipulation, visualization, and machine learning.
- **ML Model**: K-Means

## Dataset

The Spotify dataset used for training TuneSuggest is sourced from Kaggle, provided by [Vatsal Mavani](https://www.kaggle.com/vatsalmavani). This dataset contains a comprehensive collection of tracks from Spotify, including audio features such as valence, energy, tempo, and more.

- **Dataset Source**: [Spotify Dataset 1921-2020, 160k+ Tracks](https://www.kaggle.com/datasets/vatsalmavani/spotify-dataset)

## Deployment

TuneSuggest is deployed and accessible online. Visit [TuneSuggest Website](https://tunesuggestdeployed.streamlit.app/) to use the app.

## Data Analysis

You can also check out the detailed analyis of the dataset. [Github Repo](https://github.com/sriya-singh/Spotify-Music-Recommendation-System)
