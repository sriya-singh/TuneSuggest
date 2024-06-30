# streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
from collections import defaultdict
from scipy.spatial.distance import cdist
import plotly.express as px
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# Set Spotify API credentials as environment variables
os.environ["SPOTIFY_CLIENT_ID"] = "bc707fc1e55a4b11919cf4f2f9376d1a"
os.environ["SPOTIFY_CLIENT_SECRET"] = "0f27b7f9d1d14f58a20a802cb0ced2bd"

# Spotify API client
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=os.environ["SPOTIFY_CLIENT_ID"], client_secret=os.environ["SPOTIFY_CLIENT_SECRET"]))

# Load the dataset
data = pd.read_csv('data.csv')

# Preprocessing and clustering pipeline
number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']

song_cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=20, verbose=False))], verbose=False)
X = data.select_dtypes(np.number)
song_cluster_pipeline.fit(X)
data['cluster_label'] = song_cluster_pipeline.predict(X)

# Function to get all the details and audio features of the song from Spotify API
def find_song(name, year):
    song_data = defaultdict()
    results = sp.search(q=f'track: {name} year: {year}', limit=1)
    if results['tracks']['items'] == []:
        return None

    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]

    song_data['name'] = [name]
    song_data['year'] = [year]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]

    for key, value in audio_features.items():
        song_data[key] = value

    return pd.DataFrame(song_data)

# Function to get song data
def get_song_data(song, spotify_data):
    try:
        song_data = spotify_data[(spotify_data['name'] == song['name']) & (spotify_data['year'] == song['year'])].iloc[0]
        return song_data
    except IndexError:
        return find_song(song['name'], song['year'])

# Function to get mean vector of all the features in number_col for the given song list
def get_mean_vector(song_list, spotify_data):
    song_vectors = []
    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            st.warning(f'Warning: {song["name"]} does not exist in Spotify or in database')
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)

    song_matrix = np.array(song_vectors)
    return np.mean(song_matrix, axis=0)

# Function to flatten dictionary list
def flatten_dict_list(dict_list):
    flattened_dict = defaultdict(list)
    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)
    return flattened_dict

# Function to recommend songs
def recommend_songs(song_list, spotify_data, n_songs=10):
    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)

    song_center = get_mean_vector(song_list, spotify_data)
    scaler = song_cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])
    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]

    return rec_songs.reset_index(drop=True)

# Streamlit app layout
st.title("TuneSuggest")
st.subheader("Unlock Your Soundtrack: Where Music Finds You")

# Initialize session state for song list and recommendations
if 'song_list' not in st.session_state:
    st.session_state.song_list = []

if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None

# Sidebar for song selection
with st.sidebar:
    st.header("Select Songs")
    song_name = st.text_input("Song Name", key="song_name")
    song_year = st.text_input("Song Year", key="song_year")
    
    if st.button("Add Song"):
        if song_name and song_year:
            st.session_state.song_list.append({'name': song_name, 'year': int(song_year)})
            st.success(f"Added {song_name} from {song_year} to the list")
            # st.session_state.song_name = ""  # Clear the input fields
            # st.session_state.song_year = ""
        else:
            st.warning("Please provide both song name and year")
    
    if st.session_state.song_list:
        st.subheader("Selected Songs")
        df_selected_songs = pd.DataFrame(st.session_state.song_list)
        # Format year column to remove thousands separator
        df_selected_songs['year'] = df_selected_songs['year'].astype(int).astype(str)

        # Display DataFrame without index column
        # st.write(pd.DataFrame(st.session_state.song_list))
        st.write(df_selected_songs)

    if st.button("Clear All"):
        st.session_state.song_list = []
        st.session_state.recommendations = None
        st.experimental_rerun()  # Rerun to update the state

    n_recommendations = st.number_input("Number of Recommendations", min_value=1, max_value=20, value=10)

    if st.button("Get Recommendations") and st.session_state.song_list:
        recommendations = recommend_songs(st.session_state.song_list, data, n_songs=n_recommendations)
        st.session_state.recommendations = recommendations

# Display recommendations
if st.session_state.recommendations is not None:
    # st.subheader("Recommendations")
    cols = st.columns(4)  # Create 4 columns for a grid layout
    for idx, row in st.session_state.recommendations.iterrows():
        with cols[idx % 4]:
            # Format artist names
            artists = ', '.join(row['artists']) if isinstance(row['artists'], list) else row['artists']

            # Remove square brackets and quotes from artist names
            artists = artists.replace("[", "").replace("]", "").replace("'", "").replace('"', '')

            # Use HTML/CSS to add hover effects and display image and song details
            st.markdown(
                f"""
                <style>
                .song {{
                    position: relative;
                    display: inline-block;
                    margin-bottom: 20px;
                    text-align: center;
                }}
                .song img {{
                    transition: transform 0.2s, filter 0.2s;
                    width: 150px;
                    height: 150px;
                }}
                .song:hover img {{
                    transform: scale(1.05);
                    filter: brightness(50%);
                }}
                .song-details {{
                    visibility: hidden;
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    flex-direction: column;
                    color: white;
                    text-align: center;
                    font-size: 14px;
                    white-space: nowrap;
                    font-weight: bold;
                }}
                .song:hover .song-details {{
                    visibility: visible;
                }}
                .song-name {{
                    margin-top: 10px;
                    font-weight: bold;
                }}
                </style>
                <div class="song">
                    <img src="{sp.track(row['id'])['album']['images'][0]['url']}" alt="{row['name']}">
                    <div class="song-details">
                        <p style="font-weight: bold;">{artists}</p>
                        <p style="font-weight: bold;">{row['year']}</p>
                    </div>
                    <div class="song-name">
                        <p>{row['name']}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True
            )
