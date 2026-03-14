import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

st.title("🎵 Music Genre Grouping using Hierarchical Clustering")

st.write("This app groups songs based on audio features like tempo, energy, loudness, and danceability.")

data = pd.read_excel("spotify_music_small.xlsx")
st.subheader("Dataset Preview")
st.write(data.head())

features = data[['tempo','energy','loudness','danceability']]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

linked = linkage(scaled_data, method='ward')

st.subheader("Song Clustering Dendrogram")

fig, ax = plt.subplots(figsize=(10,5))

dendrogram(
    linked,
    labels=data['track_name'].values,
    leaf_rotation=90,
    ax=ax
)

plt.xlabel("Songs")
plt.ylabel("Distance")

st.pyplot(fig)
