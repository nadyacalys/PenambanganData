import sys
print(sys.executable)
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px

st.set_page_config(page_title="COVID-19 Clustering Dashboard", layout="wide")
st.title("Clustering Dashboard COVID-19")

@st.cache_data
def load_data():
    df = pd.read_csv("covid_19_indonesia_time_series_all.csv")
    return df

df = load_data()

# Pastikan kolom 'Date' dalam format datetime
df['Date'] = pd.to_datetime(df['Date'])

# Filter kolom yang digunakan
df = df[['Date', 'Location', 'Total Cases', 'Total Deaths', 'Total Recovered', 'Population Density']]
df.dropna(inplace=True)

# Sidebar pilih lokasi
unique_locations = df['Location'].unique()
selected_location = st.sidebar.selectbox("Pilih Lokasi", unique_locations)
location_data = df[df['Location'] == selected_location]

# Visualisasi tren kasus harian (berdasarkan total harian)
st.subheader(f"Tren Kasus Harian di {selected_location}")
fig, ax = plt.subplots(figsize=(10,4))
daily_cases = location_data.groupby("Date")['Total Cases'].sum()
daily_cases.plot(ax=ax, color='red')
ax.set_ylabel("Total Cases")
ax.set_xlabel("Date")
st.pyplot(fig)

# === Clustering === #
st.subheader("Hasil Clustering Wilayah")

# Ambil snapshot terbaru per lokasi (bukan rata-rata time series)
df_latest = df.sort_values('Date').groupby('Location').last()

# Ambil fitur clustering
cluster_features = df_latest[['Total Cases', 'Total Deaths', 'Total Recovered', 'Population Density']].copy()

# Standardisasi
scaler = StandardScaler()
scaled_features = scaler.fit_transform(cluster_features)

# KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(scaled_features)
cluster_features['Cluster'] = clusters

# Gabungkan ke dataframe awal untuk mapping jika perlu
df_clustered = df.merge(cluster_features['Cluster'], on='Location')

# Koordinat beberapa lokasi untuk peta
kordinat = pd.DataFrame({
    'Location': [
        'DKI Jakarta', 'Jawa Barat', 'Jawa Tengah', 'Jawa Timur',
        'Bali', 'Sumatera Utara', 'Kalimantan Timur', 'Sulawesi Selatan'
    ],
    'lat': [
        -6.2088, -6.9039, -7.1500, -7.2504,
        -8.4095, 3.5952, 0.5383, -5.1477
    ],
    'lon': [
        106.8456, 107.6186, 110.1403, 112.7688,
        115.1889, 98.6722, 116.4194, 119.4327
    ]
})

# Gabungkan untuk peta
map_df = cluster_features.reset_index().merge(kordinat, on='Location', how='left')

st.subheader("Peta Interaktif Clustering Wilayah")
ffig_map = px.scatter_mapbox(
    map_df,
    lat="lat", lon="lon",
    hover_name="Location",
    color="Cluster",  # ← tetap pakai Cluster untuk kategorisasi
    size="Total Cases",
    zoom=4,
    height=500,
    mapbox_style="carto-positron",
    color_discrete_map={
        0: "red",
        1: "yellow",
        2: "green",
        3: "blue"
    }
)
st.plotly_chart(fig_map, use_container_width=True)

# Tampilkan tabel cluster
st.subheader("Ringkasan Risiko Wilayah Berdasarkan Cluster")
st.dataframe(cluster_features.sort_values("Cluster"))
