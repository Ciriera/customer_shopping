### data_processing.py (Veri yükleme ve ön işleme için)
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import gc
import streamlit as st

@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    selected_columns = ["shopping_mall", "category", "quantity"]
    return data[selected_columns]

@st.cache_data
def preprocess_data(data):
    data_encoded = pd.get_dummies(data, drop_first=True)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_encoded)
    return scaled_data

def cleanup():
    gc.collect()

### __init__.py (Modül dosyası)
# Bu dosya klasörü bir Python modülü haline getirir. İçerik olarak boş bırakılabilir.

### clustering_algorithms.py (Kümeleme algoritmalarını içerir)
import time
import psutil
from sklearn.cluster import MiniBatchKMeans
from hdbscan import HDBSCAN
import fastcluster
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import streamlit as st

@st.cache_resource
def measure_cpu_time():
    process = psutil.Process()
    return process.cpu_times().user

@st.cache_resource
def run_kmeans(scaled_data, n_clusters=3):
    start_time = measure_cpu_time()
    param_dist_kmeans = {'n_clusters': [n_clusters], 'batch_size': [100, 200]}
    kmeans = MiniBatchKMeans(random_state=42)
    kmeans_search = RandomizedSearchCV(kmeans, param_distributions=param_dist_kmeans, n_iter=3, cv=2)
    kmeans_search.fit(scaled_data)
    labels = kmeans_search.best_estimator_.predict(scaled_data)
    cpu_time = measure_cpu_time() - start_time
    return labels, cpu_time

@st.cache_resource
def run_hdbscan(scaled_data):
    start_time = measure_cpu_time()
    hdbscan = HDBSCAN(min_cluster_size=10, min_samples=5)
    labels = hdbscan.fit_predict(scaled_data)
    cpu_time = measure_cpu_time() - start_time
    return labels, cpu_time

@st.cache_resource
def run_fastcluster(scaled_data, n_clusters=3):
    start_time = measure_cpu_time()
    linkage_matrix = fastcluster.linkage_vector(scaled_data, method="ward")
    clusters = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")
    cpu_time = measure_cpu_time() - start_time
    return clusters, cpu_time

### streamlit_app.py (Ana GUI dosyası)
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
import sys

# Dosya yolu ayarı
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Modül içe aktarma
try:
    from data_processing_Kopya import load_data, preprocess_data, cleanup
    from clustering_algorithms import run_kmeans, run_hdbscan, run_fastcluster
except ImportError as e:
    st.error(f"Modül içe aktarma hatası: {e}")
    st.stop()

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

st.set_page_config(layout="wide")

# Sidebar ile Dosya Yükleme ve Algoritma Seçimi
st.sidebar.title("Kümeleme Analizi Ayarları")
file_path = st.sidebar.file_uploader("Lütfen CSV dosyanızı yükleyin")
algorithm_choice = st.sidebar.selectbox("Kümeleme Algoritması Seçin", ["MiniBatch K-Means", "HDBSCAN", "Fastcluster"])
num_clusters = st.sidebar.slider("Küme Sayısı Seçin", min_value=2, max_value=10, value=3)

# Örnek veri seti kullanımı
if st.sidebar.button("Örnek Veri Seti Kullan"):
    data = pd.DataFrame({
        "shopping_mall": ["Mall A", "Mall B", "Mall C", "Mall A", "Mall B"],
        "category": ["Electronics", "Clothing", "Food", "Electronics", "Clothing"],
        "quantity": [100, 200, 150, 180, 220]
    })
    st.write("**Örnek Veri Seti Yüklendi.**")

if file_path:
    st.title("Alışveriş Merkezi Kümeleme ve Satış Analizi")
    data = load_data(file_path)
    st.subheader("Veri Önizleme")
    st.dataframe(data.head())
    scaled_data = preprocess_data(data)
    cleanup()

    # Algoritma Seçimi
    if algorithm_choice == "MiniBatch K-Means":
        labels, cpu_time = run_kmeans(scaled_data, num_clusters)
    elif algorithm_choice == "HDBSCAN":
        labels, cpu_time = run_hdbscan(scaled_data)
    else:
        labels, cpu_time = run_fastcluster(scaled_data, num_clusters)

    # Performans Skorları
    silhouette = silhouette_score(scaled_data, labels)
    davies_bouldin = davies_bouldin_score(scaled_data, labels)
    calinski_harabasz = calinski_harabasz_score(scaled_data, labels)

    st.sidebar.subheader("Performans Metrikleri")
    st.sidebar.write(f"**CPU Time:** {cpu_time:.2f} s")
    st.sidebar.write(f"**Silhouette Score:** {silhouette:.4f}")
    st.sidebar.write(f"**Davies-Bouldin Score:** {davies_bouldin:.4f}")
    st.sidebar.write(f"**Calinski-Harabasz Score:** {calinski_harabasz:.4f}")

    # Hata Düzeltmesi: MultiIndex için pivot_table kullanımı
    filtered_data = data.copy()
    sales_heatmap_grouped = filtered_data.pivot_table(
        values="quantity",
        index="shopping_mall",
        columns="category",
        aggfunc="sum",
        fill_value=0
    )

    # Gerçek Zamanlı İlerleme Çubuğu
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.02)
        progress_bar.progress(i + 1)

    # Algoritma Karşılaştırma Çubuk Grafiği
    comparison_metrics = pd.DataFrame({
        "Algorithm": ["MiniBatch K-Means", "HDBSCAN", "Fastcluster"],
        "Silhouette Score": [silhouette, silhouette, silhouette],
        "Davies-Bouldin Score": [davies_bouldin, davies_bouldin, davies_bouldin],
        "Calinski-Harabasz Score": [calinski_harabasz, calinski_harabasz, calinski_harabasz]
    })
    comparison_fig = px.bar(
        comparison_metrics,
        x="Algorithm",
        y=["Silhouette Score", "Davies-Bouldin Score", "Calinski-Harabasz Score"],
        barmode="group"
    )
    st.plotly_chart(comparison_fig)

    st.success("Analiz Tamamlandı! Sonuçlar Önbelleğe Alındı!")
    cleanup()
