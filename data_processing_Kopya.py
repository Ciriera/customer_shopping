import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from hdbscan import HDBSCAN
import fastcluster
from scipy.cluster.hierarchy import fcluster
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

def measure_cpu_time():
    return time()

# Veri Yükleme
file_path = "E:\\Bilgiye Erişim Proje\\customer_shopping_data_Kopya.csv"
data = pd.read_csv(file_path)
selected_columns = ["shopping_mall", "category", "quantity"]
data = data[selected_columns]

# Kategorik değişkenleri sayısal hale getirme
data_encoded = pd.get_dummies(data, drop_first=True)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_encoded)

# MiniBatch K-Means Hiper Parametre Optimizasyonu
kmeans = MiniBatchKMeans(n_clusters=3, batch_size=100, random_state=42)
kmeans.fit(scaled_data)
kmeans_labels = kmeans.predict(scaled_data)

# HDBSCAN için Hiper Parametre Optimizasyonu
hdbscan = HDBSCAN(min_cluster_size=10, min_samples=5)
hdbscan_labels = hdbscan.fit_predict(scaled_data)

# Fastcluster için Hiper Parametre Optimizasyonu
linkage_matrix = fastcluster.linkage_vector(scaled_data, method="ward")
clusters = fcluster(linkage_matrix, t=3, criterion="maxclust")

# Satış Yoğunluğu Isı Haritası (Tek Bir Grafikte Tüm Kategoriler)
for algorithm, labels in zip(["MiniBatch K-Means", "HDBSCAN", "Fastcluster"], [kmeans_labels, hdbscan_labels, clusters]):
    sales_heatmap = data.copy()
    sales_heatmap["cluster"] = labels
    sales_heatmap_grouped = sales_heatmap.pivot_table(values="quantity", index="shopping_mall", columns="category", aggfunc="sum", fill_value=0)

    plt.figure(figsize=(14, 10))
    sns.heatmap(sales_heatmap_grouped, annot=True, fmt=".0f", cmap="coolwarm")
    plt.title(f"{algorithm} Satış Yoğunluğu (Tüm Kategoriler Bir Arada)")
    plt.xlabel("Ürün Kategorisi")
    plt.ylabel("Alışveriş Merkezi")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
