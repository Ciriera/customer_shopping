print("Kod Başladı")

import pandas as pd
import numpy as np
import gc
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from hdbscan import HDBSCAN
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import silhouette_score

# Uyarıları bastırma
warnings.filterwarnings("ignore")

# CPU Zaman Ölçüm Fonksiyonu
def measure_cpu_time():
    return time()

# Veri yükleme
file_path = "E:\\Bilgiye Erişim Proje\\customer_shopping_data_Kopya.csv"
data = pd.read_csv(file_path)

# Gerekli sütunları seçme
selected_columns = ["shopping_mall", "category", "quantity"]
data = data[selected_columns]

# Belleği temizleme
gc.collect()

# Kategorik değişkenleri sayısal hale getirme
data_encoded = pd.get_dummies(data, drop_first=True)

# Veriyi normalize etme
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_encoded)

# Belleği temizleme
gc.collect()

# MiniBatch K-Means için Hiper Parametre Optimizasyonu (Çok Hızlı)
start_time = measure_cpu_time()
kmeans = MiniBatchKMeans(n_clusters=3, batch_size=100, random_state=42)
kmeans.fit(scaled_data)
kmeans_labels = kmeans.predict(scaled_data)
kmeans_cpu_time = measure_cpu_time() - start_time
print(f"MiniBatch K-Means CPU Zamanı: {kmeans_cpu_time:.4f} saniye")

# HDBSCAN için Hiper Parametre Optimizasyonu (Çok Hızlı)
start_time = measure_cpu_time()
optimal_hdbscan = HDBSCAN(min_cluster_size=10, min_samples=5)
hdbscan_labels = optimal_hdbscan.fit_predict(scaled_data)
hdbscan_cpu_time = measure_cpu_time() - start_time
print(f"HDBSCAN CPU Zamanı: {hdbscan_cpu_time:.4f} saniye")

# Scipy.cluster.hierarchy için Hiper Parametre Optimizasyonu
start_time = measure_cpu_time()
linkage_matrix = linkage(scaled_data, method="ward")
clusters = fcluster(linkage_matrix, t=3, criterion="maxclust")
fastcluster_cpu_time = measure_cpu_time() - start_time
print(f"Scipy Fastcluster CPU Zamanı: {fastcluster_cpu_time:.4f} saniye")

# Performans karşılaştırma tablosu
performance_metrics = pd.DataFrame({
    'Algorithm': ['MiniBatch K-Means', 'HDBSCAN', 'Scipy Fastcluster'],
    'CPU Time (s)': [kmeans_cpu_time, hdbscan_cpu_time, fastcluster_cpu_time]
})
print(performance_metrics)

# Satış Yoğunluğu Tablosu (Her Algoritma İçin Ayrı Grafik, Tüm Kategoriler Bir Arada)
for algorithm, labels in zip(["MiniBatch K-Means", "HDBSCAN", "Scipy Fastcluster"], [kmeans_labels, hdbscan_labels, clusters]):
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

gc.collect()
print("Kod Bitti")
