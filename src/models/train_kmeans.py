import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.insert(0,"D:\\2M\D.Mining\Data-Mining-Project")
from models.K_Means import KMeans
from src.utils import split_data, compute_metrics, plot_confusion_matrix, silhouette_score

# ----------------------------------------------------------------#
# Load data
# ----------------------------------------------------------------#
df = pd.read_csv("../../data/interim/03_static_dataset_features_built.csv", index_col=0)

# ----------------------------------------------------------------#
# Data
# ----------------------------------------------------------------#
X, y = df.drop(columns=["Fertility"]), df["Fertility"]

# ----------------------------------------------------------------#
# Standardize Data
# ----------------------------------------------------------------#
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)

# ----------------------------------------------------------------#
# Pick the right K
# ----------------------------------------------------------------#
# Silhoeutte method
k_range = range(2, 20) 
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(k=k)
    kmeans.fit(X, plot_steps=False)
    labels = kmeans.predict(X)
    silhouette_avg = silhouette_score(X, labels)
    print(f"silhouette score for {k} clusters: {silhouette_avg}")
    silhouette_scores.append(silhouette_avg)

# Plot the silhouette scores
plt.plot(k_range, silhouette_scores, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Different Values of k')
plt.xticks(np.arange(min(k_range), max(k_range)+1, 1.0))
plt.savefig("../../reports/figures/Part_2/KMeans_silhouette_method.png")
plt.show()

# Elbow method
k_range = range(2, 20)
inertia_scores = []

for k in k_range:
    kmeans = KMeans(k=k)
    kmeans.fit(X, plot_steps=False)
    inertia_scores.append(kmeans.inertia)

# Plot the inertia scores
plt.plot(k_range, inertia_scores, marker='o')   
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia Score')
plt.title('Inertia Score for Different Values of k')
plt.xticks(np.arange(min(k_range), max(k_range)+1, 1.0))
plt.savefig("../../reports/figures/Part_2/KMeans_elbow_method.png")
plt.show()

# ----------------------------------------------------------------
# Train KMeans with the best hyperparameters
# ----------------------------------------------------------------
kmeans = KMeans(k=2)
kmeans.fit(X, combined_plot=True)
labels = kmeans.predict(X)
print(f"Number of clusters: {len(np.unique(labels))}")
print(f"Silhouette score: {silhouette_score(X, labels)}")

kmeans_3 = KMeans(k=3)
kmeans_3.fit(X)
labels_3 = kmeans_3.predict(X)
print(f"Number of clusters: {len(np.unique(labels_3))}")
print(f"Silhouette score: {silhouette_score(X, labels_3)}")