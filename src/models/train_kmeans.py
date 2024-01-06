import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

# sys.path.insert(0,"D:\\2M\D.Mining\Data-Mining-Project")
sys.path.insert(0, "../../../Data-Mining-Project")
from models.K_Means import KMeans
from src.utils import (
    split_data,
    compute_metrics,
    plot_confusion_matrix,
    silhouette_score,
)

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
plt.plot(k_range, silhouette_scores, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for Different Values of k")
plt.xticks(np.arange(min(k_range), max(k_range) + 1, 1.0))
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
plt.plot(k_range, inertia_scores, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia Score")
plt.title("Inertia Score for Different Values of k")
plt.xticks(np.arange(min(k_range), max(k_range) + 1, 1.0))
plt.savefig("../../reports/figures/Part_2/KMeans_elbow_method.png")
plt.show()

# ----------------------------------------------------------------
# Train KMeans with the best hyperparameters
# ----------------------------------------------------------------
kmeans = KMeans(k=2)
kmeans.fit(X, plot_steps=True)
labels = kmeans.predict(X)

print(f"Number of clusters: {len(np.unique(labels))}")
print(f"Silhouette score: {silhouette_score(X, labels)}")

kmeans_3 = KMeans(k=3)
kmeans_3.fit(X, plot_steps=True)
labels_3 = kmeans_3.predict(X)
print(f"Number of clusters: {len(np.unique(labels_3))}")
print(f"Silhouette score: {silhouette_score(X, labels_3)}")

# ----------------------------------------------------------------
# Dimensionality reduction to visualize the clusters
# ----------------------------------------------------------------
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# ----------------------------------------------------------------
# Plot the clusters
# ----------------------------------------------------------------
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

scatter = ax.scatter(
    X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels, cmap="viridis", marker="o"
)
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")
ax.set_title("3D Scatter Plot of Clusters in PCA Space")

# Add a colorbar to show the mapping of labels to colors
colorbar = plt.colorbar(scatter, ax=ax)
colorbar.set_label("Cluster Labels")
plt.savefig("../../reports/figures/Part_2/KMeans_clusters_pca.png")
plt.show()

features_names = df.drop(columns=["Fertility"]).columns

for i, pc in enumerate(pca.components_):
    top_features_indices = pc.argsort()[-5:][
        ::-1
    ]  # Adjust the number of top features to display
    top_features = [features_names[index] for index in top_features_indices]
    print(f"Top features for Principal Component {i+1}: {top_features}")

# ----------------------------------------------------------------
# Visualize the distribution of the clusters
# ----------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.bar(labels, bins=len(np.unique(labels)))
plt.xlabel("Cluster Label")
plt.ylabel("Number of Data Points")
plt.title("Distribution of Clusters")
plt.show()

# ----------------------------------------------------------------
# Plot the clusters K = 3
# ----------------------------------------------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

scatter = ax.scatter(
    X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels_3, cmap="viridis", marker="o"
)
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")
ax.set_title("3D Scatter Plot of Clusters in PCA Space")

# Add a colorbar to show the mapping of labels to colors
colorbar = plt.colorbar(scatter, ax=ax)
colorbar.set_label("Cluster Labels")
plt.savefig("../../reports/figures/Part_2/KMeans_clusters_pca_labels_3.png")
plt.show()
