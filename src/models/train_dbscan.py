import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# project_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.insert(0, project_directory)

sys.path.insert(0, r"C:\Users\HI\My-Github\Data-Mining-Project")
from models.DBScan import DBScan
from src.utils import (
    split_data,
    compute_metrics,
    plot_confusion_matrix,
    silhouette_score,
)

# ----------------------------------------------------------------
# Load data
# ----------------------------------------------------------------
df = pd.read_csv("../../data/interim/03_static_dataset_features_built.csv", index_col=0)

# ----------------------------------------------------------------
# Data
# ----------------------------------------------------------------
X, y = df.drop(columns=["Fertility"]).values, df["Fertility"].values

# ----------------------------------------------------------------
# Hyperparameters tuning for dbscan
# ----------------------------------------------------------------

eps_range = np.arange(0.1, 1.0, 0.1)
min_samples_range = range(2, 10)
silhouette_scores = []

for eps in eps_range:
    for min_samples in min_samples_range:
        dbscan = DBScan(eps=eps, min_samples=min_samples)
        dbscan.fit(X)
        labels = dbscan.cluster_labels
        silhouette_avg = silhouette_score(X, labels)
        silhouette_scores.append(silhouette_avg)
        print(f"silhouette score for eps={eps:.1f} and min_samples={min_samples}: {silhouette_avg}")

# Reshape the silhouette_scores to match the shape of the heatmap
silhouette_scores = np.array(silhouette_scores).reshape(
    len(eps_range), len(min_samples_range)
)

# Create a heatmap
plt.figure(figsize=(10, 6))
eps_labels = [f"{eps:.1f}" for eps in eps_range]
min_samples_labels = [str(min_samples) for min_samples in min_samples_range]
plt.imshow(
    silhouette_scores,
    cmap="viridis",
    origin="lower",
    extent=[
        min_samples_range[0] - 0.5,
        min_samples_range[-1] + 0.5,
        eps_range[0] - 0.05,
        eps_range[-1] + 0.05,
    ],
)
plt.colorbar(label="Silhouette Score")
plt.xticks(min_samples_range, min_samples_labels)
plt.yticks(eps_range, eps_labels)
plt.xlabel("Min Samples")
plt.ylabel("Eps")
plt.title("Silhouette Score Heatmap for Different Eps and Min Samples")

# make the hitmap wider in the y axis
plt.gca().set_aspect("auto")
# Find the hyperparameters that give the maximum silhouette score in the plot
max_idx = np.unravel_index(
    np.argmax(silhouette_scores, axis=None), silhouette_scores.shape
)
plt.scatter(
    min_samples_range[max_idx[1]],
    eps_range[max_idx[0]],
    marker="*",
    color="red",
    label=f"Maximum Silhouette Score: eps={eps_range[max_idx[0]]:.1f}, min_samples={min_samples_range[max_idx[1]]}",
)
# display the best hyperparameters in the plot
plt.legend()
plt.savefig("../../reports/figures/Part_2/DBScan_hyperparameters_tuning.png")
plt.show()
# save the plot
print(
    f"The following hyperparameters give the maximum silhouette score: eps={eps_range[max_idx[0]]:.1f}, min_samples={min_samples_range[max_idx[1]]}"
)

# print the number of clusters
print(f"Number of clusters: {len(np.unique(dbscan.cluster_labels))}")

# ----------------------------------------------------------------
# Train DBscan with the best hyperparameters
# ----------------------------------------------------------------
eps = 0.9
min_samples = 2
dbscan = DBScan(eps=eps, min_samples=min_samples)
dbscan.fit(X, plot_steps=True)
labels = dbscan.cluster_labels
# print the number of clusters
print(f"Number of clusters: {len(np.unique(labels))}")

# compute the silhouette score
silhouette_avg = silhouette_score(X, labels)

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
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels, cmap='viridis', marker='o')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('DBScan Clustering')

# Add a colorbar to show the mapping of labels to colors
colorbar = plt.colorbar(scatter, ax=ax)
colorbar.set_label('Cluster Labels')
plt.savefig("../../reports/figures/Part_2/DBScan_clusters_pca.png")
plt.show()

features_names = df.drop(columns=["Fertility"]).columns

for i, pc in enumerate(pca.components_):
    top_features_indices = pc.argsort()[-5:][::-1]  # Adjust the number of top features to display
    top_features = [features_names[index] for index in top_features_indices]
    print(f"Top features for Principal Component {i+1}: {top_features}")

# ----------------------------------------------------------------
# Visualize the distribution of the clusters
# ----------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.hist(labels, bins=len(np.unique(labels)))
plt.xlabel('Cluster Label')
plt.ylabel('Number of Data Points')
plt.title('Distribution of Clusters')
plt.show()

# ----------------------------------------------------------------
# DBScan with SKlearn
# ----------------------------------------------------------------
from sklearn.cluster import DBSCAN

dbscan_sk = DBSCAN(eps=0.9, min_samples=2)
dbscan_sk.fit(X)
labels_sk = dbscan_sk.labels_
# print the number of clusters
print(f"Number of clusters: {len(np.unique(labels))}")

# plot the clusters
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels_sk, cmap='viridis', marker='o')
ax.set_xlabel('Principal Component 1')