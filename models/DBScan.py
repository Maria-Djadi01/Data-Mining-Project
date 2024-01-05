import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA

class DBScan:
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples
        self.cluster_labels = None
        self.n_samples = None
        self.n_features = None
        self.silhouette_scores = None

    def fit(self, X, plot_steps=False):
        self.plot_steps = plot_steps
        self.X = X
        self.n_samples, self.n_features = X.shape
        # X_pca to be used for plotting
        if X.shape[1] > 3:
            pca = PCA(n_components=3)
            self.X_pca = pca.fit_transform(X)
        self.cluster_labels = self._dbscan()
        

    def _dbscan(self):
        cluster_labels = np.zeros(self.n_samples, dtype=int)
        cluster_idx = 1
        
        for i in range(self.n_samples):
            if cluster_labels[i] == 0:
                if self._expand_cluster(cluster_labels, i, cluster_idx):
                    cluster_idx += 1

        return cluster_labels

    def _expand_cluster(self, cluster_labels, sample_idx, cluster_idx):
        if not self._is_core_sample(sample_idx):
            cluster_labels[sample_idx] = -1  # Mark as noise
            return False

        cluster_labels[sample_idx] = cluster_idx
        queue = [sample_idx]

        i = 0
        while queue:
            current_idx = queue.pop(0)
            neighbor_indices = self._get_neighbors(current_idx)
            for neighbor_idx in neighbor_indices:
                if cluster_labels[neighbor_idx] == 0:
                    cluster_labels[neighbor_idx] = cluster_idx
                    if self._is_core_sample(neighbor_idx):
                        queue.append(neighbor_idx)
                elif cluster_labels[neighbor_idx] == -1:
                    cluster_labels[neighbor_idx] = cluster_idx  # Reassign noise to the cluster
            # plot clusters
            if i % 20 == 0 and self.plot_steps:
                self._plot_clusters(cluster_labels, i)
            i += 1
        return True

    def _is_core_sample(self, sample_idx):
        return len(self._get_neighbors(sample_idx)) >= self.min_samples

    def _get_neighbors(self, sample_idx):
        return np.where(self._get_distances(sample_idx) <= self.eps)[0]

    def _get_distances(self, sample_idx):
        return np.sqrt(np.sum((self.X - self.X[sample_idx]) ** 2, axis=1))

    def _plot_clusters(self, cluster_labels, iteration):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        # plot the clusters
        unique_labels = np.unique(cluster_labels)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        for cluster_idx, cluster in enumerate(unique_labels):
            cluster_points = self.X_pca[cluster_labels == cluster]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], color=colors[cluster_idx])
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        ax.set_title(f"Iteration {iteration}")
        plt.show()

    def print(self):
        print("cluster_labels: ", self.cluster_labels)


if __name__ == "__main__":
    # Test the DBScan class
    from sklearn import datasets
    from sklearn.metrics import silhouette_samples
    from sklearn.cluster import DBSCAN
    
    # Load the data
    X, y = datasets.make_moons(n_samples=1000, noise=0.05)
    
    # Our model
    our_dbscan = DBScan(0.3, 5)
    our_dbscan.fit(X)
    our_dbscan.print()
    our_labels = our_dbscan.cluster_labels
    # print("Our Silhouette Score: ", np.mean(our_dbscan.silhouette_scores))
