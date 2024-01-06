import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA


class KMeans:
    def __init__(self, k):
        self.k = k
        self.centroids = None
        self.clusters = None
        self.labels = None
        self.inertia = None

    def fit(self, X, max_iter=100, distance="euclidean", plot_steps=False):
        self.distance = distance
        X = np.array(X)

        # X_pca to be used for plotting
        if X.shape[1] > 3:
            self.pca = PCA(n_components=3)
            X_pca = self.pca.fit_transform(X)

        self.centroids = self._init_centroids_kmeans_plus_plus(X, self.k)
        for iteration in range(max_iter):
            self.clusters = self._create_clusters(X, self.centroids)
            previous_centroids = self.centroids
            self.centroids = self._get_centroids(self.clusters, X)
            if self._is_converged(previous_centroids, self.centroids):
                break
            if plot_steps:
                if iteration % 10 == 0:
                    self._plot_iteration(X_pca, iteration)

        self.labels = self._get_cluster_labels(self.clusters, X)
        self.inertia = self._get_inertia(self.clusters, self.centroids, X)

        if plot_steps:
            self._plot_iteration(X, iteration + 1)

    def predict(self, X):
        X = np.array(X)  # Convert X to a numpy array
        clusters = self._create_clusters(X, self.centroids)
        return self._get_cluster_labels(clusters, X)

    def _plot_iteration(self, X_pca, iteration):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # plot the clusters and centroids
        # transform the centroids into pca space
        centroids_pca = self.pca.transform(self.centroids)
        # plot centroids
        ax.scatter(
            centroids_pca[:, 0],
            centroids_pca[:, 1],
            centroids_pca[:, 2],
            c="red",
            marker="x",
            s=100,
            label="Centroids",
        )

        # # plot the clusters
        # for cluster_idx, cluster in enumerate(self.clusters):
        #     cluster_points = X_pca[cluster]
        #     scatter = ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2])

        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_zlabel("Principal Component 3")
        ax.set_title(f"Iteration {iteration}")
        # Add a colorbar to show the mapping of labels to colors
        # colorbar = plt.colorbar(scatter, ax=ax)
        # colorbar.set_label('Cluster Labels')

        plt.show()

    def predict(self, X):
        X = np.array(X)  # Convert X to a numpy array
        clusters = self._create_clusters(X, self.centroids)
        return self._get_cluster_labels(clusters, X)

    def _init_centroids_randomly(self, X, n_clusters):
        n_samples, n_features = X.shape
        centroids = np.zeros((n_clusters, n_features))
        for k in range(n_clusters):
            centroid = X[np.random.choice(range(n_samples))]
            centroids[k] = centroid
        return centroids

    def _init_centroids_kmeans_plus_plus(self, X, n_clusters):
        n_samples, n_features = X.shape
        centroids = np.zeros((n_clusters, n_features))
        centroid = X[np.random.choice(range(n_samples))]
        centroids[0] = centroid
        for k in range(1, n_clusters):
            distances = np.zeros((n_samples, k))
            for i in range(n_samples):
                for j in range(k):
                    distances[i][j] = self._euclidean_distance(X[i], centroids[j])
            min_distances = np.min(distances, axis=1)
            max_distance = np.argmax(min_distances)
            centroids[k] = X[max_distance]
        return centroids

    def _create_clusters(self, X, centroids):
        n_samples = X.shape[0]
        clusters = [[] for _ in range(len(centroids))]
        for idx, sample in enumerate(X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        if self.distance == "euclidean":
            distances = [self._euclidean_distance(sample, point) for point in centroids]
        elif self.distance == "manhattan":
            distances = [self._manhattan_distance(sample, point) for point in centroids]
        else:
            self.distances = [
                self._minowski_distance(sample, point) for point in centroids
            ]
        closest_idx = np.argmin(distances)
        return closest_idx

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _manhattan_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2))

    def _minowski_distance(self, x1, x2):
        return self._euclidean_distance(x1, x2) ** 3

    def _get_centroids(self, clusters, X):
        n_features = X.shape[1]
        centroids = np.zeros((len(clusters), n_features))
        for idx, cluster in enumerate(clusters):
            centroid = np.mean(X[cluster], axis=0)
            centroids[idx] = centroid
        return centroids

    def _is_converged(self, previous_centroids, centroids):
        distances = [
            self._euclidean_distance(centroids[i], previous_centroids[i])
            for i in range(len(centroids))
        ]
        return sum(distances) == 0

    def _get_cluster_labels(self, clusters, X):
        labels = np.empty(X.shape[0])
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels

    def _get_inertia(self, clusters, centroids, X):
        inertia = 0
        for cluster_idx, cluster in enumerate(clusters):
            for sample in X[cluster]:
                inertia += self._euclidean_distance(sample, centroids[cluster_idx])
        return inertia

    def print(self):
        print("centroids: ", self.centroids)
        print("clusters: ", self.clusters)
        print("labels: ", self.labels)
        print("inertia: ", self.inertia)


if __name__ == "__main__":
    # Testing
    from sklearn import datasets
    import matplotlib.pyplot as plt

    X, y = datasets.make_blobs()

    k = KMeans(k=3)
    k.fit(X, distance="euclidean", plot_steps=True)
    k.print()

    plt.scatter(X[:, 0], X[:, 1], c=k.labels)
    plt.scatter(k.centroids[:, 0], k.centroids[:, 1], c="red")
    plt.show()
