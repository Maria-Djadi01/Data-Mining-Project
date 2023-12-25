import numpy as np

class KMeans():
    def __init__(self,):
        self.centroids = None
        self.clusters = None
        self.labels = None
        self.inertia = None

    def fit(self, X, n_clusters, max_iter=100, distance='euclidean'):
        self.distance = distance
        X = np.array(X)  # Convert X to a numpy array
        self.centroids = self._init_centroids(X, n_clusters)
        for _ in range(max_iter):
            self.clusters = self._create_clusters(X, self.centroids)
            previous_centroids = self.centroids
            self.centroids = self._get_centroids(self.clusters, X)
            if self._is_converged(previous_centroids, self.centroids):
                break
        self.labels = self._get_cluster_labels(self.clusters)
        self.inertia = self._get_inertia(self.clusters, self.centroids, X)

    def predict(self, X):
        X = np.array(X)  # Convert X to a numpy array
        clusters = self._create_clusters(X, self.centroids)
        return self._get_cluster_labels(clusters)
    
    def _init_centroids(self, X, n_clusters):
        n_samples, n_features = X.shape
        centroids = np.zeros((n_clusters, n_features))
        for k in range(n_clusters):
            centroid = X[np.random.choice(range(n_samples))]
            centroids[k] = centroid
        return centroids
    
    def _create_clusters(self, X, centroids):
        n_samples = X.shape[0]
        clusters = [[] for _ in range(len(centroids))]
        for idx, sample in enumerate(X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters
    
    def _closest_centroid(self, sample, centroids):
        if self.distance == 'euclidean':
            distances = [self._euclidean_distance(sample, point) for point in centroids]
        elif self.distance == 'manhattan':
            distances = [self._manhattan_distance(sample, point) for point in centroids]
        else:
            self.distances = [self._minowski_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx
    
    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))
    
    def _manhattan_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2))
    
    def _minowski_distance(self, x1, x2):
        return self._euclidean_distance(x1, x2)**3
    
    def _get_centroids(self, clusters, X):
        n_features = X.shape[1]
        centroids = np.zeros((len(clusters), n_features))
        for idx, cluster in enumerate(clusters):
            centroid = np.mean(X[cluster], axis=0)
            centroids[idx] = centroid
        return centroids
    
    def _is_converged(self, previous_centroids, centroids):
        distances = [self._euclidean_distance(centroids[i], previous_centroids[i]) for i in range(len(centroids))]
        return sum(distances) == 0
    
    def _get_cluster_labels(self, clusters):
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
    
    k = KMeans()
    k.fit(X, 3, distance='euclidean')
    k.print()
    
    plt.scatter(X[:, 0], X[:, 1], c=k.labels)
    plt.scatter(k.centroids[:, 0], k.centroids[:, 1], c='red')
    plt.show()
