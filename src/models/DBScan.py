import numpy as np

class DBScan():
    def __init__(self, eps=1, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.clusters = []
        self.visited = []
        self.noise = []
        self.X = None
        self.labels = None
        self.n_clusters = None
        self.core_samples = None
        self.adjacency_matrix = None
        self.distance_matrix = None
        self.n_samples = None
        self.n_features = None
        self.n_clusters = None
        self.n_noise = None
        self.cluster_labels = None
        self.core_samples_indices = None
        self.components = None
        self.components_indices = None
        
    def fit(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape
        self.n_clusters = 0
        self.n_noise = 0
        self.cluster_labels = [0] * self.n_samples
        self.core_samples_indices = []
        self.components = []
        self.components_indices = []
        self._get_adjacency_matrix()
        self._get_distance_matrix()
        self._get_core_samples()
        self._get_components()
        self._get_clusters()
        self._get_noise()
        self._get_labels()
        
    def _get_adjacency_matrix(self):
        self.adjacency_matrix = np.zeros((self.n_samples, self.n_samples))
        for i in range(self.n_samples):
            for j in range(self.n_samples):
                if self._euclidean_distance(self.X[i], self.X[j]) <= self.eps:
                    self.adjacency_matrix[i][j] = 1
                    self.adjacency_matrix[j][i] = 1
                    
    def _get_distance_matrix(self):
        self.distance_matrix = np.zeros((self.n_samples, self.n_samples))
        for i in range(self.n_samples):
            for j in range(self.n_samples):
                self.distance_matrix[i][j] = self._euclidean_distance(self.X[i], self.X[j])
                
    def _get_core_samples(self):
        for i in range(self.n_samples):
            if np.sum(self.adjacency_matrix[i]) >= self.min_samples:
                self.core_samples_indices.append(i)
                
    def _get_components(self):
        for i in range(self.n_samples):
            if i in self.core_samples_indices:
                self.components.append(self.X[i])
                self.components_indices.append(i)
            else:
                for j in self.core_samples_indices:
                    if self.adjacency_matrix[i][j] == 1:
                        self.components.append(self.X[i])
                        self.components_indices.append(i)
                        break
                    
    def _get_clusters(self):
        self.clusters = []
        self.visited = []
        for i in range(self.n_samples):
            if i in self.core_samples_indices and i not in self.visited:
                self.clusters.append([])
                self._expand_cluster(i, self.clusters[-1])
                
    def _expand_cluster(self, i, cluster):
        self.visited.append(i)
        cluster.append(i)
        for j in range(self.n_samples):
            if self.adjacency_matrix[i][j] == 1:
                if j not in self.visited:
                    self._expand_cluster(j, cluster)
                elif j not in self.core_samples_indices and j not in cluster:
                    cluster.append(j)
                    
    def _get_noise(self):
        self.noise = []
        for i in range(self.n_samples):
            if i not in self.core_samples_indices:
                self.noise.append(i)
                
    def _get_labels(self):
        for i, cluster in enumerate(self.clusters):
            for j in cluster:
                self.cluster_labels[j] = i
        for i in self.noise:
            self.cluster_labels[i] = -1
            
    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))
    
    def print(self):
        print("adjacency_matrix: ", self.adjacency_matrix)
        print("distance_matrix: ", self.distance_matrix)
        print("core_samples_indices: ", self.core_samples_indices)
        print("components_indices: ", self.components_indices)
        print("clusters: ", self.clusters)
        print("noise: ", self.noise)
        print("cluster_labels: ", self.cluster_labels)
        print("n_clusters: ", self.n_clusters)
        print("n_noise: ", self.n_noise)
        print("components: ", self.components)
        print("labels: ", self.labels)

if __name__ == "__main__":
    # Test DBScan
    import pandas as pd
    import numpy as np
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt
    
    X, y = make_blobs(n_samples=1000, centers=3, n_features=2, random_state=0)
    plt.scatter(X[:,0], X[:,1])
    plt.show()
    
    dbscan = DBScan(eps=0.5, min_samples=5)
    dbscan.fit(X)
    dbscan.print()
    
    plt.scatter(X[:,0], X[:,1], c=dbscan.cluster_labels)
    plt.show()
    
    