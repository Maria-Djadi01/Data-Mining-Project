from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt


class KNN:
    def __init__(self, k):
        self.k = k
        self.X = None
        self.y = None
        self.n_samples = None
        self.n_features = None
        self.n_classes = None
        self.distances = None
        self.k_nearest_neighbors = None
        self.predictions = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.n_samples, self.n_features = X.shape
        self.n_classes = len(np.unique(y))

    def predict(self, X, visualize=True):
        self.distances = self._get_distances(X)
        self.k_nearest_neighbors = self._get_k_nearest_neighbors()
        self.predictions = self._get_predictions()
        if visualize:
            self._plot_iterations_visualization(X)

        return self.predictions.astype(int)

    def visualize_steps(self, X, nb_plots=10):
        fig, axes = plt.subplots(2, nb_plots // 2, figsize=(15, 10))

        for i in range(nb_plots):
            neighbors = self.k_nearest_neighbors[i].astype(int)
            ax = axes[i // (nb_plots // 2), i % (nb_plots // 2)]

            ax.scatter(
                self.X[:, 0],
                self.X[:, 1],
                c=self.y,
                cmap="viridis",
                label="Training Data",
            )
            ax.scatter(X[i, 0], X[i, 1], marker="*", s=200, c="red", label="Test Point")

            for neighbor in neighbors:
                ax.scatter(
                    self.X[neighbor, 0],
                    self.X[neighbor, 1],
                    marker="o",
                    s=100,
                    edgecolors="k",
                    facecolors="none",
                    label="Neighbor",
                )

            ax.set_title(
                f"Test Point {i + 1}, Predicted Class: {int(self.predictions[i])}"
            )
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")
            ax.legend()

        plt.tight_layout()
        plt.show()

    def _get_distances(self, X):
        distances = np.zeros((len(X), self.n_samples))
        for i, sample in enumerate(X):
            for j, x in enumerate(self.X):
                distances[i][j] = self._euclidean_distance(sample, x)
        return distances

    def _get_k_nearest_neighbors(self):
        k_nearest_neighbors = np.zeros((len(self.distances), self.k))
        for i, distance in enumerate(self.distances):
            k_nearest_neighbors[i] = np.argsort(distance)[: self.k]
        return k_nearest_neighbors

    def _get_predictions(self):
        predictions = np.zeros(len(self.k_nearest_neighbors))
        for i, k_nearest_neighbors in enumerate(self.k_nearest_neighbors):
            k_nearest_neighbors = self.y[k_nearest_neighbors.astype(int)]
            predictions[i] = np.argmax(np.bincount(k_nearest_neighbors.astype(int)))
        return predictions

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _manhattan_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2))

    def _minkowski_distance(self, x1, x2):
        return self._minowski_distance(x1, x2) ** 3

    def _plot_iterations_visualization(self, X):
        cmap = ListedColormap(
            ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF"]
        )
        for i, sample in enumerate(X):
            if len(sample.shape) == 0:
                sample = np.array([sample])  # Convert scalar to 1D array
            sample = np.asarray(sample).reshape(1, -1)  # Ensure sample is a 2D array

            plt.figure(figsize=(15, 5))

            # Plot all points
            plt.subplot(131)
            plt.scatter(
                self.X[:, 0], self.X[:, 1], c=self.y, cmap=cmap, edgecolors="k", s=50
            )
            plt.scatter(
                sample[:, 0],
                sample[:, 1],
                c="red",
                marker="x",
                s=100,
                label="Test Sample",
            )
            plt.title("All Points")

            # Plot the distances to all other points
            plt.subplot(132)
            plt.scatter(
                self.X[:, 0],
                self.X[:, 1],
                c=self.distances[i],
                cmap=cmap,
                edgecolors="k",
                s=50,
            )
            plt.title("Distances")

            # Plot the k-nearest neighbors
            plt.subplot(133)
            neighbors = self.k_nearest_neighbors[i]
            plt.scatter(
                self.X[:, 0], self.X[:, 1], c=self.y, cmap=cmap, edgecolors="k", s=50
            )
            plt.scatter(
                self.X[neighbors, 0],
                self.X[neighbors, 1],
                facecolors="none",
                edgecolors="black",
                s=150,
                label="Nearest Neighbors",
            )
            plt.scatter(
                sample[:, 0],
                sample[:, 1],
                c="red",
                marker="x",
                s=100,
                label="Test Sample",
            )
            plt.title("Nearest Neighbors")

            plt.tight_layout()
            plt.suptitle(f"Sample {i+1} Prediction Visualization")
            plt.show()


if __name__ == "__main__":
    # Test the KNN class
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X, y = datasets.make_classification(
        n_samples=1000, n_features=10, n_classes=2, random_state=123
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    clf = KNN(k=5)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, predictions))

    # Add the visualization step
    clf.visualize_steps_3d(X_test, nb_plots=4)
