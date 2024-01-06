import numpy as np
import sys

sys.path.insert(1, "../../Data-Mining-Project")
from models.decisionTree import DecisionTree


class RandomForest:
    def __init__(self, n_trees, max_depth, min_samples_split):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        self.trees = []
        for i in range(self.n_trees):
            tree = DecisionTree(self.max_depth, self.min_samples_split)
            tree.fit(X, y)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(predictions, axis=0).round().astype(int)

    # def predict_proba(self, X):
    #     predictions = np.array([tree.predict_proba(X) for tree in self.trees])
    #     return np.mean(predictions, axis=0)
