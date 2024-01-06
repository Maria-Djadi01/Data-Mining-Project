import numpy as np
from statistics import mode


class Node:
    def __init__(self, feature, threshold, left, right, value):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def split_data(self, X, y, feature, threshold):
        left_index = X[:, feature] <= threshold
        right_index = X[:, feature] > threshold
        return X[left_index], X[right_index], y[left_index], y[right_index]

    def entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        prob = counts / counts.sum()
        return sum(prob * -np.log2(prob))

    def information_gain(self, parent, left_child, right_child):
        p = len(left_child) / len(parent)
        return (
            self.entropy(parent)
            - p * self.entropy(left_child)
            - (1 - p) * self.entropy(right_child)
        )

    def best_split(self, X, y):
        best_feature, best_threshold, best_gain = None, None, 0
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_X, right_X, left_y, right_y = self.split_data(
                    X, y, feature, threshold
                )
                gain = self.information_gain(y, left_y, right_y)
                if gain > best_gain:
                    best_feature, best_threshold, best_gain = (
                        feature,
                        threshold,
                        gain,
                    )
        return best_feature, best_threshold, best_gain

    def calculate_leaf_value(self, y):
        return mode(y)

    def build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or n_labels == 1
        ):
            leaf_value = self.calculate_leaf_value(y)
            return Node(None, None, None, None, leaf_value)
        else:
            depth += 1
            best_feature, best_threshold, best_gain = self.best_split(X, y)
            left_X, right_X, left_y, right_y = self.split_data(
                X, y, best_feature, best_threshold
            )
            left = self.build_tree(left_X, left_y, depth)
            right = self.build_tree(right_X, right_y, depth)
            return Node(best_feature, best_threshold, left, right, None)

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def predict(self, X):
        return [self.traverse_tree(x, self.tree) for x in X]

    def traverse_tree(self, x, node):
        if node.value != None:
            return node.value
        else:
            if x[node.feature] <= node.threshold:
                return self.traverse_tree(x, node.left)
            else:
                return self.traverse_tree(x, node.right)
