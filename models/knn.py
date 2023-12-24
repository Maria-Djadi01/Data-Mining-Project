import numpy as np
import pandas as pd


class KNNClassifier:
    def __init__(self, k, distance_metric="euclidean"):
        self.k = k
        self.distance_metric = distance_metric

    def calculate_distance(self, instance1, instance2):
        instance1 = np.array(instance1)
        instance2 = np.array(instance2)

        if self.distance_metric == "manhattan":
            return np.sum(np.abs(instance1 - instance2))
        elif self.distance_metric == "euclidean":
            return np.sqrt(np.sum((instance1 - instance2) ** 2))
        elif self.distance_metric == "minkowski":
            p = 3
            return np.power(np.sum(np.power(np.abs(instance1 - instance2), p)), 1 / p)
        elif self.distance_metric == "cosine":
            dot_product = np.dot(instance1, instance2)
            norm_instance1 = np.linalg.norm(instance1)
            norm_instance2 = np.linalg.norm(instance2)
            return 1 - (dot_product / (norm_instance1 * norm_instance2))
        elif self.distance_metric == "hamming":
            return np.sum(instance1 != instance2) / len(instance1)
        else:
            raise ValueError("Invalid distance metric")

    def sort_instances_by_distance(self, test_instance, dataset):
        distances = [
            (self.calculate_distance(test_instance, row[:]), row["label"])
            for _, row in dataset.iterrows()
        ]
        distances.sort(key=lambda x: x[0])
        return distances

    def get_majority_class(self, distances):
        k_nearest = distances[: self.k]
        classes = [cls for (_, cls) in k_nearest]
        unique_classes, counts = np.unique(classes, return_counts=True)
        index = np.argmax(counts)
        return unique_classes[index]

    def predict(self, test_instance, dataset):
        sorted_distances = self.sort_instances_by_distance(test_instance, dataset)
        predicted_class = self.get_majority_class(sorted_distances)
        return predicted_class


# import numpy as np
# import pandas as pd
# from collections import Counter


# class KNNClassifier:
#     def __init__(self, k=3, distance_metric="euclidean"):
#         self.k = k
#         self.distance_metric = distance_metric

#     def calculate_distance(self, instance1, instance2):
#         instance1 = np.array(instance1)
#         instance2 = np.array(instance2)

#         if self.distance_metric == "euclidean":
#             return np.sqrt(np.sum((instance1 - instance2) ** 2))
#         elif self.distance_metric == "manhattan":
#             return np.sum(np.abs(instance1 - instance2))
#         else:
#             raise ValueError("Invalid distance metric")

#     def predict(self, test_instance, dataset):
#         distances = [
#             (self.calculate_distance(test_instance, row[:-1]), row["label"])
#             for _, row in dataset.iterrows()
#         ]
#         distances.sort(key=lambda x: x[0])
#         k_nearest = distances[: self.k]
#         classes = [cls for (_, cls) in k_nearest]
#         majority_class = Counter(classes).most_common(1)[0][0]
#         return majority_class
