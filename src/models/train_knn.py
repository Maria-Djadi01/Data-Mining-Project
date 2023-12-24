import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sys

sys.path.insert(0, "../../../Data-Mining-Project")
from models.knn import KNNClassifier
from src.utils import split_data

# ----------------------------------------------------------------#
# Load data
# ----------------------------------------------------------------#

df = pd.read_csv("../../data/interim/03_static_dataset_features_built.csv", index_col=0)

# ----------------------------------------------------------------#
# Split Data
# ----------------------------------------------------------------#

X_train, X_test, y_train, y_test = split_data(df)

# ----------------------------------------------------------------#
# Our KNN
# ----------------------------------------------------------------#

test_sample = [
    0.123,
    0.567,
    0.890,
    0.432,
    0.765,
    0.321,
    0.654,
    0.987,
    0.234,
    0.876,
    0.543,
    0.109,
]

knn_3 = KNNClassifier(k=3, distance_metric="euclidean")

result_k3 = knn_3.predict(test_sample, df)

# KNN avec k=5 et distance euclidienne
knn_5 = KNNClassifier(k=5, distance_metric="euclidean")

# Prédiction de la classe
result_k5 = knn_5.predict(test_sample, df)

print("Prédiction avec K=3:", result_k3)
print("Prédiction avec K=5:", result_k5)

# ----------------------------------------------------------------#
# SKLearn KNN
# ----------------------------------------------------------------#
SKforest = RandomForestClassifier()

SKforest.fit(X_train, y_train)
y_pred = SKforest.predict(X_test)
print("SK_Accuracy:", (y_pred == y_test).mean())
