import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix, roc_curve, auc
import time
import sys

sys.path.insert(0, "../../../Data-Mining-Project")
# from models.knn import KNNClassifier
from src.models.KNN import KNN
from src.utils import split_data, compute_metrics, plot_confusion_matrix

# ----------------------------------------------------------------#
# Load data
# ----------------------------------------------------------------#

df = pd.read_csv("../../data/interim/03_static_dataset_features_built.csv", index_col=0)

# ----------------------------------------------------------------#
# Split Data
# ----------------------------------------------------------------#

X_train, X_test, y_train, y_test = split_data(df)
X_train.shape, y_train.shape
# ----------------------------------------------------------------#
# Undersampling
# ----------------------------------------------------------------#
import numpy as np

desired_num_samples = 34
sampling_strategy_dict = {
    class_label: desired_num_samples
    for class_label, desired_num_samples in zip(*np.unique(y_train, return_counts=True))
}

# Apply random undersampling to the training data
undersampler = RandomUnderSampler(
    sampling_strategy=sampling_strategy_dict, random_state=42
)
X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)
X_resampled.shape, y_resampled.shape

# ----------------------------------------------------------------#
# Our KNN
# ----------------------------------------------------------------#

knn_3 = KNN(k=20)
# knn_3 = KNNClassifier(k=3, distance_metric="euclidean")


start_time = time.time()
knn_3.fit(X_resampled, y_resampled)
# knn_3.fit(X_train, y_train)
y_pred = knn_3.predict(X_test)
end_time = time.time()
RF_exec_time = end_time - start_time

compute_metrics(y_test, y_pred)
print("Execution Time: ", RF_exec_time)

cm = confusion_matrix(y_test, y_pred)

plot_confusion_matrix(cm)


result_k3 = knn_3.predict(X_test)

# KNN avec k=5 et distance euclidienne
knn_5 = KNN(k=5, distance_metric="euclidean")

# Prédiction de la classe
result_k5 = knn_5.predict(X_test, df)

print("Prédiction avec K=3:", result_k3)
print("Prédiction avec K=5:", result_k5)

# ----------------------------------------------------------------#
# SKLearn KNN
# ----------------------------------------------------------------#
SKforest = RandomForestClassifier()

SKforest.fit(X_train, y_train)
y_pred = SKforest.predict(X_test)
print("SK_Accuracy:", (y_pred == y_test).mean())
