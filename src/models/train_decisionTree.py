import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
import time
import sys

sys.path.insert(0, "../../../Data-Mining-Project")
from models.decisionTree import DecisionTree
from src.utils import split_data, compute_metrics, plot_confusion_matrix

# ----------------------------------------------------------------#
# Load data
# ----------------------------------------------------------------#

df = pd.read_csv("../../data/interim/03_static_dataset_features_built.csv", index_col=0)

# ----------------------------------------------------------------#
# Split Data
# ----------------------------------------------------------------#

X_train, X_test, y_train, y_test = split_data(df)

# ----------------------------------------------------------------#
# Our Decision Tree
# ----------------------------------------------------------------#

tree = DecisionTree(max_depth=10, min_samples_split=2)
start_time = time.time()
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
end_time = time.time()
DT_exec_time = end_time - start_time

compute_metrics(y_test, y_pred)
print("Execution Time: ", DT_exec_time)

cm = confusion_matrix(y_test, y_pred)

plot_confusion_matrix(cm)


# ----------------------------------------------------------------#
# SKLearn Decision Tree
# ----------------------------------------------------------------#
SKtree = DecisionTreeClassifier()

SKtree.fit(X_train, y_train)
y_pred = SKtree.predict(X_test)
print("SK_Accuracy:", (y_pred == y_test).mean())
