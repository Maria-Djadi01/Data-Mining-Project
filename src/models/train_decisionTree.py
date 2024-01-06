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
X_train.shape, y_train.shape

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


import matplotlib.pyplot as plt

# Step 1: Define a meshgrid
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Step 2: Predict class labels for each point in the meshgrid
Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Step 3: Plot the decision boundaries
plt.contourf(xx, yy, Z, alpha=0.3)

# Step 4: Plot the training data
plt.scatter(
    X_train[:, 0],
    X_train[:, 1],
    c=y_train,
    edgecolors="k",
    marker="o",
    s=50,
    linewidth=1,
    cmap=plt.cm.Paired,
)

# Add labels and title
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Decision Tree Decision Boundaries")

plt.show()

# ----------------------------------------------------------------#
# SKLearn Decision Tree
# ----------------------------------------------------------------#
SKtree = DecisionTreeClassifier()

SKtree.fit(X_train, y_train)
y_pred = SKtree.predict(X_test)
print("SK_Accuracy:", (y_pred == y_test).mean())
