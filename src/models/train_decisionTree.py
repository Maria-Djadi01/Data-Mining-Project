import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import sys

sys.path.insert(0, "../../../Data-Mining-Project")
from models.decisionTree import DecisionTree

# ----------------------------------------------------------------#
# Load data
# ----------------------------------------------------------------#

df = pd.read_csv("../../data/interim/03_static_dataset_features_built.csv", index_col=0)

# ----------------------------------------------------------------#
# Split Data
# ----------------------------------------------------------------#

X = df.drop(columns=["Fertility"]).values
y = df["Fertility"].values


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=40
)

tree = DecisionTree(max_depth=10, min_samples_split=2)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
print("Accuracy:", (y_pred == y_test).mean())

import numpy as np


def check_class_balance(y):
    unique_classes, class_counts = np.unique(y, return_counts=True)

    for cls, count in zip(unique_classes, class_counts):
        print(f"Class {cls}: {count} instances")

    # Check if the counts are roughly equal
    is_balanced = np.all(np.isclose(class_counts, np.mean(class_counts)))

    if is_balanced:
        print("The dataset is balanced.")
    else:
        print("The dataset is imbalanced.")


check_class_balance(y)


# ----------------------------------------------------------------#
# SKLearn Decision Tree
# ----------------------------------------------------------------#
SKtree = DecisionTreeClassifier()

SKtree.fit(X_train, y_train)
y_pred = SKtree.predict(X_test)
print("SK_Accuracy:", (y_pred == y_test).mean())
