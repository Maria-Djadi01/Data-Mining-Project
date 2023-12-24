import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
import time
import sys

sys.path.insert(0, "../../../Data-Mining-Project")
from models.randomForest import RandomForest
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
# Our Random Forest
# ----------------------------------------------------------------#

forest = RandomForest(max_depth=5, min_samples_split=2)
start_time = time.time()
forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)
end_time = time.time()
RF_exec_time = end_time - start_time

compute_metrics(y_test, y_pred)
print("Execution Time: ", RF_exec_time)

cm = confusion_matrix(y_test, y_pred)

plot_confusion_matrix(cm)

import matplotlib.pyplot as plt


def plot_roc_curve(classifier, X_test, Y_test):
    # Get predicted probabilities for the positive class
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(Y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], "k--")  # Plot the diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.show()


plot_roc_curve(forest, X_test, y_test)

# ----------------------------------------------------------------#
# SKLearn Random Forest
# ----------------------------------------------------------------#
SKforest = RandomForestClassifier()

SKforest.fit(X_train, y_train)
y_pred = SKforest.predict(X_test)
print("SK_Accuracy:", (y_pred == y_test).mean())
