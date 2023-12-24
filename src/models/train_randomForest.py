import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sys

sys.path.insert(0, "../../../Data-Mining-Project")
from models.randomForest import RandomForest

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

tree = RandomForest(max_depth=5, min_samples_split=2)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
print("Accuracy:", (y_pred == y_test).mean())

# ----------------------------------------------------------------#
# SKLearn Decision Tree
# ----------------------------------------------------------------#
SKforest = RandomForestClassifier()

SKforest.fit(X_train, y_train)
y_pred = SKforest.predict(X_test)
print("SK_Accuracy:", (y_pred == y_test).mean())
