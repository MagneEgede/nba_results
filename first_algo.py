import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import warnings
import random

warnings.filterwarnings("ignore")
random.seed(10)


df = pd.read_csv("nba_games.csv", index_col=0)
df = df.sort_values("date")
df = df.reset_index(drop=True)

del df["mp.1"]
del df["mp_opp.1"]
del df["index_opp"]

def add_target(group):
    group["target"] = group["won"].shift(-1)
    return group

df = df.groupby("team", group_keys=False).apply(add_target)


removed_columns = ["season", "date", "won", "target", "team", "team_opp"]
selected_columns = df.columns[~df.columns.isin(removed_columns)]

df["target"][pd.isnull(df["target"])] = 2
df["target"] = df["target"].astype(int, errors="ignore")

# Assuming 'df' is your DataFrame and 'selected_columns' contains the features
X = df[selected_columns]
y = df['target']

# Create an instance of TimeSeriesSplit with the desired number of splits
n_splits = 5  # Adjust the number of splits as needed
random_seed = 42
tscv = TimeSeriesSplit(n_splits=n_splits)

# Create an XGBoost classifier instance
# Create an XGBoost classifier instance with the random seed
bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic', random_state=random_seed)

# Initialize an empty list to store accuracy scores
accuracy_scores = []

# Perform time series cross-validation
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Fit the model on the training data
    bst.fit(X_train, y_train)

    # Make predictions on the test data
    preds = bst.predict(X_test)

    # Compute accuracy and store it in the list
    accuracy = accuracy_score(y_test, preds)
    accuracy_scores.append(accuracy)

# Print the average accuracy across all folds
print(f'Average Accuracy: {sum(accuracy_scores) / n_splits:.4f}')

