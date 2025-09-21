import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from utils.metrics import evaluate_model


def get_logistic_regression():
    """Return an untrained Logistic Regression model (for CV/ensembles)."""
    return LogisticRegression(max_iter=500, random_state=42)

def train_logistic():
    X_train = pd.read_csv("train_X.csv")
    y_train = pd.read_csv("train_y.csv").values.ravel()
    X_test = pd.read_csv("test_X.csv")
    y_test = pd.read_csv("test_y.csv").values.ravel()

    model = get_logistic_regression() #LogisticRegression(max_iter=500, random_state=42)
    model.fit(X_train, y_train)

    results = evaluate_model(model, X_test, y_test, "Logistic Regression")

    joblib.dump(model, "models/logistic_regression.pkl")
    return results

if __name__ == "__main__":
    train_logistic()


'''
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from feature_engineering import run_feature_pipeline

def train_logistic():
    # Load raw data
    df = pd.read_csv("framingham.csv")

    # Run feature pipeline
    df_encoded, _ = run_feature_pipeline(df)

    # Split features & target
    X = df_encoded.drop("TenYearCHD", axis=1)  # replace with actual target column
    y = df_encoded["TenYearCHD"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    print("Training complete!")

if __name__ == "__main__":
    train_logistic()
    '''