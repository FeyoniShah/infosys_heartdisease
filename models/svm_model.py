import pandas as pd
import joblib
from sklearn.svm import SVC
from utils.metrics import evaluate_model

def get_svm():
    """Return an untrained Random Forest model."""
    return SVC(kernel="rbf", probability=True, random_state=42)

def train_svm():
    X_train = pd.read_csv("train_X.csv")
    y_train = pd.read_csv("train_y.csv").values.ravel()
    X_test = pd.read_csv("test_X.csv")
    y_test = pd.read_csv("test_y.csv").values.ravel()

    model =get_svm() # SVC(kernel="rbf", probability=True, random_state=42)
    model.fit(X_train, y_train)

    results = evaluate_model(model, X_test, y_test, "SVM (RBF)")

    joblib.dump(model, "models/svm_model.pkl")
    return results

if __name__ == "__main__":
    train_svm()
