import pandas as pd
import joblib
from sklearn.neural_network import MLPClassifier
from utils.metrics import evaluate_model

def get_neural_network():
    """Return an untrained Logistic Regression model (for CV/ensembles)."""
    return MLPClassifier(hidden_layer_sizes=(64, 32), 
                          activation="relu", 
                          solver="adam", 
                          max_iter=500, 
                          random_state=42)

def train_neural_network():
    X_train = pd.read_csv("train_X.csv")
    y_train = pd.read_csv("train_y.csv").values.ravel()
    X_test = pd.read_csv("test_X.csv")
    y_test = pd.read_csv("test_y.csv").values.ravel()

    model = get_neural_network()  #MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", max_iter=500, random_state=42)
    model.fit(X_train, y_train)

    results = evaluate_model(model, X_test, y_test, "Neural Network")

    joblib.dump(model, "models/neural_network.pkl")
    return results

if __name__ == "__main__":
    train_neural_network()
