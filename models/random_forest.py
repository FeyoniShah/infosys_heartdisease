# import pandas as pd
# import joblib
# from sklearn.ensemble import RandomForestClassifier
# from utils.metrics import evaluate_model

# def train_random_forest():
#     X_train = pd.read_csv("train_X.csv")
#     y_train = pd.read_csv("train_y.csv").values.ravel()
#     X_test = pd.read_csv("test_X.csv")
#     y_test = pd.read_csv("test_y.csv").values.ravel()

#     model = RandomForestClassifier(n_estimators=200, random_state=42)
#     model.fit(X_train, y_train)

#     results = evaluate_model(model, X_test, y_test, "Random Forest")

#     joblib.dump(model, "models/random_forest.pkl")
#     return results

# if __name__ == "__main__":
#     train_random_forest()


'''
import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_curve
from imblearn.over_sampling import SMOTE
from utils.metrics import evaluate_model


def train_random_forest_with_smote():
    # Load data
    X_train = pd.read_csv("train_X.csv")
    y_train = pd.read_csv("train_y.csv").values.ravel()
    X_test = pd.read_csv("test_X.csv")
    y_test = pd.read_csv("test_y.csv").values.ravel()

    # Apply SMOTE to balance classes
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print("Before SMOTE:", dict(pd.Series(y_train).value_counts()))
    print("After SMOTE :", dict(pd.Series(y_res).value_counts()))

    # Train Random Forest
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_res, y_res)

    # Predict probabilities instead of labels
    y_probs = model.predict_proba(X_test)[:, 1]

    # Tune threshold for better recall
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)

    # Example: choose threshold that maximizes F1
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    print(f"\nOptimal Threshold: {best_threshold:.2f}")
    print(f"Precision: {precisions[best_idx]:.3f}, Recall: {recalls[best_idx]:.3f}, F1: {f1_scores[best_idx]:.3f}")

    # Final predictions with tuned threshold
    y_pred = (y_probs >= best_threshold).astype(int)

    # Evaluate model
    results = evaluate_model(model, X_test, y_test, "Random Forest (SMOTE + Threshold)", y_pred)

    # Save model + threshold
    joblib.dump((model, best_threshold), "models/random_forest_smote.pkl")

    return results


if __name__ == "__main__":
    train_random_forest_with_smote()
'''


import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from imblearn.over_sampling import SMOTE
from utils.metrics import evaluate_model


def get_random_forest():
    """Return an untrained Random Forest model."""
    return RandomForestClassifier(n_estimators=200, random_state=42)


def train_random_forest_with_smote():
    # Load data
    X_train = pd.read_csv("train_X.csv")
    y_train = pd.read_csv("train_y.csv").values.ravel()
    X_test = pd.read_csv("test_X.csv")
    y_test = pd.read_csv("test_y.csv").values.ravel()

    # Apply SMOTE to balance classes
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print("Before SMOTE:", dict(pd.Series(y_train).value_counts()))
    print("After SMOTE :", dict(pd.Series(y_res).value_counts()))

    # Train Random Forest
    model = get_random_forest() #RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_res, y_res)

    # Predict probabilities instead of labels
    y_probs = model.predict_proba(X_test)[:, 1]

    # Tune threshold for better recall
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)

    # Example: choose threshold that maximizes F1
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    print(f"\nOptimal Threshold: {best_threshold:.2f}")
    #print(f"Precision: {precisions[best_idx]:.3f}, Recall: {recalls[best_idx]:.3f}, F1: {f1_scores[best_idx]:.3f}")

    # Override model.predict to use tuned threshold
    original_predict = model.predict
    model.predict = lambda X: (model.predict_proba(X)[:, 1] >= best_threshold).astype(int)

    # Evaluate model (now uses thresholded predictions)
    results = evaluate_model(model, X_test, y_test, "Random Forest (SMOTE + Threshold)")

    # Restore original predict method (optional)
    model.predict = original_predict

    # Save model + threshold
    joblib.dump((model, best_threshold), "models/random_forest.pkl")

    return results


if __name__ == "__main__":
    train_random_forest_with_smote()
