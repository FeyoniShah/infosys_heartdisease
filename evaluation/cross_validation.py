import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate

# Import models from your models/ folder
from models.logistic_regression import get_logistic_regression
from models.random_forest import get_random_forest
from models.svm_model import get_svm
from models.neural_network import get_neural_network


def run_all_cv(X, y, k=5, use_smote=True):
    """
    Run stratified k-fold cross-validation for multiple models.
    Optionally applies SMOTE inside each fold.
    """
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    models = {
        "Logistic Regression": get_logistic_regression(),
        "Random Forest": get_random_forest(),
        "SVM": get_svm(),
        "Neural Network": get_neural_network(),
    }

    results = {}

    for name, model in models.items():
        if use_smote:
            pipeline = Pipeline([
                ("smote", SMOTE(random_state=42)),
                ("model", model)
            ])
        else:
            pipeline = model

        scoring = ["accuracy", "precision", "recall", "f1"]

        scores = cross_validate(
        pipeline, X, y,
        cv=skf,
        scoring=scoring
        )

        print(f"\n {name} Cross-Validation Results (avg across {k} folds):")
        for metric in scoring:
            mean = scores[f'test_{metric}'].mean()
            std = scores[f'test_{metric}'].std()
            print(f"   {metric.capitalize():<9}: {mean:.4f} (+/- {std:.4f})") 

        #  Only use F1 for leaderboard
        f1_mean = scores["test_f1"].mean()
        f1_std = scores["test_f1"].std()

        results[name] = f1_mean
        print(f"{name} (F1): {f1_mean:.4f} (+/- {f1_std:.4f})")

        # if use_smote:
        #     pipeline = Pipeline([
        #         ("smote", SMOTE(random_state=42)),
        #         ("model", model)
        #     ])
        # else:
        #     pipeline = model

        # # scores = cross_val_score(
        # #     pipeline, X, y,
        # #     cv=skf,
        # #     scoring="f1"   # you can also try "accuracy", "recall", etc.
        # # )

        # scoring = ["accuracy", "precision", "recall", "f1"]

        # scores = cross_validate(
        #     pipeline, X, y,
        #     cv=skf,
        #     scoring=scoring
        # )

        # print(f"\n{name} Cross-Validation Results (avg across {k} folds):")
        # for metric in scoring:
        #     mean = scores[f'test_{metric}'].mean()
        #     std = scores[f'test_{metric}'].std()
        #     print(f"   {metric.capitalize():<9}: {mean:.4f} (+/- {std:.4f})") 

        # results[name] = scores.mean()

        # print(f"name}: {scores.mean():.4f} (+/- {scores.std():.4f})")

    # Print leaderboard
    print("\n=== Model CV Performance Leaderboard (avg F1-score) ===")
    for name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{name:<20} {score:.4f}")


   

    # scoring = ["accuracy", "precision", "recall", "f1"]

    # scores = cross_validate(
    #     pipeline, X, y,
    #     cv=skf,
    #     scoring=scoring
    # )

    # print(f"\n {name} Cross-Validation Results (avg across {k} folds):")
    # for metric in scoring:
    #     mean = scores[f'test_{metric}'].mean()
    #     std = scores[f'test_{metric}'].std()
    #     print(f"   {metric.capitalize():<9}: {mean:.4f} (+/- {std:.4f})")    


def run_cross_validation():
    """
    Load training split and run cross-validation on all models.
    """
    X_train = pd.read_csv("train_X.csv")
    y_train = pd.read_csv("train_y.csv").squeeze("columns")

    print(" Starting cross-validation on training set...")
    run_all_cv(X_train, y_train, k=5, use_smote=True)


if __name__ == "__main__":
    run_cross_validation()
