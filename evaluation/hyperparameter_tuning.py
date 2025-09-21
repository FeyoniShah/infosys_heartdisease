import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import RandomizedSearchCV


# Import your model getters
from models.logistic_regression import get_logistic_regression
from models.random_forest import get_random_forest
from models.svm_model import get_svm
from models.neural_network import get_neural_network

# -------------------------------
# Load training split
# -------------------------------
X_train = pd.read_csv("train_X.csv")
y_train = pd.read_csv("train_y.csv").squeeze("columns")

# -------------------------------
# Define hyperparameter grids
# -------------------------------
param_grids = {
    "logistic_regression": {
        "model__C": [0.01, 0.1, 1, 10, 100],
        "model__max_iter": [200, 500, 1000]
    },
    "random_forest": {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_split": [2, 5, 10]
    },
    "svm": {
        "model__C": [0.1, 1, 10],
        "model__kernel": ["linear", "rbf"],
        "model__gamma": ["scale", "auto"]
    },
    "neural_network": {
    "model__hidden_layer_sizes": [(50,), (100,), (50,50)],
    "model__activation": ["relu", "tanh"],
    "model__alpha": [0.0001, 0.001, 0.01],
    "model__learning_rate_init": [0.001, 0.01, 0.1]
    }

}

# -------------------------------
# Define models
# -------------------------------
models = {
    "logistic_regression": get_logistic_regression(),
    "random_forest": get_random_forest(),
    "svm": get_svm(),
    "neural_network": get_neural_network()
}

# -------------------------------
# Custom F1 scorer
# -------------------------------
f1_scorer = make_scorer(f1_score)

# -------------------------------
# Run GridSearchCV with SMOTE
# -------------------------------
best_estimators = {}

for name, model in models.items():
    print(f"\n Tuning hyperparameters for {name}...")
    
    pipeline = Pipeline([
        ("smote", SMOTE(random_state=42)),
        ("model", model)
    ])

    grid = param_grids[name]

    if name == "svm":
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=grid,
            scoring=f1_scorer,
            cv=5,
            n_iter=20,  # choose number of random combinations
            n_jobs=-1,
            verbose=2,
            random_state=42
        )
    else:
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=grid,
            scoring=f1_scorer,
            cv=5,
            n_jobs=-1,
            verbose=2
        )

    search.fit(X_train, y_train)

    print(f" Best params for {name}: {search.best_params_}")
    print(f"Best F1-score: {search.best_score_:.4f}")
    
    best_estimators[name] = search.best_estimator_
    
    # grid = param_grids[name]
    
    # grid_search = GridSearchCV(
    #     estimator=pipeline,
    #     param_grid=grid,
    #     scoring=f1_scorer,
    #     cv=5,
    #     n_jobs=-1,
    #     verbose=2
    # )
    
    # grid_search.fit(X_train, y_train)
    
    # print(f" Best params for {name}: {grid_search.best_params_}")
    # print(f"Best F1-score: {grid_search.best_score_:.4f}")
    
    # best_estimators[name] = grid_search.best_estimator_

# -------------------------------
# Optionally save best estimators
# -------------------------------
import joblib
joblib.dump(best_estimators, "models/best_estimators.pkl")
print("\n All best estimators saved!")
