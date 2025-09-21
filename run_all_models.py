
import joblib
import pandas as pd
from utils.visualization import plot_confusion_matrix, plot_roc_curve, plot_pr_curve
from deployment.risk_categorization import assign_risk, risk_summary
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# -------------------------------
# Load test data
# -------------------------------
X_test = pd.read_csv("test_X.csv")
y_test = pd.read_csv("test_y.csv").squeeze("columns")

# -------------------------------
# Load best estimators
# -------------------------------
best_estimators = joblib.load("models/best_estimators.pkl")

baseline_models = {
    "Logistic Regression": best_estimators["logistic_regression"],
    "Random Forest": best_estimators["random_forest"],
    "SVM": best_estimators["svm"],
    "Neural Network": best_estimators["neural_network"]
}

# Load ensemble pipeline + threshold
ensemble_pipeline, ensemble_threshold = joblib.load("models/ensemble_model.pkl")
baseline_models["Ensemble (Voting)"] = ensemble_pipeline

# -------------------------------
# Evaluation function
# -------------------------------
def evaluate_model(model, X_test, y_test, threshold=None):
    if threshold is not None:
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
    else:
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)
        except:
            y_pred = model.predict(X_test)
            y_proba = None

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0)
    }
    return y_pred, y_proba, metrics

# -------------------------------
# Run evaluation for all models
# -------------------------------
results_summary = []

for name, model in baseline_models.items():
    print(f"\n🔹 Evaluating {name}...")
    if name == "Ensemble (Voting)":
        y_pred, y_proba, metrics = evaluate_model(model, X_test, y_test, threshold=ensemble_threshold)
    else:
        y_pred, y_proba, metrics = evaluate_model(model, X_test, y_test)

    # Print metrics
    print(f"Metrics: {metrics}")

    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred, model_name=name)

    # ROC & PR curves (only if probabilities available)
    if y_proba is not None:
        plot_roc_curve(model, X_test, y_test, model_name=name)
        plot_pr_curve(model, X_test, y_test, model_name=name)

    # Risk categorization (only for ensemble)
    if name == "Ensemble (Voting)":
        risks = assign_risk(y_proba)
        print("\nRisk Summary:")
        print(risk_summary(risks))

    # Append summary
    results_summary.append({
        "Model": name,
        **metrics
    })

# -------------------------------
# Final summary table
# -------------------------------
summary_df = pd.DataFrame(results_summary)
print("\nFinal Model Comparison:")
print(summary_df.sort_values(by="F1 Score", ascending=False))
summary_df.to_csv("models/model_comparison_summary.csv", index=False)
print("\nModel comparison saved to 'models/model_comparison_summary.csv'")


