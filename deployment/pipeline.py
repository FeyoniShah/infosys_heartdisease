import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import joblib
import pandas as pd
from deployment.risk_categorization import assign_risk, risk_summary
from utils.visualization import plot_confusion_matrix, plot_roc_curve, plot_pr_curve
from utils.interpretability import shap_summary, plot_feature_importance

# -------------------------------
# Load saved ensemble pipeline + threshold
# -------------------------------
pipeline, threshold = joblib.load("models/ensemble_model.pkl")  # ensemble with SMOTE + auto threshold

# -------------------------------
# Predict function
# -------------------------------
def predict_and_categorize(X):
    """
    Args:
        X (DataFrame): New input features
    
    Returns:
        DataFrame: Predictions, probabilities, and risk levels
    """
    # Predicted probabilities for positive class
    y_proba = pipeline.predict_proba(X)[:, 1]
    
    # Predictions based on threshold
    y_pred = (y_proba >= threshold).astype(int)
    
    # Risk levels
    risk_levels = assign_risk(y_proba, thresholds=(0.3, 0.7))
    
    # Combine results
    results = X.copy()
    results["Predicted"] = y_pred
    results["Probability"] = y_proba
    results["Risk"] = risk_levels
    
    return results

# -------------------------------
# Evaluation function
# -------------------------------
def evaluate_pipeline(X_test, y_test):
    """
    Evaluate the deployment pipeline on test data.
    """
    results = predict_and_categorize(X_test)
    
    y_pred = results["Predicted"].values
    y_proba = results["Probability"].values
    
    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred, model_name="Pipeline")
    
    # ROC curve
    plot_roc_curve(pipeline, X_test, y_test, model_name="Pipeline")
    
    # Precision-Recall curve
    plot_pr_curve(pipeline, X_test, y_test, model_name="Pipeline")
    
    # Risk summary
    print("\nRisk Summary:")
    print(risk_summary(results["Risk"]))
    
    return results

# -------------------------------
# Explainability function
# -------------------------------
def explain_pipeline(X_train, top_n=10):
    """
    Generate SHAP summary and feature importance plots.
    """
    shap_summary(pipeline, X_train, model_name="Pipeline")
    plot_feature_importance(pipeline.named_steps["ensemble"], X_train, top_n=top_n, model_name="Pipeline")



# if __name__ == "__main__":
#     import pandas as pd

#     # Load test data
#     X_test = pd.read_csv("test_X.csv")
#     y_test = pd.read_csv("test_y.csv").squeeze("columns")

#     # Evaluate pipeline
#     results = evaluate_pipeline(X_test, y_test)

#     # Optionally show first few predictions
#     print("\nSample predictions:")
#     print(results.head())
