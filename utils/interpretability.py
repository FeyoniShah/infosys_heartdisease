import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# -------------------------------
# SHAP Summary Plot
# -------------------------------
def shap_summary(model, X_train, model_name="Model"):
    """
    Compute and plot SHAP summary for tree-based or linear models.
    """
    # Tree models use TreeExplainer, others use KernelExplainer
    try:
        explainer = shap.Explainer(model, X_train)
    except Exception:
        explainer = shap.KernelExplainer(model.predict_proba, X_train)
    
    shap_values = explainer(X_train)
    
    shap.summary_plot(shap_values, X_train, show=True, plot_size=(8,6))
    print(f"SHAP summary plot generated for {model_name}")
    
# -------------------------------
# Feature Importance
# -------------------------------
def plot_feature_importance(model, X_train, top_n=10, model_name="Model"):
    """
    Plots top_n feature importances for tree-based or linear models.
    """
    try:
        if hasattr(model, "feature_importances_"):  # Tree-based
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):  # Linear
            importances = np.abs(model.coef_[0])
        else:
            print(f"Model type not supported for feature importance: {model_name}")
            return

        feature_importance = pd.Series(importances, index=X_train.columns)
        feature_importance = feature_importance.sort_values(ascending=False).head(top_n)

        plt.figure(figsize=(8,6))
        feature_importance.plot(kind='bar', color='skyblue')
        plt.title(f'Top {top_n} Feature Importances - {model_name}')
        plt.ylabel('Importance')
        plt.show()
        print(f"Feature importance plot generated for {model_name}")

    except Exception as e:
        print(f"Error generating feature importance: {e}")
