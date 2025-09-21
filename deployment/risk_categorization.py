import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

def assign_risk(y_proba, thresholds=(0.3, 0.7)):
    """
    Assign risk levels based on probability thresholds.

    Args:
        y_proba (array-like): Predicted probabilities for the positive class.
        thresholds (tuple): (low_to_medium, medium_to_high) thresholds.

    Returns:
        list of risk levels: 'Low', 'Medium', 'High'
    """
    low_thresh, high_thresh = thresholds
    risk_levels = []

    for prob in y_proba:
        if prob < low_thresh:
            risk_levels.append("Low")
        elif prob < high_thresh:
            risk_levels.append("Medium")
        else:
            risk_levels.append("High")
    return risk_levels


def risk_summary(risk_levels):
    """
    Return a summary dataframe with counts of each risk level.
    """
    df = pd.DataFrame(risk_levels, columns=["Risk"])
    summary = df.value_counts().reset_index()
    summary.columns = ["Risk Level", "Count"]
    return summary
