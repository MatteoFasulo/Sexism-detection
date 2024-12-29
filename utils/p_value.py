from statsmodels.stats.contingency_tables import mcnemar
import numpy as np
import pandas as pd

def compute_p_value(pipeline1_preds : pd.Series, pipeline2_preds : pd.Series, true_labels: pd.Series):
    """
    Compute the p-value for statistical significance of improvement between two binary classification pipelines.

    Parameters:
        pipeline1_preds (pd.Series): Predictions from pipeline 1 (binary values: 0 or 1).
        pipeline2_preds (pd.Series): Predictions from pipeline 2 (binary values: 0 or 1).
        true_labels (pd.Series): Ground truth labels (binary values: 0 or 1).

    Returns:
        float: The p-value for the McNemar's test.
    """
    # Convert inputs to numpy arrays
    pipeline1_preds = np.array(pipeline1_preds)
    pipeline2_preds = np.array(pipeline2_preds)
    true_labels = np.array(true_labels)

    # Create the contingency table
    both_correct = np.sum((pipeline1_preds == true_labels) & (pipeline2_preds == true_labels))
    only_pipeline1_correct = np.sum((pipeline1_preds == true_labels) & (pipeline2_preds != true_labels))
    only_pipeline2_correct = np.sum((pipeline1_preds != true_labels) & (pipeline2_preds == true_labels))
    both_incorrect = np.sum((pipeline1_preds != true_labels) & (pipeline2_preds != true_labels))

    contingency_table = np.array([[both_correct, only_pipeline2_correct],
                                   [only_pipeline1_correct, both_incorrect]])

    # Perform McNemar's test
    result = mcnemar(contingency_table, exact=True)

    return result.pvalue

# Example usage
df = pd.read_csv('csv\models_predictions.csv')
pipeline1_preds = df["Mistralv3_4_shot_labels"]
pipeline2_preds = df["Mistralv3_8_shot_labels"]
true_labels = pd.Series([0] * len(pipeline1_preds))  # fake true, I was lazy to load it
p_value = compute_p_value(pipeline1_preds, pipeline2_preds, true_labels)

significance_level = 0.05 # or 0.01 for even more significance probability 

if p_value < significance_level:
    print(f"Pipelines ARE significantly different (p-value = {p_value:.4f}).")
else:
    print(f"Pipelines are NOT significantly different (p-value = {p_value:.4f}).")