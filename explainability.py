# explainability.py
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def explain_with_shap(model, X_train, sample, class_idx=0):
    """Generate SHAP explanation for a given sample."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    shap.initjs()
    shap.force_plot(
        explainer.expected_value[class_idx],
        shap_values[class_idx][0, :],
        sample
    )
    return shap_values


def explain_with_lime(model, X_train, sample):
    """Generate LIME explanation for a given sample."""
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=X_train.columns,
        mode='classification'
    )
    exp = explainer.explain_instance(
        data_row=sample.iloc[0],
        predict_fn=model.predict_proba
    )
    exp.show_in_notebook(show_table=True)
    return exp


def show_feature_importance(model, X_train):
    """Plot feature importance for tree-based models."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feat_importance = pd.DataFrame({
            "Feature": X_train.columns,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)
        feat_importance.plot.barh(x="Feature", y="Importance", figsize=(10, 6))
        plt.title("Feature Importance")
        plt.show()
        return feat_importance
    else:
        print("Model does not have feature_importances_.")
        return None
