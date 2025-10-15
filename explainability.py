import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Generate SHAP explanations for a given sample
def explain_with_shap(model, X_train, sample, class_idx=0):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    shap.initjs()
    shap.force_plot(
        explainer.expected_value[class_idx],
        shap_values[class_idx][0, :],
        sample
    )
    return shap_values

# Plot feature importance for tree based models
def show_feature_importance(model, X_train):
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
