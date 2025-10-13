import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load model (voting ensemble)
model = joblib.load('models/best_voting_ensemble.joblib')

# Load test feature and label files
X_test = pd.read_csv('data/processed/X_test.csv')
y_test = pd.read_csv('data/processed/y_test.csv')

# Handle label column if it's a dataframe with one column
if y_test.shape[1] == 1:
    y_test = y_test.iloc[:, 0]

# Predict fetal health classes
predictions = model.predict(X_test)

# Evaluate model performance
acc = accuracy_score(y_test, predictions)
print(f"Accuracy: {acc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, predictions))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))

# Care recommendation mapping
def recommend_care(pred):
    if pred == 0:
        return "Routine monitoring (low risk)"
    elif pred == 1:
        return "Enhanced observation / further testing"
    else:
        return "Immediate medical attention (high risk)"

care = [recommend_care(p) for p in predictions]

# Save predictions + recommendations
results = X_test.copy()
results['True_Label'] = y_test.values
results['Predicted_Label'] = predictions
results['Care_Recommendation'] = care

results.to_csv('results/test_predictions_with_care.csv', index=False)
print("\nInference complete. Results saved to results/test_predictions_with_care.csv")
