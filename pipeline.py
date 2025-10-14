import joblib
import numpy as np
import pandas as pd
from explainability import explain_with_lime, explain_with_shap

model = joblib.load("models/best_voting_ensemble.joblib")
scaler = joblib.load("data/processed/scaler.joblib")

FEATURES = [
    'LB', 'AC.1', 'FM.1', 'UC.1', 'DL.1', 'DS.1', 'DP.1', 'ASTV', 'MSTV',
    'ALTV', 'MLTV', 'Width', 'Min', 'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean',
    'Median', 'Variance', 'Tendency'
]

CLASS_MAP = {1: "Normal", 2: "Suspect", 3: "Pathological"}

CARE_SUGGESTIONS = {
    "Normal": (
        "Routine monitoring (low risk)"
    ),
    "Suspect": (
        "Enhanced observation / further testing"
    ),
    "Pathological": (
        "Immediate medical attention (high risk)"
    )
}

def preprocess_uploaded_df(input_df: pd.DataFrame) -> pd.DataFrame:

    if all(isinstance(x, str) for x in input_df.iloc[0]):
        input_df = input_df[1:].reset_index(drop=True)

    input_df = input_df.apply(pd.to_numeric, errors='coerce')

    input_df = input_df.fillna(input_df.median(numeric_only=True))

    missing_cols = [col for col in FEATURES if col not in input_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in uploaded CSV: {missing_cols}")

    input_df = input_df[FEATURES]

    return input_df

def predict_fetal_health(input_df: pd.DataFrame):

    df = preprocess_uploaded_df(input_df)

    df_scaled = scaler.transform(df)

    probs = model.predict_proba(df_scaled)
    preds = np.argmax(probs, axis=1) + 1

    results = []
    for i, pred_class in enumerate(preds):
        class_name = CLASS_MAP[pred_class]
        result = {
            "Predicted Class": class_name,
            "Probabilities": {
                CLASS_MAP[j + 1]: float(probs[i][j]) for j in range(len(CLASS_MAP))
            },
            "Care Suggestion": CARE_SUGGESTIONS[class_name]
        }
        results.append(result)

    results_df = pd.DataFrame(results)
    return results_df
