# app.py
import streamlit as st
import pandas as pd
from pipeline import predict_fetal_health

st.set_page_config(page_title="Fetal Health Classification", layout="centered")
st.title("Fetal Health Classification System")
st.write("Upload your fetal monitoring data (CSV) to predict fetal health status.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, skiprows=1)
    st.write("### Uploaded Data Preview:")
    st.dataframe(df.head())

    if st.button("Run Prediction"):
        with st.spinner("Running predictions..."):
            try:
                results = predict_fetal_health(df)
            except ValueError as e:
                st.error(str(e))
            else:
                for i, res in enumerate(results):
                    st.markdown(f"### Sample {i+1} Prediction")
                    st.markdown(f"**Predicted Class:** {res['Predicted Class']}")
                    
                    st.markdown("**Probabilities:**")
                    for cls, prob in res["Probabilities"].items():
                        st.progress(prob)
                        st.write(f"{cls}: {prob:.2f}")
                    
                    st.markdown("**Suggested Care Instructions:**")
                    st.info(res["Care Suggestion"])
                    st.markdown("---")
