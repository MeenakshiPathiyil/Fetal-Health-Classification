# app.py
import streamlit as st
import pandas as pd
from pipeline import predict_fetal_health

st.set_page_config(page_title="Fetal Health Classification", layout="centered")
st.title("Fetal Health Classification System")
st.write("Upload your fetal monitoring data (CSV) to predict fetal health status.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview:")
    st.dataframe(df.head())

    if st.button("Run Prediction"):
        with st.spinner("Running predictions..."):
            results_df = predict_fetal_health(df)

        st.success("Prediction completed successfully!")

        # Show preview in Streamlit
        st.write("### Prediction Results:")
        st.dataframe(results_df)

        # Save results to a CSV file (in memory)
        csv = results_df.to_csv(index=False).encode("utf-8")

        # Provide download button
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="fetal_health_predictions.csv",
            mime="text/csv",
        )



