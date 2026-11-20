import pandas as pd
import streamlit as st

from glazevision.config import ROOT

st.title("Model comparison")
results = pd.read_csv(ROOT / "reports/results.csv")
if results.empty:
    st.info("No verified experiment results have been recorded.")
else:
    st.dataframe(results.sort_values("macro_f1", ascending=False))
    st.caption("Compare modalities only within the same task and evaluation cohort.")
