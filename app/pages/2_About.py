import streamlit as st

st.title("About GlazeVision")
st.write("A DEPI IBM Data Science graduation project for ceramic glaze finish prediction.")
st.write("Lighting, camera settings and source-data coverage can affect predictions.")
st.write("Predictions support human review of finish. They cannot establish product safety.")
st.link_button("GlazyBench dataset", "https://huggingface.co/datasets/AlpachinoNLP/GlazyBench")
