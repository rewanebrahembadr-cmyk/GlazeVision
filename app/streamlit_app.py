import pandas as pd
import streamlit as st
from PIL import Image

from glazevision.config import CONFIG, ROOT
from glazevision.predict import load_predictor, predict, similar_glazes

st.set_page_config(page_title="GlazeVision")
st.title("GlazeVision")
st.caption("Predict ceramic glaze finish from a recipe, a photo, or both.")
if st.cache_resource(load_predictor)() is None:
    st.info("Placeholder model active. Predictions are simulated and carry no performance claim.")
source = st.radio("Image source", ["Upload", "Camera"], horizontal=True)
photo = (st.camera_input("Photograph the sample") if source == "Camera" else
         st.file_uploader("Upload a glaze photo", type=["jpg", "jpeg", "png"]))
include_recipe = st.checkbox("Enter recipe and firing conditions")
recipe = None
if include_recipe:
    ingredients = st.text_area("Ingredients, one per line (material, amount)")
    cone = st.text_input("Orton cone or cone range")
    atmosphere = st.selectbox("Firing atmosphere", ["", "Oxidation", "Reduction"])
if st.button("Predict"):
    try:
        if include_recipe:
            entries = []
            for line in ingredients.splitlines():
                if line.strip():
                    material, amount = line.rsplit(",", 1)
                    if not material.strip() or float(amount) <= 0:
                        raise ValueError("Each ingredient needs a name and positive amount")
                    entries.append({"material": material.strip(), "amount": float(amount)})
            if not (entries or cone or atmosphere):
                raise ValueError("Enter at least one recipe value")
            recipe = dict(ingredients=entries, cone=cone, atmosphere=atmosphere)
        image = Image.open(photo).convert("RGB") if photo else None
        result = predict(image, recipe)
    except (ValueError, OSError) as error:
        st.error(str(error))
    else:
        if image is not None:
            st.image(image, width=300)
        for task in CONFIG["classes"]:
            st.subheader(task.replace("_", " ").title())
            st.bar_chart(pd.Series(result[task], name="Probability"))
        st.write(f"Top finish score: {result['confidence']:.0%}")
        if result["uncertain"]:
            st.warning("Low confidence. Send this sample for human review.")
        if result["gradcam"] is not None:
            import numpy as np
            from matplotlib import colormaps

            heatmap = result["gradcam"]
            colors = (colormaps["inferno"](heatmap)[..., :3] * 255).astype("uint8")
            overlay = Image.blend(image.resize((heatmap.shape[1], heatmap.shape[0])),
                                  Image.fromarray(colors), .45)
            st.image(overlay, caption="Grad-CAM from the separate image model", width=300)
        if recipe and (ROOT / CONFIG["split"]).exists() and (ROOT / CONFIG["raw"]).exists():
            matches = similar_glazes(recipe)
            if matches:
                st.subheader("Similar recipes in the training data")
                st.dataframe(matches)
st.caption("Research prototype for finish review. Never use it to make product-safety decisions.")
