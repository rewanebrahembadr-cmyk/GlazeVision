import re

from .data import ingredients_vector

OXIDES = ("SiO2", "Al2O3", "Na2O", "K2O", "MgO", "CaO", "SrO", "BaO", "ZnO", "Li2O",
          "TiO2", "Fe2O3", "ZrO2", "B2O3")
FLUXES = OXIDES[2:10]


def recipe_features(recipe, chemistry=False):
    features = {f"ingredient:{key}": value for key, value in ingredients_vector(
        recipe.get("ingredients")).items()}
    atmosphere = recipe.get("atmosphere")
    if isinstance(atmosphere, str) and atmosphere.strip():
        features["atmosphere"] = atmosphere.strip().lower()
    cone = recipe.get("cone")
    if isinstance(cone, str):
        # Preserve leading zero: cone 06 and cone 6 represent different heatwork.
        cones = re.findall(r"\d+", cone)
        if cones:
            features["cone_min"] = cones[0]
            features["cone_max"] = cones[-1]
    amounts = recipe.get("chemical_composition")
    if isinstance(amounts, dict):
        for oxide in OXIDES:
            if isinstance(amounts.get(oxide), (int, float)):
                features[f"oxide:{oxide}"] = amounts[oxide]
    if chemistry and isinstance(recipe.get("umf"), dict):
        umf = recipe["umf"]
        features.update({f"umf:{oxide}": umf[oxide] for oxide in OXIDES if oxide in umf})
        features["umf_fluxes"] = sum(umf.get(oxide, 0) for oxide in FLUXES)
        features["umf_stabilizer"] = umf.get("Al2O3", 0)
        features["umf_glass_formers"] = umf.get("SiO2", 0) + umf.get("B2O3", 0)
        if umf.get("Al2O3"):
            features["umf_silica_alumina"] = umf.get("SiO2", 0) / umf["Al2O3"]
    return features
