import hashlib
import json
from functools import lru_cache

import numpy as np

from .config import CONFIG, ROOT

CHECKPOINT = ROOT / "models/best_fusion_model.pt"
THRESHOLD = .65


@lru_cache(maxsize=1)
def load_predictor():
    if not CHECKPOINT.is_file():
        return None
    import torch

    from .fusion import FusionModel

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model = FusionModel(len(checkpoint["columns"]), backbone=checkpoint["backbone"],
                        mode=checkpoint["mode"], pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint["columns"]


def recipe_row(features, columns):
    encoded = {}
    for key, value in features.items():
        if isinstance(value, str):
            encoded[f"{key}={value}"] = 1
        else:
            encoded[key] = value
    return [encoded.get(key, 0) for key in columns]


def predict(image=None, recipe=None):
    if image is None and recipe is None:
        raise ValueError("Provide a glaze image or recipe")
    predictor = load_predictor()
    heatmap = None
    if predictor is None:
        # Deterministic demo output. Mark as placeholder at every UI entry.
        seed = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode())
        if image is not None:
            seed.update(image.tobytes())
        rng = np.random.default_rng(int.from_bytes(seed.digest()[:8], "big"))
        probabilities = {task: rng.dirichlet(np.ones(len(classes))).tolist()
                         for task, classes in CONFIG["classes"].items()}
    else:
        import torch

        from .features import recipe_features
        from .image_data import image_transform

        model, columns = predictor
        visual = image_transform()(image.convert("RGB")).unsqueeze(0) if image is not None else None
        tabular = torch.tensor([recipe_row(recipe_features(recipe, chemistry=True), columns)],
                               dtype=torch.float32) if recipe is not None else None
        with torch.inference_mode():
            scores = model(visual, tabular)
            probabilities = {task: logits.softmax(1)[0].tolist() for task, logits in scores.items()}
        image_checkpoint = ROOT / "models/best_image_model.pt"
        if visual is not None and image_checkpoint.is_file():
            from .explain import gradcam
            from .image_models import load_image_model

            visual_model = load_image_model(image_checkpoint)
            if any(isinstance(layer, torch.nn.Conv2d) for layer in visual_model.encoder.modules()):
                heatmap = gradcam(visual_model, visual)
    result = {task: dict(zip(CONFIG["classes"][task], values)) for task, values in probabilities.items()}
    confidence = max(result["surface"].values())
    return dict(result, confidence=confidence, uncertain=confidence < THRESHOLD, gradcam=heatmap)


def similar_glazes(recipe, limit=5):
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    from .data import ingredients_vector, joined_samples

    query = ingredients_vector(recipe.get("ingredients"))
    if not query:
        return []
    rows = joined_samples().query("split == 'train'").dropna(subset=["surface"])
    vectors = rows.ingredients.map(ingredients_vector).tolist()
    matrix = DictVectorizer().fit_transform([query, *vectors])
    similarity = cosine_similarity(matrix[:1], matrix[1:])[0]
    return [{"sample_id": rows.iloc[index].sample_id, "surface": rows.iloc[index].surface,
             "similarity": float(similarity[index])}
            for index in similarity.argsort()[::-1][:limit] if similarity[index] > 0]
