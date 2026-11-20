import json
import re

import pandas as pd

from .config import CONFIG, ROOT


def read_records(path):
    with path.open(encoding="utf-8") as source:
        records = json.load(source)
    values = records.values() if isinstance(records, dict) else records
    return {str(record["id"]): record for record in values}


def load_samples(raw=None):
    raw = raw or ROOT / CONFIG["raw"]
    samples = {}
    for partition in ("train", "test"):
        folder = raw / "property_prediction" / partition
        targets = read_records(folder / "targets.json")
        for sample_id, recipe in read_records(folder / "recipes.json").items():
            target = targets.get(sample_id, {})
            samples[sample_id] = dict(recipe, sample_id=sample_id,
                                      **{task: target.get(task) for task in CONFIG["classes"]})
    for partition in ("train", "test"):
        folder = raw / "image_generation" / partition
        if not (folder / "metadata.json").exists():
            continue
        for sample_id, metadata in read_records(folder / "metadata.json").items():
            row = samples.setdefault(sample_id, {"sample_id": sample_id})
            for key in ("ingredients", "chemical_composition", "cone", "atmosphere"):
                if metadata.get(key) and not row.get(key):
                    row[key] = metadata[key]
            row.update({f"image_{task}": metadata.get(task) for task in CONFIG["classes"]})
            filename = metadata.get("image_filename")
            if filename:
                row["image_path"] = str(folder / "images" / filename)
    frame = pd.DataFrame(samples.values())
    for name in ("surface", "image_surface"):
        if name in frame:
            frame[name] = frame[name].map(CONFIG["surface_mapping"])
    return frame


def load_split(path=None):
    return pd.read_csv(path or ROOT / CONFIG["split"], dtype={"sample_id": str, "group_id": str})


def joined_samples():
    return load_samples().merge(load_split(), on="sample_id", validate="one_to_one")


def ingredients_vector(ingredients):
    if not isinstance(ingredients, list):
        return {}
    amounts = {}
    for ingredient in ingredients:
        material = re.sub(r"\s+", " ", ingredient["material"].strip().lower())
        if material in {"total", "total base recipe"}:
            continue
        amount = ingredient.get("amount")
        if material and isinstance(amount, (float, int)) and amount > 0:
            amounts[material] = amounts.get(material, 0) + amount
    total = sum(amounts.values())
    return {material: amount / total for material, amount in amounts.items()}
