import numpy as np
import pandas as pd

from .config import CONFIG, ROOT
from .data import joined_samples
from .evaluate import evaluate, review_predictions
from .features import recipe_features
from .utils import save_results, set_seed


def record(name, owner, input_type, task, ids, truth, probabilities, cohort):
    classes = np.array(CONFIG["classes"][task])
    probabilities = np.asarray(probabilities)
    if not len(truth):
        raise ValueError(f"No labeled test examples for {task}")
    predicted = classes[probabilities.argmax(axis=1)]
    metrics = evaluate(truth, predicted, f"{name}_{task}", task)
    review_predictions(ids, truth, predicted, probabilities.max(axis=1)).to_csv(
        ROOT / "reports" / f"{name}_{task}_review.csv", index=False)
    save_results(name, owner, task, input_type, metrics, notes=cohort)
    return metrics


def run_tabular(chemistry=False, names=None, evaluation_split="val"):
    from .tabular_models import classifiers

    set_seed()
    frame = joined_samples().dropna(subset=["surface"])
    features = frame.apply(lambda row: recipe_features(row, chemistry), axis=1)
    train, test = frame.split.eq("train"), frame.split.eq(evaluation_split)
    classes = CONFIG["classes"]["surface"]
    for name, model in classifiers().items():
        if names is not None and name not in names:
            continue
        model.fit(features[train].tolist(), frame.surface[train])
        probabilities = pd.DataFrame(model.predict_proba(features[test].tolist()), columns=model.classes_)
        probabilities = probabilities.reindex(columns=classes, fill_value=0)
        print(name, record(f"{name}{'_chemistry' if chemistry else ''}", "P3" if chemistry else "P2",
                           "tabular", "surface", frame.sample_id[test].tolist(),
                           frame.surface[test].tolist(), probabilities, f"all labeled tabular {evaluation_split} rows"))


def run_deep(backbone="tiny", mode=None, epochs=2, freeze=False, balanced=False,
             owner="P4", export=False, conformal=False, evaluation_split="val"):
    import torch
    from sklearn.feature_extraction import DictVectorizer

    from .fusion import FusionModel
    from .image_data import loader
    from .image_models import ImageClassifier
    from .train import collect, train_epoch
    from .uncertainty import conformal_threshold, prediction_sets

    if conformal and evaluation_split != "test":
        raise ValueError("Evaluate conformal coverage on test, separate from calibration on val")
    set_seed()
    frame = joined_samples().dropna(subset=["image_path", "image_surface"])
    if mode is not None:
        frame = frame[frame.ingredients.map(lambda value: isinstance(value, list) and bool(value))]
    partitions = {split: frame[frame.split.eq(split)] for split in ("train", "val", "test")}
    train = partitions["train"]
    if any(part.empty for part in partitions.values()):
        raise ValueError("Every partition needs labeled paired samples for this experiment")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if mode is None:
        model = ImageClassifier(backbone, freeze=freeze)
        matrices = dict.fromkeys(partitions)
    else:
        features = {split: part.apply(lambda row: recipe_features(row, True), axis=1).tolist()
                    for split, part in partitions.items()}
        vectorizer = DictVectorizer(sparse=False, dtype=np.float32).fit(features["train"])
        matrices = {split: vectorizer.transform(rows) for split, rows in features.items()}
        model = FusionModel(len(vectorizer.feature_names_), backbone, mode)
    model.to(device)
    batches = {split: loader(part, split == "train", matrices[split]) for split, part in partitions.items()}
    counts = train.image_surface.value_counts().reindex(CONFIG["classes"]["surface"], fill_value=0)
    weights = torch.tensor(1 / counts.clip(lower=1).values, dtype=torch.float32,
                           device=device) if balanced else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003)
    for epoch in range(epochs):
        train_epoch(model, batches["train"], optimizer, device, weights)
        if freeze and epoch == 0 and epochs > 1:
            model.encoder.requires_grad_(True)
    tasks = ("surface",) if mode is None else CONFIG["classes"]
    prefix = f"{owner}_{backbone}_{mode or 'image'}_{evaluation_split}"
    for task in tasks:
        ids, targets, probabilities = collect(model, batches[evaluation_split], device, task)
        truth = np.array(CONFIG["classes"][task])[targets].tolist()
        input_type = "image" if mode in {None, "image"} else "tabular" if mode == "tabular" else "fusion"
        print(task, record(prefix, owner, input_type, task, ids, truth, probabilities,
                           f"labeled image {evaluation_split} cohort" if mode is None else
                           f"identical paired {evaluation_split} ablation cohort"))
        if conformal and task == "surface":
            _, validation, scores = collect(model, batches["val"], device)
            cutoff = conformal_threshold(scores, validation)
            sets = prediction_sets(probabilities, cutoff)
            classes = np.array(CONFIG["classes"][task])
            pd.DataFrame({"sample_id": ids, "prediction_set": [" | ".join(classes[row]) for row in sets],
                          "covered": sets[np.arange(len(targets)), targets]}).to_csv(
                ROOT / "reports" / f"{prefix}_conformal.csv", index=False)
    if export:
        checkpoint = {"backbone": backbone, "state_dict": model.state_dict()}
        filename = "best_image_model.pt"
        if mode is not None:
            if mode in {"tabular", "image"}:
                raise ValueError("Export a fusion mode supporting both single-input fallbacks")
            checkpoint.update(mode=mode, columns=vectorizer.feature_names_)
            filename = "best_fusion_model.pt"
        torch.save(checkpoint, ROOT / "models" / filename)
    return model
