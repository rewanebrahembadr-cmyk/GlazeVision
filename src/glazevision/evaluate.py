import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

from .config import CONFIG, ROOT


def evaluate(y_true, y_pred, name, task="surface"):
    y_true, y_pred = list(y_true), list(y_pred)
    labels = CONFIG["classes"][task]
    if not labels or set(y_true) - set(labels) or set(y_pred) - set(labels):
        raise ValueError("Configure all evaluation labels before scoring")
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    path = ROOT / "reports/figures" / f"{name}_confusion.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    display = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=labels)
    display.figure_.savefig(path, bbox_inches="tight")
    plt.close(display.figure_)
    per_class = {label: report[label] for label in labels}
    return {
        "per_class": per_class,
        "accuracy": float((pd.Series(y_true) == pd.Series(y_pred)).mean()),
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    }


def review_predictions(ids, truth, predictions, confidence, threshold=0.65):
    review = pd.DataFrame(dict(sample_id=ids, observed=truth, predicted=predictions,
                               confidence=confidence))
    review["uncertain"] = review.confidence < threshold
    review["misclassified"] = review.observed != review.predicted
    return review.loc[review.uncertain | review.misclassified]
