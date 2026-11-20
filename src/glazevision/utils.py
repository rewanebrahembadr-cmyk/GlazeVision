import csv
from filelock import FileLock
import json
import logging
import random
from datetime import date

import numpy as np

from .config import CONFIG, ROOT

logger = logging.getLogger("glazevision")


def set_seed(seed=CONFIG["seed"]):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)


def save_results(model_name, owner, task, input_type, metrics, notes=""):
    if input_type not in {"tabular", "image", "fusion"}:
        raise ValueError("Unsupported input type")
    path = ROOT / "reports/results.csv"
    row = [model_name, owner, task, input_type, metrics["accuracy"],
           metrics["macro_f1"], metrics["weighted_f1"],
           json.dumps(metrics["per_class"]), notes, date.today().isoformat()]
    with FileLock(str(path) + ".lock"):
        with path.open("a", newline="", encoding="utf-8") as output:
            csv.writer(output).writerow(row)
