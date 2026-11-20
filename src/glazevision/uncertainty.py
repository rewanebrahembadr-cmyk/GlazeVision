import math

import numpy as np


def conformal_threshold(probabilities, labels, alpha=.1):
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels, dtype=int)
    if not len(labels) or not 0 < alpha < 1:
        raise ValueError("Calibration requires labels and alpha in (0, 1)")
    scores = np.sort(1 - probabilities[np.arange(len(labels)), labels])
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    return 1.0 if rank > len(scores) else float(scores[rank - 1])


def prediction_sets(probabilities, threshold):
    return 1 - np.asarray(probabilities) <= threshold
