# P5 — Advanced deep learning, explainability and uncertainty

**Difficulty** Advanced

**Course** Model evaluation and deep learning extension

## Goal

Advanced deep learning, explainability and uncertainty

## Inputs

Image loader and baseline from P4; split from P1

## Tasks

- [ ] Compare ConvNeXt, ViT and DINOv2 with a common budget.
- [ ] Compare class-weighted cross-entropy to unweighted training.
- [ ] Implement and inspect Grad-CAM on the CNN path.
- [ ] Calibrate conformal prediction on validation. Audit coverage and set sizes on test.
- [ ] Export the selected image checkpoint and document its loading interface.

## Outputs

- `src/glazevision/explain.py`
- `src/glazevision/uncertainty.py`
- `models/best_image_model.pt`
- `notebooks/05_advanced_dl_explainability.ipynb`

## Definition of done

Checkpoint loads; per-class metrics, explanations and uncertainty examples are available.

## Bonus

Investigate transformer-specific attribution and compare to the CNN explanations.
