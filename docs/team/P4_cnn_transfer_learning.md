# P4 — Image pipeline and transfer learning

**Difficulty** Intermediate

**Course** Python, supervised classification and deep learning extension

## Goal

Image pipeline and transfer learning

## Inputs

Images, annotations and split from P1

## Tasks

- [ ] Validate resolved image paths and provide reusable PyTorch batches.
- [ ] Train a small CNN baseline.
- [ ] Train EfficientNet-B0 with frozen encoder first, then fine-tune.
- [ ] Document preprocessing and pass the loader interface to P5 and P6.

## Outputs

- `src/glazevision/image_data.py`
- `src/glazevision/image_models.py`
- `notebooks/04_cnn_transfer_learning.ipynb`

## Definition of done

Held-out image metrics and dataloader reproduce, with no color jitter.

## Bonus

Try YOLO defect detection on separately annotated data after the core submission.
