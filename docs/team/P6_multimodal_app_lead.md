# P6 — Multimodal fusion, app and presentation

**Difficulty** Most advanced

**Course** Model evaluation, deployment and deep learning extension

## Goal

Multimodal fusion, app and presentation

## Inputs

P1 split, P2 baseline, P3 recipe features, P4 data loader and P5 explanation

## Tasks

- [ ] Train two-branch model with surface, transparency and color-family heads.
- [ ] Compare early, late and attention fusion to image-only and tabular-only on identical samples.
- [ ] Validate single-input fallback and export a compatible fusion checkpoint.
- [ ] Integrate upload, camera, manual recipe, confidence, Grad-CAM, similar recipes and comparison.
- [ ] Review PRs, collect two or three slides per member and rehearse the demo.
- [ ] Record a demo backup and collect reproducible final results and limitations.

## Outputs

- `src/glazevision/fusion.py`
- `src/glazevision/predict.py`
- `app/streamlit_app.py`
- `notebooks/06_multimodal_fusion.ipynb`
- `presentation/outline.md`

## Definition of done

All modalities run, ablation uses one cohort, README reruns, and demo has a recorded fallback.

## Bonus

Deploy to Streamlit Community Cloud or Hugging Face Spaces.
