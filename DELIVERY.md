# Delivery and verification

This handoff contains the repository scaffold, seven notebooks, six role briefs,
shared model interfaces, a runnable Streamlit demo, and GitHub setup automation.

## Verified in the delivery workspace

- Downloaded the pinned GlazyBench snapshot and extracted both image archives.
  All 4,933 metadata-linked image files resolved.
- Generated the shared split over 21,691 recipe IDs. Verified zero group overlap
  across train, validation and test under the documented distance rule.
- Fit the majority and logistic baselines using real tabular data. Exercised
  per-class metrics, confusion matrices, error-review export and results logging.
- Exercised the tiny CNN and five model modes on a small real-image cohort.
  Checked checkpoint export and reload, paired and single-input inference,
  Grad-CAM and conformal set generation.
- Ran CPU forward passes with untrained EfficientNet, ConvNeXt, ViT and DINOv2.
- Opened all three Streamlit pages and exercised the placeholder recipe flow.
  Parsed every notebook code cell and checked all Python modules compile.

Full training sweeps, final evaluation, pretrained weight downloads, Google Colab
GPU execution, and live camera capture remain to be run. Smoke-test outputs and
weights were kept outside the deliverable. The results table is header-only.

## Assumptions and TODOs

- Schema keys are verified against the downloaded revision. There are no unverified
  column-name TODOs. Recipe distance and coarse label mapping are project choices.
- P1–P6 notebooks each include a TODO to record actual settings, findings and limitations.
- P2 must choose the final classical models using validation.
- P4 must choose the final image model using validation.
- P5 must choose the backbone before test and use a CNN for the supplied Grad-CAM path.
- P6 must lock the fusion strategy and training budget before final ablation.
- Confirm with the instructor that recipe-similarity groups are an acceptable substitute
  for physical batch IDs, which the processed data does not expose.

## Publish to GitHub

Create an empty repository named GlazeVision on your GitHub account. Choose the
visibility appropriate for your team and DEPI. From the extracted GlazeVision folder:

```bash
git init -b main
git add .
git commit -m "feat: initialize GlazeVision graduation project"
git remote add origin https://github.com/rewanebrahembadr-cmyk/GlazeVision.git
git push -u origin main
```

With GitHub CLI authenticated, create the six issues and four milestones:

```bash
python scripts/setup_github.py rewanebrahembadr-cmyk/GlazeVision
```

The GitHub endpoint for this repository returned 404 during delivery. The remote,
issues and milestones were therefore not created in this session.

## File inventory

- `.github/ISSUE_TEMPLATE/task.md`
- `.github/pull_request_template.md`
- `.gitignore`
- `CONTRIBUTING.md`
- `DELIVERY.md`
- `LICENSE`
- `README.md`
- `app/pages/1_Model_Comparison.py`
- `app/pages/2_About.py`
- `app/streamlit_app.py`
- `configs/config.yaml`
- `data/README.md`
- `docs/team/P1_data_eda.md`
- `docs/team/P2_classical_ml.md`
- `docs/team/P3_features_clustering.md`
- `docs/team/P4_cnn_transfer_learning.md`
- `docs/team/P5_advanced_dl.md`
- `docs/team/P6_multimodal_app_lead.md`
- `models/.gitkeep`
- `notebooks/00_quickstart_colab.ipynb`
- `notebooks/01_data_eda.ipynb`
- `notebooks/02_classical_ml.ipynb`
- `notebooks/03_features_clustering.ipynb`
- `notebooks/04_cnn_transfer_learning.ipynb`
- `notebooks/05_advanced_dl_explainability.ipynb`
- `notebooks/06_multimodal_fusion.ipynb`
- `presentation/outline.md`
- `pyproject.toml`
- `reports/figures/.gitkeep`
- `reports/findings_and_limitations.md`
- `reports/results.csv`
- `requirements.txt`
- `scripts/download_data.py`
- `scripts/make_split.py`
- `scripts/setup_github.py`
- `src/glazevision/__init__.py`
- `src/glazevision/config.py`
- `src/glazevision/data.py`
- `src/glazevision/evaluate.py`
- `src/glazevision/experiments.py`
- `src/glazevision/explain.py`
- `src/glazevision/features.py`
- `src/glazevision/fusion.py`
- `src/glazevision/image_data.py`
- `src/glazevision/image_models.py`
- `src/glazevision/predict.py`
- `src/glazevision/tabular_models.py`
- `src/glazevision/train.py`
- `src/glazevision/uncertainty.py`
- `src/glazevision/utils.py`
