# P3 — Chemistry features and clustering

**Difficulty** Basic to intermediate

**Course** Machine Learning with Python, session 6

## Goal

Chemistry features and clustering

## Inputs

Recipe fields and split from P1; reference classifiers from P2

## Tasks

- [ ] Build UMF features, oxide groups and silica/alumina ratio.
- [ ] Run controlled ablation with and without additional chemistry features.
- [ ] Fit K-Means and hierarchical clustering without target labels.
- [ ] Plot PCA or UMAP and compare clusters to observed finish after fitting.

## Outputs

- `src/glazevision/features.py`
- `notebooks/03_features_clustering.ipynb`
- `reports/results.csv`

## Definition of done

Feature comparison uses the same split and clustering fit excludes labels.

## Bonus

Try RGB regression or permutation importance.
