# P1 — Data preparation and EDA

**Difficulty** Basic

**Course** Data Analysis and Data Visualization with Python

## Goal

Data preparation and EDA

## Inputs

GlazyBench exports and the DEPI brief

## Tasks

- [ ] Download and inspect every benchmark file used by the loaders. Document keys and types.
- [ ] Audit missing data, duplicated IDs, quantities, labels and firing units.
- [ ] Publish class counts and justify the nine-to-three finish mapping.
- [ ] Generate and audit the shared group split. Notify P2–P6 before they train.
- [ ] Plot class balance, atmosphere and cone coverage.

## Outputs

- `data/README.md`
- `scripts/make_split.py`
- `data/splits/split.csv`
- `notebooks/01_data_eda.ipynb`

## Definition of done

Split regenerates with unchanged IDs and every recipe group stays in one partition.

## Bonus

Check blurry, duplicate and tiny images.
