# GlazyBench data

Run `python scripts/download_data.py` from the repository root. Use
`--metadata-only` for EDA and classical ML. Use `--revision <commit>` to
reproduce a specific snapshot. The image ZIPs are extracted into each
partition's `images/` directory. Then run `python scripts/make_split.py`.

| Verified source | Keys used |
|---|---|
| `property_prediction/{train,test}/recipes.json` | `id`, `ingredients` (material and amount), `chemical_composition`, `umf`, `cone`, `atmosphere` |
| `property_prediction/{train,test}/targets.json` | `id`, `surface`, `transparency`, `color_family` |
| `image_generation/{train,test}/metadata.json` | keyed by ID; `id`, `image_filename`, `surface`, `transparency`, `color_family`, `ingredients`, `chemical_composition`, `cone`, `atmosphere` |

Join tracks by `id`. Preserve the image annotations as separate fields because
their labels may differ from the recipe benchmark. Missing labels are excluded
from that task's evaluation. Never use label fields or target-derived metadata
as recipe features. Model features preserve leading zeros in Orton cones.

The starter mapping collapses nine finish labels into Glossy, Satin and Matte.
It keeps the three broad finishes interpretable and pools smaller subclasses.
The upstream image annotations map to 2,218 Glossy, 600 Satin and 894 Matte
examples before availability checks and our group split.

No kiln batch identifier or cooling duration was found in the inspected processed
recipe and image schemas. Recipe groups proxy similarity, not physical batches.
Near duplicates are defined as connected components of normalized ingredient
vectors within the Euclidean radius in `configs/config.yaml`. Ingredient aliases,
unknown quantities, firing schedules and recipe provenance limit this proxy.
The supplied source partitions are used to locate files. The generated split is
specific to this project and its scores are separate from the published benchmark.
