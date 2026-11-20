# Contributing

Use the role brief in `docs/team/`. Branch from `main` and use one of
`p1-data`, `p2-classical`, `p3-features`, `p4-image`, `p5-advanced`,
or `p6-fusion-app`. Use commits such as `feat: add recipe features` and
`fix: handle missing labels`. Open PRs into `main` for P6 review.

Use snake_case for Python files and function names. Keep notebooks free of large
outputs, and leave raw data, weights, credentials and caches out of Git.

## Shared contracts

- Use `data/splits/split.csv` with `sample_id,split,group_id` and seed 42. Ask P1 for
  a new version before changing the split or mapping.
- Keep labels and mappings in `configs/config.yaml`. Use validation for development.
- Use `utils.save_results()` to append one verified row per model and task:
  `model_name,owner,task,input_type,accuracy,macro_f1,weighted_f1,per_class_json,notes,date`.
  Use `tabular`, `image` or `fusion` for input type and describe the cohort in notes.
  Before merging, rebase and preserve unrelated rows from other contributors.
- The app calls `predict(image=None, recipe=None)`. Return class probability maps for
  surface, transparency and color family, plus confidence, uncertain and gradcam.

After publishing the repository, an authenticated maintainer can run
`python scripts/setup_github.py OWNER/GlazeVision` to create the six role issues
and four milestones. The command skips existing issues and milestones by title.
