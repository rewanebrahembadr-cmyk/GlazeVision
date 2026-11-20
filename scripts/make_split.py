import argparse
from pathlib import Path

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors

from glazevision.config import CONFIG, ROOT
from glazevision.data import ingredients_vector, load_samples


def recipe_groups(samples):
    vectors = samples.ingredients.map(ingredients_vector).tolist()
    # Records without comparable ingredients share a group to keep them together.
    matrix = DictVectorizer().fit_transform(vectors)
    parents = list(range(len(vectors)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    neighbors = NearestNeighbors(radius=CONFIG["recipe_group_distance"],
                                 metric="euclidean", algorithm="brute").fit(matrix)
    for start in range(0, len(vectors), 256):
        matches = neighbors.radius_neighbors(matrix[start:start + 256], return_distance=False)
        for index, related in enumerate(matches, start):
            for other in related:
                if other < index:
                    parents[root(index)] = root(int(other))
    return [f"recipe-{root(index)}" for index in range(len(vectors))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=ROOT / CONFIG["raw"])
    parser.add_argument("--output", type=Path, default=ROOT / CONFIG["split"])
    args = parser.parse_args()
    samples = load_samples(args.raw).sort_values("sample_id").reset_index(drop=True)
    samples["group_id"] = recipe_groups(samples)
    reserve = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=CONFIG["seed"])
    train_val, test = next(reserve.split(samples, groups=samples.group_id))
    validation = GroupShuffleSplit(n_splits=1, test_size=.125, random_state=CONFIG["seed"])
    _, val = next(validation.split(samples.iloc[train_val], groups=samples.iloc[train_val].group_id))
    samples["split"] = "train"
    samples.loc[test, "split"] = "test"
    samples.loc[train_val[val], "split"] = "val"
    assert samples.groupby("group_id").split.nunique().max() == 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    samples[["sample_id", "split", "group_id"]].to_csv(args.output, index=False)
    print(samples.groupby("split").size().to_string())


if __name__ == "__main__":
    main()
