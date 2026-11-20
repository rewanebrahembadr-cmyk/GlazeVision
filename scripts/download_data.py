import argparse
import shutil
from pathlib import Path
from zipfile import ZipFile

from huggingface_hub import snapshot_download

from glazevision.config import CONFIG, ROOT

DATASET = "AlpachinoNLP/GlazyBench"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--revision", default="a73a8f4ad8412c175f466f94b235a751ab8c3122")
    parser.add_argument("--output", type=Path, default=ROOT / CONFIG["raw"])
    args = parser.parse_args()
    patterns = ["property_prediction/*/*.json", "image_generation/*/*.json"]
    if not args.metadata_only:
        patterns.extend(["image_generation/*/images/*", "image_generation/*/images.zip"])
    snapshot_download(repo_id=DATASET, repo_type="dataset", revision=args.revision,
                      allow_patterns=patterns, local_dir=args.output)
    for archive in args.output.glob("image_generation/*/images.zip"):
        destination = archive.parent / "images"
        destination.mkdir(exist_ok=True)
        with ZipFile(archive) as images:
            for entry in images.infolist():
                name = Path(entry.filename).name
                if Path(name).suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"} or name.startswith("."):
                    continue
                with images.open(entry) as source, (destination / name).open("wb") as output:
                    shutil.copyfileobj(source, output)
    print(f"Downloaded {DATASET}@{args.revision} into {args.output}")


if __name__ == "__main__":
    main()
