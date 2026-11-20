"""Create the project board's milestones and role issues on a published repo."""

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MILESTONES = (
    "Phase 1: Data & Split",
    "Phase 2: Models",
    "Phase 3: Fusion & App",
    "Phase 4: Presentation",
)


def github(*arguments):
    return subprocess.check_output(["gh", *arguments], text=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repository", help="GitHub repository in owner/name format")
    repository = parser.parse_args().repository
    base = f"repos/{repository}"
    milestones = {row["title"] for row in json.loads(github(
        "api", f"{base}/milestones?state=all&per_page=100"))}
    for title in MILESTONES:
        if title not in milestones:
            github("api", f"{base}/milestones", "--method", "POST", "-f", f"title={title}")
    existing = json.loads(github("issue", "list", "--repo", repository, "--state", "all",
                                 "--limit", "1000", "--json", "title,url"))
    known = {row["title"]: row["url"] for row in existing}
    for path in sorted((ROOT / "docs/team").glob("P*.md")):
        title = path.read_text().splitlines()[0].removeprefix("# ")
        role = int(path.name[1])
        milestone = MILESTONES[0 if role == 1 else 2 if role == 6 else 1]
        print(known.get(title) or github("issue", "create", "--repo", repository, "--title", title,
                                       "--body-file", str(path), "--milestone", milestone).strip())


if __name__ == "__main__":
    main()
