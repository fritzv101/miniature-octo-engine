"""Create/update a Hugging Face Docker Space serving the oncomap API.

    HF_TOKEN=... python deploy/huggingface/deploy.py --space user/oncomap \
        --model model.joblib --compounds compounds.csv [--api-key KEY]

The API key (if given) is stored as the Space secret ONCOMAP_API_KEY, never in files.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", required=True, help="owner/name")
    parser.add_argument("--model", required=True)
    parser.add_argument("--compounds", required=True)
    parser.add_argument("--api-key", default=os.environ.get("ONCOMAP_API_KEY"))
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(
        args.space, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True
    )
    if args.api_key:
        api.add_space_secret(args.space, "ONCOMAP_API_KEY", args.api_key)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        shutil.copytree(ROOT / "src", out / "src", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy(ROOT / "pyproject.toml", out)
        shutil.copy(HERE / "Dockerfile", out / "Dockerfile")
        shutil.copy(HERE / "SPACE_README.md", out / "README.md")
        (out / "models").mkdir()
        shutil.copy(args.model, out / "models" / "model.joblib")
        shutil.copy(args.compounds, out / "models" / "compounds.csv")
        (out / ".gitattributes").write_text("*.joblib filter=lfs diff=lfs merge=lfs -text\n")
        api.upload_folder(
            repo_id=args.space,
            repo_type="space",
            folder_path=str(out),
            commit_message="Deploy oncomap API",
            delete_patterns=["src/**"],
        )
    owner, name = args.space.split("/")
    print(f"https://{owner.lower()}-{name.lower().replace('_', '-')}.hf.space")


if __name__ == "__main__":
    main()
