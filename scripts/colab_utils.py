# AI-assisted (Claude Code, claude.ai) — https://claude.ai
"""Shared helpers for running nocapchicken notebooks on Google Colab.

Adapted from spatialft/spatialft.github.io colab_utils pattern.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

REPO_URL = "https://github.com/nocapchicken/nocapchicken.github.io.git"
DEFAULT_REPO_DIR = Path("/content/nocapchicken.github.io")


def prepare_notebook(
    repo_dir: str | Path = DEFAULT_REPO_DIR,
    branch: str = "main",
) -> Path:
    """Clone the repo into Colab workspace, install deps, and return repo root."""
    repo_path = Path(repo_dir)
    if not repo_path.exists():
        subprocess.run(["git", "clone", REPO_URL, str(repo_path)], check=True)

    if branch != "main":
        subprocess.run(["git", "checkout", branch], check=True, cwd=repo_path)

    # Install project dependencies
    req = repo_path / "requirements.txt"
    if req.exists():
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(req)],
            check=True,
        )

    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    os.chdir(repo_path)
    return repo_path


def publish_artifacts(
    paths: Iterable[str | Path],
    message: str,
    repo_dir: str | Path = DEFAULT_REPO_DIR,
    branch: str = "main",
) -> bool:
    """Commit and push generated artifacts from Colab back to GitHub.

    Requires a Colab secret named GITHUB_TOKEN_NOCAPCHICKEN with repo write access.
    Returns True when a commit was created and pushed.
    """
    try:
        from google.colab import userdata
    except ImportError as exc:
        raise RuntimeError("publish_artifacts only works from Google Colab.") from exc

    token = userdata.get("GITHUB_TOKEN_NOCAPCHICKEN")
    if not token:
        raise RuntimeError(
            "Missing Colab secret GITHUB_TOKEN_NOCAPCHICKEN. "
            "Add it in Colab: key icon (left sidebar) → Add secret."
        )

    repo_path = Path(repo_dir)
    rel_paths = [str(Path(p)) for p in paths]

    missing = [p for p in rel_paths if not (repo_path / p).exists()]
    if missing:
        raise FileNotFoundError(f"Cannot publish — files not found: {', '.join(missing)}")

    repo_url = f"https://x-access-token:{token}@github.com/nocapchicken/nocapchicken.github.io.git"

    subprocess.run(["git", "config", "user.email", "colab-bot@nocapchicken"], check=True, cwd=repo_path)
    subprocess.run(["git", "config", "user.name", "Colab Bot"], check=True, cwd=repo_path)
    subprocess.run(["git", "remote", "set-url", "origin", repo_url], check=True, cwd=repo_path)

    # Stash artifacts, rebase on latest, re-apply
    stash_result = subprocess.run(
        ["git", "stash", "push", "--include-untracked", "-m", "colab-artifacts", "--", *rel_paths],
        check=True, cwd=repo_path, capture_output=True, text=True,
    )
    stashed = "No local changes to save" not in stash_result.stdout

    subprocess.run(["git", "pull", "--rebase", "origin", branch], check=True, cwd=repo_path)

    if stashed:
        subprocess.run(["git", "stash", "pop"], check=True, cwd=repo_path)

    subprocess.run(["git", "add", "--", *rel_paths], check=True, cwd=repo_path)

    diff = subprocess.run(["git", "diff", "--cached", "--quiet", "--", *rel_paths], cwd=repo_path)
    if diff.returncode == 0:
        print("No artifact changes to commit.")
        return False

    subprocess.run(["git", "commit", "-m", message], check=True, cwd=repo_path)
    subprocess.run(["git", "push", "origin", branch], check=True, cwd=repo_path)
    print(f"Pushed artifacts: {', '.join(rel_paths)}")
    return True
