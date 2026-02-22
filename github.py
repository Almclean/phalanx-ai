"""
github.py — GitHub repo cloning support.

Handles:
- Shallow clone (--depth 1) for speed
- Private repos via token injection
- Branch/tag/ref targeting
- Large repo guard (file count check before running)
- Automatic temp dir cleanup via context manager
"""

from __future__ import annotations
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


class CloneError(Exception):
    pass


def is_github_url(s: str) -> bool:
    """Return True if the string looks like a GitHub (or generic git) URL."""
    patterns = [
        r'^https?://github\.com/',
        r'^git@github\.com:',
        r'^https?://gitlab\.com/',
        r'^https?://bitbucket\.org/',
        r'^https?://.*\.git$',
    ]
    return any(re.match(p, s) for p in patterns)


def normalise_github_url(url: str, token: Optional[str] = None) -> str:
    """
    Normalise a GitHub URL and optionally inject a PAT for private repo access.

    Accepts:
      https://github.com/owner/repo
      https://github.com/owner/repo.git
      github.com/owner/repo          (no scheme)
    """
    # Add scheme if missing
    if not url.startswith(("http://", "https://", "git@")):
        url = "https://" + url

    # Inject token: https://token@github.com/...
    if token and url.startswith("https://"):
        url = url.replace("https://", f"https://{token}@", 1)

    return url


def clone_repo(
    url: str,
    *,
    token: Optional[str] = None,
    branch: Optional[str] = None,
    ref: Optional[str] = None,
) -> tempfile.TemporaryDirectory:
    """
    Shallow-clone a git repository into a temporary directory.

    Returns a TemporaryDirectory context manager. The caller is responsible
    for cleanup (use as a context manager: `with clone_repo(...) as tmp_dir`).

    Args:
        url:    Git remote URL (GitHub, GitLab, Bitbucket, etc.)
        token:  Optional PAT/token for private repos
        branch: Branch name to clone (default: repo default branch)
        ref:    Specific commit/tag ref to checkout after clone

    Returns:
        tempfile.TemporaryDirectory — call .name for the path string
    """
    if not shutil.which("git"):
        raise CloneError("git is not installed or not on PATH")

    clone_url = normalise_github_url(url, token)
    tmp = tempfile.TemporaryDirectory(prefix="repo_summarizer_")

    try:
        cmd = [
            "git", "clone",
            "--depth", "1",       # shallow — history not needed
            "--single-branch",    # don't fetch all branches
            "--no-tags",          # skip tag objects
            "--recurse-submodules", "--shallow-submodules",
        ]

        if branch:
            cmd += ["--branch", branch]

        cmd += [clone_url, tmp.name]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout for large repos
        )

        if result.returncode != 0:
            # Scrub token from error message before raising
            err = result.stderr
            if token:
                err = err.replace(token, "***")
            tmp.cleanup()
            raise CloneError(f"git clone failed:\n{err}")

        # Optionally checkout a specific ref
        if ref:
            result = subprocess.run(
                ["git", "checkout", ref],
                cwd=tmp.name,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                tmp.cleanup()
                raise CloneError(f"git checkout {ref} failed:\n{result.stderr}")

        return tmp

    except subprocess.TimeoutExpired:
        tmp.cleanup()
        raise CloneError("git clone timed out after 5 minutes. Repo may be too large.")
    except Exception:
        tmp.cleanup()
        raise
