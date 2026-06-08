"""
Repository cloner and file tree traversal.

Clones a Github repository into a session-scoped workspace
directory and produces a structures FileMetadata and RepoMetadata objects
for downstream processing.
"""

import subprocess
import uuid
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from config import EXTENSION_MAP
from settings import settings
from logger import setup_logger

logger = setup_logger(__name__)

#CONSTANTS
IGNORED_DIRECTORIES: set[str] = {
    "node_modules", ".git", "__pycache__", ".venv", "venv",
    "env", ".env", "dist", "build", ".next", ".nuxt",
    "vendor", "target", ".gradle", ".idea", ".vscode",
    "coverage", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "eggs", ".eggs", "htmlcov", "site-packages",
}

IGNORED_EXTENSIONS: set[str] = {
    ".pyc", ".pyo", ".pyd",
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".webp",
    ".pdf", ".zip", ".tar", ".gz", ".rar", ".7z",
    ".woff", ".woff2", ".ttf", ".eot",
    ".map", ".lock",
    ".bin", ".exe", ".dll", ".so", ".dylib",
    ".db", ".sqlite", ".sqlite3",
}

TEST_PATH_PATTERNS: tuple[str, ...] = (
    "test_", "_test", "/tests/", "/test/", "/spec/", "_spec",
    "\\tests\\", "\\test\\", "\\spec\\",
)

ENTRY_POINT_NAMES: set[str] = {
    "main.py", "app.py", "__main__.py", "manage.py", "wsgi.py",
    "asgi.py", "server.py", "run.py", "index.ts", "index.js",
    "main.ts", "main.js", "app.ts", "app.js", "server.ts",
    "main.go", "main.rs", "Main.java", "Program.cs",
}


# Data models
class FileMetadata(BaseModel):
    """structured metadata for a single file in a cloned repository."""
    path: str       # relative path from repo root
    absolute_path: str     # full path on disk
    language: Optional[str]    # detected from extension
    extension: str       # file extension including dot
    size_bytes: int      # raw file size
    line_count: int     # number of lines
    is_test_file: bool   # detected from path patterns
    is_entry_point: bool  # detected from file name
    is_supported: bool     # true if language is in EXTENSION_MAP


class RepoMetadata(BaseModel):
    """structured metadata for a cloned repository."""
    session_id: str     ##UUID for this clone session
    url: str      # original github url
    owner: str    # github owner (user/org)
    name: str     #repo name
    local_path: str     #absolute path to cloned repo on disk
    cloned_at: str     #ISO timestamp of clone operation
    total_files: int    # all files found (including ignored)
    indexed_files: int   # files that passed all filters
    languages: dict[str, int]   # language -> file count
    files: list[FileMetadata]   # only the indexed files



# helper functions
def _parse_github_url(url: str) -> tuple[str, str]:
    """
    Extract owner and repo name from a Github URL

    Handles:
        https://github.com/owner/repo
        https://github.com/owner/repo.git
        github.com/owner/repo

    Returns:
        (owner, name) tuple

    Raises:
        ValueError if URL cannot be parsed
    """
    url = url.rstrip("/").removesuffix(".git")

    # strip protocol
    if "://" in url:
        url = url.split("://", 1)[1]

    parts = url.split("/")

    if len(parts) < 3 or parts[0] not in ("github.com",):
        raise ValueError(
            f"Cannot parse Github URL: '{url}. "
            "Expected format: https://github.com/owner/repo"
        )
    
    owner = parts[1]
    name = parts[2]
    return owner,name


def _is_ignored_path(path: Path) -> bool:
    """
    Return true if the path should be skipped entirely
    Checks directory components and file extension against ignore lists.
    """

    # check every directory component in the path
    for part in path.parts:
        if part in IGNORED_DIRECTORIES:
            return True
        
    # check extension
    suffix = path.suffix.lower()
    if suffix in IGNORED_EXTENSIONS:
        return True
    
    return False

def _is_test_file(relative_path: str) -> bool:
    """
    Detect test files from common naming conventions.
    """
    normalized = relative_path.replace("\\", "/").lower()
    return any(pattern in normalized for pattern in TEST_PATH_PATTERNS)


def _is_entry_point(filename: str) -> bool:
    """
    Detect entry points from common filename conventions."""
    return filename in ENTRY_POINT_NAMES


def _count_lines(file_path: Path) -> int:
    """
    Count  lines in a file safely.
    Returns 0 if the file cannot be read as text (binary content)
    """
    try:
        return sum(1 for _ in file_path.open(encoding="utf-8", errors="ignore"))
    except OSError:
        return 0
    

def _detect_language(extension: str) -> Optional[str]:
    """
    Map file extension to language using shared EXTENSION_MAP."""
    return EXTENSION_MAP.get(extension.lower())


# core functions
def _clone_repo(url: str, target_path: Path) -> bool:
    """
    Shallow-clone a repo to target_path using subprocess git.
    Returns true on success, false on failure.
    """
    target_path.parent.mkdir(parents=True, exist_ok = True)

    command = [
        "git", "clone",
        "--depth=1",
        "--single-branch",
        "--no-tags",
        url,
        str(target_path),
    ]

    logger.info(f"Cloning {url} into {target_path}")

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"Git clone failed: {result.stderr.strip()}")
        return False
    
    logger.info(f"Clone complete: {target_path}")
    return True


# file tree traversal
def _traverse(repo_path: Path) -> list[FileMetadata]:
    """
    Walk the cloned repo and collect FileMetadata for every non-ignored file.

    Only collects files that:
    - Are not in ignored directories
    - Do not have ignored extensions
    - Are within the size limit
    - Are non-empty
    """
    files: list[FileMetadata] = []
    total_seen = 0

    for file_path in repo_path.rglob("*"):
        if not file_path.is_file():
            continue

        total_seen += 1

        # Get path relative to repo root for clean reporting
        try:
            relative = file_path.relative_to(repo_path)
        except ValueError:
            continue

        # Apply ignore rules
        if _is_ignored_path(relative):
            continue

        # Size check
        try:
            size = file_path.stat().st_size
        except OSError:
            continue

        if size == 0:
            continue

        if size > settings.MAX_FILE_SIZE_BYTES:
            logger.debug(f"Skipping oversized file: {relative} ({size} bytes)")
            continue

        extension = file_path.suffix.lower()
        language = _detect_language(extension)
        relative_str = str(relative)

        metadata = FileMetadata(
            path=relative_str,
            absolute_path=str(file_path),
            language=language,
            extension=extension,
            size_bytes=size,
            line_count=_count_lines(file_path),
            is_test_file=_is_test_file(relative_str),
            is_entry_point=_is_entry_point(file_path.name),
            is_supported=language is not None,
        )

        files.append(metadata)

    logger.info(f"Traversal complete: {len(files)} files indexed from {total_seen} total")
    return files



# Public interface
class RepoCloner:
    """
    Clones a GitHub repository and produces structured RepoMetadata.

    Usage:
        cloner = RepoCloner()
        metadata = cloner.clone("https://github.com/tiangolo/fastapi")
        print(metadata.indexed_files)
        print(metadata.languages)

    The cloner manages its own workspace directory under
    settings.REPO_WORKSPACE_DIR. Each clone gets a unique session ID.

    Call cleanup(session_id) to delete a cloned repo from disk when done.
    """

    def __init__(self, workspace_dir: Optional[str] = None) -> None:
        self._workspace = Path(workspace_dir or settings.REPO_WORKSPACE_DIR)

    def clone(self, url: str) -> RepoMetadata:
        """
        Clone a GitHub repository and return structured metadata.

        Args:
            url: GitHub repository URL

        Returns:
            RepoMetadata with full file inventory

        Raises:
            ValueError: if URL cannot be parsed
            RuntimeError: if git clone fails
        """
        owner, name = _parse_github_url(url)
        session_id = str(uuid.uuid4())
        session_dir = self._workspace / session_id[:8]
        repo_path = session_dir / name

        success = _clone_repo(url, repo_path)

        if not success:
            raise RuntimeError(f"Failed to clone repository: {url}")

        files = _traverse(repo_path)

        # Build language frequency map from indexed files only
        languages: dict[str, int] = {}
        for f in files:
            if f.language:
                languages[f.language] = languages.get(f.language, 0) + 1

        # Count total files on disk for reporting
        total_on_disk = sum(1 for p in repo_path.rglob("*") if p.is_file())

        metadata = RepoMetadata(
            session_id=session_id,
            url=url,
            owner=owner,
            name=name,
            local_path=str(repo_path),
            cloned_at=datetime.utcnow().isoformat(),
            total_files=total_on_disk,
            indexed_files=len(files),
            languages=languages,
            files=files,
        )

        logger.info(
            f"Session {session_id[:8]} | {owner}/{name} | "
            f"{metadata.indexed_files} files | "
            f"languages: {list(languages.keys())}"
        )

        return metadata

    def cleanup(self, session_id: str) -> bool:
        """
        Delete the cloned repository for a given session.

        Args:
            session_id: The session_id from RepoMetadata

        Returns:
            True if deleted, False if session directory not found
        """
        session_dir = self._workspace / session_id[:8]

        if not session_dir.exists():
            logger.warning(f"Session directory not found: {session_dir}")
            return False

        shutil.rmtree(session_dir)
        logger.info(f"Cleaned up session: {session_id[:8]}")
        return True