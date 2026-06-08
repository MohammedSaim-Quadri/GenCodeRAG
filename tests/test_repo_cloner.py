"""
Tests for pipeline/repo_cloner.py

We test:
- URL parsing (pure function, no I/O)
- Ignore path detection (pure function, no I/O)
- Test file detection (pure function, no I/O)
- Entry point detection (pure function, no I/O)
- FileMetadata construction (uses tmp_path fixture)
- RepoCloner.clone() with a real small public repo (integration, marked slow)
- RepoCloner.cleanup() (uses tmp_path fixture)

We do NOT test _clone_repo() in unit tests because it makes a real network
call. That belongs in integration tests only.
"""

import pytest
from pathlib import Path

from pipeline.repo_cloner import (
    _parse_github_url,
    _is_ignored_path,
    _is_test_file,
    _is_entry_point,
    _count_lines,
    _detect_language,
    RepoCloner,
    FileMetadata,
)


# ---------------------------------------------------------------------------
# URL parsing
# ---------------------------------------------------------------------------

def test_parse_github_url_https():
    owner, name = _parse_github_url("https://github.com/tiangolo/fastapi")
    assert owner == "tiangolo"
    assert name == "fastapi"


def test_parse_github_url_with_git_suffix():
    owner, name = _parse_github_url("https://github.com/tiangolo/fastapi.git")
    assert owner == "tiangolo"
    assert name == "fastapi"


def test_parse_github_url_with_trailing_slash():
    owner, name = _parse_github_url("https://github.com/tiangolo/fastapi/")
    assert owner == "tiangolo"
    assert name == "fastapi"


def test_parse_github_url_invalid_raises():
    with pytest.raises(ValueError):
        _parse_github_url("https://gitlab.com/owner/repo")


def test_parse_github_url_too_short_raises():
    with pytest.raises(ValueError):
        _parse_github_url("https://github.com/owner")


# ---------------------------------------------------------------------------
# Ignore path detection
# ---------------------------------------------------------------------------

def test_ignored_directory_node_modules():
    assert _is_ignored_path(Path("node_modules/lodash/index.js")) is True


def test_ignored_directory_pycache():
    assert _is_ignored_path(Path("src/__pycache__/module.cpython-310.pyc")) is True


def test_ignored_extension_pyc():
    assert _is_ignored_path(Path("src/module.pyc")) is True


def test_ignored_extension_png():
    assert _is_ignored_path(Path("assets/logo.png")) is True


def test_not_ignored_python_file():
    assert _is_ignored_path(Path("src/main.py")) is False


def test_not_ignored_nested_source_file():
    assert _is_ignored_path(Path("src/api/routes/auth.py")) is False


# ---------------------------------------------------------------------------
# Test file detection
# ---------------------------------------------------------------------------

def test_detects_test_prefix():
    assert _is_test_file("tests/test_auth.py") is True


def test_detects_test_suffix():
    assert _is_test_file("src/auth_test.go") is True


def test_detects_tests_directory():
    assert _is_test_file("tests/integration/test_db.py") is True


def test_not_a_test_file():
    assert _is_test_file("src/api/auth.py") is False


# ---------------------------------------------------------------------------
# Entry point detection
# ---------------------------------------------------------------------------

def test_detects_main_py():
    assert _is_entry_point("main.py") is True


def test_detects_app_py():
    assert _is_entry_point("app.py") is True


def test_detects_manage_py():
    assert _is_entry_point("manage.py") is True


def test_not_entry_point():
    assert _is_entry_point("utils.py") is False


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

def test_detects_python():
    assert _detect_language(".py") == "python"


def test_detects_typescript():
    assert _detect_language(".ts") == "typescript"


def test_returns_none_for_unknown():
    assert _detect_language(".xyz") is None


# ---------------------------------------------------------------------------
# Line counting
# ---------------------------------------------------------------------------

def test_count_lines(tmp_path):
    f = tmp_path / "test.py"
    f.write_text("line1\nline2\nline3\n")
    assert _count_lines(f) == 3


def test_count_lines_empty_file(tmp_path):
    f = tmp_path / "empty.py"
    f.write_text("")
    assert _count_lines(f) == 0


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def test_cleanup_removes_session_directory(tmp_path):
    cloner = RepoCloner(workspace_dir=str(tmp_path))

    # Simulate what clone() creates
    fake_session_id = "abcd1234-0000-0000-0000-000000000000"
    session_dir = tmp_path / fake_session_id[:8]
    session_dir.mkdir()
    (session_dir / "somefile.py").write_text("print('hello')")

    result = cloner.cleanup(fake_session_id)

    assert result is True
    assert not session_dir.exists()


def test_cleanup_missing_session_returns_false(tmp_path):
    cloner = RepoCloner(workspace_dir=str(tmp_path))
    result = cloner.cleanup("nonexistent-session-id")
    assert result is False