"""Tests for aegle/env_info.py environment and git info utilities."""

import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aegle.env_info import (
    get_git_info,
    get_environment_info,
    get_system_info,
    get_full_env_info,
    log_env_info,
    save_run_metadata,
    _get_package_version,
    _run_git_command,
)


class TestGetPackageVersion(unittest.TestCase):
    """Tests for _get_package_version helper."""

    def test_installed_package(self):
        """Should return version string for installed package."""
        version = _get_package_version("numpy")
        self.assertIsNotNone(version)
        self.assertIsInstance(version, str)

    def test_missing_package(self):
        """Should return None for non-existent package."""
        version = _get_package_version("nonexistent_package_xyz_12345")
        self.assertIsNone(version)


class TestGetGitInfo(unittest.TestCase):
    """Tests for get_git_info function."""

    def test_returns_dict_structure(self):
        """Should always return dict with expected keys."""
        info = get_git_info(use_cache=False)
        self.assertIn("is_git_repo", info)
        self.assertIn("commit_sha", info)
        self.assertIn("commit_sha_short", info)
        self.assertIn("branch", info)
        self.assertIn("is_dirty", info)
        self.assertIn("dirty_files", info)

    def test_in_git_repo(self):
        """When in git repo, should return commit info."""
        info = get_git_info(use_cache=False)
        # This test runs from the repo, so we expect git info
        if info["is_git_repo"]:
            self.assertIsNotNone(info["commit_sha"])
            self.assertEqual(len(info["commit_sha"]), 40)  # Full SHA
            self.assertEqual(len(info["commit_sha_short"]), 7)  # Short SHA
            self.assertIsNotNone(info["branch"])

    def test_caching(self):
        """Should cache results when use_cache=True."""
        info1 = get_git_info(use_cache=True)
        info2 = get_git_info(use_cache=True)
        # Should be the same object (cached)
        self.assertIs(info1, info2)

    @patch("aegle.env_info._run_git_command")
    def test_handles_git_not_available(self, mock_run):
        """Should handle git not available gracefully."""
        mock_run.return_value = None
        info = get_git_info(use_cache=False)
        # Should still return dict with is_git_repo=False
        self.assertIsInstance(info, dict)
        self.assertFalse(info["is_git_repo"])


class TestGetEnvironmentInfo(unittest.TestCase):
    """Tests for get_environment_info function."""

    def test_returns_dict_structure(self):
        """Should return dict with expected keys."""
        info = get_environment_info(use_cache=False)
        self.assertIn("python_version", info)
        self.assertIn("python_executable", info)
        self.assertIn("packages", info)

    def test_python_version_format(self):
        """Python version should be in expected format."""
        info = get_environment_info(use_cache=False)
        version = info["python_version"]
        # Should be like "3.8.10" or similar
        parts = version.split(".")
        self.assertGreaterEqual(len(parts), 2)
        self.assertTrue(all(p.isdigit() for p in parts[:2]))

    def test_packages_includes_numpy(self):
        """Should include numpy in packages (it's always installed)."""
        info = get_environment_info(use_cache=False)
        self.assertIn("numpy", info["packages"])
        self.assertIsNotNone(info["packages"]["numpy"])

    def test_caching(self):
        """Should cache results when use_cache=True."""
        info1 = get_environment_info(use_cache=True)
        info2 = get_environment_info(use_cache=True)
        self.assertIs(info1, info2)


class TestGetSystemInfo(unittest.TestCase):
    """Tests for get_system_info function."""

    def test_returns_dict_structure(self):
        """Should return dict with expected keys."""
        info = get_system_info(use_cache=False)
        self.assertIn("hostname", info)
        self.assertIn("os", info)
        self.assertIn("os_version", info)
        self.assertIn("platform", info)
        self.assertIn("architecture", info)

    def test_hostname_not_empty(self):
        """Hostname should not be empty."""
        info = get_system_info(use_cache=False)
        self.assertTrue(len(info["hostname"]) > 0)

    def test_caching(self):
        """Should cache results when use_cache=True."""
        info1 = get_system_info(use_cache=True)
        info2 = get_system_info(use_cache=True)
        self.assertIs(info1, info2)


class TestGetFullEnvInfo(unittest.TestCase):
    """Tests for get_full_env_info function."""

    def test_returns_combined_structure(self):
        """Should return combined dict with all sections."""
        info = get_full_env_info(use_cache=False)
        self.assertIn("timestamp", info)
        self.assertIn("git", info)
        self.assertIn("python", info)
        self.assertIn("packages", info)
        self.assertIn("system", info)

    def test_timestamp_format(self):
        """Timestamp should be ISO format."""
        info = get_full_env_info(use_cache=False)
        timestamp = info["timestamp"]
        # Should be parseable as ISO format (contains T separator)
        self.assertIn("T", timestamp)


class TestLogEnvInfo(unittest.TestCase):
    """Tests for log_env_info function."""

    def test_logs_and_returns(self):
        """Should log info and return dict."""
        with patch.object(logging.getLogger("aegle.env_info"), "info") as mock_log:
            result = log_env_info()
            self.assertIsInstance(result, dict)
            self.assertIn("git", result)
            self.assertIn("python", result)
            self.assertIn("system", result)
            # Should have logged multiple lines (banner format)
            self.assertTrue(mock_log.called)
            self.assertGreater(mock_log.call_count, 3)  # At least banner + 3 info lines

    def test_accepts_custom_logger(self):
        """Should use provided logger."""
        custom_logger = logging.getLogger("test_custom")
        with patch.object(custom_logger, "info") as mock_log:
            log_env_info(log=custom_logger)
            self.assertTrue(mock_log.called)


class TestSaveRunMetadata(unittest.TestCase):
    """Tests for save_run_metadata function."""

    def test_saves_json_file(self):
        """Should save valid JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_info = get_full_env_info()
            result = save_run_metadata(tmpdir, env_info)

            self.assertIsNotNone(result)
            self.assertTrue(Path(result).exists())

            # Verify it's valid JSON
            with open(result) as f:
                loaded = json.load(f)
            self.assertIn("git", loaded)
            self.assertIn("python", loaded)

    def test_returns_none_for_missing_dir(self):
        """Should return None if directory doesn't exist."""
        result = save_run_metadata("/nonexistent/path/12345", {})
        self.assertIsNone(result)

    def test_collects_fresh_info_if_none_provided(self):
        """Should collect fresh info if env_info is None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = save_run_metadata(tmpdir, None)

            self.assertIsNotNone(result)
            with open(result) as f:
                loaded = json.load(f)
            # Should have all sections
            self.assertIn("timestamp", loaded)
            self.assertIn("git", loaded)


class TestRunGitCommand(unittest.TestCase):
    """Tests for _run_git_command helper."""

    def test_returns_none_on_failure(self):
        """Should return None for invalid git command."""
        result = _run_git_command(["git", "invalid-command-xyz"])
        self.assertIsNone(result)

    def test_returns_string_on_success(self):
        """Should return stripped string on success."""
        result = _run_git_command(["git", "--version"])
        if result is not None:
            self.assertIsInstance(result, str)
            self.assertTrue(result.startswith("git version"))


if __name__ == "__main__":
    unittest.main()
