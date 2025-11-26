"""Environment and git information utilities.

This module provides utilities for logging and retrieving information about
the runtime environment and git repository state for reproducibility.
"""

import json
import logging
import platform
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Packages to check versions for
_KEY_PACKAGES = [
    # Core pipeline dependencies
    "numpy",
    "pandas",
    "scipy",
    "scikit-image",
    "tifffile",
    "matplotlib",
    "Pillow",
    "pyyaml",
    "zstandard",
    "tqdm",
    # GPU/ML (optional)
    "tensorflow",
    "cupy",
    "deepcell",
    "torch",
    # Our package
    "aegle",
]

# Cached values
_CACHED_GIT_INFO: Optional[Dict] = None
_CACHED_ENV_INFO: Optional[Dict] = None
_CACHED_SYSTEM_INFO: Optional[Dict] = None


def _get_repo_root() -> Path:
    """Get the repository root directory."""
    return Path(__file__).parent.parent


def _run_git_command(args: List[str], cwd: Path = None) -> Optional[str]:
    """Run a git command and return stripped output.

    Args:
        args: Git command arguments
        cwd: Working directory (defaults to repo root)

    Returns:
        Command output stripped, or None on failure
    """
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=True,
            cwd=cwd or _get_repo_root(),
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.debug(f"Git command failed: {args}, error: {e}")
        return None


def _get_package_version(package_name: str) -> Optional[str]:
    """Get version of installed package.

    Args:
        package_name: Name of the package

    Returns:
        Version string, or None if not installed
    """
    try:
        import importlib.metadata

        return importlib.metadata.version(package_name)
    except Exception:
        return None


def get_git_info(use_cache: bool = True) -> Dict:
    """Get git repository information.

    Args:
        use_cache: If True, return cached result on subsequent calls.

    Returns:
        dict: Git information with keys:
            - is_git_repo: bool
            - commit_sha: str or None (full SHA)
            - commit_sha_short: str or None (7-char SHA)
            - branch: str or None
            - is_dirty: bool or None
            - dirty_files: list[str] or None
    """
    global _CACHED_GIT_INFO
    if use_cache and _CACHED_GIT_INFO is not None:
        return _CACHED_GIT_INFO

    result = {
        "is_git_repo": False,
        "commit_sha": None,
        "commit_sha_short": None,
        "branch": None,
        "is_dirty": None,
        "dirty_files": None,
    }

    # Check if we're in a git repo
    git_dir = _run_git_command(["git", "rev-parse", "--git-dir"])
    if git_dir is None:
        logger.debug("Not a git repository or git not available")
        if use_cache:
            _CACHED_GIT_INFO = result
        return result

    result["is_git_repo"] = True

    # Get commit SHA
    sha = _run_git_command(["git", "rev-parse", "HEAD"])
    if sha:
        result["commit_sha"] = sha
        result["commit_sha_short"] = sha[:7]

    # Get branch name
    branch = _run_git_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    if branch:
        result["branch"] = branch

    # Get dirty status
    status = _run_git_command(["git", "status", "--porcelain"])
    if status is not None:
        dirty_files = [line.strip() for line in status.split("\n") if line.strip()]
        result["is_dirty"] = len(dirty_files) > 0
        result["dirty_files"] = dirty_files if dirty_files else []

    if use_cache:
        _CACHED_GIT_INFO = result
    return result


def get_environment_info(use_cache: bool = True) -> Dict:
    """Get Python environment and package version information.

    Args:
        use_cache: If True, return cached result on subsequent calls.

    Returns:
        dict: Environment information with keys:
            - python_version: str
            - python_executable: str
            - packages: dict[str, str or None] (package_name -> version)
    """
    global _CACHED_ENV_INFO
    if use_cache and _CACHED_ENV_INFO is not None:
        return _CACHED_ENV_INFO

    result = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "packages": {},
    }

    # Get package versions
    for package in _KEY_PACKAGES:
        result["packages"][package] = _get_package_version(package)

    if use_cache:
        _CACHED_ENV_INFO = result
    return result


def get_system_info(use_cache: bool = True) -> Dict:
    """Get system information.

    Args:
        use_cache: If True, return cached result on subsequent calls.

    Returns:
        dict: System information with keys:
            - hostname: str
            - os: str
            - os_version: str
            - platform: str
            - architecture: str
    """
    global _CACHED_SYSTEM_INFO
    if use_cache and _CACHED_SYSTEM_INFO is not None:
        return _CACHED_SYSTEM_INFO

    result = {
        "hostname": socket.gethostname(),
        "os": platform.system(),
        "os_version": platform.release(),
        "platform": sys.platform,
        "architecture": platform.machine(),
    }

    if use_cache:
        _CACHED_SYSTEM_INFO = result
    return result


def get_full_env_info(use_cache: bool = True) -> Dict:
    """Get combined git, environment, and system information.

    Args:
        use_cache: If True, return cached results for sub-components.

    Returns:
        dict: Combined information with keys:
            - timestamp: str (ISO format)
            - git: dict (from get_git_info)
            - python: dict (version and executable)
            - packages: dict (package versions)
            - system: dict (from get_system_info)
    """
    env_info = get_environment_info(use_cache=use_cache)

    return {
        "timestamp": datetime.now().isoformat(),
        "git": get_git_info(use_cache=use_cache),
        "python": {
            "version": env_info["python_version"],
            "executable": env_info["python_executable"],
        },
        "packages": env_info["packages"],
        "system": get_system_info(use_cache=use_cache),
    }


def log_env_info(log: logging.Logger = None) -> Dict:
    """Log environment and git information in banner format.

    This is the main entry point for CLI scripts. Logs a banner-style
    block of environment information and returns the data.

    Args:
        log: Logger instance to use. If None, uses module logger.

    Returns:
        dict: Full environment info (same as get_full_env_info)
    """
    if log is None:
        log = logger

    info = get_full_env_info()

    # Build banner
    separator = "=" * 60
    log.info(separator)
    log.info("AEGLE PIPELINE - RUN ENVIRONMENT")
    log.info(separator)

    # Git info
    git = info["git"]
    if git["is_git_repo"]:
        dirty_str = " (dirty)" if git["is_dirty"] else " (clean)"
        log.info(f"Git:        {git['commit_sha_short']} on {git['branch']}{dirty_str}")
        if git["is_dirty"] and git["dirty_files"]:
            # Show first few dirty files
            files_preview = git["dirty_files"][:3]
            if len(git["dirty_files"]) > 3:
                files_preview.append(f"... and {len(git['dirty_files']) - 3} more")
            log.info(f"  Modified: {', '.join(files_preview)}")
    else:
        log.info("Git:        Not a git repository")

    # Python info
    log.info(f"Python:     {info['python']['version']}")

    # System info
    system = info["system"]
    log.info(f"Platform:   {system['os']} {system['os_version']} ({system['architecture']})")
    log.info(f"Hostname:   {system['hostname']}")

    # Package versions (only installed ones, formatted nicely)
    packages = info["packages"]
    installed = {k: v for k, v in packages.items() if v is not None}
    if installed:
        # Format as comma-separated list, wrap if too long
        pkg_strs = [f"{k}={v}" for k, v in installed.items()]
        log.info(f"Packages:   {', '.join(pkg_strs)}")

    log.info(separator)

    return info


def save_run_metadata(output_dir: str, env_info: Dict = None) -> Optional[str]:
    """Save environment info to run_metadata.json in the output directory.

    Args:
        output_dir: Directory to save the metadata file
        env_info: Environment info dict (from get_full_env_info or log_env_info).
                  If None, will collect fresh info.

    Returns:
        str: Path to the saved file, or None on failure
    """
    if env_info is None:
        env_info = get_full_env_info()

    output_path = Path(output_dir)
    if not output_path.exists():
        logger.warning(f"Output directory does not exist: {output_dir}")
        return None

    metadata_file = output_path / "run_metadata.json"

    try:
        with open(metadata_file, "w") as f:
            json.dump(env_info, f, indent=2)
        logger.debug(f"Saved run metadata to {metadata_file}")
        return str(metadata_file)
    except Exception as e:
        logger.warning(f"Failed to save run metadata: {e}")
        return None
