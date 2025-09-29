from __future__ import annotations

from pathlib import Path

import pytest

from exps import test_config_generator as smoke


@pytest.mark.usefixtures("tmp_path")
def test_main_smoke(tmp_path: Path) -> None:
    smoke.case_main_smoke(tmp_path / "main")


@pytest.mark.usefixtures("tmp_path")
def test_preprocess_manual_mask(tmp_path: Path) -> None:
    smoke.case_preprocess_manual_mask(tmp_path / "pre_manual")


@pytest.mark.usefixtures("tmp_path")
def test_preprocess_missing_downscale_should_fail(tmp_path: Path) -> None:
    smoke.case_preprocess_missing_downscale_should_fail(tmp_path / "pre_missing")
