#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import yaml

EXPS_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = EXPS_DIR / "templates"
SCHEMAS_DIR = EXPS_DIR / "schemas"
GENERATOR = EXPS_DIR / "config_generator.py"


class SmokeTestError(RuntimeError):
    """Raised when a smoke test fails."""


@dataclass
class CommandResult:
    name: str
    returncode: int
    stdout: str
    stderr: str

    @property
    def output(self) -> str:
        return f"{self.stdout}{self.stderr}"


def run_generator(
    *,
    analysis_step: str,
    experiment_set: str,
    csv_path: Path,
    template_path: Path,
    schema_path: Path,
    output_dir: Path,
    python_bin: str | None = None,
) -> CommandResult:
    python_bin = python_bin or sys.executable
    cmd = [
        python_bin,
        str(GENERATOR),
        "--analysis-step",
        analysis_step,
        "--experiment-set",
        experiment_set,
        "--base-dir",
        str(EXPS_DIR),
        "--csv",
        str(csv_path),
        "--template",
        str(template_path),
        "--schema",
        str(schema_path),
        "--output-dir",
        str(output_dir),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    return CommandResult(
        name=experiment_set,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _write_csv(path: Path, header: str, rows: Iterable[str]) -> None:
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")


def case_main_smoke(tmp_dir: Path) -> Path:
    tmp_dir = tmp_dir.resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    image = tmp_dir / "image.tiff"
    antibodies = tmp_dir / "antibodies.tsv"
    model_dir = tmp_dir / "model"
    image.touch()
    antibodies.touch()
    model_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tmp_dir / "main_design.csv"
    header = (
        "exp_id,sample_id,data::file_name,data::antibodies_file,data::image_mpp,"
        "channels::nuclear_channel,channels::wholecell_channel,"
        "patching::split_mode,segmentation::model_path"
    )
    row = (
        f"main_smoke,sample1,{image},{antibodies},0.5,"
        f"DAPI,\"CD45,CD3\",full_image,{model_dir}"
    )
    _write_csv(csv_path, header, [row])

    output_dir = tmp_dir / "out"
    result = run_generator(
        analysis_step="main",
        experiment_set="main_smoke_tests",
        csv_path=csv_path,
        template_path=TEMPLATES_DIR / "main_template.yaml",
        schema_path=SCHEMAS_DIR / "main.yaml",
        output_dir=output_dir,
    )
    if result.returncode != 0:
        raise SmokeTestError(
            f"main_smoke failed with code {result.returncode}:\n{result.output}"
        )

    config_path = output_dir / "main_smoke" / "config.yaml"
    if not config_path.is_file():
        raise SmokeTestError(
            "Expected config.yaml to be created for main_smoke test"
        )
    return config_path


def case_preprocess_manual_mask(tmp_dir: Path) -> Path:
    tmp_dir = tmp_dir.resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    image = tmp_dir / "image.qptiff"
    mask = tmp_dir / "mask.json"
    output_dir_value = tmp_dir / "outputs"
    image.touch()
    mask.write_text("{\"regions\": []}\n", encoding="utf-8")

    csv_path = tmp_dir / "preprocess_design.csv"
    header = (
        "exp_id,data::file_name,tissue_extraction::output_dir,"
        "tissue_extraction::manual_mask_json,tissue_extraction::visualize"
    )
    row = f"pre_manual,{image},{output_dir_value},{mask},TRUE"
    _write_csv(csv_path, header, [row])

    output_dir = tmp_dir / "out"
    result = run_generator(
        analysis_step="preprocess",
        experiment_set="preprocess_manual_mask",
        csv_path=csv_path,
        template_path=TEMPLATES_DIR / "preprocess_template.yaml",
        schema_path=SCHEMAS_DIR / "preprocess.yaml",
        output_dir=output_dir,
    )
    if result.returncode != 0:
        raise SmokeTestError(
            "preprocess_manual_mask failed with code "
            f"{result.returncode}:\n{result.output}"
        )

    config_path = output_dir / "pre_manual" / "config.yaml"
    if not config_path.is_file():
        raise SmokeTestError(
            "Expected config.yaml to be created for preprocess manual mask test"
        )

    with config_path.open(encoding="utf-8") as f:
        generated_config = yaml.safe_load(f)

    downscale = generated_config.get("tissue_extraction", {}).get("downscale_factor")
    if downscale != 64:
        raise SmokeTestError(
            f"Expected downscale_factor to default to 64 when visualize=true, got {downscale}"
        )
    return config_path


def case_preprocess_missing_downscale_should_fail(tmp_dir: Path) -> CommandResult:
    tmp_dir = tmp_dir.resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    image = tmp_dir / "image.qptiff"
    output_dir_value = tmp_dir / "outputs"
    image.touch()

    csv_path = tmp_dir / "preprocess_missing_design.csv"
    header = (
        "exp_id,data::file_name,tissue_extraction::output_dir,"
        "tissue_extraction::manual_mask_json"
    )
    row = f"pre_fail,{image},{output_dir_value},NULL"
    _write_csv(csv_path, header, [row])

    output_dir = tmp_dir / "out"
    result = run_generator(
        analysis_step="preprocess",
        experiment_set="preprocess_missing_fields",
        csv_path=csv_path,
        template_path=TEMPLATES_DIR / "preprocess_template.yaml",
        schema_path=SCHEMAS_DIR / "preprocess.yaml",
        output_dir=output_dir,
    )

    if result.returncode == 0:
        raise SmokeTestError(
            "Expected schema validation failure but generator succeeded"
        )
    expected_tokens = (
        "tissue_extraction::downscale_factor",
        "value is required",
    )
    combined = result.output
    if not any(token in combined for token in expected_tokens):
        raise SmokeTestError(
            "Schema validation error did not mention missing downscale parameters:"
            f"\n{combined}"
        )
    return result


def case_analysis_invalid_seg_format_should_fail(tmp_dir: Path) -> CommandResult:
    """Segmentation format must be one of the allowed schema choices."""
    tmp_dir = tmp_dir.resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_dir / "analysis_bad_seg.csv"
    header = "exp_id,analysis::data_dir,analysis::segmentation_format"
    row = "analysis_bad,exp_out,invalid_format"
    _write_csv(csv_path, header, [row])

    output_dir = tmp_dir / "out"
    result = run_generator(
        analysis_step="analysis",
        experiment_set="analysis_bad_seg",
        csv_path=csv_path,
        template_path=TEMPLATES_DIR / "analysis_template.yaml",
        schema_path=SCHEMAS_DIR / "analysis.yaml",
        output_dir=output_dir,
    )
    if result.returncode == 0:
        raise SmokeTestError("Expected invalid segmentation_format to fail schema validation")
    if "segmentation_format" not in result.output:
        raise SmokeTestError("Expected error message to mention segmentation_format")
    return result


def case_analysis_missing_data_dir_should_fail(tmp_dir: Path) -> CommandResult:
    """analysis::data_dir is required; omission should fail schema validation."""
    tmp_dir = tmp_dir.resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_dir / "analysis_missing_data_dir.csv"
    header = "exp_id,analysis::segmentation_format"
    row = "analysis_missing,pickle.zst"
    _write_csv(csv_path, header, [row])

    output_dir = tmp_dir / "out"
    result = run_generator(
        analysis_step="analysis",
        experiment_set="analysis_missing_data_dir",
        csv_path=csv_path,
        template_path=TEMPLATES_DIR / "analysis_template.yaml",
        schema_path=SCHEMAS_DIR / "analysis.yaml",
        output_dir=output_dir,
    )
    if result.returncode == 0:
        raise SmokeTestError("Expected missing analysis::data_dir to fail schema validation")
    if "analysis::data_dir" not in result.output:
        raise SmokeTestError("Expected error message to mention analysis::data_dir")
    return result


def case_analysis_smoke(tmp_dir: Path) -> Path:
    """Generate analysis configs from CSV and ensure they materialise."""
    tmp_dir = tmp_dir.resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_dir / "analysis_design.csv"
    header = "exp_id,analysis::data_dir,analysis::output_subdir,analysis::segmentation_format"
    row = "analysis_smoke,exp_out,analysis_out,pickle.zst"
    _write_csv(csv_path, header, [row])

    output_dir = tmp_dir / "out"
    result = run_generator(
        analysis_step="analysis",
        experiment_set="analysis_smoke_tests",
        csv_path=csv_path,
        template_path=TEMPLATES_DIR / "analysis_template.yaml",
        schema_path=SCHEMAS_DIR / "analysis.yaml",
        output_dir=output_dir,
    )
    if result.returncode != 0:
        raise SmokeTestError(
            f"analysis_smoke failed with code {result.returncode}:\n{result.output}"
        )

    config_path = output_dir / "analysis_smoke" / "config.yaml"
    if not config_path.is_file():
        raise SmokeTestError(
            "Expected config.yaml to be created for analysis_smoke test"
        )
    with config_path.open(encoding="utf-8") as f:
        generated_config = yaml.safe_load(f)
    if generated_config.get("analysis", {}).get("segmentation_format") != "pickle.zst":
        raise SmokeTestError("segmentation_format should be set from CSV row")
    return config_path


def run_smoke_tests(tmp_dir: Optional[Path] = None) -> None:
    created_tmp: Optional[tempfile.TemporaryDirectory[str]] = None
    if tmp_dir is None:
        created_tmp = tempfile.TemporaryDirectory(prefix="config-generator-smoke-")
        tmp_root = Path(created_tmp.name)
    else:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_root = tmp_dir

    successes: List[str] = []
    failures: List[str] = []

    def record(name: str, func) -> None:
        target_dir = tmp_root / name
        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            func(target_dir)
        except Exception as exc:  # pragma: no cover - recorded for CLI output
            failures.append(f"{name}: {exc}")
        else:
            successes.append(name)

    record("main", case_main_smoke)
    record("preprocess_manual", case_preprocess_manual_mask)
    record("preprocess_missing", case_preprocess_missing_downscale_should_fail)
    record("analysis", case_analysis_smoke)

    if created_tmp is not None:
        created_tmp.cleanup()

    if failures:
        summary = "\n".join(failures)
        raise SmokeTestError(f"Smoke tests failed:\n{summary}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run config generator smoke tests"
    )
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        help="Optional directory to reuse for temporary files (not cleaned up).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        run_smoke_tests(args.tmp_dir)
    except SmokeTestError as exc:
        print(exc, file=sys.stderr)
        return 1
    else:
        print("All smoke tests passed.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
