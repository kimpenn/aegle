#!/usr/bin/env python3
"""Build or validate an immutable reviewed oocyte release package."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aegle.oocyte.release import build_oocyte_release, validate_oocyte_release


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or validate reviewed oocyte delivery artifacts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build",
        help="Build a new immutable release from a YAML or JSON specification.",
    )
    build.add_argument("--spec", type=Path, required=True)
    build.add_argument("--out-dir", type=Path, required=True)

    validate = subparsers.add_parser(
        "validate",
        help="Verify release checksums and cross-file invariants.",
    )
    validate.add_argument("--release-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args()
    if args.command == "build":
        result = build_oocyte_release(args.spec, args.out_dir)
        validation = result.validation
        logging.info("Release: %s", result.release_dir)
        logging.info("Samples: %d", result.sample_count)
        logging.info("Positive oocytes: %d", result.oocyte_count)
    else:
        validation = validate_oocyte_release(args.release_dir)
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
