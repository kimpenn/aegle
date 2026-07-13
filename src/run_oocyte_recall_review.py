#!/usr/bin/env python3
"""Launch or analyze the standalone oocyte recall-review workflow."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aegle.oocyte.recall_review import (
    DEFAULT_OVERVIEW_DOWNSAMPLE,
    DEFAULT_WINDOW_RADIUS_PX,
    DEFAULT_WINDOW_STRIDE_PX,
    analyze_recall_review,
    generate_recall_review_bundle,
    serve_recall_review,
)
from aegle.oocyte.recall_review_batch import (
    generate_batch_recall_review_bundle,
    serve_batch_recall_review,
)
from aegle.oocyte.recall_manual_boundary_finalize import (
    finalize_recall_manual_boundary_review,
)
from aegle.oocyte.recall_manual_boundary_review import (
    generate_recall_manual_boundary_review,
)
from aegle.oocyte.manual_seed_finalize import finalize_manual_seed_review
from aegle.oocyte.precision_boundary_review import (
    generate_precision_boundary_review,
)
from aegle.oocyte.precision_boundary_finalize import (
    finalize_precision_boundary_review,
)
from aegle.oocyte.precision_manual_boundary_review import (
    generate_precision_manual_boundary_review,
)
from aegle.oocyte.precision_manual_boundary_finalize import (
    finalize_precision_manual_boundary_review,
)
from aegle.oocyte.shape_recovery_review import generate_shape_recovery_review
from aegle.oocyte.shape_recovery_finalize import finalize_shape_recovery_review


def _add_coverage_geometry_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--window-radius",
        type=int,
        default=DEFAULT_WINDOW_RADIUS_PX,
        help="Full-resolution Recall window radius in pixels (default: %(default)s).",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=DEFAULT_WINDOW_STRIDE_PX,
        help="Full-resolution Recall grid stride in pixels (default: %(default)s).",
    )
    parser.add_argument(
        "--overview-downsample",
        type=int,
        default=DEFAULT_OVERVIEW_DOWNSAMPLE,
        help="Whole-slide navigator downsample factor (default: %(default)s).",
    )


def _serve_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and serve one sample's raw-UCHL1 recall review page."
    )
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument(
        "--overlay-dir",
        type=Path,
        help="Immutable reviewed Precision delivery to use for Recall masks.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    _add_coverage_geometry_arguments(parser)
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Generate the review bundle without starting the HTTP server.",
    )
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Serve an existing identity-matched bundle without regenerating it.",
    )
    return parser


def _analyze_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest an exported recall review and classify missing clicks."
    )
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--review-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--overlay-dir",
        type=Path,
        help="Reviewed overlay bound to the exported Recall JSON; inferred by default.",
    )
    return parser


def _serve_batch_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and serve multiple sample review consoles on one port."
    )
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument(
        "--sample-id",
        action="append",
        dest="sample_ids",
        help="Sample ID to include; repeat for multiple samples. Defaults to all samples.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    _add_coverage_geometry_arguments(parser)
    parser.add_argument(
        "--overlay",
        action="append",
        dest="overlay_specs",
        metavar="SAMPLE_ID=PATH",
        help="Reviewed Precision overlay for one sample; repeat as needed.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--generate-only",
        action="store_true",
        help="Generate all selected sample bundles and the batch index, then exit.",
    )
    mode.add_argument(
        "--no-generate",
        action="store_true",
        help="Serve existing identity-matched bundles without regenerating them.",
    )
    return parser


def _parse_overlay_specs(values: Sequence[str] | None) -> dict[str, Path]:
    overlays: dict[str, Path] = {}
    for raw_value in values or ():
        sample_id, separator, raw_path = str(raw_value).partition("=")
        sample_id = sample_id.strip()
        raw_path = raw_path.strip()
        if not separator or not sample_id or not raw_path:
            raise ValueError(
                f"invalid --overlay {raw_value!r}; expected SAMPLE_ID=PATH"
            )
        if sample_id in overlays:
            raise ValueError(f"duplicate --overlay for sample {sample_id}")
        overlays[sample_id] = Path(raw_path).resolve()
    return overlays


def _finalize_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Finalize a completed manual-seed mask review into label artifacts."
    )
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--review-json", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--delta-only",
        action="store_true",
        help="Write only the reviewed manual delta, without a combined label image.",
    )
    return parser


def _shape_review_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a delta review for shape-gated manual-seed masks."
    )
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--recall-review-json", type=Path, required=True)
    parser.add_argument("--manual-review-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def _precision_boundary_review_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a boundary-recovery review for Precision mask failures."
    )
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--precision-review-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--patch-radius", type=int, default=220)
    return parser


def _precision_boundary_finalize_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize completed Precision and boundary reviews into an immutable "
            "Precision-only intermediate."
        )
    )
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--precision-review-json", type=Path, required=True)
    parser.add_argument("--boundary-review-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def _precision_manual_boundary_review_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a polygon editor for unresolved Precision boundaries."
    )
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--base-resolved-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--patch-radius", type=int, default=220)
    return parser


def _precision_manual_boundary_finalize_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize reviewed Precision polygons into an immutable v2 "
            "intermediate."
        )
    )
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--base-resolved-dir", type=Path, required=True)
    parser.add_argument("--manual-review-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def _recall_manual_boundary_review_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a polygon editor for unresolved Recall boundaries."
    )
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--manual-review-json", type=Path, required=True)
    parser.add_argument("--base-finalize-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--patch-radius", type=int, default=220)
    return parser


def _recall_manual_boundary_finalize_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Finalize reviewed Recall polygons into a manual-seed v2 delivery."
    )
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--base-finalize-dir", type=Path, required=True)
    parser.add_argument("--review-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def _shape_finalize_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Finalize shape-recovery decisions into a new v2 label delivery."
    )
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--shape-review-json", type=Path, required=True)
    parser.add_argument("--base-finalize-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--delta-only",
        action="store_true",
        help="Write only the reviewed manual v2 delta, without a combined image.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if arguments and arguments[0] == "serve-batch":
        args = _serve_batch_parser().parse_args(arguments[1:])
        try:
            overlay_dirs = _parse_overlay_specs(args.overlay_specs)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if args.generate_only:
            bundle = generate_batch_recall_review_bundle(
                args.batch_dir,
                sample_ids=args.sample_ids,
                overlay_dirs=overlay_dirs,
                generate_samples=True,
                window_radius_px=args.window_radius,
                window_stride_px=args.window_stride,
                overview_downsample=args.overview_downsample,
            )
            logging.getLogger(__name__).info(
                "Batch review bundle generated: samples=%s candidates=%s "
                "windows=%s index=%s",
                len(bundle.sample_ids),
                bundle.total_candidate_count,
                bundle.total_window_count,
                bundle.index_path,
            )
            return 0
        serve_batch_recall_review(
            args.batch_dir,
            sample_ids=args.sample_ids,
            overlay_dirs=overlay_dirs,
            host=args.host,
            port=args.port,
            generate=not args.no_generate,
            window_radius_px=args.window_radius,
            window_stride_px=args.window_stride,
            overview_downsample=args.overview_downsample,
        )
        return 0
    if arguments and arguments[0] == "analyze":
        args = _analyze_parser().parse_args(arguments[1:])
        table_path = analyze_recall_review(
            args.sample_dir,
            args.review_json,
            args.out_dir,
            overlay_dir=args.overlay_dir,
        )
        logging.getLogger(__name__).info("Recall analysis written to %s", table_path)
        return 0
    if arguments and arguments[0] == "finalize":
        args = _finalize_parser().parse_args(arguments[1:])
        result = finalize_manual_seed_review(
            args.sample_dir,
            args.review_json,
            args.out_dir,
            analysis_dir=args.analysis_dir,
            write_combined_labels=not args.delta_only,
        )
        logging.getLogger(__name__).info(
            "Manual-seed review finalized: accepted=%s warnings=%s manifest=%s",
            result.accepted_count,
            result.boundary_warning_count,
            result.manifest_path,
        )
        return 0
    if arguments and arguments[0] == "shape-review":
        args = _shape_review_parser().parse_args(arguments[1:])
        result = generate_shape_recovery_review(
            args.sample_dir,
            args.recall_review_json,
            args.manual_review_json,
            args.out_dir,
        )
        logging.getLogger(__name__).info(
            "Shape-recovery review generated: cards=%s page=%s",
            result.card_count,
            result.page_path,
        )
        return 0
    if arguments and arguments[0] == "precision-boundary-review":
        args = _precision_boundary_review_parser().parse_args(arguments[1:])
        result = generate_precision_boundary_review(
            args.sample_dir,
            args.precision_review_json,
            args.out_dir,
            patch_radius_px=args.patch_radius,
        )
        logging.getLogger(__name__).info(
            "Precision boundary review generated: cards=%s proposals=%s "
            "manual_only=%s page=%s automatic_review=%s",
            result.card_count,
            result.proposal_count,
            result.manual_only_count,
            result.page_path,
            result.automatic_review_path,
        )
        return 0
    if arguments and arguments[0] == "precision-boundary-finalize":
        args = _precision_boundary_finalize_parser().parse_args(arguments[1:])
        result = finalize_precision_boundary_review(
            args.sample_dir,
            args.precision_review_json,
            args.boundary_review_json,
            args.out_dir,
        )
        logging.getLogger(__name__).info(
            "Precision boundary review finalized: resolved=%s unresolved_manual=%s "
            "excluded=%s manifest=%s",
            result.resolved_count,
            result.unresolved_manual_count,
            result.excluded_count,
            result.manifest_path,
        )
        return 0
    if arguments and arguments[0] == "precision-manual-boundary-review":
        args = _precision_manual_boundary_review_parser().parse_args(arguments[1:])
        result = generate_precision_manual_boundary_review(
            args.sample_dir,
            args.base_resolved_dir,
            args.out_dir,
            patch_radius_px=args.patch_radius,
        )
        logging.getLogger(__name__).info(
            "Precision manual-boundary review generated: cards=%s page=%s",
            result.card_count,
            result.page_path,
        )
        return 0
    if arguments and arguments[0] == "precision-manual-boundary-finalize":
        args = _precision_manual_boundary_finalize_parser().parse_args(arguments[1:])
        result = finalize_precision_manual_boundary_review(
            args.sample_dir,
            args.base_resolved_dir,
            args.manual_review_json,
            args.out_dir,
        )
        logging.getLogger(__name__).info(
            "Precision manual boundaries finalized: labels=%s added=%s "
            "excluded=%s manifest=%s",
            result.resolved_count,
            result.manual_added_count,
            result.manual_excluded_count,
            result.manifest_path,
        )
        return 0
    if arguments and arguments[0] == "recall-manual-boundary-review":
        args = _recall_manual_boundary_review_parser().parse_args(arguments[1:])
        result = generate_recall_manual_boundary_review(
            args.sample_dir,
            args.manual_review_json,
            args.base_finalize_dir,
            args.out_dir,
            patch_radius_px=args.patch_radius,
        )
        logging.getLogger(__name__).info(
            "Recall manual-boundary review generated: cards=%s page=%s",
            result.card_count,
            result.page_path,
        )
        return 0
    if arguments and arguments[0] == "recall-manual-boundary-finalize":
        args = _recall_manual_boundary_finalize_parser().parse_args(arguments[1:])
        result = finalize_recall_manual_boundary_review(
            args.sample_dir,
            args.base_finalize_dir,
            args.review_json,
            args.out_dir,
        )
        logging.getLogger(__name__).info(
            "Recall manual boundary finalized: labels=%s added=%s excluded=%s "
            "manifest=%s",
            result.combined_label_count,
            result.manual_added_count,
            result.manual_excluded_count,
            result.manifest_path,
        )
        return 0
    if arguments and arguments[0] == "shape-finalize":
        args = _shape_finalize_parser().parse_args(arguments[1:])
        result = finalize_shape_recovery_review(
            args.sample_dir,
            args.shape_review_json,
            args.base_finalize_dir,
            args.out_dir,
            write_combined_labels=not args.delta_only,
        )
        logging.getLogger(__name__).info(
            "Shape-recovery review finalized: accepted=%s warnings=%s manifest=%s",
            result.accepted_count,
            result.boundary_warning_count,
            result.manifest_path,
        )
        return 0
    if arguments and arguments[0] == "serve":
        arguments = arguments[1:]
    args = _serve_parser().parse_args(arguments)
    if args.generate_only:
        bundle = generate_recall_review_bundle(
            args.sample_dir,
            overlay_dir=args.overlay_dir,
            window_radius_px=args.window_radius,
            window_stride_px=args.window_stride,
            overview_downsample=args.overview_downsample,
        )
        logging.getLogger(__name__).info(
            "Recall review bundle generated: %s (%s windows)",
            bundle.page_path,
            bundle.window_count,
        )
        return 0
    serve_recall_review(
        args.sample_dir,
        overlay_dir=args.overlay_dir,
        host=args.host,
        port=args.port,
        generate=not args.no_generate,
        window_radius_px=args.window_radius,
        window_stride_px=args.window_stride,
        overview_downsample=args.overview_downsample,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
