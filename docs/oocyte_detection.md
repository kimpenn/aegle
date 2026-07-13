# Standalone raw-UCHL1 oocyte detection

`aegle.oocyte` detects and segments oocytes directly from the registered raw
UCHL1 channel. It does not read or require DeepCell masks, nucleus masks, cell
metadata, or any other Aegle pipeline output.

For Codex-assisted operation, start with `aegle/oocyte/AGENTS.md`. It defines
the human/agent decision boundary, the required state audit at the beginning of
each session, durable review evidence, versioning rules, and a reusable handoff
prompt. This document remains the command-level technical reference.

## Installation

From the repository root:

```bash
python -m pip install -e '.[oocyte]'
```

The direct `src/run_oocyte.py` wrapper also works from a source checkout without
an editable install.

## D11/D13 panel1 run

The checked-in manifest contains the eight current ovary sections and resolves
its image paths relative to the workspace layout:

```bash
python src/run_oocyte.py \
  --config exps/configs/oocyte_d11_d13_panel1/config.yaml \
  --manifest exps/configs/oocyte_d11_d13_panel1/samples.csv \
  --out-dir /workspaces/1-spatial_frs_analysis/oocyte-output/d11_d13_panel1
```

Completed matching samples resume by default. Pass `--no-resume` to recompute
them. `jobs` controls sample-level processes; candidate evaluation within one
sample remains deterministic and single-process.

`donor13_v6` is the immutable numerical baseline. The separately named
`donor13_v6_rescue_v1` profile retains every v6 acceptance, retries rejected
seeds with a P95 annulus floor, discovers multiple components in crowded
coarse patches, and deduplicates the resulting persisted masks against v6.

Required manifest columns are `sample_id`, `image_path`, and `pixel_size_um`.
Each enabled row must also provide either `channel_index` or `antibodies_path`.
Optional columns are `channel_name`, `enabled`, and `profile`.

## Outputs

Each completed sample writes:

```text
<sample_id>/
  run_manifest.json
  summary.json
  runtime.json
  coarse_candidates.csv
  candidates.csv
  masks/<candidate_id>.npz
  oocyte_labels.ome.tiff
  oocyte_labels.csv
  overview.png
  accepted_duplicate_suspects.csv
  rescue_diagnostics.csv              # rescue profile only
  oocytes.html
  html_candidates.csv
  html_assets/*.webp
  html_assets/manifest.json
```

Every NPZ contains the image-bounded boolean mask, full-image bounding box,
source image shape, detector profile fingerprint, and metrics from the same
segmentation evaluation. `oocyte_labels.ome.tiff` is a tiled, compressed
`uint16` image composed from accepted persisted masks. Its mapping CSV records
label IDs and overlap pixels. It can be regenerated without rereading UCHL1.

The batch root writes `batch_manifest.json`, `batch_summary.csv`, and
`batch_summary.json`, plus `spatial_qc/all_samples_overview.png`, an overview
index, and a combined duplicate-suspect table. A failed sample is recorded
without deleting successful sample outputs.

It also writes `oocyte_detection_algorithm.html` and
`oocyte_review_index.html`. The algorithm page is self-contained and uses
inline SVG. Each sample page displays raw UCHL1 with the exact persisted cyan
mask, a whole-slide coordinate map, baseline/rescue filters, review-priority
flags, local browser review state, notes, and CSV/JSON export.

`html_assets/manifest.json` binds every rank-named WebP to the raw source stat,
UCHL1 channel, patch center and radius, renderer version, and exact NPZ mask
SHA-256. Report regeneration reuses a thumbnail only when both that render
fingerprint and the WebP SHA-256 match. Candidate insertion or suppression can
therefore change display ranks without silently reusing another candidate's
image.

When a rescue delta is supplied, donor13 samples also receive
`candidates_rescue_v1_combined.csv`, `oocyte_labels_rescue_v1.ome.tiff`, and
`oocyte_labels_rescue_v1.csv`. These combine baseline and rescue exact masks;
the original frozen v6 artifacts remain unchanged.

## Final-label profiling

Profile marker intensities only after selecting the final reviewed label image.
The profiler enumerates objects from that label image, not from a nucleus mask,
DeepCell output, or the optional candidate table. The candidate table can enrich
provenance columns but cannot add or remove expression rows.

For example, profile the reviewed `13-23` v2 labels with:

```bash
python src/run_oocyte_profile.py \
  --sample-id 13-23 \
  --image /path/to/registered.ome.tiff \
  --antibodies /path/to/antibodies.tsv \
  --labels /path/to/final/oocyte_labels.ome.tiff \
  --mapping /path/to/final/oocyte_labels.csv \
  --candidates /path/to/final/oocyte_candidates.csv \
  --out-dir /path/to/profiling \
  --pixel-size-um 0.5
```

The output directory contains:

```text
oocyte_by_marker.csv
oocyte_metadata.csv
oocyte_overview.csv
channel_manifest.csv
profiling_manifest.json
```

`oocyte_by_marker.csv` has one row per positive final label and one column per
registered channel. Values are raw within-mask means, matching the whole-cell
mean semantics of Aegle's `cell_by_marker.csv`. They are not background
subtracted, transformed, or normalized. All acquisition channels are retained;
`channel_manifest.csv` classifies DAPI as a nuclear stain rather than protein.

The profiler validates that the label image and mapping contain exactly the same
positive labels, checks assigned pixel counts and bounding boxes, and records
source identities plus SHA-256 hashes for the label, mapping, antibody, candidate,
and output files. It scans labels and raw channels through bounded active regions
instead of materializing the full multichannel image. Empty negative-control
labels produce header-only marker and metadata tables with zero rows.

## Review

When review is enabled, the CLI writes:

```text
review/
  accepted_candidates.csv
  accepted_pages/
  novel_candidates.csv
  novel_pages/
  missed_references.csv
  missed_pages/
  reference_comparison.csv
  summary.json
```

Every montage panel has a `#NNN` index that maps to `review_rank` in the CSV.
Cyan is the exact persisted candidate mask. Lime outlines are nearby accepted
masks and help identify crowded fields or duplicate candidates.

In the per-sample Precision HTML, every candidate card has a `Hide mask`
button. It loads the matching raw UCHL1 patch on demand from the dynamic review
server and changes to `Show mask`; toggling affects only that card and does not
change its Accept, Reject, Unsure, or notes state. Open Precision through the
batch review console on port `8767` for this control. A plain static file server
can display the persisted-mask thumbnails but cannot provide on-demand raw
patches.

Precision browser state is keyed by stable detection pass plus detector
component ID and is namespaced by the SHA-256 of `html_candidates.csv`; it is
not keyed by the displayed `#NNN` rank. `Export JSON` writes an identity-bound
`oocyte_precision_review` record. Use `Import JSON` to resume from that durable
record. Import rejects a different sample, raw-source stat, detector profile,
implementation version, or candidate-table SHA instead of applying decisions
to a changed queue. CSV remains a convenient flat view, but JSON is the release
input.

Recommended manual values:

- `manual_is_oocyte`: `yes`, `no`, or `uncertain`.
- `manual_mask_quality`: `good`, `undersegmented`, `oversegmented`, or `truncated`.
- `manual_duplicate_group`: assign the same short ID to duplicate panels.
- `manual_notes`: free text for unusual morphology or review context.

The optional legacy JSONL only defines near-reference, novel, and missed review
buckets. It never changes detection, segmentation, scoring, acceptance, or
deduplication and must not be treated as biological ground truth.

### Precision boundary recovery

When Precision review marks a true oocyte as `Reject` with
`true_oocyte; mask_truncated; mask_off_target`, generate a separate replacement
review instead of lowering the frozen detector threshold globally:

```bash
python src/run_oocyte_recall_review.py precision-boundary-review \
  --sample-dir /path/to/batch/13-21 \
  --precision-review-json /path/to/13-21_oocyte_review.json \
  --out-dir /path/to/batch/13-21/recall_analysis_precision_boundary_v1 \
  --patch-radius 220
```

The command validates the Precision export against the current raw source,
profile, implementation version, and candidate-table SHA-256. It selects only
reviewed true-oocyte boundary failures, scans lower annulus percentiles, and
retains a proposal only when it contains at least 95% of the current component,
grows by a bounded amount, stays within physical and centroid gates, and avoids
other confirmed oocyte centers. It writes exact conservative/expanded NPZs, a
candidate CSV, WebP comparisons, an identity-bound HTML page, and a summary. It
does not modify current masks or label images.

Each comparison shows raw context plus all current cyan masks, the frozen
current mask in yellow, a safe conservative proposal in green, and a safe
expanded proposal in orange. Choose `Use conservative` or `Use expanded` only
when that contour isolates the intended oocyte without absorbing a neighbor.
Use `Needs manual` when neither proposal is satisfactory, `Keep current` only
when the original mask is acceptable on reinspection, and `Not oocyte` only to
correct the Precision biological classification. Export JSON as the durable
record; final labels remain unchanged until a separate identity-validated
finalization step ingests those choices.

Finalize a completed Precision boundary export into a self-contained,
Precision-only intermediate with:

```bash
python src/run_oocyte_recall_review.py precision-boundary-finalize \
  --sample-dir /path/to/batch/13-21 \
  --precision-review-json /path/to/13-21_oocyte_review.json \
  --boundary-review-json /path/to/13-21_precision_boundary_review.json \
  --out-dir /path/to/batch/13-21/precision_resolved_v1
```

The finalizer requires complete Precision decisions and complete, non-`Unsure`
boundary choices. It verifies both review identities and SHA-256 values,
validates every selected proposal against the bound candidate table and NPZ
metadata, and blocks masks with at least 25% smaller-object overlap. Accepted
current and replacement masks are copied into `reviewed_masks/`; source detector
and review-pack assets remain unchanged.

The output includes `precision_resolved_candidates.csv`, all 132 decisions in
`precision_review_decisions.csv`, unresolved objects in
`manual_boundary_queue.csv`, a mask-overlap audit, an OME-TIFF plus label
mapping, exact copies of all review inputs, and
`precision_resolved_manifest.json` with artifact hashes. `Needs manual` objects
are deliberately absent from the label image until their boundaries are drawn
and reviewed. This intermediate always records `release_ready=false` and
`recall_complete=false`; do not run final expression profiling from it or call
it a sample release until the manual queue and whole-slide Recall review are
resolved.

If `precision_resolved_v1/manual_boundary_queue.csv` is non-empty, generate a
native-resolution polygon editor without changing v1:

```bash
python src/run_oocyte_recall_review.py precision-manual-boundary-review \
  --sample-dir /path/to/batch/13-21 \
  --base-resolved-dir /path/to/batch/13-21/precision_resolved_v1 \
  --out-dir /path/to/batch/13-21/recall_analysis_precision_manual_boundary_v1 \
  --patch-radius 220
```

The page shows resolved neighbors in cyan, the rejected current target mask in
yellow, and the editable polygon in orange. Click to add vertices, drag a
vertex to refine it, use `Undo point` or `Clear` when needed, and toggle masks
to inspect raw UCHL1. The first vertex is yellow. Trace the intended outer
oocyte boundary, excluding adjacent oocytes and follicular halo, then select
`Accept contour`. Every contour edit returns the card to `Unreviewed`; accept
it again before exporting. `Export JSON` records full-image X/Y vertices and a
review identity. Browser local storage is not a final mask.

After every card has a resolved choice, finalize the exported polygons into a
new immutable version:

```bash
python src/run_oocyte_recall_review.py precision-manual-boundary-finalize \
  --sample-dir /path/to/batch/13-21 \
  --base-resolved-dir /path/to/batch/13-21/precision_resolved_v1 \
  --manual-review-json /path/to/13-21_precision_manual_boundary_review.json \
  --out-dir /path/to/batch/13-21/precision_resolved_v2
```

The Python finalizer, not JavaScript, rasterizes the polygons. It rejects stale
base or candidate identities, missing and `Unsure` decisions, fewer than three
unique vertices, self-intersection, contours outside the reviewed patch,
contours that miss the reviewed center, equivalent diameters outside 10-100
um, and any overlap pixel with a resolved mask. It copies every v1 mask into a
self-contained v2 directory, writes exact manual NPZs and a new whole-slide
label OME-TIFF, and records all hashes. A successful v2 sets
`manual_boundary_complete=true` but remains `release_ready=false` and
`recall_complete=false` until whole-slide Recall review is finalized.

## Recall review and human-seeded diagnostics

The candidate-card page is a precision workflow: it can reject false-positive
masks but cannot reveal an oocyte for which no accepted mask exists. The
separate recall reviewer covers image space with deterministic focal windows.
It displays a globally normalized whole-slide UCHL1 navigator, reads each
native-resolution focal patch on demand, overlays every intersecting persisted
mask, and records full-resolution missing-oocyte clicks.

Generate and serve the `13-23` proof of concept with:

```bash
python src/run_oocyte_recall_review.py \
  --sample-dir /workspaces/1-spatial_frs_analysis/oocyte-output/d11_d13_panel1/13-23 \
  --host 0.0.0.0 \
  --port 8767
```

Forward `8767` from the devcontainer in Cursor's `Ports` panel and open
`http://localhost:8767/recall_review.html`. Unlike the static precision pack,
this page requires the dedicated process because `/api/patch.webp`,
`/api/overlay.png`, and `/api/probe` read bounded raw-image regions and exact
NPZ masks on demand.

After Precision finalization, Recall must use the immutable reviewed delivery
rather than the original `html_candidates.csv` masks. Generate a reviewed-
overlay bundle with:

```bash
python src/run_oocyte_recall_review.py serve \
  --sample-dir /path/to/batch/13-21 \
  --overlay-dir /path/to/batch/13-21/precision_resolved_v2 \
  --generate-only
```

Only complete `precision_resolved_v1` or `precision_resolved_v2` deliveries
with zero unresolved manual boundaries are accepted. Startup verifies the
delivery manifest, every artifact SHA-256 and size, candidate count, NPZ hash,
sample identity, image geometry, and bounding boxes. The reviewed candidates
then drive the cyan overlays, whole-slide points, accepted counts, nearest-
accepted distance, and exact `already_covered` classification. Frozen refined,
coarse, and duplicate-suppressed detector tables remain the diagnostic source
for classifying genuine misses.

The Recall page displays the bound overlay name and count. Its exported JSON
includes the reviewed manifest and candidate-table hashes. Browser state from
an automatic or different reviewed overlay is discarded, and analysis rejects
an export that does not match the currently generated Recall bundle. Analyze a
matching export normally; `--overlay-dir` is optional because the reviewed path
is recovered and revalidated from the bound identity:

```bash
python src/run_oocyte_recall_review.py analyze \
  --sample-dir /path/to/batch/13-21 \
  --review-json /path/to/13-21_recall_review.json \
  --out-dir /path/to/batch/13-21/recall_analysis_v1
```

For cohort review, generate the four donor13 consoles and one batch index with:

```bash
python src/run_oocyte_recall_review.py serve-batch \
  --batch-dir /workspaces/1-spatial_frs_analysis/oocyte-output/d11_d13_panel1 \
  --sample-id 13-21 \
  --sample-id 13-22 \
  --sample-id 13-23 \
  --sample-id 13-24 \
  --generate-only
```

When generating several samples together, bind reviewed overlays explicitly as
`SAMPLE_ID=PATH` entries. Samples without an entry keep their automatic masks:

```bash
python src/run_oocyte_recall_review.py serve-batch \
  --batch-dir /path/to/batch \
  --sample-id 13-21 --sample-id 13-22 \
  --overlay 13-21=/path/to/batch/13-21/precision_resolved_v2 \
  --host 0.0.0.0 --port 8767
```

With `--no-generate`, each sample recovers its previously bound overlay from
`recall_review/metadata.json`; static Recall and console pages are refreshed
without rereading raw images or changing metadata.

Then serve the identity-matched bundles on one port without regenerating them:

```bash
python src/run_oocyte_recall_review.py serve-batch \
  --batch-dir /workspaces/1-spatial_frs_analysis/oocyte-output/d11_d13_panel1 \
  --sample-id 13-21 \
  --sample-id 13-22 \
  --sample-id 13-23 \
  --sample-id 13-24 \
  --host 0.0.0.0 \
  --port 8767 \
  --no-generate
```

Open `http://localhost:8767/` after forwarding the port. The root page links to
each sample's `review_console.html`, precision page, and recall page. Sample API
routes are isolated under `/<sample_id>/api/...`; the server validates every
sample against its source image and candidate-table identity before accepting
requests. Page visits do not mark review work complete. Exported precision and
recall JSON remain the durable records.

The review queue prioritizes near-threshold rejected candidates and proposal-
dense windows while still covering the entire image with overlapping windows.
For every inspected window, record `Complete`, `Has misses`, or `Unsure`.
Click `Add missed oocyte` and then the focal image to add a manual center. The
page immediately reports the nearest detector stages plus conservative and
expanded click-targeted segmentation diagnostics. A click is not an accepted
boundary and never modifies a production label image.

Review state lives in browser local storage until exported. Export JSON for
scientific retention and offline analysis; CSV is a convenient flat audit
view. Analyze an exported JSON file with:

```bash
python src/run_oocyte_recall_review.py analyze \
  --sample-dir /workspaces/1-spatial_frs_analysis/oocyte-output/d11_d13_panel1/13-23 \
  --review-json /path/to/13-23_recall_review.json \
  --out-dir /path/to/13-23_recall_analysis
```

The analysis writes `recall_failure_analysis.csv`, `summary.json`, two
provisional masks per successful click under `provisional_masks/`, and a
`manual_seed_review.html` page. For each manual center, segmentation sweeps
annulus percentiles P95 through P60 in a compact native-resolution patch. The
highest-percentile component passing click-distance, centroid, and physical-size
gates is the conservative mask. The expanded mask is the largest lower-threshold
version that retains at least 70% overlap with the conservative component and
grows by no more than fourfold. Multi-lobed connected components are split by a
distance-transform watershed when they contain multiple cell-sized basins. In
offline analysis, every other manual center is also an exclusion point, so a
candidate cannot expand across another clicked oocyte.

When the analysis directory is under the served sample directory, open its page
through the same dynamic server, for example
`http://localhost:8767/recall_analysis_v4/manual_seed_review.html`. Each card
shows raw UCHL1 plus existing cyan masks and the manual center on the left, the
green conservative mask in the middle, and the yellow expanded mask on the
right. Select `Use conservative`, `Use expanded`, `Neither`, `Duplicate`, or
`Unsure`, then export JSON. The selection export, not browser local storage, is
the durable review record.

Failure classes identify the earliest actionable detector stage:
`proposal_miss`, `segmentation_miss`, `acceptance_miss`, `dedup_error`, or
`already_covered`. Both candidate masks remain diagnostic until this second
review is exported; analysis never changes a production label image.

Finalize a completed second-stage JSON review with:

```bash
python src/run_oocyte_recall_review.py finalize \
  --sample-dir /workspaces/1-spatial_frs_analysis/oocyte-output/d11_d13_panel1/13-23 \
  --review-json /path/to/13-23_manual_seed_mask_review.json \
  --analysis-dir /workspaces/1-spatial_frs_analysis/oocyte-output/d11_d13_panel1/13-23/recall_analysis_v4 \
  --out-dir /path/to/reviewed_manual_seed_delta_v1
```

Finalization validates the sample identity and analysis SHA-256, joins choices
to the trusted analysis table by `annotation_id`, and ignores mask paths supplied
by the browser export. When Recall was generated against a reviewed Precision
overlay, the finalizer recovers that overlay from the export identity and carries
forward only those reviewed masks; it never falls back to rejected automatic
candidates. Accepted masks are copied into immutable reviewed NPZs.
`Neither`, `Duplicate`, and `Unsure` rows remain in the decision audit but do not
become labels. A 25% smaller-mask overlap with another reviewed or production
mask blocks finalization; smaller contacts are recorded in
`mask_overlap_audit.csv`.

The output contains a manual-only label OME-TIFF, a separately named combined
rescue-v1-plus-manual OME-TIFF, candidate and mapping CSVs, all review decisions,
and `manual_seed_finalize_manifest.json` with hashes and sizes. Notes indicating
an incomplete or excessive boundary become `boundary_warning=true`. Existing
baseline and rescue-v1 labels are never overwritten. Use `--delta-only` when a
combined label image is not wanted.

If `Neither` means a confirmed oocyte with two unacceptable threshold masks,
include `needs_manual_boundary` in its note. Freeze all safe choices as v1, then
generate a polygon editor only for those confirmed unresolved objects:

```bash
python src/run_oocyte_recall_review.py recall-manual-boundary-review \
  --sample-dir /workspaces/1-spatial_frs_analysis/oocyte-output/d11_d13_panel1/13-21 \
  --manual-review-json /path/to/13-21_manual_seed_mask_review.json \
  --base-finalize-dir /path/to/reviewed_manual_seed_delta_v1 \
  --out-dir /path/to/recall_analysis_manual_boundary_v1
```

The contour page shows all v1 masks in cyan and the rejected conservative mask
in yellow. Trace the intended boundary, choose `Accept contour`, and export the
identity-bound JSON. Finalize it into a separately named v2 delivery with:

```bash
python src/run_oocyte_recall_review.py recall-manual-boundary-finalize \
  --sample-dir /workspaces/1-spatial_frs_analysis/oocyte-output/d11_d13_panel1/13-21 \
  --base-finalize-dir /path/to/reviewed_manual_seed_delta_v1 \
  --review-json /path/to/13-21_recall_manual_boundary_review.json \
  --out-dir /path/to/reviewed_manual_seed_delta_v2
```

The finalizer independently validates a simple polygon, requires the reviewed
center inside it, enforces a 10-100 um equivalent-diameter range, and rejects any
overlap pixel with resolved masks or another manual contour. It verifies every
v1 artifact before composing v2 and never modifies v1.

If review identifies a fragmented expanded mask, generate the opt-in shape
recovery delta with:

```bash
python src/run_oocyte_recall_review.py shape-review \
  --sample-dir /workspaces/1-spatial_frs_analysis/oocyte-output/d11_d13_panel1/13-23 \
  --recall-review-json /path/to/13-23_recall_review.json \
  --manual-review-json /path/to/13-23_manual_seed_mask_review.json \
  --out-dir /workspaces/1-spatial_frs_analysis/oocyte-output/d11_d13_panel1/13-23/recall_analysis_v4/shape_recovery_v1
```

This mode leaves the standard 4x expansion cap unchanged. Its candidate may
grow by up to 6x only when it still overlaps at least 70% of the conservative
mask, has diameter at least 20 um, circularity at least 0.80, solidity at least
0.90, centroid offset at most 25 px, and does not reach another manual center.
The generated page contains only changed masks. Yellow is the frozen v4 result;
orange is shape recovery. Export `Keep v4`, `Use recovery`, `Exclude`, or
`Unsure` decisions before creating a separately named combined v2.

Finalize the exported shape review against the immutable v1 delivery with:

```bash
python src/run_oocyte_recall_review.py shape-finalize \
  --sample-dir /workspaces/1-spatial_frs_analysis/oocyte-output/d11_d13_panel1/13-23 \
  --shape-review-json /path/to/13-23_shape_recovery_review.json \
  --base-finalize-dir /path/to/reviewed_manual_seed_delta_v1 \
  --out-dir /path/to/reviewed_manual_seed_delta_v2
```

Shape finalization verifies every v1 manifest artifact before reading it. `Use
recovery` replaces an accepted v1 mask or adds a mask that v1 excluded; `Keep
v4` carries the v1 decision forward; `Exclude` removes it; unresolved `Unsure`
decisions block finalization. It reruns manual/manual and manual/production
overlap checks, then writes self-contained reviewed NPZs, a manual-only v2
OME-TIFF, and an independently named combined v2 OME-TIFF. Neither v1 nor the
frozen detector outputs are modified.

Prioritized hard-case review can improve the detector but does not by itself
measure global recall. A global recall claim requires explicit disposition of
every relevant tissue window. Any learned profile must preserve the frozen v6
and rescue-v1 outputs and must be checked against precision review plus donor11
as a biologically confirmed no-oocyte negative control.

## Reviewed release packaging

Final delivery is separate from detector and review working directories. The
release builder copies only final reviewed labels and masks, raw within-mask
profiles, exported review evidence, and provenance manifests into a new
immutable directory. It references raw OME-TIFFs by size and modification-time
identity rather than duplicating them.

Build the reviewed donor11/donor13 panel1 release with embedded static review
images using:

```bash
python src/run_oocyte_release.py build \
  --spec exps/configs/oocyte_d11_d13_panel1/release_v6.yaml \
  --out-dir /workspaces/1-spatial_frs_analysis/oocyte-output/d11_d13_panel1_delivery_v6
```

The build fails if the destination already exists, any source hash or profiling
identity is inconsistent, a positive sample is empty, or a donor11 negative
control has a nonzero label or accepted detector/rescue diagnostic. It validates
the temporary package before atomically publishing the destination.

Verify a completed package independently with:

```bash
python src/run_oocyte_release.py validate \
  --release-dir /workspaces/1-spatial_frs_analysis/oocyte-output/d11_d13_panel1_delivery_v6
```

The validator checks every packaged SHA-256, label-to-mapping pixel counts,
candidate and profiling IDs, marker schemas, sample totals, and cohort table
composition. `batch_oocyte_by_marker.csv` is the cohort-level combined protein
expression matrix; `batch_oocyte_metadata.csv` contains geometry and provenance.
Per-sample deliverables are under `samples/<sample-id>/`, and
`oocyte_review_index.html` is the release summary page.

Every sample's `review_console.html` begins with an embedded, downsampled
whole-slide raw-UCHL1 view. Positive pages initially show the exact final NPZ
boundaries in cyan; Hide masks/Show masks switches between masked and raw
overviews, and clicking a mask location jumps to the nearest matching cell card.
Each positive card also embeds raw UCHL1 and the same patch with the exact
delivered mask, with its own Hide mask/Show mask control and provenance filters.
Negative-control pages show the raw whole-slide overview but no boundaries,
hotspots, or cards, and explicitly report zero final oocytes. All controls work
when the HTML is opened directly through `file://` and make no API or network
request. Relative links expose expression, metadata, labels, mapping, candidates,
review evidence, and checksums. Raw OME-TIFFs remain outside the package and are
not required to view any embedded image.

## Incremental rescue validation

The rescue profile can be reviewed without rerunning the expensive frozen v6
refinement. The delta command loads completed v6 candidates and exact masks,
reads only local raw-UCHL1 patches, and writes a rescue-only standard review
batch:

```bash
python src/run_oocyte_rescue_delta.py \
  --baseline-dir /workspaces/1-spatial_frs_analysis/oocyte-output/d11_d13_panel1 \
  --out-dir /workspaces/1-spatial_frs_analysis/oocyte-output/d11_d13_panel1_rescue_v1_delta
```

Regenerate the combined per-sample HTML separately when needed:

```bash
python src/run_oocyte_report.py \
  --batch-dir /workspaces/1-spatial_frs_analysis/oocyte-output/d11_d13_panel1 \
  --rescue-delta-dir /workspaces/1-spatial_frs_analysis/oocyte-output/d11_d13_panel1_rescue_v1_delta \
  --references /workspaces/1-spatial_frs_analysis/frs-atlas-phenocycler/output/ovary_panel1/v2_4/oocyte_review_final.jsonl
```

## Validation and limitations

The frozen `donor13_v6` profile reproduces sample `13-23` with 237 coarse
proposals, 318 refined candidates, and 135 accepted candidates. The representative
`13-23/#680` mask matches the research mask pixel-for-pixel. The production
`13-23` run took about 15.8 minutes and used bounded proposal-local caches.

Across donor13, frozen v6 produces `352` accepts. The rescue delta adds `99`
visually inspected candidates (`31`, `8`, `58`, and `2` by section). Actual-mask
overlap then suppresses `22` lower-score duplicate v6 rows, giving `132`, `63`,
`187`, and `47` combined review candidates, or `429` total. In the provisional
legacy-miss self-audit, rescue recovered all `21` visually definite examples and
none of `28` visually false references; this is development evidence, not
biological ground truth or a formal sensitivity/specificity estimate.

The four combined donor13 OME label images match their raw source dimensions,
have one mapping row per review candidate, and contain no assigned overlap
pixels. All eight sample HTML pages and the batch index were checked, including
desktop/mobile rendering of the 187-card `13-23` page. The complete baseline
directory including HTML assets occupies about 434 MB; the rescue delta occupies
about 58 MB.

Donor13 has engineering visual-review evidence. Some open-ring or crescent
rescue masks remain explicitly review-priority cases. Donor11 was confirmed as
a no-oocyte negative control. The frozen detector and rescue diagnostics yield
zero accepted objects across sections 11-21 through 11-24. Machine-accepted
counts in development directories remain review candidates, not final oocyte
counts.

Run the fast suite with:

```bash
python -m unittest discover -s tests/oocyte -v
```

Run the local donor13 parity tests only where the research fixtures and raw image
are available:

```bash
AEGLE_RUN_OOCYTE_LOCAL_REGRESSION=1 \
  python -m unittest tests.oocyte.test_local_regression -v
```
