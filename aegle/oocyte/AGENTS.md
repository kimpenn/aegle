# Oocyte Module Collaboration Guide

This directory implements standalone raw-UCHL1 oocyte detection, review, final
mask export, expression profiling, and release packaging. It does not require
DeepCell labels or another Aegle pipeline output. Keep the module runnable from
raw registered OME-TIFFs plus the acquisition antibody table.

## Delivery Model

This module is intentionally delivered as a deterministic pipeline with
human-in-the-loop biological review and optional Codex orchestration. Codex is
allowed to inspect state, run deterministic commands, validate identities,
generate review consoles, ingest exported review JSON, preserve versioned
artifacts, and explain the next checkpoint. A biologist remains the authority
for whether an object is an oocyte and whether its boundary is acceptable.

Do not make the current delivery depend on an autonomous workflow engine. The
purpose of these instructions is to let another researcher give Codex the repo,
raw-image paths, and review exports and reproduce the same sequence safely.
Algorithm optimization, threshold retuning, and automatic replacement of human
choices are follow-up work unless the user explicitly opens a separate issue.

## Sources Of Truth

Use these sources in order rather than relying on conversation history:

1. `aegle/oocyte/AGENTS.md` defines the orchestration and safety contract.
2. `docs/oocyte_detection.md` defines command syntax and artifact semantics.
3. The selected YAML config, sample CSV, run manifests, profile fingerprint,
   candidate tables, and NPZ metadata define the computation that actually ran.
4. Exported identity-bound review JSON defines durable human decisions. Browser
   local storage, screenshots, chat messages, and CSV exports are not final
   review authority.
5. Finalization, profiling, and release manifests plus their SHA-256 records
   define the delivered labels and expression tables.

If prose conflicts with an identity-validated manifest, stop and investigate;
do not silently choose whichever artifact is more convenient.

## Starting A Codex Session

At the beginning of a new oocyte session, Codex must:

1. Read this file and the relevant sections of `docs/oocyte_detection.md`.
2. Inspect `git status --short` and avoid staging, reverting, or modifying
   unrelated worktree changes.
3. Resolve the requested sample IDs, raw OME-TIFFs, antibody table, detector
   config, output root, and all review JSON paths from disk.
4. Audit existing manifests and versioned directories instead of assuming the
   last conversational step completed successfully.
5. Report the current phase for every requested sample, the evidence used to
   determine it, unresolved biological decisions, and the exact next action.
6. Confirm the output destination is new before running any finalizer, profiler,
   or release builder. Never overwrite an existing reviewed version.

For a code change, run the focused tests before handing work back:

```bash
python -m unittest discover -s tests/oocyte -v
```

Routine operation on unchanged code does not require rerunning the entire suite,
but every generated release must pass the independent release validator.

## Human And Codex Responsibilities

Codex owns the mechanical work:

- Validate input paths, channel resolution, image geometry, and profile identity.
- Run detection, review generation, review ingestion, finalization, profiling,
  release construction, and independent validation commands.
- Keep detector outputs, reviewed deltas, final labels, and releases in separate
  versioned directories.
- Summarize counts, incomplete decisions, identity mismatches, overlap failures,
  and generated artifact paths without changing biological classifications.
- Start the dynamic review server on `0.0.0.0:8767` when Precision or Recall
  needs native raw-image access and tell the user which URL to open.

The biologist owns the scientific decisions:

- Precision: Accept, Reject, or Unsure for each proposed object.
- Boundary review: choose a safe proposal, request a manual contour, or exclude.
- Recall: classify every survey window and click every visible oocyte lacking a
  satisfactory current mask.
- Provisional-mask and polygon review: select the intended boundary or leave the
  object unresolved.

Codex must never convert `Unsure`, `Neither`, an incomplete queue, a note alone,
or unexported browser state into a final label. When a human checkpoint is
pending, provide the console URL and expected export filename, then stop the
scientific transition until the exported JSON is available.

## Orchestration State Model

Treat each sample as moving through these explicit phases. Some samples have no
boundary-recovery subphase, but phase ordering must not be reversed.

| Phase | Codex action | Durable evidence | Human checkpoint |
| --- | --- | --- | --- |
| Input audit | Validate manifest, raw image, antibodies, pixel size, profile, and output destination | Resolved config and sample manifest | Confirm sample scope and profile |
| Detection | Run `src/run_oocyte.py` and generate exact candidate NPZs, labels, reports, and manifests | `run_manifest.json`, `candidates.csv`, `masks/`, `oocyte_labels.ome.tiff` | None |
| Precision | Serve the candidate console and preserve its exported JSON | Identity-bound Precision JSON | Classify every candidate and mask quality |
| Precision boundary resolution | Generate threshold alternatives or a polygon queue, then run the matching finalizer | Versioned `precision_resolved_vN` manifest and copied review JSON | Select alternatives or draw contours |
| Recall | Generate the reviewed-overlay survey, serve all windows, and analyze the exported JSON | Geometry-bound Recall JSON and analysis manifest | Classify every window and click all misses |
| Miss boundary resolution | Generate conservative/expanded masks and, when needed, manual contours; finalize accepted choices | Versioned reviewed manual-seed directory and decision audit | Select or draw each missing boundary |
| Final labels | Verify there are no unresolved required decisions and identify the exact final OME label and mapping | Finalization manifest, label OME-TIFF, mapping CSV, exact NPZ masks | Confirm release set |
| Profiling | Measure raw within-mask means from final labels only | `profiling_manifest.json`, marker and metadata tables | None |
| Release | Build into a new immutable directory and run `validate` independently | Release and sample manifests, SHA-256 artifact index | Approve package for sharing |

After the user reports that a review is complete, Codex must inspect the stated
JSON file, verify sample and review identity, summarize completion counts, run
only the matching analyzer/finalizer, and report the next human checkpoint. Do
not skip directly from a browser export to profiling or release.

## Operating Principles

- Treat detector accepts as review candidates, not final biological labels.
- Never overwrite detector, review, or finalized outputs. Use a new versioned
  directory for every iteration.
- Keep precision and recall evidence distinct. Precision review decides whether
  proposed objects and boundaries are acceptable. Recall review surveys the
  whole tissue and records missing objects.
- A manual center is not a final mask. Generate provisional boundaries, obtain a
  second review, and finalize only explicit accepted choices.
- Do not silently convert `Unsure`, `Neither`, incomplete reviews, or browser
  local storage into final labels.
- Preserve source identities, exported review JSON, artifact hashes, and final
  mapping tables so every label can be audited.
- Keep donor11 sections as zero-oocyte negative controls for the panel1 release.
  Any accepted detector or rescue object in donor11 must block release creation.

## Standard Workflow

1. Run standalone detection from the raw OME-TIFF and antibody table with
   `src/run_oocyte.py`. Use the frozen study profile unless an issue explicitly
   defines and validates a new profile.
2. Open the sample review console. Complete Precision first, rejecting false
   objects and flagging true oocytes with unacceptable boundaries.
3. Resolve precision boundary cases through the generated boundary review. Use
   manual contours only when threshold-derived alternatives cannot represent the
   intended cell.
4. Complete the whole-slide Recall survey. Click every oocyte that lacks a
   satisfactory current mask and export the review JSON.
5. Analyze manual centers into conservative and expanded provisional masks,
   review every proposal, and finalize only accepted masks. Resolve remaining
   confirmed boundary failures with the manual-contour workflow.
6. Run `src/run_oocyte_profile.py` only against the final reviewed label image
   and mapping. The default measurement is the raw within-mask mean for every
   registered channel.
7. Build a release with `src/run_oocyte_release.py build`, then run the separate
   `validate` command. Never edit files inside a completed release.

The command-level options for every boundary and Recall subphase are maintained
in `docs/oocyte_detection.md`; do not duplicate or invent flags from memory.

## Reference Panel1 Run

The completed D11/D13 panel1 run is the reproducibility reference, not a claim
that the same profile is validated for every ovary cohort:

```text
Detector config: exps/configs/oocyte_d11_d13_panel1/config.yaml
Sample manifest: exps/configs/oocyte_d11_d13_panel1/samples.csv
Detector profile: donor13_v6
Secondary profile: donor13_v6_rescue_v1
Final release spec: exps/configs/oocyte_d11_d13_panel1/release_v6.yaml
Final donor13 counts: 13-21=129, 13-22=68, 13-23=219, 13-24=43
Negative controls: 11-21, 11-22, 11-23, 11-24, all with zero final labels
```

The final reviewed release contains 459 donor13 labels, raw within-mask values
for 36 channels, four zero-label donor11 controls, complete review evidence, and
static per-sample consoles. The release validator reports 598 hashed artifacts.
Use these invariants to detect accidental orchestration or packaging drift.

The detector and release directories used in one workstation are not portable
inputs. A colleague should provide their own raw-data root and output root while
retaining the checked-in config/profile semantics. The sample CSV resolves raw
paths relative to its own location; the release v6 spec is cohort provenance and
must be adapted only when rebuilding from another filesystem layout.

## Reusing The Workflow On New Samples

Copy both templates rather than editing the reference cohort in place:

```text
exps/templates/oocyte_template.yaml
exps/templates/oocyte_samples_template.csv
```

A new sample needs a registered CYX OME-TIFF, an acquisition antibody table or
explicit UCHL1 channel index, pixel size, a unique sample ID, and a new output
directory. The detector can use `donor13_v6` as an initial candidate generator,
but a different donor, panel, staining run, or tissue context requires complete
Precision and whole-slide Recall before any sensitivity claim. If review across
multiple samples justifies new numerical behavior, create a new named profile
and a separate algorithm-validation issue; never mutate `donor13_v6`.

## Prompt For A Colleague

Use this prompt when handing the repository to another Codex session. Replace
the bracketed values with real paths and sample IDs:

```text
Operate the standalone raw-UCHL1 oocyte workflow in this Aegle checkout.
First read aegle/oocyte/AGENTS.md and docs/oocyte_detection.md. Do not optimize
the detector or change frozen profiles in this task. Audit the on-disk state for
samples [SAMPLE_IDS], using config [CONFIG], manifest [SAMPLE_CSV], output root
[OUTPUT_ROOT], and review exports under [REVIEW_DIR]. Report each sample's
current workflow phase and evidence before running anything. Then execute the
next deterministic step, preserve all existing outputs, and stop whenever a
biological review is required. After each exported review JSON is provided,
validate its identity, ingest it through the documented command, and generate
the next review checkpoint. Profile only final reviewed labels. Build releases
in new versioned directories and independently validate them before sharing.
```

For an interrupted session, append the exact last successful output directory,
review JSON path, and unresolved choice. Codex must re-audit those artifacts and
must not trust the stated phase without checking manifests.

## Review Iteration

Use review outcomes in two ways:

- Delivery correction: reviewed false positives, misses, and manual boundaries
  directly determine the current sample's final masks.
- Algorithm improvement: aggregate failure classes across completed samples,
  change detector parameters in a new named profile, and rerun precision plus
  whole-slide recall. Do not tune against one clicked object and claim a global
  detector improvement.

Before changing shared segmentation behavior, compare the proposed profile with
the frozen baseline on all reviewed donor13 sections and all donor11 negative
controls. Record count deltas, gained and lost reviewed objects, duplicate
behavior, and boundary regressions.

For the initial publication PR, algorithm improvement is explicitly deferred.
Review exports are included as provenance in the release package, not converted
into a learned model or global parameter update. A colleague may open a new
issue for optimization after reproducing the current reference workflow.

## Delivery Contract

The release builder consumes a YAML or JSON spec and creates an immutable tree:

```text
release/
  release_manifest.json
  batch_summary.csv
  batch_oocyte_by_marker.csv
  batch_oocyte_metadata.csv
  samples/<sample-id>/
    final/oocyte_labels.ome.tiff
    final/oocyte_labels.csv
    final/oocyte_candidates.csv
    final/masks/*.npz
    profiling/*.csv
    profiling/profiling_manifest.json
    review/review_manifest.json
    review_console.html
    sample_release_manifest.json
```

`release_manifest.json` is the authority for file hashes and sample totals.
Validation must confirm label-to-mapping pixel counts, unique IDs, profiling ID
alignment, uniform marker schemas, all package hashes, positive sample counts,
and zero labels plus zero accepted diagnostics for negative controls.

## Commands

```bash
python src/run_oocyte_release.py build \
  --spec exps/configs/oocyte_d11_d13_panel1/release_v6.yaml \
  --out-dir /workspaces/1-spatial_frs_analysis/oocyte-output/d11_d13_panel1_delivery_v6

python src/run_oocyte_release.py validate \
  --release-dir /workspaces/1-spatial_frs_analysis/oocyte-output/d11_d13_panel1_delivery_v6

python -m unittest discover -s tests/oocyte -v
```

When handing work to another Codex session, provide the active issue, exact
sample IDs, current versioned output directories, exported review JSON paths,
known unresolved choices, and the validation command. Do not describe an
unreviewed detector directory as a final delivery.
