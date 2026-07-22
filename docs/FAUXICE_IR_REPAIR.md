# Optional IR dust repair (digital-fauxice)

`negpy.services.repair` provides the repair runtime used by the roll workflow
and by the optional post-import file helper. It calls
[digital-fauxice](https://github.com/rohanpandula/digital-fauxice), an
independently reverse-engineered implementation whose pinned exact path was
validated sample-for-sample against Nikon output. Hybrid is a separate,
explicitly disclosed model-assisted path and is not labeled byte-exact Nikon
output. NegPy does not implement the repair math itself; it validates and
binds the acquisition, runtime, output, and receipts around the engine.

The standalone post-import helper is off unless a caller explicitly enables
it. The LS-5000 roll sidebar is different: its new-user Color + DICE workflow
enables Repaired and Positive and selects Hybrid by default while preserving
any previously saved choices. Nothing in the normal image render path runs
repair implicitly.

## What it needs

Digital ICE does not repair a single scan. It needs two RGBI captures of the
same physical frame: a 285 dpi prepass and a 4000 dpi main scan, with focus,
exposure, frame position, and crop held fixed between them. The prepass
carries per-frame calibration the main pass depends on. It is not a
downsampled convenience copy and cannot be reconstructed from the main scan
after the fact.

NegPy's generic single-frame/import pipeline captures no paired prepass. On a
frame imported that way, this module reports a skipped status rather than
inventing one. The current LS-5000 roll workflow can supply a real, frame-bound
prepass, main pass, infrared-validity array, and acquisition provenance from
the reviewed coolscanpy overlay. The two Tier-1 TIFFs alone do not retain all
of those inputs and cannot later be promoted to an exact repair acquisition.

## Retained roll-acquisition replay archive

For an infrared-bearing Tier-1 roll frame, NegPy attempts to retain those
otherwise transient inputs at write time. The frame receipt exposes the
result as `outputs.repair_acquisition_evidence`. Success is explicit:
`retained` and `replayable` are both `true`, and `schema` is
`negpy.dice-acquisition-replay-v1`.

The archive adds three files to the RGB/IR TIFF pair:

- `acquisition-binding.json`, represented by a `binding` object with `path`,
  `bytes`, and `sha256`;
- `prepass.rgbi16.npy`, the scanner-native RGBI prepass; and
- `ir-validity.npy`, the scanner-native per-pixel validity map.

The receipt lists the two NPY inputs under `artifacts` and the Tier-1 RGB and
infrared TIFFs under `sources`. Each row binds its expected layout and content
and includes a relative path. The binding declares the source TIFFs as
`upright-storage` and its `replay.requires` contract names five requirements:
`storage_rgb_tiff`, `storage_ir_tiff`, `prepass_rgbi`, `ir_validity`, and
`acquisition_provenance`.

To reconstruct scanner-native main RGBI, replay stacks the upright-storage
RGB and infrared planes and applies `rot90(k=-1)`. The
`load_repair_acquisition_evidence` loader stable-reads the binding and every
referenced input, verifies their hashes and geometry, and re-derives the
Coolscan producer acquisition identity and evidence hash. Relative paths let
an operator move the whole output archive and replay it in the new location;
the TIFFs and hidden evidence directories must stay together. Copying only
the TIFF pair or only the binding directory does not satisfy the contract.
After a move, pass `acquisition-binding.json` at its new location; the
capture-time absolute `binding.path` in the outer frame receipt is not a
relocation pointer.

The replay advertises `integrity-bound-not-signed`. Its hashes and canonical
producer binding detect substitution or corruption, but are not a publisher
signature. Retention is fail-soft: if validation or writing fails, Tier 1 can
still survive while the receipt records `retained: false`,
`replayable: false`, and an `unavailable` reason. That state must never be
treated as replayable or upgraded from the TIFF pair later.

## Install

Two optional dependency groups exist in `pyproject.toml`, alongside the
existing `scanner` group that carries `python-sane`:

```
uv sync --group fauxice           # core engine only, exact mode
uv sync --group fauxice-hybrid    # adds the hybrid companion
```

Neither package is on PyPI. Both install from a pinned digital-fauxice GitHub
release. The local macOS parity app bundles the core runtime and its frozen
smoke fails if the roll repair bridge does not register. A source installation
without the optional group can still run NegPy, but repair reports unavailable.

Hybrid mode additionally needs a separately installed IOPaint 1.6.0 runtime
and the `big-lama.pt` weights, in their own virtual environment. Neither is
installed by the `fauxice-hybrid` group. See digital-fauxice's own
`hybrid/docs/hybrid-repair.md` for that setup; NegPy only points at the
paths once they exist.

## Two modes

The setting offers `exact` and `hybrid`. The standalone post-import helper
defaults to `exact`; the LS-5000 roll sidebar's first-run Color + DICE workflow
defaults to `hybrid` and visibly labels its model-generated fill. Existing
saved roll choices are preserved.

`exact` reproduces Nikon's own Digital ICE output value for value. The
engine's validation compared two complete frames against Nikon's real
output, 68,447,316 16-bit samples per frame, with zero mismatches.

`hybrid` additionally routes the frame's worst damage, the regions where the
engine's own defect signal maxes out, to a LaMa inpainting model, and
composites the result back into the exact output. It needs the separate
`fauxice-hybrid` package and its IOPaint runtime. If hybrid is requested but
either is missing, or its validated run fails, repair falls back to `exact`
and the receipt records why. Explicit cancellation remains cancellation; it
does not silently start a potentially long exact repair afterward.

Do not use the old whole-frame timing numbers as a promise: runtime varies with
selected backend, frame geometry, routed damage, model runtime, and host
configuration. Hybrid can be materially slower because it invokes the
separately pinned inpainting environment for routed regions; the desktop
reports progress and supports cancellation instead of estimating a fixed
completion time.

## Progress and cancellation

`repair_ir_dust` and `repair_frame_files` accept an optional `progress`
callback and `cancel` event, in the same shape `ScannerService.run_scan`
already uses. Exact forwards cooperative progress and cancellation to the
engine. Hybrid reports coarse verified-run checkpoints and polls the external
process; cancellation terminates its process group and returns a cancelled
result instead of leaving the CLI running in the background.

## Output and provenance

A repair never rewrites the source master. NegPy keys stored edits by the
source file's content hash, so silently changing that file would orphan
every edit already saved against it. The post-import file helper writes a new
companion, `<basename>_FAUXICE.tif`, next to the untouched original; its
`_IR.tif` companion is only ever opened for reading. The roll workflow instead
uses its Tier-2 names and frame receipt described in
`COOLSCANPY_ROLL_SCANNING.md`.

Every post-import file-helper call also writes
`<basename>_FAUXICE.json`, whether or not the repair was applied, so a caller
can show why. It records:

- the status: applied, skipped, or unavailable, and a plain-text reason
- the mode requested and the mode actually used
- the engine's package name and installed version
- the compute backend requested and the one actually used
- for a hybrid run, the routing counts from the tool's own receipt (how
  many regions were routed, how many pixels were synthesized, out of how
  many pixels in the frame) and the SHA-256 of the disclosure mask PNG

## Hybrid fill is disclosed, not hidden

Hybrid mode does not claim its filled pixels are recovered film information.
They are generated by a model. Every synthesized pixel is recorded in a mask
PNG written alongside the output (`<basename>_FAUXICE_SYNTH.png`), where a
white pixel marks a synthesized one. NegPy stable-reads and validates the
canonical hybrid receipt, output and mask hashes, geometry, routing counts,
synthesis accounting, runtime manifests, model identity, and the receipt's
within-budget verdict. It does not turn those checks into a broader claim that
the entire hybrid image is byte-identical Nikon output.

## Limits

This only works on color negative film with a real infrared channel
captured on the same physical frame as the visible-light scan. Traditional
silver-based black and white film and some Kodachrome stocks block infrared
light outright, so neither this module nor Nikon's own Digital ICE can
clean them; there is no IR signal to read.

The validated profile is narrow: a Nikon Super Coolscan 5000 ED running
Digital ICE Normal at the resolution metrics used in digital-fauxice's own
validation. An acquisition outside that profile is rejected before any
processing runs.

Ordinary imports and generic one-pass scans still have no paired prepass and
therefore report a skipped status. The LS-5000 roll acquisition path supplies
the required pair and binds it to one frame. If that binding, validity data,
runtime, or receipt is missing or inconsistent, repair fails closed; it does
not reconstruct calibration from a main scan or from Tier-1 TIFFs.
