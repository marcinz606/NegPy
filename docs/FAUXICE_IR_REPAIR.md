# Optional IR dust repair (digital-fauxice)

`negpy.services.repair` adds an optional post-import step that repairs
infrared-flagged dust and scratches in a scanned frame. It calls
[digital-fauxice](https://github.com/rohanpandula/digital-fauxice), an
independently reverse-engineered, byte-exact reproduction of the Nikon Super
Coolscan's Digital ICE processing. NegPy does not implement any dust repair
logic itself here; it hands a frame to the engine and writes back what comes
out.

The feature is off by default and stays off unless a caller explicitly
enables it. Nothing in the normal import or render path touches it.

## What it needs

Digital ICE does not repair a single scan. It needs two RGBI captures of the
same physical frame: a 285 dpi prepass and a 4000 dpi main scan, with focus,
exposure, frame position, and crop held fixed between them. The prepass
carries per-frame calibration the main pass depends on. It is not a
downsampled convenience copy and cannot be reconstructed from the main scan
after the fact.

NegPy's current scanning pipeline captures one pass per frame and keeps no
prepass. On a frame imported that way, this module reports a skipped status
rather than inventing one. The calling contract exists so that a future
acquisition path, such as one built on
[coolscanpy](https://github.com/rohanpandula/coolscanpy), can supply a real
prepass and have repair start working without any change to this module.

## Install

Two optional dependency groups exist in `pyproject.toml`, alongside the
existing `scanner` group that carries `python-sane`:

```
uv sync --group fauxice           # core engine only, exact mode
uv sync --group fauxice-hybrid    # adds the hybrid companion
```

Neither package is on PyPI. Both install from a digital-fauxice GitHub
release, the same way `python-sane` is an optional group rather than a hard
dependency: NegPy runs without either installed, and every function in this
module reports a clear status instead of raising when they are missing.

Hybrid mode additionally needs a separately installed IOPaint 1.6.0 runtime
and the `big-lama.pt` weights, in their own virtual environment. Neither is
installed by the `fauxice-hybrid` group. See digital-fauxice's own
`hybrid/docs/hybrid-repair.md` for that setup; NegPy only points at the
paths once they exist.

## Two modes

The setting offers `exact` and `hybrid`. `exact` is the default.

`exact` reproduces Nikon's own Digital ICE output value for value. The
engine's validation compared two complete frames against Nikon's real
output, 68,447,316 16-bit samples per frame, with zero mismatches.

`hybrid` additionally routes the frame's worst damage, the regions where the
engine's own defect signal maxes out, to a LaMa inpainting model, and
composites the result back into the exact output. It needs the separate
`fauxice-hybrid` package and its IOPaint runtime. If hybrid is requested but
either is missing, or the run fails for any reason, repair falls back to
`exact` and the sidecar records why. Hybrid mode never fails the whole
repair; it degrades to exact.

Exact is the default because of the time difference, measured by
digital-fauxice on its own validation frames on an Apple M4:

| | Frame 1 (lightly marked) | Frame 2 (badly marked) |
|---|---:|---:|
| Exact repair, cpu-fast | about 10 s | about 10 s |
| Hybrid, whole frame | 72 s | 210 s |

That is roughly seven times the exact path on a lightly marked frame and
twenty times on a badly marked one, because the model runs once per routed
region on CPU and the routed area tracks how damaged the frame is, not how
large it is. Over a 36-frame roll the difference is about six minutes
against something between forty minutes and two hours. The sensible pattern
is to run exact across a whole roll and reach for hybrid only on the few
frames that still show visible ICE residue afterward.

## Progress and cancellation

`repair_ir_dust` and `repair_frame_files` accept an optional `progress`
callback and `cancel` event, in the same shape
`ScannerService.run_scan` already uses. Both only cover the exact path: the
engine supports cooperative progress and cancellation natively there. Hybrid
mode is one blocking call to an external CLI, so `progress` is never called
during a hybrid run, and `cancel` is only checked before that call starts,
not while it runs.

## Output and provenance

A repair never rewrites the source master. NegPy keys stored edits by the
source file's content hash, so silently changing that file would orphan
every edit already saved against it. A successful repair instead writes a
new companion, `<basename>_FAUXICE.tif`, next to the untouched original. The
`_IR.tif` companion is only ever opened for reading.

Every call also writes `<basename>_FAUXICE.json`, whether or not the repair
was applied, so a caller can show why. It records:

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
white pixel marks a synthesized one. Outside that mask, the hybrid output is
byte-identical to the exact output; digital-fauxice's own receipt proves
this pixel by pixel before NegPy ever sees the result.

The routing that decides what gets filled is conservative on purpose and
fails closed: if the routed regions would cover more than 2% of the frame,
digital-fauxice refuses to run hybrid mode on that frame at all rather than
fill an unbounded area. On its own two validation frames the tool routed
0.07% and 1.36% of the frame respectively.

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

Until NegPy's scanning pipeline captures a paired prepass, this module has
nothing to repair on ordinary imports and will report a skipped status. The
config, the sidecar format, and the tests in this repository exist so that
gap can close without another round of adapter work.
