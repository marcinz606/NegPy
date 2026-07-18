# Portable Digital ICE integration

NegPy can acquire the two scanner sources required by
[Portable Digital ICE](https://github.com/rohanpandula/digital-fauxice)
and pass them to its exact CPU or CUDA backend before any rotation, negative
inversion, or color processing.

This is deliberately separate from NegPy's existing file-based infrared dust
tool. Portable Digital ICE needs two scans of the same physical frame:

1. a dedicated 285 dpi, 16-bit, single-sample RGBI prepass; and
2. a locked 4000 dpi, 16-bit, single-sample RGBI main scan.

Focus, exposure, frame position, crop, and orientation must stay fixed between
the two scans. An averaged RGB4x scan, a metering thumbnail, or an infrared
plane captured later is not a substitute for either source.

## What this integration provides

The acquisition runner:

- performs both scans through one SANE device handle;
- meters focus and per-channel exposure on the prepass, then replays that state
  with autofocus and auto-exposure disabled for the main scan;
- reads raw `sane_read` bytes so the final RGBI row is preserved;
- records every option write and readback;
- keeps the scanner-native RGBI arrays immutable; and
- publishes the pair only after its manifest, hashes, shapes, ordering, and
  payloads verify.

The processing adapter exposes four explicit choices: `off`, `cpu`, `cpu-fast`,
and `cuda`. It never exposes the engine's automatic fallback mode, and a backend
that cannot run fails with a clear error rather than quietly becoming a different
one — a CUDA request does not become an hour-long reference CPU job, and a
`cpu-fast` request does not either. Raw RGBI sources remain untouched and the
cleaned RGB image is returned as a separate result with backend, output, RNG, and
startup receipt data.

`cpu` is the engine's reference implementation. `cpu-fast` is its compiled
equivalent: the engine proves the two byte-identical on a synthetic job at
startup and refuses the compiled path if that proof fails, so the choice is a
speed decision rather than an accuracy one.

## Install the optional engine

Portable Digital ICE is not a required NegPy dependency. Install the published
wheel only when you want this path:

```sh
python -m pip install \
  https://github.com/rohanpandula/digital-fauxice/releases/download/v0.2.0/portable_digital_ice-0.2.0-py3-none-any.whl
```

That base install provides `cpu`. The compiled `cpu-fast` backend needs the
`fast` extra, which adds numba:

```sh
python -m pip install \
  "portable-digital-ice[fast] @ https://github.com/rohanpandula/digital-fauxice/releases/download/v0.2.0/portable_digital_ice-0.2.0-py3-none-any.whl"
```

The extra accepts numba 0.65 or 0.66, so it resolves against the numba NegPy
already pins and does not move it.

For the NVIDIA backend, install the CUDA extra and ensure a compatible driver is
available:

```sh
python -m pip install \
  "portable-digital-ice[cuda] @ https://github.com/rohanpandula/digital-fauxice/releases/download/v0.2.0/portable_digital_ice-0.2.0-py3-none-any.whl"
```

## Inspect or run the capture boundary

The command is scanner-free by default. It prints the exact live command and
the capture contract without touching hardware:

```sh
uv run python -m negpy.infrastructure.scanners.dice_dual_source_runner
```

Live acquisition requires an explicit confirmation that the film is loaded,
aligned, and not in use by another scanner client:

```sh
uv run python -m negpy.infrastructure.scanners.dice_dual_source_runner \
  --live \
  --confirm-film-stationary \
  --transport mounted \
  --out-dir dice-dual-rgbi-results
```

The returned bundle must be processed only after the scanner handle is closed.
Per-frame times at the native 4000 dpi frame size, from the engine's own
published measurements:

| Backend | Time per frame | Host |
| --- | --- | --- |
| `cpu` (reference) | roughly an hour | Apple M4 |
| `cpu-fast` | 9.2 to 9.5 s | Apple M4, default thread count |
| `cuda` | 5.3 to 5.8 s | NVIDIA RTX A4000 |

The reference backend is the one that makes the closed-handle rule matter most,
but the rule holds for all of them: processing is a separate job from
acquisition, and no backend should run while the scanner is still reserved.

## Roll workflow

The roll contact sheet can capture selected frames for Digital ICE directly.
With color negative material chosen, enable **Digital ICE (dual RGBI)** next to
the scan material and pick a processing backend; the selector shows which
backends are installed and defaults to `cpu-fast` when available. The Scan
button's free-space gate switches to the ICE budget: one transient dual-RGBI
bundle at a time plus one RGB-only master per frame.

Each selected slot then becomes its own complete cycle:

1. The reviewed roll fingerprint is re-verified, exactly as for the packed
   RGB4x path, and the slot's Film Spacing Offset resolves to coolscan3
   frame-and-subframe geometry. An offset that moved the frame past a slot
   boundary refuses rather than scanning the wrong slot.
2. One dedicated SANE session captures the 285 dpi prepass and the locked
   4000 dpi main, writes the verified bundle, and closes. The requested
   backend was already proven runnable before the scanner was touched.
3. With the scanner released, the engine repairs the frame and NegPy
   publishes one cleaned 16-bit scanner-linear TIFF beside a
   `_SCAN.json` receipt recording the slot, offset, plan, bundle hash,
   backend selection, and the engine's full receipt.

Completed frames import automatically as color negatives with the file-based
IR repair left off — their dust is already repaired and they carry no IR
sidecar. If processing fails, the frame's verified bundle stays on disk under
the attempts folder and the error names it, so the repair can be retried
without a rescan. Stop after current frame works between frames, as
everywhere else in the roll workflow.

Choosing ICE trades the RGB 4x noise averaging for the engine's exact
single-sample input contract; that trade is stated in the panel when the
toggle is on. Run a normal RGB4x+IR batch instead when the archival
multi-sample master matters more than exact ICE repair.

## Current boundary

The engine's exact complete-frame receipts cover two independent mounted C-41
frames from a Nikon Super Coolscan 5000 ED in the Digital ICE Normal path. The
NegPy capture runner and the roll workflow above also support registered roll
positions, but roll geometry still needs its own independent full-frame parity
receipt before it should carry the same exactness label.

This path is intended for infrared-compatible color film. Traditional
silver-based black-and-white film, and some Kodachrome material, can block
infrared light in the image itself and are not suitable for this repair method.
