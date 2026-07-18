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

## Current boundary

The engine's exact complete-frame receipts cover two independent mounted C-41
frames from a Nikon Super Coolscan 5000 ED in the Digital ICE Normal path. The
NegPy capture runner also supports a registered roll position, but roll geometry
still needs its own independent full-frame parity receipt before it should carry
the same exactness label.

This path is intended for infrared-compatible color film. Traditional
silver-based black-and-white film, and some Kodachrome material, can block
infrared light in the image itself and are not suitable for this repair method.

This change adds the acquisition and processing boundaries. Product UI,
transactional TIFF publication, and post-scanner job scheduling remain separate
follow-up work, so that processing is always scheduled after the scanner
reservation has been released rather than during it.
