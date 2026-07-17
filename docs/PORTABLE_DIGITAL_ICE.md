# Portable Digital ICE integration

NegPy can acquire the two scanner sources required by
[Portable Digital ICE](https://github.com/rohanpandula/portable-digital-ice)
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

The processing adapter exposes only three explicit choices: `off`, `cpu`, and
`cuda`. It never exposes the engine's automatic fallback mode. A CUDA request
that cannot run fails with a clear error instead of silently starting an hour-long
CPU job. Raw RGBI sources remain untouched and the cleaned RGB image is returned
as a separate result with backend, output, RNG, and startup receipt data.

## Install the optional engine

Portable Digital ICE is not a required NegPy dependency. Install the published
wheel only when you want this path:

```sh
python -m pip install \
  https://github.com/rohanpandula/portable-digital-ice/releases/download/v0.1.0/portable_digital_ice-0.1.0-py3-none-any.whl
```

For the NVIDIA backend, install the CUDA extra and ensure a compatible driver is
available:

```sh
python -m pip install \
  "portable-digital-ice[cuda] @ https://github.com/rohanpandula/portable-digital-ice/releases/download/v0.1.0/portable_digital_ice-0.1.0-py3-none-any.whl"
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
This matters for the CPU backend, which took roughly 49 to 72 minutes per native
frame in the validation campaign. The exact CUDA backend took about 21 seconds
on an NVIDIA RTX A4000.

## Current boundary

The engine's exact complete-frame receipts cover two independent mounted C-41
frames from a Nikon Super Coolscan 5000 ED in the Digital ICE Normal path. The
NegPy capture runner also supports a registered roll position, but roll geometry
still needs its own independent full-frame parity receipt before it should carry
the same exactness label.

This path is intended for infrared-compatible color film. Traditional
silver-based black-and-white film, and some Kodachrome material, can block
infrared light in the image itself and are not suitable for this repair method.

The current draft adds the acquisition and processing boundaries. Product UI,
transactional TIFF publication, and post-scanner job scheduling remain separate
follow-up work so a long CPU run can never hold the scanner reservation.
