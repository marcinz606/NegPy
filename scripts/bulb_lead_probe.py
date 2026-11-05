"""Measure how soon the camera's shutter is open after NegPy presses Bulb over USB.

Shoots red-only Blend exposures with a shorter dark lead each time and compares each red level with the
longest lead's. A lead shorter than the shutter's lag cuts the start of the red light, and the level drops.
Close NegPy first: the camera and the light take one program at a time. Leave the film still.

    uv run python scripts/bulb_lead_probe.py
"""

from __future__ import annotations

import argparse
import os
import statistics
import tempfile
import time

import numpy as np

import negpy.services.capture.service as service
from negpy.infrastructure.capture.base import CaptureSettings
from negpy.infrastructure.capture.gphoto import GphotoCamera
from negpy.infrastructure.capture.lights import open_light
from negpy.infrastructure.capture.raw_demosaic import linear_demosaic

LEADS = (0.3, 0.2, 0.15, 0.1, 0.075, 0.05, 0.025, 0.0)
SECONDS = 0.2  # the red time: short, so a few ms of lag show as a clear drop
TARGET = 0.4  # red level as a share of full scale, clear of clipping and noise
TOLERANCE = 0.01  # a 1 % drop at 0.2 s is 2 ms of lost light
MARGIN_S = 0.05


def _red(path: str) -> float:
    img = linear_demosaic(path, half_size=True)
    h, w = img.shape[:2]
    return float(img[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4, 0].mean()) / 65535.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--repeats", type=int, default=3, help="exposures per lead")
    parser.add_argument("--level", type=int, default=120, help="the first red level; the probe then aims at 40 %% of full scale")
    args = parser.parse_args()

    camera = GphotoCamera()
    camera.open()
    light = open_light()
    # Time every light command, sent and returned, so a varying light time (the light's reply
    # latency) shows apart from the shutter's lag.
    calls: list[tuple[str, float, float]] = []
    for name in ("set_color", "set_color_now", "off"):
        original = getattr(light, name, None)
        if original is None:
            continue

        def _timed(*a, _original=original, _name=name, **k):
            start = time.perf_counter()
            try:
                return _original(*a, **k)
            finally:
                lit = _name != "off" and (any(a) or any(k.values()))
                calls.append(("on" if lit else "off", start, time.perf_counter()))

        setattr(light, name, _timed)
    on_times: list[float] = []
    on_latency: list[float] = []
    off_latency: list[float] = []
    folder = tempfile.mkdtemp(prefix="negpy-bulb-lead-")
    capture = service.CaptureService(light, camera)
    count = 0

    def shoot(lead: float, level: int) -> float:
        nonlocal count
        count += 1
        service.BULB_LEAD_S = lead
        settings = CaptureSettings(roll_name="probe", frame_number=count, output_folder=folder, levels=(level, 0, 0))
        del calls[:]
        path = capture.capture_blend(settings, seconds=SECONDS, keep_bulb=True)
        ons = [c for c in calls if c[0] == "on"]
        if ons:
            on = ons[0]
            off = next((c for c in calls if c[0] == "off" and c[1] >= on[2]), None)
            on_latency.append(on[2] - on[1])
            if off is not None:
                off_latency.append(off[2] - off[1])
                # The light is lit from somewhere inside the on command to somewhere inside the off command.
                on_times.append(off[1] - on[1])
        try:
            return _red(path)
        finally:
            os.remove(path)

    try:
        first = shoot(LEADS[0], args.level)
        if first < 0.005:
            raise SystemExit("The red light reads almost nothing. Open the aperture or raise the ISO, then run the probe again.")
        level = int(np.clip(round(args.level * TARGET / first), 10, 255))
        print(f"Red at level {level} for {SECONDS} s, {args.repeats} exposures per lead.\n")
        results = {lead: [shoot(lead, level) for _ in range(args.repeats)] for lead in LEADS}
        base = statistics.mean(results[LEADS[0]])
        if base > 0.9:
            print("Warning: the red level is near clipping, so a drop can hide. Close the aperture one stop and run again.\n")
        print(" lead (s)   red / longest lead")
        safe = None
        for lead in LEADS:
            ratios = [value / base for value in results[lead]]
            passed = min(ratios) >= 1.0 - TOLERANCE
            print(f"  {lead:5.3f}    " + "  ".join(f"{r:6.3f}" for r in ratios) + ("" if passed else "   <- light lost"))
            if passed and (safe is None or safe == LEADS[LEADS.index(lead) - 1]):
                safe = lead
        if on_times:

            def ms(xs: list[float]) -> str:
                return f"min {min(xs) * 1000:6.1f}  mean {statistics.mean(xs) * 1000:6.1f}  max {max(xs) * 1000:6.1f} ms"

            print(f"\nLight command reply time, on:  {ms(on_latency)}")
            print(f"Light command reply time, off: {ms(off_latency)}")
            print(f"Red lit, on sent to off sent:  {ms(on_times)}  (asked for {SECONDS * 1000:.0f} ms)")
        if safe is None:
            print("\nEven the longest lead varies by more than 1 %. Check that the film and the light are still, then run again.")
        else:
            print(f"\nThe shutter is open by a lead of {safe:.3f} s. With a margin, use a lead of {safe + MARGIN_S:.3f} s.")
    finally:
        try:
            camera.release_bulb()
            light.off()
        finally:
            light.close()
            camera.close()


if __name__ == "__main__":
    main()
