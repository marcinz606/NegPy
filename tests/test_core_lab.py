"""Pins of the lab and toning kernels. Each test hashes the output for fixed seeded inputs."""

import cv2
import negpy_core
import numpy as np
from core_oracle import cv2_oracle
from core_pin import digest

from negpy.features.lab.logic import apply_clahe, apply_glow_and_halation, apply_rl_sharpening, gaussian_kernel_1d
from negpy.features.toning.logic import apply_split_toning
from negpy.kernel.image.logic import WORKING_LAB

pytestmark = cv2_oracle


def test_chroma_denoise_output_is_pinned():
    rng = np.random.default_rng(17)
    outs = []
    for shape in [(97, 131), (3, 4), (300, 200)]:
        for sigma in [0.5, 2.0, 6.0]:
            img = rng.random((*shape, 3), dtype=np.float32)
            outs.append(negpy_core.chroma_denoise(img, sigma, gaussian_kernel_1d(sigma), *WORKING_LAB))
    assert digest(*outs) == "8c4dce1aab011563c1fcc53782cc17515c971fddf155186690c55f427e0e2552"


def test_glow_and_halation_output_is_pinned():
    rng = np.random.default_rng(17)
    outs = []
    for shape in [(97, 131), (5, 7)]:
        for glow, halation, scale in [(0.3, 0.0, 1.0), (0.0, 0.5, 1.0), (0.4, 0.2, 0.5), (1.0, 1.0, 2.0)]:
            img = rng.random((*shape, 3), dtype=np.float32)
            outs.append(apply_glow_and_halation(img, glow, halation, scale))
    assert digest(*outs) == "932811cd3a0aaadcc57401245d6789f2f53061b7fe9642f774fe293fa254d03f"


def test_clahe_output_is_pinned():
    rng = np.random.default_rng(17)
    outs = []
    for shape in [(97, 131), (5, 7), (1, 1), (300, 200)]:
        for strength in [0.1, 0.5, 1.0, 3.0]:
            outs.append(apply_clahe(rng.random((*shape, 3), dtype=np.float32), strength))
    assert digest(*outs) == "17146c69aa9150fb55a18f51c745c6ec3c8c1a15113f5145fcc33da79e53883c"


def test_rl_sharpen_output_is_pinned():
    rng = np.random.default_rng(17)
    outs = []
    for shape in [(97, 131), (6, 7), (200, 300)]:
        for amount, radius, masking in [(0.25, 1.0, 0.0), (1.0, 0.6, 0.0), (0.6, 1.4, 0.5)]:
            img = cv2.GaussianBlur(rng.random((*shape, 3), dtype=np.float32), (0, 0), 1.0) - np.float32(0.05)
            outs.append(apply_rl_sharpening(img, amount, radius, masking))
    assert digest(*outs) == "0a57bcb9ec23a79ec40bfaedd4945ada401134f8a950ad78fafce97e90743429"


def test_split_toning_output_is_pinned():
    rng = np.random.default_rng(17)
    outs = []
    for args in [(30.0, 0.4, 210.0, 0.0), (0.0, 0.0, 45.0, 0.7), (300.0, 1.0, 120.0, 0.5)]:
        outs.append(apply_split_toning(rng.random((61, 83, 3), dtype=np.float32), *args))
    assert digest(*outs) == "fbfd4bd43bf8e10e2ccdd20afc3eccaa3060fa4b51e8b350194153d6f6fb1b55"
