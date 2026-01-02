import numpy as np
import pytest
from src.backend.utils import ensure_rgb, get_luminance

def test_ensure_rgb_2d():
    # Test grayscale (2D) to RGB (3D)
    img_2d = np.zeros((10, 10))
    img_rgb = ensure_rgb(img_2d)
    assert img_rgb.shape == (10, 10, 3)
    assert np.array_equal(img_rgb[:,:,0], img_2d)
    assert np.array_equal(img_rgb[:,:,1], img_2d)
    assert np.array_equal(img_rgb[:,:,2], img_2d)

def test_ensure_rgb_3d_single_channel():
    # Test 3D single channel to RGB
    img_1ch = np.zeros((10, 10, 1))
    img_rgb = ensure_rgb(img_1ch)
    assert img_rgb.shape == (10, 10, 3)

def test_ensure_rgb_already_rgb():
    # Test if RGB stays RGB
    img_rgb = np.zeros((10, 10, 3))
    img_out = ensure_rgb(img_rgb)
    assert img_out.shape == (10, 10, 3)
    assert img_out is img_rgb

def test_get_luminance_rgb():
    # Test luminance calculation
    img = np.zeros((1, 1, 3))
    img[0, 0] = [1.0, 1.0, 1.0] # White
    lum = get_luminance(img)
    assert np.isclose(lum[0, 0], 1.0)
    
    img[0, 0] = [1.0, 0.0, 0.0] # Red
    lum = get_luminance(img)
    assert np.isclose(lum[0, 0], 0.2126)

def test_get_luminance_2d_array():
    # Test luminance for flattened pixels (N, 3)
    pixels = np.array([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
    lum = get_luminance(pixels)
    assert lum.shape == (2,)
    assert np.isclose(lum[0], 1.0)
    assert np.isclose(lum[1], 0.2126)
