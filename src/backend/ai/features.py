import numpy as np
import cv2

def extract_features(image_data: np.ndarray) -> np.ndarray:
    """
    Extracts a feature vector from a raw image numpy array (H, W, 3).
    Input image is expected to be normalized (0.0 - 1.0) RGB.
    
    Returns:
        np.ndarray: A 1D array of features.
    """
    # 1. Resize for speed and consistency
    target_size = (256, 256)
    if image_data.shape[0] > target_size[0] or image_data.shape[1] > target_size[1]:
        img_small = cv2.resize(image_data, target_size, interpolation=cv2.INTER_AREA)
    else:
        img_small = image_data.copy()

    features = []

    # 2. Global Statistics (RGB)
    for i in range(3):
        channel = img_small[:, :, i]
        features.extend([
            np.mean(channel),
            np.std(channel),
            np.percentile(channel, 1),
            np.percentile(channel, 5),
            np.percentile(channel, 50),
            np.percentile(channel, 95),
            np.percentile(channel, 99)
        ])

    # 3. Histograms (RGB)
    # 32 bins per channel
    for i in range(3):
        hist = cv2.calcHist([img_small.astype('float32')], [i], None, [32], [0, 1])
        hist = hist.flatten() / (img_small.shape[0] * img_small.shape[1]) # Normalize
        features.extend(hist)

    # 4. Color Space Statistics (LAB) - perceptual lightness/color
    img_lab = cv2.cvtColor(img_small.astype('float32'), cv2.COLOR_RGB2LAB)
    # LAB in OpenCV float is L: 0-100, A: -128-127, B: -128-127
    # We normalize to 0-1 for consistency
    l_chan, a_chan, b_chan = cv2.split(img_lab)
    l_chan = l_chan / 100.0
    a_chan = (a_chan + 128) / 255.0
    b_chan = (b_chan + 128) / 255.0

    for channel in [l_chan, a_chan, b_chan]:
        features.extend([
            np.mean(channel),
            np.std(channel),
            np.percentile(channel, 10),
            np.percentile(channel, 90)
        ])

    return np.array(features, dtype=np.float32)
