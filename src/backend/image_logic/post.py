import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
from src.backend.image_logic.color import apply_color_separation
from src.domain_objects import ImageSettings


def apply_post_color_grading(
    pil_img: Image.Image, params: ImageSettings
) -> Image.Image:
    """
    Applies post-processing color grading:
    1. Color Separation (via numpy/custom logic)
    2. Saturation (via PIL ImageEnhance)

    Args:
        pil_img (Image.Image): Input PIL Image (RGB).
        params (ImageSettings): Processing parameters.

    Returns:
        Image.Image: Processed PIL Image.
    """
    is_bw = params.is_bw
    if not is_bw:
        # 1. Color Separation
        img_arr = np.array(pil_img)
        img_sep = apply_color_separation(img_arr, params.color_separation)
        pil_img = Image.fromarray(img_sep)

        # 2. Classic Saturation
        sat = params.saturation
        if sat != 1.0:
            enhancer = ImageEnhance.Color(pil_img)
            pil_img = enhancer.enhance(sat)
    else:
        # B&W Mode: Ensure it's grayscale if it isn't already handled
        pass

    return pil_img


def apply_output_sharpening(pil_img: Image.Image, amount: float) -> Image.Image:
    """
    Applies Unsharp Mask sharpening to the Lightness channel.

    Args:
        pil_img (Image.Image): Input PIL Image.
        amount (float): Sharpening amount (0.0 to 1.0+).

    Returns:
        Image.Image: Sharpened PIL Image.
    """
    if amount <= 0:
        return pil_img

    if pil_img.mode != "RGB":
        # If B&W (L mode), convert to RGB to use LAB logic or just sharpen L directly?
        # Previous logic:
        # img_lab = cv2.cvtColor(np.array(pil_prev.convert("RGB")), cv2.COLOR_RGB2LAB)
        # So it handles conversion.
        pil_working = pil_img.convert("RGB")
    else:
        pil_working = pil_img

    img_lab = cv2.cvtColor(np.array(pil_working), cv2.COLOR_RGB2LAB)
    l_chan, a, b = cv2.split(img_lab)
    l_pil = Image.fromarray(l_chan)

    # Radius=1.0, Threshold=5 are fixed in original code
    l_sharpened = l_pil.filter(
        ImageFilter.UnsharpMask(radius=1.0, percent=int(amount * 250), threshold=5)
    )

    img_lab_sharpened = cv2.merge([np.array(l_sharpened), a, b])
    result_rgb = Image.fromarray(cv2.cvtColor(img_lab_sharpened, cv2.COLOR_LAB2RGB))

    # If input was L, return L
    if pil_img.mode == "L":
        return result_rgb.convert("L")

    return result_rgb
