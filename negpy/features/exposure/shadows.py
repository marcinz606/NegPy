import numpy as np
from negpy.domain.types import ImageBuffer
from negpy.kernel.image.validation import ensure_image

def analyze_shadow_cast(image: ImageBuffer, threshold: float = 0.75) -> tuple[float, float, float]:
    density = np.mean(image, axis=-1)
    mask = density > threshold

    if np.any(mask):
        avg_rgb = np.mean(image[mask], axis=0)
        target_density = np.mean(avg_rgb)
        correction_vector = target_density - avg_rgb
        return (float(correction_vector[0]), float(correction_vector[1]), float(correction_vector[2]))

    return (0.0, 0.0, 0.0)

def apply_shadow_cast_correction(
    image: ImageBuffer, correction_vector: tuple[float, float, float], strength: float = 1.0
) -> ImageBuffer:
    if strength <= 0:
        return image

    density = np.mean(image, axis=-1, keepdims=True)
    cv = np.array(correction_vector, dtype=np.float32)
    correction = cv * (density**1.5) * strength
    corrected_image = image + correction

    return ensure_image(np.clip(corrected_image, 0.0, 1.0))
