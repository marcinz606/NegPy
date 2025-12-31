
import numpy as np

def apply_wb_gains_to_sliders(r: float, g: float, b: float):
    """
    Translates raw RGB gains (from Auto-WB) into CMY filtration (0-170).
    Mapping: val = log10(gain / min_gain) * 100.
    This ensures at least one slider is at 0 (Darkroom Workflow).
    """
    # Find minimum gain to anchor at 0
    min_g = min(r, g, b)
    
    # Calculate filtration in CC (Color Correction) units
    # 100 CC = 1.0 Density = 10x Gain
    c = int(np.log10(r / min_g) * 100.0)
    m = int(np.log10(g / min_g) * 100.0)
    y = int(np.log10(b / min_g) * 100.0)
    
    return {
        'wb_cyan': int(np.clip(c, 0, 170)),
        'wb_magenta': int(np.clip(m, 0, 170)),
        'wb_yellow': int(np.clip(y, 0, 170)),
        'cr_balance': 1.0,
        'mg_balance': 1.0,
        'yb_balance': 1.0
    }
