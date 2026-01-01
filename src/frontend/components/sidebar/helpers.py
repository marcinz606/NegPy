
import numpy as np

def apply_wb_gains_to_sliders(r: float, g: float, b: float):
    """
    Translates raw RGB gains (from Auto-WB) into CMY filtration (0-170).
    Cyan is anchored at 0. Magenta and Yellow are logarithmic (100 * log10(gain)).
    """
    # Auto-WB already returns gains relative to Red (r=1.0).
    # m = 100 * log10(g_gain), y = 100 * log10(b_gain)
    m = int(np.log10(max(g, 1.0)) * 100.0)
    y = int(np.log10(max(b, 1.0)) * 100.0)
    
    return {
        'wb_cyan': 0,
        'wb_magenta': int(np.clip(m, 0, 170)),
        'wb_yellow': int(np.clip(y, 0, 170)),
        'cr_balance': 1.0,
        'mg_balance': 1.0,
        'yb_balance': 1.0
    }
