import platform
import sys

import pytest

# opencv-python's AVX2 build on Linux x86-64 is the reference negpy_core copies. Its other
# builds differ in the last bits (NEON lanes, FMA contraction), so they are not oracles.
cv2_oracle = pytest.mark.skipif(
    not (sys.platform == "linux" and platform.machine() == "x86_64"),
    reason="cv2 is the oracle only on its Linux x86-64 build",
)
