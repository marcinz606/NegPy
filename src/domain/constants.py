from typing import List


def _reverse_aspect_ratios(ratios: List[str]) -> List[str]:
    return [":".join(r.split(":")[::-1]) for r in ratios if ":" in r and r != "1:1"]


SUPPORTED_ASPECT_RATIOS: List[str] = [
    "Free",
    "3:2",
    "4:3",
    "5:4",
    "6:7",
    "1:1",
    "65:24",
]
VERTICAL_ASPECT_RATIOS: List[str] = _reverse_aspect_ratios(SUPPORTED_ASPECT_RATIOS)
