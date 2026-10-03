"""Frames scanned more than once: rescans of one negative, and copies of one scan.

Byte-identical files never reach the Film Strip (`session.add_files` skips a repeated
content hash). This finds the rest by picture: a coarse fingerprint pairs candidates, and a
registered correlation confirms them, since a rescan sits a few pixels off and crops
differently. A neighbouring frame of the same scene does not register, which is what keeps
two shots of one subject apart from two scans of one frame.
"""

import itertools
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

#: Long edge of the image a fingerprint is measured on.
FINGERPRINT_EDGE = 256
#: Coarse similarity a pair needs before it is registered.
PREFILTER = 0.9
#: Registered correlation that makes a pair one frame. Rescans measure above it and
#: neighbouring frames of one scene below it.
MATCH = 0.95
#: Above this the two files hold the same scan (an export, a resized copy), so stacking
#: them averages nothing.
SAME_SCAN = 0.995
#: Aspect ratios further apart than this are different crops of different things.
_MAX_ASPECT_DIFF = 0.08


@dataclass(frozen=True)
class Fingerprint:
    image: np.ndarray  # log-density, zero mean, unit variance, FINGERPRINT_EDGE long
    coarse: np.ndarray  # 16x16 of the same, unit length


@dataclass(frozen=True)
class DuplicateGroup:
    paths: Tuple[str, ...]
    #: Weakest confirmed link in the group.
    score: float

    @property
    def same_scan(self) -> bool:
        return self.score >= SAME_SCAN


def fingerprint(pixels: np.ndarray) -> Fingerprint:
    """Fingerprint of an image (any size, gray or RGB, any bit depth)."""
    a = np.asarray(pixels, dtype=np.float32)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2)
    s = FINGERPRINT_EDGE / max(a.shape[:2])
    if s < 1.0:
        a = cv2.resize(a, (max(1, round(a.shape[1] * s)), max(1, round(a.shape[0] * s))), interpolation=cv2.INTER_AREA)
    # Log of the level, so a scan and its gamma-encoded copy compare on the same footing.
    a = np.log1p(a / max(float(a.max()), 1e-6) * 1000.0)
    a = (a - a.mean()) / (a.std() + 1e-6)
    c = cv2.resize(a, (16, 16), interpolation=cv2.INTER_AREA).ravel()
    c = c - c.mean()
    return Fingerprint(image=a.astype(np.float32), coarse=(c / (np.linalg.norm(c) + 1e-9)).astype(np.float32))


def registered_similarity(a: Fingerprint, b: Fingerprint) -> float:
    """Correlation of two fingerprints once ``b`` is shifted onto ``a``. -1 when they cannot
    be the same frame (different aspect)."""
    ia, ib = a.image, b.image
    ha, wa = ia.shape
    hb, wb = ib.shape
    if abs(ha / wa - hb / wb) > _MAX_ASPECT_DIFF:
        return -1.0
    if ib.shape != ia.shape:
        ib = cv2.resize(ib, (wa, ha), interpolation=cv2.INTER_AREA)
    win = cv2.createHanningWindow((wa, ha), cv2.CV_32F)
    (dx, dy), _resp = cv2.phaseCorrelate(ia, ib, win)
    m = int(max(abs(dx), abs(dy))) + 4
    if 2 * m >= min(ha, wa):
        return -1.0
    shifted = cv2.warpAffine(ib, np.float32([[1, 0, -dx], [0, 1, -dy]]), (wa, ha))
    x, y = ia[m:-m, m:-m].ravel(), shifted[m:-m, m:-m].ravel()
    return float(np.corrcoef(x, y)[0, 1])


def find_groups(prints: Dict[str, Fingerprint]) -> List[DuplicateGroup]:
    """Groups of paths that hold one frame, in path order. Linked pairwise, so a group is
    any chain of confirmed matches."""
    paths = sorted(prints)
    parent = {p: p for p in paths}

    def root(p: str) -> str:
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    links: List[Tuple[str, str, float]] = []
    for p, q in itertools.combinations(paths, 2):
        if float(prints[p].coarse @ prints[q].coarse) < PREFILTER:
            continue
        score = registered_similarity(prints[p], prints[q])
        if score >= MATCH:
            links.append((p, q, score))
            parent[root(p)] = root(q)

    members: Dict[str, List[str]] = {}
    for p in paths:
        members.setdefault(root(p), []).append(p)
    weakest: Dict[str, float] = {}
    for p, _q, score in links:
        r = root(p)
        weakest[r] = min(weakest.get(r, 1.0), score)
    groups = [DuplicateGroup(tuple(m), weakest[r]) for r, m in members.items() if len(m) > 1]
    return sorted(groups, key=lambda g: g.paths)


def ordered_pairs(groups: Sequence[DuplicateGroup]) -> int:
    """Files a decision on every group would act on besides the one kept."""
    return sum(len(g.paths) - 1 for g in groups)
