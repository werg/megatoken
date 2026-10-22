"""Numeric core of the collection index: quantized codes, centroids and probe calibration.

Pure numpy. Keys are unit teacher embeddings. An inverted-file index groups them
into cells around spherical k-means centroids; each key also has a symmetric int8
code (``key ~= scale * code``) so a probed cell can be ranked without reading
the float keys, which are read only as far as the exact top-k needs.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

MAX_CELLS = 4096
MAX_FIT_KEYS = 32768
CALIBRATION_QUERIES = 256
# Norm of the noise added to a key to make a stand-in for a student's query.
CALIBRATION_NOISE = 0.75
PROBE_LADDER = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 1024, 2048)


def unit(values, *, width: int | None = None) -> np.ndarray:
    """A finite float32 unit vector (or matrix of them, row by row)."""
    array = np.asarray(values, dtype="<f4")
    if width is not None and array.shape[-1:] != (width,):
        raise ValueError(f"Expected vectors of width {width}")
    norms = np.linalg.norm(array.astype(np.float64), axis=-1, keepdims=True)
    if not np.isfinite(array).all() or (norms < 1e-8).any():
        raise ValueError("A search key must be finite and nonzero")
    return np.ascontiguousarray(array / norms.astype(np.float32), dtype="<f4")


def quantize(key: np.ndarray) -> tuple[float, np.ndarray]:
    """Symmetric int8 code of one unit key: ``key ~= scale * code``."""
    scale = float(np.abs(key).max()) / 127
    return scale, np.rint(key / scale).astype(np.int8)


def fit_centroids(
    keys: np.ndarray, cells: int, *, iterations: int = 12, seed: int = 12345
) -> np.ndarray:
    """Deterministic spherical k-means over (a sample of) the keys."""
    if not 1 <= cells <= len(keys):
        raise ValueError("Cannot fit more cells than keys")
    rng = np.random.default_rng(seed)
    if len(keys) > MAX_FIT_KEYS:
        keys = keys[np.sort(rng.choice(len(keys), size=MAX_FIT_KEYS, replace=False))]
    centers = keys[rng.choice(len(keys), size=cells, replace=False)].copy()
    for _ in range(iterations):
        totals = np.zeros_like(centers)
        counts = np.zeros(cells, dtype=np.int64)
        for start in range(0, len(keys), 2048):
            block = keys[start : start + 2048]
            closest = np.argmax(block @ centers.T, axis=1)
            np.add.at(totals, closest, block)
            np.add.at(counts, closest, 1)
        occupied = counts > 0
        centers[occupied] = unit(totals[occupied])
        # An empty cell is reseeded from a key, so the cell count stays what was asked.
        if not occupied.all():
            refill = rng.choice(len(keys), size=int((~occupied).sum()), replace=False)
            centers[~occupied] = keys[refill]
    return unit(centers)


def calibrate_nprobe(
    keys: np.ndarray,
    centroids: np.ndarray,
    queries: np.ndarray,
    *,
    k: int = 10,
    target_recall: float = 0.97,
) -> tuple[int, dict[int, float]]:
    """Smallest probe count whose IVF top-k overlaps the exact top-k at ``target_recall``.

    Every row is a unit vector. Returns the choice and the measured overlap of
    each probe count tried.
    """
    if not 0 < target_recall <= 1 or len(queries) < 1 or not 1 <= k < len(keys):
        raise ValueError("Calibration needs queries, more keys than k and a recall in (0, 1]")
    assignment = np.argmax(keys @ centroids.T, axis=1)
    cells = len(centroids)
    ladder = sorted({n for n in PROBE_LADDER if n < cells} | {cells})
    overlap = dict.fromkeys(ladder, 0.0)
    for start in range(0, len(queries), 128):
        block = queries[start : start + 128]
        scores = block @ keys.T
        exact = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        order = np.argsort(-(block @ centroids.T), axis=1, kind="stable")
        for row in range(len(block)):
            reference = set(exact[row].tolist())
            for nprobe in ladder:
                candidates = np.flatnonzero(np.isin(assignment, order[row, :nprobe]))
                top = candidates[np.argsort(-scores[row, candidates], kind="stable")[:k]]
                overlap[nprobe] += len(reference.intersection(top.tolist())) / k
    curve = {nprobe: overlap[nprobe] / len(queries) for nprobe in ladder}
    for nprobe in ladder:
        if curve[nprobe] >= target_recall:
            return nprobe, curve
    return cells, curve


def noisy_queries(keys: np.ndarray, *, seed: int = 12345) -> np.ndarray:
    """Stand-ins for a student's imperfect queries: sampled keys plus fixed-norm noise."""
    rng = np.random.default_rng(seed)
    chosen = keys[rng.choice(len(keys), size=min(CALIBRATION_QUERIES, len(keys)), replace=False)]
    noise = rng.standard_normal(chosen.shape).astype(np.float32)
    noise *= CALIBRATION_NOISE / np.linalg.norm(noise, axis=1, keepdims=True)
    return unit(chosen + noise)


def rank_exact(
    ids: np.ndarray,
    scales: np.ndarray,
    codes: np.ndarray,
    query: np.ndarray,
    limit: int,
    read_keys: Callable[[np.ndarray], np.ndarray],
) -> list[tuple[int, float]]:
    """The exact top ``limit`` (id, score) of the candidates, ties broken by id.

    Candidates are ordered by their int8 scores and their float keys read in that
    order until the ``limit``-th exact score beats every unread candidate's best
    possible score: its int8 score plus the quantization error bound
    ``scale / 2 * |q|_1``, widened by 2% for float32 accumulation. The result
    equals exact rescoring of every candidate.
    """
    if limit < 1 or not len(ids):
        return []
    q = np.asarray(query, dtype=np.float64)
    approx = (codes.astype(np.float32) @ q.astype(np.float32)) * scales
    bound = approx + scales * (0.51 * np.abs(q).sum())
    order = np.argsort(-approx, kind="stable")
    # ceiling[i]: the best exact score any candidate of order[i:] can have.
    ceiling = np.append(np.maximum.accumulate(bound[order][::-1])[::-1], -np.inf)
    exact: dict[int, float] = {}
    read, window = 0, max(4 * limit, 64)
    while True:
        chosen = ids[order[read : read + window]]
        read, window = read + len(chosen), 2 * window
        for item, key in zip(chosen, read_keys(chosen), strict=True):
            exact[int(item)] = float(key.astype(np.float64) @ q)
        ranked = sorted(exact, key=lambda item: (-exact[item], item))
        if read >= len(order) or (
            len(ranked) >= limit and exact[ranked[limit - 1]] > ceiling[read]
        ):
            return [(item, exact[item]) for item in ranked[:limit]]
