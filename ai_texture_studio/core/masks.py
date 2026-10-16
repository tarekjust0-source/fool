"""Mask morphology, feathering and hole filling (numpy only)."""

from __future__ import annotations

import numpy as np


def _shift(a: np.ndarray, dy: int, dx: int, fill) -> np.ndarray:
    out = np.empty_like(a)
    out[...] = fill
    h, w = a.shape[:2]
    ys_src = slice(max(0, -dy), h - max(0, dy))
    ys_dst = slice(max(0, dy), h - max(0, -dy))
    xs_src = slice(max(0, -dx), w - max(0, dx))
    xs_dst = slice(max(0, dx), w - max(0, -dx))
    out[ys_dst, xs_dst] = a[ys_src, xs_src]
    return out


_N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    m = mask.astype(bool)
    for _ in range(max(0, int(iterations))):
        grown = m.copy()
        grown[1:] |= m[:-1]
        grown[:-1] |= m[1:]
        grown[:, 1:] |= m[:, :-1]
        grown[:, :-1] |= m[:, 1:]
        m = grown
    return m


def erode(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    return ~dilate(~mask.astype(bool), iterations)


def distance_from(mask: np.ndarray, max_steps: int) -> np.ndarray:
    """Approximate (city-block) distance in pixels from True pixels.

    Pixels inside ``mask`` get 0; pixels further than ``max_steps`` get
    ``max_steps``.
    """
    m = mask.astype(bool)
    dist = np.full(m.shape, float(max_steps), dtype=np.float32)
    dist[m] = 0.0
    cur = m
    for step in range(1, int(max_steps)):
        nxt = dilate(cur, 1)
        ring = nxt & ~cur
        if not ring.any():
            break
        dist[ring] = float(step)
        cur = nxt
    return dist


def box_blur(a: np.ndarray, radius: int) -> np.ndarray:
    """Separable box blur with edge clamping using prefix sums."""
    r = int(radius)
    if r <= 0:
        return a.astype(np.float32)
    f = a.astype(np.float32)
    for axis in (0, 1):
        n = f.shape[axis]
        pad = [(0, 0)] * f.ndim
        pad[axis] = (r + 1, r)
        p = np.pad(f, pad, mode="edge")
        cs = np.cumsum(p, axis=axis, dtype=np.float64)
        hi = np.take(cs, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
        lo = np.take(cs, np.arange(0, n), axis=axis)
        f = ((hi - lo) / (2 * r + 1)).astype(np.float32)
    return f


def feather(mask: np.ndarray, radius: int) -> np.ndarray:
    """Soft mask in [0,1]. Two box passes approximate a Gaussian."""
    f = mask.astype(np.float32)
    if radius <= 0:
        return f
    r = max(1, int(radius) // 2)
    return np.clip(box_blur(box_blur(f, r), r), 0.0, 1.0)


def _dilate8(m: np.ndarray) -> np.ndarray:
    g = m.copy()
    g[1:] |= m[:-1]
    g[:-1] |= m[1:]
    g[:, 1:] |= m[:, :-1]
    g[:, :-1] |= m[:, 1:]
    g[1:, 1:] |= m[:-1, :-1]
    g[1:, :-1] |= m[:-1, 1:]
    g[:-1, 1:] |= m[1:, :-1]
    g[:-1, :-1] |= m[1:, 1:]
    return g


def fill_gutter(color: np.ndarray, valid: np.ndarray, iterations: int) -> tuple[np.ndarray, np.ndarray]:
    """Grow colours outward from ``valid`` pixels by averaging neighbours.

    Standard texture "edge padding" so mip-maps and bilinear filtering do not
    pull in background colour at UV island borders.  Each step only touches
    the one-pixel frontier around the valid region, so cost scales with the
    island perimeter rather than the texture area.  Returns (color, valid).
    """
    col = color.astype(np.float32).copy()
    squeeze = col.ndim == 2
    if squeeze:
        col = col[:, :, None]
    v = valid.astype(bool).copy()
    H, W = v.shape
    for _ in range(max(0, int(iterations))):
        front = _dilate8(v) & ~v
        idx = np.flatnonzero(front)
        if idx.size == 0:
            break
        ys, xs = np.divmod(idx, W)
        acc = np.zeros((idx.size, col.shape[2]), np.float32)
        cnt = np.zeros(idx.size, np.float32)
        for dy, dx in _N8:
            yy = ys + dy
            xx = xs + dx
            inside = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
            yy = np.clip(yy, 0, H - 1)
            xx = np.clip(xx, 0, W - 1)
            nv = v[yy, xx] & inside
            acc += col[yy, xx] * nv[:, None]
            cnt += nv
        col[ys, xs] = acc / np.maximum(cnt, 1.0)[:, None]
        v[ys, xs] = True
    if squeeze:
        col = col[:, :, 0]
    return col, v


def push_pull_fill(color: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill every invalid pixel with a smooth extension of valid pixels.

    Classic push-pull: build a weighted pyramid down to 1x1, then pull back up
    filling holes.  Used for surface regions no camera could see.
    """
    col = color.astype(np.float32)
    squeeze = col.ndim == 2
    if squeeze:
        col = col[:, :, None]
    w = valid.astype(np.float32)
    if w.sum() == 0:
        out = np.full_like(col, 0.5)
        return out[:, :, 0] if squeeze else out
    levels = [(col * w[:, :, None], w)]
    while max(levels[-1][1].shape) > 1:
        c, ww = levels[-1]
        h, wd = ww.shape
        h2, w2 = (h + 1) // 2, (wd + 1) // 2
        cp = np.zeros((h2 * 2, w2 * 2, c.shape[2]), np.float32)
        wp = np.zeros((h2 * 2, w2 * 2), np.float32)
        cp[:h, :wd] = c
        wp[:h, :wd] = ww
        cd = cp.reshape(h2, 2, w2, 2, -1).sum(axis=(1, 3))
        wdn = wp.reshape(h2, 2, w2, 2).sum(axis=(1, 3))
        levels.append((cd, wdn))
    # Pull: walk back up, filling where weight is zero.
    c_top, w_top = levels[-1]
    filled = c_top / np.maximum(w_top, 1e-8)[:, :, None]
    for c, ww in reversed(levels[:-1]):
        h, wd = ww.shape
        up = np.repeat(np.repeat(filled, 2, axis=0), 2, axis=1)[:h, :wd]
        here = c / np.maximum(ww, 1e-8)[:, :, None]
        alpha = np.clip(ww, 0, 1)[:, :, None]
        filled = here * alpha + up * (1 - alpha)
    out = np.where(valid.astype(bool)[:, :, None], col, filled)
    return out[:, :, 0] if squeeze else out


def bbox(mask: np.ndarray):
    """Return (y0, y1, x0, x1) exclusive bounds of True pixels or None."""
    ys = np.flatnonzero(mask.any(axis=1))
    if ys.size == 0:
        return None
    xs = np.flatnonzero(mask.any(axis=0))
    return int(ys[0]), int(ys[-1]) + 1, int(xs[0]), int(xs[-1]) + 1
