"""Vectorised triangle rasteriser (numpy).

Used for two jobs:

* UV space: find, for every texel, which triangle covers it and where
  (barycentric coordinates) so we can recover its 3D position and normal.
* Screen space: render the mesh from a camera with a depth buffer, producing
  per-pixel triangle ids and perspective-correct barycentrics.

Triangles are processed in batches; each triangle's bounding box is split
into row strips so a single huge triangle cannot explode memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Number of candidate pixels processed per numpy batch.
DEFAULT_BATCH = 1_500_000


@dataclass
class Raster:
    tri: np.ndarray    # (H, W) int32, -1 where empty
    bary: np.ndarray   # (H, W, 3) float32 barycentrics (perspective-correct when requested)
    depth: np.ndarray | None = None  # (H, W) float32 interpolated depth (inf where empty)

    @property
    def mask(self) -> np.ndarray:
        return self.tri >= 0


def _strips(x0, x1, y0, y1, cap):
    bw = x1 - x0 + 1
    bh = y1 - y0 + 1
    rows = np.maximum(1, cap // np.maximum(bw, 1))
    nstrip = (bh + rows - 1) // rows
    idx = np.repeat(np.arange(len(bw)), nstrip)
    first = np.cumsum(nstrip) - nstrip
    k = np.arange(len(idx)) - np.repeat(first, nstrip)
    sy0 = y0[idx] + k * rows[idx]
    sy1 = np.minimum(sy0 + rows[idx] - 1, y1[idx])
    return idx, sy0, sy1


def rasterize(
    xy: np.ndarray,
    width: int,
    height: int,
    *,
    z: np.ndarray | None = None,
    inv_w: np.ndarray | None = None,
    eps: float = 1e-7,
    batch: int = DEFAULT_BATCH,
    tri_ids: np.ndarray | None = None,
) -> Raster:
    """Rasterise triangles.

    xy:     (T, 3, 2) pixel coordinates (pixel (i, j) has centre (i+0.5, j+0.5)).
    z:      optional (T, 3) depth used for z-buffering (smaller = closer).
            Must be affine in screen space (e.g. NDC z).
    inv_w:  optional (T, 3) 1/w for perspective-correct barycentrics.
    tri_ids: optional ids written to the output instead of 0..T-1.
    Without ``z`` later triangles overwrite earlier ones (UV space).
    """
    xy = np.asarray(xy, dtype=np.float64)
    T = xy.shape[0]
    tri_out = np.full((height, width), -1, dtype=np.int32)
    bary_out = np.zeros((height, width, 3), dtype=np.float32)
    zbuf = np.full((height, width), np.inf, dtype=np.float64) if z is not None else None
    if T == 0:
        return Raster(tri_out, bary_out, zbuf.astype(np.float32) if zbuf is not None else None)
    if tri_ids is None:
        tri_ids = np.arange(T, dtype=np.int32)

    ax, ay = xy[:, 0, 0], xy[:, 0, 1]
    bx, by = xy[:, 1, 0], xy[:, 1, 1]
    cx, cy = xy[:, 2, 0], xy[:, 2, 1]
    denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    finite = np.isfinite(xy).all(axis=(1, 2))
    ok = finite & (np.abs(denom) > 1e-12)

    minx = np.floor(np.min(xy[:, :, 0], axis=1) - 0.5)
    maxx = np.floor(np.max(xy[:, :, 0], axis=1) - 0.5) + 1
    miny = np.floor(np.min(xy[:, :, 1], axis=1) - 0.5)
    maxy = np.floor(np.max(xy[:, :, 1], axis=1) - 0.5) + 1
    x0 = np.clip(np.where(ok, minx, 0), 0, width - 1).astype(np.int64)
    x1 = np.clip(np.where(ok, maxx, -1), -1, width - 1).astype(np.int64)
    y0 = np.clip(np.where(ok, miny, 0), 0, height - 1).astype(np.int64)
    y1 = np.clip(np.where(ok, maxy, -1), -1, height - 1).astype(np.int64)
    ok &= (x1 >= x0) & (y1 >= y0) & (maxx >= 0) & (minx <= width) & (maxy >= 0) & (miny <= height)
    sel = np.flatnonzero(ok)
    if sel.size == 0:
        return Raster(tri_out, bary_out, zbuf.astype(np.float32) if zbuf is not None else None)

    d = denom[sel]
    A0 = (by[sel] - cy[sel]) / d
    B0 = (cx[sel] - bx[sel]) / d
    A1 = (cy[sel] - ay[sel]) / d
    B1 = (ax[sel] - cx[sel]) / d
    CX = cx[sel]
    CY = cy[sel]
    sx0, sx1, sy0_t, sy1_t = x0[sel], x1[sel], y0[sel], y1[sel]
    zz = z[sel].astype(np.float64) if z is not None else None
    iw = inv_w[sel].astype(np.float64) if inv_w is not None else None

    strip_tri, sy0, sy1 = _strips(sx0, sx1, sy0_t, sy1_t, batch)
    sbw = sx1[strip_tri] - sx0[strip_tri] + 1
    area = sbw * (sy1 - sy0 + 1)
    # Group strips into batches of bounded total area.
    cum = np.cumsum(area)
    batch_id = (cum - 1) // batch
    bounds = np.flatnonzero(np.diff(batch_id)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(area)]])

    for s, e in zip(starts, ends):
        st = strip_tri[s:e]
        cnt = area[s:e]
        total = int(cnt.sum())
        if total == 0:
            continue
        t_local = np.repeat(st, cnt)
        offs = np.repeat(np.cumsum(cnt) - cnt, cnt)
        local = np.arange(total, dtype=np.int64) - offs
        bw = np.repeat(sbw[s:e], cnt)
        px = sx0[t_local] + local % bw
        py = np.repeat(sy0[s:e], cnt) + local // bw
        fx = px + 0.5 - CX[t_local]
        fy = py + 0.5 - CY[t_local]
        l0 = A0[t_local] * fx + B0[t_local] * fy
        l1 = A1[t_local] * fx + B1[t_local] * fy
        l2 = 1.0 - l0 - l1
        inside = (l0 >= -eps) & (l1 >= -eps) & (l2 >= -eps)
        if not inside.any():
            continue
        t_local = t_local[inside]
        px = px[inside]
        py = py[inside]
        l0 = l0[inside]
        l1 = l1[inside]
        l2 = l2[inside]
        pix = py * width + px
        if zz is not None:
            zv = l0 * zz[t_local, 0] + l1 * zz[t_local, 1] + l2 * zz[t_local, 2]
            order = np.lexsort((zv, pix))
            pix_s = pix[order]
            first = np.ones(len(pix_s), dtype=bool)
            first[1:] = pix_s[1:] != pix_s[:-1]
            keep = order[first]
            pix = pix[keep]
            zv = zv[keep]
            t_local = t_local[keep]
            l0, l1, l2 = l0[keep], l1[keep], l2[keep]
            flat_z = zbuf.reshape(-1)
            closer = zv < flat_z[pix]
            pix = pix[closer]
            t_local = t_local[closer]
            l0, l1, l2 = l0[closer], l1[closer], l2[closer]
            flat_z[pix] = zv[closer]
        if iw is not None:
            w0 = l0 * iw[t_local, 0]
            w1 = l1 * iw[t_local, 1]
            w2 = l2 * iw[t_local, 2]
            ssum = w0 + w1 + w2
            ssum = np.where(np.abs(ssum) < 1e-20, 1e-20, ssum)
            l0, l1, l2 = w0 / ssum, w1 / ssum, w2 / ssum
        tri_out.reshape(-1)[pix] = tri_ids[sel[t_local]]
        bflat = bary_out.reshape(-1, 3)
        bflat[pix, 0] = l0
        bflat[pix, 1] = l1
        bflat[pix, 2] = l2

    return Raster(tri_out, bary_out, zbuf.astype(np.float32) if zbuf is not None else None)


@dataclass
class UVSamples:
    """Texels covered by the mesh in a UV map at a given resolution."""

    size: int
    index: np.ndarray   # (N,) flat texel index (row-major, row 0 = top = v=1)
    tri: np.ndarray     # (N,) int32 triangle index
    bary: np.ndarray    # (N, 3) float32
    coverage: float     # fraction of the texture area covered

    def covered_mask(self) -> np.ndarray:
        m = np.zeros(self.size * self.size, dtype=bool)
        m[self.index] = True
        return m.reshape(self.size, self.size)


def uv_to_pixels(uv: np.ndarray, size: int) -> np.ndarray:
    """UV (0..1, v up) -> pixel coords (x right, y down) at ``size``."""
    out = np.empty(uv.shape, dtype=np.float64)
    out[..., 0] = uv[..., 0] * size
    out[..., 1] = (1.0 - uv[..., 1]) * size
    return out


def rasterize_uv(tri_uvs: np.ndarray, size: int, tri_subset: np.ndarray | None = None) -> UVSamples:
    """Rasterise triangles in UV space at ``size`` x ``size`` texels."""
    if tri_subset is not None:
        ids = np.asarray(tri_subset, dtype=np.int32)
        uvs = tri_uvs[ids]
    else:
        ids = np.arange(len(tri_uvs), dtype=np.int32)
        uvs = tri_uvs
    r = rasterize(uv_to_pixels(uvs, size), size, size, tri_ids=ids, eps=1e-5)
    flat_tri = r.tri.reshape(-1)
    index = np.flatnonzero(flat_tri >= 0)
    return UVSamples(
        size=size,
        index=index,
        tri=flat_tri[index].astype(np.int32),
        bary=r.bary.reshape(-1, 3)[index],
        coverage=float(index.size) / float(size * size),
    )


def uv_overlap_fraction(tri_uvs: np.ndarray, size: int = 256) -> float:
    """Fraction of covered texels hit by more than one triangle (approx.).

    Counts how many triangles strictly contain each texel centre; shared
    edges are excluded by a strict interior test so only real overlaps count.
    """
    T = len(tri_uvs)
    if T == 0:
        return 0.0
    counts = np.zeros(size * size, dtype=np.int32)
    xy = uv_to_pixels(tri_uvs, size)
    chunk = 4096
    for s in range(0, T, chunk):
        counts += _coverage_counts(xy[s:s + chunk], size)
    covered = counts > 0
    if not covered.any():
        return 0.0
    return float((counts > 1).sum()) / float(covered.sum())


def _coverage_counts(xy: np.ndarray, size: int) -> np.ndarray:
    """Number of triangles strictly covering each texel centre."""
    out = np.zeros(size * size, dtype=np.int32)
    ax, ay = xy[:, 0, 0], xy[:, 0, 1]
    bx, by = xy[:, 1, 0], xy[:, 1, 1]
    cx, cy = xy[:, 2, 0], xy[:, 2, 1]
    denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    ok = np.abs(denom) > 1e-12
    x0 = np.clip(np.floor(xy[:, :, 0].min(1) - 0.5), 0, size - 1).astype(np.int64)
    x1 = np.clip(np.floor(xy[:, :, 0].max(1) - 0.5) + 1, -1, size - 1).astype(np.int64)
    y0 = np.clip(np.floor(xy[:, :, 1].min(1) - 0.5), 0, size - 1).astype(np.int64)
    y1 = np.clip(np.floor(xy[:, :, 1].max(1) - 0.5) + 1, -1, size - 1).astype(np.int64)
    ok &= (x1 >= x0) & (y1 >= y0)
    sel = np.flatnonzero(ok)
    if sel.size == 0:
        return out
    bw = x1[sel] - x0[sel] + 1
    bh = y1[sel] - y0[sel] + 1
    cnt = bw * bh
    t = np.repeat(sel, cnt)
    local = np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    bwr = np.repeat(bw, cnt)
    px = x0[t] + local % bwr
    py = y0[t] + local // bwr
    d = denom[t]
    fx = px + 0.5 - cx[t]
    fy = py + 0.5 - cy[t]
    l0 = ((by[t] - cy[t]) * fx + (cx[t] - bx[t]) * fy) / d
    l1 = ((cy[t] - ay[t]) * fx + (ax[t] - cx[t]) * fy) / d
    l2 = 1 - l0 - l1
    e = 1e-3
    inside = (l0 > e) & (l1 > e) & (l2 > e)
    np.add.at(out, (py * size + px)[inside], 1)
    return out
