"""Back-projection of generated views into a UV texture.

Algorithm (per view, in order):

1. Render the mesh from the view with the texture painted so far.
   Untextured surface shows as shaded clay.
2. The AI mask = visible untextured pixels, grown by ``band_px`` so the model
   repaints a thin strip of existing texture -> seamless continuation.
3. Optional alignment (phase correlation) and colour matching of the
   generated image against the already-textured context ring.
4. For every texel of the UV map: compute its 3D point, test visibility
   (depth buffer + facing), and sample the generated image.
   * untextured texels seen at a good angle (facing >= min_facing, away from
     silhouettes) are *claimed* (alpha 1);
   * already-textured texels inside the repainted band get a ramped alpha
     (cross-fade across the seam);
   * untextured texels seen only at grazing angles are kept as low-priority
     fallback colours, used after all views if no better view claimed them.
5. Each view becomes its own RGBA layer in UV space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import imageio, masks, raster
from .camera import ViewCamera
from .meshdata import MeshData
from .render import ViewRender, edge_weight, render_geometry, render_view, texel_visibility


@dataclass
class ProjectionSettings:
    texture_size: int = 1024
    view_size: int = 1024
    min_facing: float = 0.3
    band_px: int = 12
    edge_falloff_px: int = 6
    depth_tol: float = 0.01        # relative to object diameter
    skip_fraction: float = 0.004   # skip a view if less untextured area visible
    auto_align: bool = True
    max_align_shift: float = 0.04  # fraction of the view size
    color_match: bool = True
    fill_unseen: bool = True
    gutter_px: int = 4             # padding used for sampling the texture in renders


class TextureState:
    """In-memory texture being built (UV space, row 0 = top)."""

    def __init__(self, mesh: MeshData, size: int, rgb: np.ndarray | None = None, alpha: np.ndarray | None = None,
                 protected: np.ndarray | None = None, grazing: np.ndarray | None = None):
        """``rgb``/``alpha``: confident paint.  ``protected``: texels never written.
        ``grazing``: RGBA of earlier low-angle AI pixels - shown as context and
        kept unless a view paints those texels from a better angle."""
        self.size = size
        self.uv = raster.rasterize_uv(mesh.tri_uvs, size)
        self.rgb = np.zeros((size, size, 3), np.float32) if rgb is None else imageio.to_float(rgb)[..., :3].copy()
        self.alpha = np.zeros((size, size), np.float32) if alpha is None else imageio.to_float(alpha).copy()
        if self.rgb.shape[:2] != (size, size):
            self.rgb = imageio.resize(self.rgb, size, size)
        if self.alpha.shape != (size, size):
            self.alpha = imageio.resize(self.alpha, size, size)
        n = self.uv.index.size
        self.fallback_rgb = np.zeros((n, 3), np.float32)
        self.fallback_w = np.zeros(n, np.float32)
        # Texels of protected layers (imported base, accepted corrections, paint):
        # never written by projection, not even by seam blending.
        prot = np.zeros((size, size), bool) if protected is None else np.asarray(protected, bool)
        if prot.shape != (size, size):
            prot = imageio.resize_nearest(prot, size, size)
        self.protected_flat = prot.reshape(-1)[self.uv.index]
        self.grazing_alpha = np.zeros((size, size), np.float32)
        self.grazing_rgb = np.zeros((size, size, 3), np.float32)
        if grazing is not None:
            g = imageio.to_float(grazing)
            if g.shape[:2] != (size, size):
                g = imageio.resize(g, size, size)
            self.grazing_rgb = g[..., :3].copy()
            self.grazing_alpha = g[..., 3].copy()
            ga = self.grazing_alpha.reshape(-1)[self.uv.index]
            has = ga > 0.5
            self.fallback_rgb[has] = self.grazing_rgb.reshape(-1, 3)[self.uv.index][has]
            self.fallback_w[has] = 0.05            # any real view sample beats an old low-angle one
        self._P = mesh.texel_points(self.uv.tri, self.uv.bary)
        self._N = mesh.tri_normals[self.uv.tri].astype(np.float64)
        self._display = None

    @property
    def points(self) -> np.ndarray:
        return self._P

    @property
    def normals(self) -> np.ndarray:
        return self._N

    def alpha_flat(self) -> np.ndarray:
        return self.alpha.reshape(-1)[self.uv.index]

    def display(self, gutter_px: int):
        """Gutter-padded copies used when sampling the texture for renders."""
        if self._display is None:
            ga = self.grazing_alpha * (1 - self.alpha)        # low-angle pixels sit below confident paint
            alpha = self.alpha + ga
            rgb = self.rgb * self.alpha[..., None] + self.grazing_rgb * ga[..., None]
            rgb = np.where(alpha[..., None] > 1e-6, rgb / np.maximum(alpha, 1e-6)[..., None], self.rgb)
            valid = alpha > 0.5
            col, _ = masks.fill_gutter(rgb, valid, gutter_px)
            a, _ = masks.fill_gutter(alpha, valid, gutter_px)
            self._display = (col, a)
        return self._display

    def apply_layer(self, layer_rgba: np.ndarray):
        """Alpha-over a float RGBA layer onto the state."""
        a = layer_rgba[..., 3]
        self.rgb = layer_rgba[..., :3] * a[..., None] + self.rgb * (1 - a[..., None])
        self.alpha = a + self.alpha * (1 - a)
        self._display = None

    def textured_fraction(self) -> float:
        a = self.alpha_flat()
        return float((a > 0.5).mean()) if a.size else 0.0


@dataclass
class ViewPlan:
    render: ViewRender
    ai_mask: np.ndarray        # (H,W) bool region the AI should (re)paint
    new_mask: np.ndarray       # (H,W) bool untextured visible pixels
    first: bool                # nothing visible is textured yet
    untextured_fraction: float
    skip: bool


def plan_view(mesh: MeshData, cam: ViewCamera, state: TextureState, s: ProjectionSettings) -> ViewPlan:
    col, a = state.display(s.gutter_px)
    geo = render_geometry(mesh, cam)
    vr = render_view(mesh, cam, col, a, geo)
    obj = vr.mask
    n_obj = int(obj.sum())
    new = obj & (vr.textured < 0.5)
    frac = float(new.sum()) / max(n_obj, 1)
    first = bool(new.sum() == n_obj)
    if first:
        ai = obj.copy()
    else:
        ai = masks.dilate(new, s.band_px) & obj
    skip = n_obj == 0 or frac < s.skip_fraction
    return ViewPlan(vr, ai, new, first, frac, skip)


# ---------------------------------------------------------------------------
# Post-processing of generated views
# ---------------------------------------------------------------------------

def _grad_mag(img: np.ndarray) -> np.ndarray:
    g = imageio.luminance(img) if img.ndim == 3 else img
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1] = g[2:] - g[:-2]
    return np.sqrt(gx * gx + gy * gy)


def estimate_shift(reference: np.ndarray, moving: np.ndarray, max_shift: int, work: int = 256):
    """Phase-correlate edge maps; returns (dy, dx, confidence) in full-res px.

    Positive (dy, dx) means ``moving`` must be shifted by that amount to line
    up with ``reference``.
    """
    H, W = reference.shape[:2]
    scale = work / float(max(H, W))
    ref = imageio.resize(imageio.to_float(reference), max(8, int(W * scale)), max(8, int(H * scale)))
    mov = imageio.resize(imageio.to_float(moving), ref.shape[1], ref.shape[0])
    a = _grad_mag(ref)
    b = _grad_mag(mov)
    a = a - a.mean()
    b = b - b.mean()
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1])).astype(np.float32)
    Fa = np.fft.fft2(a * win)
    Fb = np.fft.fft2(b * win)
    R = Fa * np.conj(Fb)
    R /= np.maximum(np.abs(R), 1e-9)
    r = np.real(np.fft.ifft2(R))
    peak = np.unravel_index(int(np.argmax(r)), r.shape)
    py, px = peak
    if py > r.shape[0] // 2:
        py -= r.shape[0]
    if px > r.shape[1] // 2:
        px -= r.shape[1]
    conf = float(r.max() / (np.abs(r).mean() + 1e-9))
    dy = int(round(py / scale))
    dx = int(round(px / scale))
    if abs(dy) > max_shift or abs(dx) > max_shift:
        return 0, 0, conf
    return dy, dx, conf


def match_colors(gen: np.ndarray, render: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Per-channel mean/std transfer so ``gen`` matches ``render`` in ``region``."""
    if region.sum() < 400:
        return gen
    g = gen[region]
    r = render[region]
    mg, sg = g.mean(axis=0), g.std(axis=0) + 1e-4
    mr, sr = r.mean(axis=0), r.std(axis=0) + 1e-4
    if np.abs(mg - mr).max() < 2.0 / 255.0 and np.abs(sg - sr).max() < 2.0 / 255.0:
        return gen
    gain = np.clip(sr / sg, 0.7, 1.4)
    out = (gen - mg) * gain + mr
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def prepare_generated(plan: ViewPlan, generated: np.ndarray, s: ProjectionSettings, may_misalign: bool):
    """Resize, align and colour-match a provider image. Returns (img, info)."""
    vr = plan.render
    H, W = vr.mask.shape
    gen = imageio.to_float(imageio.rgb(generated))
    info = {"shift": (0, 0), "align_confidence": None, "color_matched": False, "cropped_to_aspect": None}
    gh, gw = gen.shape[:2]
    if abs(gw / gh - W / H) > 0.02:
        # The model changed the aspect ratio: stretching would misalign every pixel.
        # Models keep the subject centred, so centre-crop to the requested aspect.
        target = W / H
        if gw / gh > target:
            nw = int(round(gh * target))
            x0 = (gw - nw) // 2
            gen = gen[:, x0:x0 + nw]
        else:
            nh = int(round(gw / target))
            y0 = (gh - nh) // 2
            gen = gen[y0:y0 + nh]
        info["cropped_to_aspect"] = [int(gw), int(gh)]
    if gen.shape[:2] != (H, W):
        gen = imageio.resize(gen, W, H)
    if s.auto_align and may_misalign:
        dy, dx, conf = estimate_shift(vr.composite, gen, int(s.max_align_shift * max(H, W)))
        info["align_confidence"] = conf
        if conf > 8.0 and (dy or dx):
            gen = imageio.shift_image(gen, dy, dx)
            info["shift"] = (dy, dx)
    if s.color_match and not plan.first:
        ring = masks.dilate(plan.ai_mask, 32) & ~plan.ai_mask & vr.mask & (vr.textured > 0.9)
        before = gen
        gen = match_colors(gen, vr.composite, ring)
        info["color_matched"] = gen is not before
    return gen, info


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def project_view(mesh: MeshData, plan: ViewPlan, gen: np.ndarray, state: TextureState,
                 s: ProjectionSettings, *, chunk: int = 2_000_000):
    """Project ``gen`` into UV space. Returns (layer RGBA float (S,S,4), stats)."""
    vr = plan.render
    cam = vr.camera
    size = state.size
    diam = mesh.diameter()
    ew = edge_weight(vr.geo, diam, s.edge_falloff_px)
    band_dist = masks.distance_from(plan.new_mask, s.band_px + 1)
    layer = np.zeros((size * size, 4), np.float32)
    alpha_flat = state.alpha_flat()
    n = state.uv.index.size
    claimed_total = 0
    band_total = 0
    for start in range(0, n, chunk):
        sl = slice(start, min(n, start + chunk))
        P = state.points[sl]
        N = state.normals[sl]
        vis, x, y = texel_visibility(cam, vr.geo, P, N, depth_tol=s.depth_tol * diam)
        if not vis.any():
            continue
        facing = np.einsum("ij,ij->i", N, cam.dir_to_camera(P)).astype(np.float32)
        e = imageio.sample_nearest(ew, x, y)
        in_ai = imageio.sample_nearest(plan.ai_mask, x, y)
        bd = imageio.sample_nearest(band_dist, x, y)
        color = imageio.sample_bilinear(gen, x, y)
        untex = (alpha_flat[sl] < 0.5) & ~state.protected_flat[sl]
        good = vis & in_ai & (facing >= s.min_facing) & (e > 0.25) & ~state.protected_flat[sl]
        claim = good & untex
        band = good & ~untex & (bd > 0)
        ramp = np.clip(1.0 - bd / float(max(s.band_px, 1)), 0.0, 1.0) * 0.85
        idx = state.uv.index[sl]
        layer[idx[claim], :3] = color[claim]
        layer[idx[claim], 3] = 1.0
        b_idx = np.flatnonzero(band & (ramp > 0.01))
        layer[idx[b_idx], :3] = color[b_idx]
        layer[idx[b_idx], 3] = ramp[b_idx] * np.clip(e[b_idx], 0, 1)
        claimed_total += int(claim.sum())
        band_total += int(b_idx.size)
        # Grazing-angle fallback candidates.
        fb = vis & in_ai & untex & ~claim
        w = np.clip(facing, 0, 1) * (0.25 + 0.75 * e)
        better = fb & (w > state.fallback_w[sl])
        if better.any():
            fw = state.fallback_w[sl]
            frgb = state.fallback_rgb[sl]
            fw[better] = w[better]
            frgb[better] = color[better]
    layer = layer.reshape(size, size, 4)
    stats = {"claimed_texels": claimed_total, "band_texels": band_total,
             "claimed_fraction": claimed_total / max(n, 1)}
    return layer, stats


def fill_by_3d_proximity(points: np.ndarray, colors: np.ndarray, valid: np.ndarray, *, levels: int = 9) -> np.ndarray:
    """Colour for every invalid point from the nearest valid surface in 3D.

    A voxel pyramid (finest 2^(levels-1) cells per axis) averages valid
    colours; each invalid point takes the finest occupied voxel containing it.
    Unlike a UV-space fill, this never pulls colour from an unrelated UV
    island that merely happens to be adjacent in the atlas.
    """
    out = colors.astype(np.float32).copy()
    todo = ~valid
    if not todo.any() or not valid.any():
        if not valid.any():
            out[:] = 0.5
        return out
    lo = points.min(axis=0)
    span = max(float((points.max(axis=0) - lo).max()), 1e-9)
    norm = (points - lo) / span                       # 0..1
    vp, vc = norm[valid], colors[valid].astype(np.float64)
    remaining = np.flatnonzero(todo)
    for lvl in range(levels - 1, -1, -1):
        if remaining.size == 0:
            break
        res = 1 << lvl
        def key(p):
            q = np.minimum((p * res).astype(np.int64), res - 1)
            return (q[:, 0] * res + q[:, 1]) * res + q[:, 2]
        k_valid = key(vp)
        uniq, inv = np.unique(k_valid, return_inverse=True)
        sums = np.zeros((uniq.size, 3))
        np.add.at(sums, inv, vc)
        counts = np.bincount(inv, minlength=uniq.size).astype(np.float64)
        means = sums / counts[:, None]
        kq = key(norm[remaining])
        pos = np.searchsorted(uniq, kq)
        pos = np.minimum(pos, uniq.size - 1)
        hit = uniq[pos] == kq
        out[remaining[hit]] = means[pos[hit]]
        remaining = remaining[~hit]
    if remaining.size:
        out[remaining] = vc.mean(axis=0)
    return out


def finalize_fill(state: TextureState, s: ProjectionSettings):
    """Layers for texels no view claimed.

    Returns (grazing_layer, fill_layer, stats); each layer is RGBA float or None.
    * grazing_layer: AI pixels seen only at flat angles (low confidence);
    * fill_layer: colour extension (NOT AI-generated) for texels no camera
      saw at all, using the nearest painted surface in 3D.
    """
    size = state.size
    a = state.alpha_flat()
    untex = (a < 0.5) & ~state.protected_flat
    fb = untex & (state.fallback_w > 0)
    unseen = untex & ~fb
    n = max(state.uv.index.size, 1)
    stats = {"fallback_fraction": float(fb.sum()) / n, "unseen_fraction": float(unseen.sum()) / n}
    idx = state.uv.index
    grazing = None
    if fb.any():
        g = np.zeros((size * size, 4), np.float32)
        g[idx[fb], :3] = state.fallback_rgb[fb]
        g[idx[fb], 3] = 1.0
        grazing = g.reshape(size, size, 4)
    fill = None
    if s.fill_unseen and unseen.any():
        cur = state.rgb.reshape(-1, 3)[idx].copy()
        cur[fb] = state.fallback_rgb[fb]
        valid = ~unseen & ((a >= 0.5) | fb | state.protected_flat)
        filled = fill_by_3d_proximity(state.points, cur, valid)
        f = np.zeros((size * size, 4), np.float32)
        f[idx[unseen], :3] = filled[unseen]
        f[idx[unseen], 3] = 1.0
        fill = f.reshape(size, size, 4)
    return grazing, fill, stats


def project_image_from_camera(mesh: MeshData, cam: ViewCamera, image: np.ndarray, uv_mask: np.ndarray,
                              size: int, *, min_facing: float = 0.05, depth_tol: float = 0.01):
    """Project ``image`` into texels selected by ``uv_mask`` (used by refinement).

    Returns (rgb (S,S,3) float, weight (S,S) float, stats).
    """
    uvs = raster.rasterize_uv(mesh.tri_uvs, size)
    sel = uv_mask.reshape(-1)[uvs.index] > 0.0
    rgb = np.zeros((size * size, 3), np.float32)
    wgt = np.zeros(size * size, np.float32)
    total = int(sel.sum())
    if total == 0:
        return rgb.reshape(size, size, 3), wgt.reshape(size, size), {"masked_texels": 0, "visible_fraction": 0.0}
    tri = uvs.tri[sel]
    bary = uvs.bary[sel]
    idx = uvs.index[sel]
    P = mesh.texel_points(tri, bary)
    N = mesh.tri_normals[tri].astype(np.float64)
    geo = render_geometry(mesh, cam)
    vis, x, y = texel_visibility(cam, geo, P, N, depth_tol=depth_tol * mesh.diameter())
    facing = np.einsum("ij,ij->i", N, cam.dir_to_camera(P))
    ok = vis & (facing >= min_facing)
    gen = imageio.to_float(imageio.rgb(image))
    if gen.shape[:2] != (cam.height, cam.width):
        gen = imageio.resize(gen, cam.width, cam.height)
    ew = edge_weight(geo, mesh.diameter(), 3)
    e = imageio.sample_nearest(ew, x, y)
    rgb[idx[ok]] = imageio.sample_bilinear(gen, x[ok], y[ok])
    wgt[idx[ok]] = np.clip(0.4 + 0.6 * e[ok], 0, 1) * np.clip(facing[ok] / 0.3, 0, 1)
    return (rgb.reshape(size, size, 3), wgt.reshape(size, size),
            {"masked_texels": total, "visible_fraction": float(ok.sum()) / total})
