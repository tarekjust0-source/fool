"""Render the real mesh from a camera (numpy) and test texel visibility.

The renders produced here are what gets sent to the AI provider ("this is
the object from the front; paint the grey clay parts"), and the exact same
geometry is used to project the answer back into UV space.  Using one
rasteriser for both directions keeps projection pixel-aligned.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import imageio, masks, raster
from .camera import ViewCamera
from .meshdata import MeshData

CLAY_COLOR = np.array([0.66, 0.65, 0.62], np.float32)
BACKGROUND = np.array([0.94, 0.94, 0.94], np.float32)


@dataclass
class Geometry:
    """Per-pixel geometric buffers for one camera."""

    tri: np.ndarray        # (H,W) int32, -1 = background
    bary: np.ndarray       # (H,W,3) perspective-correct
    depth: np.ndarray      # (H,W) view depth (inf background)

    @property
    def mask(self) -> np.ndarray:
        return self.tri >= 0


def resized_camera(cam: ViewCamera, size: int) -> ViewCamera:
    return ViewCamera(cam.name, cam.view, cam.proj, size, size, cam.ortho, cam.meta)


def render_geometry(mesh: MeshData, cam: ViewCamera) -> Geometry:
    xy, ndc_z, w, vdepth = cam.project(mesh.positions)
    tri_xy = xy[mesh.tris]
    tri_w = w[mesh.tris]
    # Cull triangles crossing / behind the near plane (simple, robust).
    front = (tri_w > 1e-6).all(axis=1)
    ids = np.flatnonzero(front).astype(np.int32)
    tz = ndc_z[mesh.tris][ids]
    inv_w = None if cam.ortho else 1.0 / tri_w[ids]
    r = raster.rasterize(tri_xy[ids], cam.width, cam.height, z=tz, inv_w=inv_w, tri_ids=ids)
    m = r.tri >= 0
    depth = np.full(m.shape, np.inf, np.float32)
    if m.any():
        t = r.tri[m]
        b = r.bary[m]
        depth[m] = np.einsum("nk,nk->n", b, vdepth[mesh.tris[t]]).astype(np.float32)
    return Geometry(r.tri, r.bary, depth)


def pixel_attributes(mesh: MeshData, geo: Geometry):
    """UVs and smooth normals for every covered pixel."""
    m = geo.mask
    t = geo.tri[m]
    b = geo.bary[m]
    uv = np.einsum("nk,nkj->nj", b, mesh.tri_uvs[t].astype(np.float32))
    n = np.einsum("nk,nkj->nj", b, mesh.corner_normals[t].astype(np.float32))
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return m, uv, n


def edge_weight(geo: Geometry, diameter: float, falloff_px: int = 6) -> np.ndarray:
    """0 at silhouettes / depth discontinuities, 1 further than ``falloff_px``."""
    m = geo.mask
    d = np.where(m, geo.depth, np.nan)
    jump = 0.02 * diameter
    disc = np.zeros(m.shape, bool)
    for axis in (0, 1):
        a = np.diff(d, axis=axis)
        big = ~(np.abs(a) <= jump)  # NaN (background boundary) counts as a jump
        if axis == 0:
            disc[1:] |= big
            disc[:-1] |= big
        else:
            disc[:, 1:] |= big
            disc[:, :-1] |= big
    disc &= m
    disc |= m & ~masks.erode(m, 1)
    dist = masks.distance_from(disc, falloff_px + 1)
    return np.clip(dist / float(max(falloff_px, 1)), 0.0, 1.0) * m


def texel_visibility(cam: ViewCamera, geo: Geometry, P: np.ndarray, N: np.ndarray, *, depth_tol: float):
    """Which world points are visible from ``cam``.

    Returns (visible (N,) bool, pixel x (N,), pixel y (N,)).
    """
    xy, _, w, vdepth = cam.project(P)
    x, y = xy[:, 0], xy[:, 1]
    inside = (w > 1e-9) & (x >= 0) & (x < cam.width) & (y >= 0) & (y < cam.height)
    xi = np.clip(np.floor(x).astype(np.int64), 0, cam.width - 1)
    yi = np.clip(np.floor(y).astype(np.int64), 0, cam.height - 1)
    zb = geo.depth[yi, xi]
    on_object = geo.tri[yi, xi] >= 0
    facing = np.einsum("ij,ij->i", N, cam.dir_to_camera(P))
    # Slope-scaled tolerance: grazing surfaces have larger depth error.
    tol = depth_tol * (1.0 + 2.0 * (1.0 - np.clip(facing, 0, 1)))
    # Back faces are culled so thin two-sided objects (shields, capes) keep
    # separate front/back paint; globally flipped meshes are re-oriented by
    # meshdata.orient_normals() before this runs.
    vis = inside & on_object & (facing > 0.0) & (vdepth <= zb + tol)
    return vis, x, y


@dataclass
class ViewRender:
    camera: ViewCamera
    geo: Geometry
    composite: np.ndarray    # (H,W,3) float: texture where textured, shaded clay elsewhere
    textured: np.ndarray     # (H,W) float 0..1 how much of the pixel is already textured
    depth_image: np.ndarray  # (H,W) float 0..1, near = 1
    normal_image: np.ndarray  # (H,W,3) float camera-space normals mapped to 0..1
    shading: np.ndarray      # (H,W) float clay shading factor

    @property
    def mask(self) -> np.ndarray:
        return self.geo.mask


def render_view(mesh: MeshData, cam: ViewCamera, tex_rgb: np.ndarray | None, tex_alpha: np.ndarray | None,
                geo: Geometry | None = None) -> ViewRender:
    """Render the mesh with the current texture (unlit) and clay elsewhere."""
    geo = geo or render_geometry(mesh, cam)
    H, W = geo.tri.shape
    m, uv, n = pixel_attributes(mesh, geo)
    comp = np.empty((H, W, 3), np.float32)
    comp[:] = BACKGROUND
    textured = np.zeros((H, W), np.float32)
    shading = np.zeros((H, W), np.float32)
    normal_img = np.zeros((H, W, 3), np.float32)
    depth_img = np.zeros((H, W), np.float32)
    if m.any():
        # Headlight from upper-left-front for readable clay shading.
        fwd = cam.forward
        right = cam.view[0, :3]
        up = cam.view[1, :3]
        light = -fwd * 0.75 - right * 0.35 + up * 0.55
        light /= np.linalg.norm(light)
        n = n * np.where((n @ -fwd) < 0, -1.0, 1.0)[:, None].astype(np.float32)  # face the camera
        lam = np.clip(n @ light, 0.0, 1.0)
        shade = (0.32 + 0.68 * lam).astype(np.float32)
        shading[m] = shade
        clay = shade[:, None] * CLAY_COLOR[None, :]
        if tex_rgb is not None and tex_alpha is not None:
            size = tex_alpha.shape[0]
            px = uv[:, 0] * size
            py = (1.0 - uv[:, 1]) * size
            a = imageio.sample_bilinear(tex_alpha, px, py)
            col = imageio.sample_bilinear(tex_rgb, px, py)
            a = np.clip(a, 0, 1)
            # Treat near-border samples generously: a texel counts as textured
            # if most of its neighbourhood is.
            textured[m] = a
            comp[m] = col * a[:, None] + clay * (1 - a[:, None])
        else:
            comp[m] = clay
        # camera-space normals (x right, y up, z towards camera)
        cn = np.stack([n @ right, n @ up, n @ (-fwd)], axis=1)
        normal_img[m] = cn * 0.5 + 0.5
        d = geo.depth[m]
        dmin, dmax = float(d.min()), float(d.max())
        depth_img[m] = 1.0 - (d - dmin) / max(dmax - dmin, 1e-9) * 0.8
    return ViewRender(cam, geo, np.clip(comp, 0, 1), textured, depth_img, normal_img, shading)
