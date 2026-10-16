"""Triangle mesh snapshot used by the pipeline, plus mesh/UV analysis.

The Blender bridge fills a :class:`MeshData` on the main thread (evaluated
mesh, world space).  All later processing works on this immutable snapshot,
which is what makes it safe to run in a worker thread.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import raster


@dataclass
class MeshData:
    name: str
    positions: np.ndarray        # (V, 3) float64 world-space vertex positions
    tris: np.ndarray             # (T, 3) int32 vertex indices
    tri_uvs: np.ndarray          # (T, 3, 2) float32 UVs of the target UV map
    tri_normals: np.ndarray      # (T, 3) float32 world-space face normals (unit)
    corner_normals: np.ndarray   # (T, 3, 3) float32 world-space split normals
    tri_material: np.ndarray     # (T,) int32 material slot index
    tri_selected: np.ndarray     # (T,) bool polygon selection state
    basis: np.ndarray = field(default_factory=lambda: np.eye(3))  # object rotation (3x3)
    uv_name: str = ""
    extra_uvs: dict = field(default_factory=dict)   # name -> (T,3,2)
    normals_flipped: bool = False                   # snapshot was re-oriented (scene untouched)

    @property
    def tri_count(self) -> int:
        return int(len(self.tris))

    def tri_positions(self) -> np.ndarray:
        return self.positions[self.tris]

    def bounds(self):
        return self.positions.min(axis=0), self.positions.max(axis=0)

    def diameter(self) -> float:
        lo, hi = self.bounds()
        return float(np.linalg.norm(hi - lo))

    def texel_points(self, tri: np.ndarray, bary: np.ndarray) -> np.ndarray:
        corners = self.positions[self.tris[tri]]  # (N,3,3)
        return np.einsum("nk,nkj->nj", bary.astype(np.float64), corners)

    def uv_points(self, tri: np.ndarray, bary: np.ndarray, uvs: np.ndarray | None = None) -> np.ndarray:
        src = self.tri_uvs if uvs is None else uvs
        return np.einsum("nk,nkj->nj", bary.astype(np.float32), src[tri])


def compute_face_normals(positions: np.ndarray, tris: np.ndarray) -> np.ndarray:
    p = positions[tris]
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return (n / np.maximum(ln, 1e-20)).astype(np.float32)


def triangle_areas(points: np.ndarray) -> np.ndarray:
    """points (T,3,D) for D in {2,3}."""
    a = points[:, 1] - points[:, 0]
    b = points[:, 2] - points[:, 0]
    if points.shape[2] == 2:
        return 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    return 0.5 * np.linalg.norm(np.cross(a, b), axis=1)


def signed_volume(mesh: MeshData) -> float:
    p = mesh.tri_positions()
    c = p.reshape(-1, 3).mean(axis=0)
    p0, p1, p2 = p[:, 0] - c, p[:, 1] - c, p[:, 2] - c
    return float(np.einsum("ij,ij->i", p0, np.cross(p1, p2)).sum() / 6.0)


def normals_inverted(mesh: MeshData) -> bool:
    """True when face normals predominantly point inwards (closed-ish meshes)."""
    if mesh.tri_count == 0:
        return False
    lo, hi = mesh.bounds()
    bbox_volume = float(np.prod(np.maximum(hi - lo, 1e-9)))
    return signed_volume(mesh) < -0.05 * bbox_volume


def orient_normals(mesh: MeshData) -> bool:
    """Flip all normals of a globally inverted mesh (snapshot only, scene untouched)."""
    if normals_inverted(mesh):
        mesh.tri_normals = -mesh.tri_normals
        mesh.corner_normals = -mesh.corner_normals
        mesh.normals_flipped = True
        return True
    return False


def uv_islands(mesh: MeshData) -> np.ndarray:
    """Island id per triangle (triangles connected through shared UV'd vertices)."""
    T = mesh.tri_count
    if T == 0:
        return np.zeros(0, np.int32)
    q = np.round(mesh.tri_uvs.reshape(-1, 2).astype(np.float64) * 1e5).astype(np.int64)
    keys = np.stack([mesh.tris.reshape(-1).astype(np.int64), q[:, 0], q[:, 1]], axis=1)
    _, corner_id = np.unique(keys, axis=0, return_inverse=True)
    corner_id = corner_id.reshape(T, 3)
    parent = np.arange(int(corner_id.max()) + 1)

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for a, b, c in corner_id.tolist():
        ra, rb, rc = find(a), find(b), find(c)
        parent[rb] = ra
        parent[find(rc)] = ra
    roots = np.array([find(x) for x in corner_id[:, 0].tolist()])
    _, island = np.unique(roots, return_inverse=True)
    return island.astype(np.int32)


@dataclass
class AnalysisReport:
    tri_count: int = 0
    vert_count: int = 0
    dimensions: tuple = (0.0, 0.0, 0.0)
    surface_area: float = 0.0
    uv_name: str = ""
    uv_islands: int = 0
    uv_coverage: float = 0.0
    uv_overlap: float = 0.0
    uv_out_of_range: float = 0.0
    zero_area_uv_tris: float = 0.0
    view_coverage: float | None = None
    messages: list = field(default_factory=list)   # (level, text) level in INFO/WARNING/ERROR

    def add(self, level: str, text: str):
        self.messages.append((level, text))

    @property
    def suitable(self) -> bool:
        return not any(level == "ERROR" for level, _ in self.messages)

    def to_dict(self) -> dict:
        return {
            "tri_count": self.tri_count, "vert_count": self.vert_count,
            "dimensions": list(self.dimensions), "surface_area": self.surface_area,
            "uv_name": self.uv_name, "uv_islands": self.uv_islands, "uv_coverage": self.uv_coverage,
            "uv_overlap": self.uv_overlap, "uv_out_of_range": self.uv_out_of_range,
            "zero_area_uv_tris": self.zero_area_uv_tris, "view_coverage": self.view_coverage,
            "messages": [list(m) for m in self.messages],
        }


def analyze(mesh: MeshData, *, check_islands: bool = True) -> AnalysisReport:
    rep = AnalysisReport()
    rep.tri_count = mesh.tri_count
    rep.vert_count = int(len(mesh.positions))
    rep.uv_name = mesh.uv_name
    if rep.tri_count == 0:
        rep.add("ERROR", "Mesh has no faces to texture.")
        return rep
    lo, hi = mesh.bounds()
    rep.dimensions = tuple(float(x) for x in (hi - lo))
    tri_p = mesh.tri_positions()
    areas3 = triangle_areas(tri_p)
    rep.surface_area = float(areas3.sum())
    if rep.surface_area <= 1e-12:
        rep.add("ERROR", "Mesh surface area is zero.")
        return rep
    if mesh.normals_flipped or normals_inverted(mesh):
        rep.add("WARNING", "Normals appear to point inwards (flipped). Texturing compensates, but fix them "
                           "(Mesh > Normals > Recalculate Outside) before exporting to a game engine.")
    if rep.tri_count > 1_500_000:
        rep.add("WARNING", f"{rep.tri_count:,} triangles: projection will be slow and memory-heavy. "
                           "Consider a decimated copy for texturing.")
    if not mesh.uv_name:
        rep.add("ERROR", "No UV map. Use 'Create UV Map' (a new map is added; existing data is kept).")
        return rep

    uvs = mesh.tri_uvs.astype(np.float64)
    out = (uvs < -1e-4) | (uvs > 1 + 1e-4)
    rep.uv_out_of_range = float(out.any(axis=(1, 2)).mean())
    a_uv = triangle_areas(uvs)
    rep.zero_area_uv_tris = float((a_uv < 1e-12).mean())
    rep.uv_overlap = raster.uv_overlap_fraction(mesh.tri_uvs, 256)
    cov = raster.rasterize_uv(mesh.tri_uvs, 256)
    rep.uv_coverage = cov.coverage
    if check_islands and rep.tri_count <= 400_000:
        rep.uv_islands = int(uv_islands(mesh).max()) + 1

    if rep.uv_out_of_range > 0.01:
        rep.add("WARNING", f"{rep.uv_out_of_range:.0%} of triangles have UVs outside 0-1 (UDIM/tiling). "
                           "Those areas cannot receive the generated atlas.")
    if rep.uv_overlap > 0.05:
        rep.add("WARNING", f"About {rep.uv_overlap:.0%} of the UV area overlaps (mirrored/stacked islands). "
                           "Overlapping surfaces will share one texture; consider 'Create UV Map'.")
    if rep.zero_area_uv_tris > 0.05:
        rep.add("WARNING", f"{rep.zero_area_uv_tris:.0%} of triangles have zero UV area and will stay untextured.")
    if rep.uv_coverage < 0.15:
        rep.add("WARNING", f"UV islands use only {rep.uv_coverage:.0%} of the texture; texel density will be low.")
    if rep.uv_coverage > 0 and rep.uv_overlap <= 0.05 and rep.uv_out_of_range <= 0.01:
        rep.add("INFO", "UV layout looks usable for projection texturing.")
    return rep


def estimate_view_coverage(mesh: MeshData, cameras, *, tex_size: int = 128, view_size: int = 192,
                           min_facing: float = 0.05) -> float:
    """Fraction of UV-covered surface visible from at least one camera."""
    from . import render  # local import to avoid a cycle

    uvs = raster.rasterize_uv(mesh.tri_uvs, tex_size)
    if uvs.index.size == 0:
        return 0.0
    P = mesh.texel_points(uvs.tri, uvs.bary)
    N = mesh.tri_normals[uvs.tri].astype(np.float64)
    seen = np.zeros(uvs.index.size, bool)
    diam = mesh.diameter()
    for cam in cameras:
        small = render.resized_camera(cam, view_size)
        geo = render.render_geometry(mesh, small)
        vis, _, _ = render.texel_visibility(small, geo, P, N, depth_tol=0.01 * diam)
        facing = np.einsum("ij,ij->i", N, small.dir_to_camera(P))
        seen |= vis & (facing >= min_facing)
    return float(seen.mean())
