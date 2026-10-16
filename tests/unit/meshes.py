"""Procedural test meshes (pure numpy) with non-overlapping UVs."""

from __future__ import annotations

import numpy as np

from ai_texture_studio.core.meshdata import MeshData, compute_face_normals


def _finish(name, pos, tris, tri_uvs, smooth=False):
    pos = np.asarray(pos, np.float64)
    tris = np.asarray(tris, np.int32)
    fn = compute_face_normals(pos, tris)
    if smooth:
        vn = np.zeros_like(pos)
        np.add.at(vn, tris.reshape(-1), np.repeat(fn, 3, axis=0))
        vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
        cn = vn[tris].astype(np.float32)
    else:
        cn = np.repeat(fn[:, None, :], 3, axis=1)
    T = len(tris)
    return MeshData(name=name, positions=pos, tris=tris, tri_uvs=np.asarray(tri_uvs, np.float32),
                    tri_normals=fn, corner_normals=cn, tri_material=np.zeros(T, np.int32),
                    tri_selected=np.zeros(T, bool), uv_name="UVMap")


def cube(size=2.0, center=(0, 0, 0)):
    """Unit cube; each face gets its own cell in a 3x2 UV grid."""
    h = size / 2
    c = np.asarray(center, float)
    faces = [  # (normal axis, sign) -> 4 corners CCW seen from outside
        [(-h, -h, -h), (h, -h, -h), (h, -h, h), (-h, -h, h)],    # front  (-Y)
        [(h, h, -h), (-h, h, -h), (-h, h, h), (h, h, h)],        # back   (+Y)
        [(h, -h, -h), (h, h, -h), (h, h, h), (h, -h, h)],        # right  (+X)
        [(-h, h, -h), (-h, -h, -h), (-h, -h, h), (-h, h, h)],    # left   (-X)
        [(-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h)],        # top    (+Z)
        [(-h, h, -h), (h, h, -h), (h, -h, -h), (-h, -h, -h)],    # bottom (-Z)
    ]
    pos, tris, uvs = [], [], []
    for i, f in enumerate(faces):
        cx, cy = i % 3, i // 3
        u0, v0 = cx / 3 + 0.01, cy / 2 + 0.01
        du, dv = 1 / 3 - 0.02, 1 / 2 - 0.02
        quv = [(u0, v0), (u0 + du, v0), (u0 + du, v0 + dv), (u0, v0 + dv)]
        base = len(pos)
        pos.extend([tuple(np.add(p, c)) for p in f])
        for a, b, cc in ((0, 1, 2), (0, 2, 3)):
            tris.append((base + a, base + b, base + cc))
            uvs.append((quv[a], quv[b], quv[cc]))
    return _finish("Cube", pos, tris, uvs)


def uv_sphere(radius=1.0, segments=32, rings=16, center=(0, 0, 0)):
    pos, tris, uvs = [], [], []
    for r in range(rings + 1):
        th = np.pi * r / rings
        for s in range(segments + 1):
            ph = 2 * np.pi * s / segments
            pos.append((center[0] + radius * np.sin(th) * np.cos(ph),
                        center[1] + radius * np.sin(th) * np.sin(ph),
                        center[2] + radius * np.cos(th)))
    W = segments + 1
    for r in range(rings):
        for s in range(segments):
            a, b2, c, d = r * W + s, r * W + s + 1, (r + 1) * W + s + 1, (r + 1) * W + s
            ua, ub = s / segments, (s + 1) / segments
            va, vb = 1 - r / rings, 1 - (r + 1) / rings
            uvm = {a: (ua, va), b2: (ub, va), c: (ub, vb), d: (ua, vb)}
            for t in ((a, d, c), (a, c, b2)):
                tris.append(t)
                uvs.append(tuple(uvm[k] for k in t))
    return _finish("Sphere", pos, tris, uvs, smooth=True)


def combine(name, *meshes):
    """Join meshes, packing each one's UVs into its own column of the atlas."""
    pos, tris, uvs = [], [], []
    n = len(meshes)
    off = 0
    for i, m in enumerate(meshes):
        pos.append(m.positions)
        tris.append(m.tris + off)
        u = m.tri_uvs.copy()
        u[..., 0] = (u[..., 0] + i) / n
        uvs.append(u)
        off += len(m.positions)
    out = _finish(name, np.concatenate(pos), np.concatenate(tris), np.concatenate(uvs))
    out.corner_normals = np.concatenate([m.corner_normals for m in meshes])
    return out
