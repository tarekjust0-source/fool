"""Quantitative projection accuracy tests with an 'oracle' provider.

The oracle paints a smooth 3D procedural colour field consistently from
every camera (what a perfect multi-view model would do).  The texture the
pipeline reconstructs in UV space is compared with the field evaluated at
each texel's true 3D position.  This isolates our projection / blending /
seam handling from model inconsistency.
"""

from __future__ import annotations

import numpy as np

from ai_texture_studio.core import camera, imageio, jobs, pipeline, projection, render
from ai_texture_studio.providers import base as pb
from meshes import combine, cube, uv_sphere


def field(P):
    """Smooth procedural colour (values 0.1..0.9)."""
    return np.clip(np.stack([
        0.5 + 0.4 * np.sin(P[:, 0] * 2.1 + P[:, 2] * 0.7),
        0.5 + 0.4 * np.sin(P[:, 1] * 1.7 - P[:, 0] * 0.5),
        0.5 + 0.4 * np.cos(P[:, 2] * 1.9 + P[:, 1] * 0.3),
    ], axis=1), 0, 1).astype(np.float32)


class Oracle(pb.Provider):
    id = "ORACLE"
    name = "Oracle"
    requires_key = False
    is_local = True

    def __init__(self, mesh, cams, shift=(0, 0), misalign=False):
        self.mesh, self.cams, self.shift, self.i, self.misalign = mesh, cams, shift, 0, misalign

    def capabilities(self, cfg):
        caps = {pb.IMAGE_EDIT, pb.MASK}
        if self.misalign:
            caps.add(pb.MAY_MISALIGN)
        return caps

    def _generate(self, req, cfg, cancel=None):
        cam = render.resized_camera(self.cams[self.i], req.width)
        self.i += 1
        geo = render.render_geometry(self.mesh, cam)
        img = np.full((req.height, req.width, 3), 0.5, np.float32)
        m = geo.mask
        t, b = geo.tri[m], geo.bary[m]
        P = np.einsum("nk,nkj->nj", b.astype(np.float64), self.mesh.positions[self.mesh.tris[t]])
        img[m] = field(P)
        img = imageio.shift_image(img, *self.shift)
        return pb.ImageResult(imageio.to_uint8(img), self.id, "oracle")


def reconstruct(mesh, keys, *, shift=(0, 0), misalign=False, auto_align=True, size=512, view=512):
    cams = [camera.preset_camera(k, camera.VIEW_PRESETS[k], mesh.positions, view) for k in keys]
    st = projection.TextureState(mesh, size)
    ps = projection.ProjectionSettings(texture_size=size, view_size=view, auto_align=auto_align, color_match=False)
    job = jobs.Job("q", "q")
    prov = Oracle(mesh, cams, shift, misalign)
    pipeline.run_multiview(job, mesh, cams, st, prov, pb.ProviderConfig(), pipeline.GenerationSpec(
        prompt="x", request_size=view), ps)
    truth = field(st.points)
    got = st.rgb.reshape(-1, 3)[st.uv.index]
    textured = st.alpha_flat() > 0.5
    err = np.abs(got - truth).mean(axis=1)
    return err, textured


ALL6 = ["FRONT", "RIGHT", "LEFT", "BACK", "TOP", "BOTTOM"]


def test_reconstruction_error_is_small_on_sphere():
    err, tex = reconstruct(uv_sphere(1.0, 48, 24), ALL6 + ["FRONT_RIGHT", "FRONT_LEFT", "BACK_RIGHT", "BACK_LEFT"])
    assert tex.mean() > 0.97
    assert err[tex].mean() < 4.0 / 255.0, err[tex].mean() * 255
    assert np.percentile(err[tex], 99) < 30.0 / 255.0


def test_reconstruction_on_occluding_scene():
    m = combine("scene", cube(2.0), uv_sphere(0.7, 32, 16, center=(0, 0, 1.7)))
    err, tex = reconstruct(m, ALL6)
    # depth test must stop the sphere's colours leaking onto the occluded cube top
    assert err[tex].mean() < 5.0 / 255.0, err[tex].mean() * 255
    assert np.percentile(err[tex], 99) < 40.0 / 255.0


def test_auto_align_recovers_shifted_outputs():
    sphere = uv_sphere(1.0, 48, 24)
    err_aligned, tex = reconstruct(sphere, ALL6, shift=(9, -7), misalign=True, auto_align=True)
    err_raw, tex2 = reconstruct(sphere, ALL6, shift=(9, -7), misalign=True, auto_align=False)
    assert err_aligned[tex].mean() < err_raw[tex2].mean() * 0.6
    assert err_aligned[tex].mean() < 5.0 / 255.0
