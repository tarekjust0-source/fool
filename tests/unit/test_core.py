"""Unit tests for the pure-Python core (no Blender required)."""

from __future__ import annotations

import struct
import zlib

import numpy as np
import pytest

from ai_texture_studio.core import (camera, compositing, imageio, jobs, masks, meshdata, pbr, pipeline, projection,
                                    prompting, raster, secrets)
from ai_texture_studio import providers
from meshes import combine, cube, uv_sphere


# ---------------------------------------------------------------------------
# image IO
# ---------------------------------------------------------------------------

def _png_with_filters(arr: np.ndarray, filters) -> bytes:
    """Encode with explicit per-row filter types (reference implementation)."""
    h, w, c = arr.shape
    bpp = c
    rows = []
    prev = np.zeros(w * c, np.int32)
    for y in range(h):
        line = arr[y].reshape(-1).astype(np.int32)
        f = filters[y % len(filters)]
        out = np.zeros_like(line)
        for i in range(len(line)):
            a = line[i - bpp] if i >= bpp else 0
            b = prev[i]
            cc = prev[i - bpp] if i >= bpp else 0
            if f == 0:
                p = 0
            elif f == 1:
                p = a
            elif f == 2:
                p = b
            elif f == 3:
                p = (a + b) // 2
            else:
                pp = a + b - cc
                pa, pb, pc = abs(pp - a), abs(pp - b), abs(pp - cc)
                p = a if (pa <= pb and pa <= pc) else (b if pb <= pc else cc)
            out[i] = (line[i] - p) & 255
        rows.append(bytes([f]) + bytes(out.astype(np.uint8)))
        prev = line
    ctype = {3: 2, 4: 6}[c]
    ihdr = struct.pack(">IIBBBBB", w, h, 8, ctype, 0, 0, 0)

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    return imageio.PNG_SIGNATURE + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(b"".join(rows))) + \
        chunk(b"IEND", b"")


@pytest.mark.parametrize("filters", [[0], [1], [2], [3], [4], [0, 1, 2, 3, 4], [4, 3]])
@pytest.mark.parametrize("channels", [3, 4])
def test_png_decoder_all_filter_types(filters, channels):
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 256, size=(23, 17, channels), dtype=np.uint8)
    out = imageio.decode_png(_png_with_filters(arr, filters))
    assert np.array_equal(out, arr)


def test_png_roundtrip_and_gray_alpha():
    rng = np.random.default_rng(2)
    for c in (1, 2, 3, 4):
        arr = rng.integers(0, 256, size=(31, 40, c), dtype=np.uint8)
        dec = imageio.decode_png(imageio.encode_png(arr))
        exp = imageio.validate_image_array(arr, reject_uniform=False)
        assert np.array_equal(dec, exp)


def test_png_corruption_detected():
    data = bytearray(imageio.encode_png(np.zeros((20, 20, 3), np.uint8) + 7))
    data[40] ^= 0xFF
    with pytest.raises(imageio.ImageDecodeError):
        imageio.decode_png(bytes(data))


def test_invalid_and_blank_images_rejected():
    with pytest.raises(imageio.InvalidImageError):
        imageio.decode_image(b"<html>error</html>" * 4)
    with pytest.raises(imageio.ImageDecodeError):
        imageio.decode_image(b"\xff\xd8\xff" + b"0" * 64)  # JPEG -> needs Blender's decoder
    with pytest.raises(imageio.InvalidImageError):
        imageio.validate_image_array(np.full((64, 64, 3), 9, np.uint8))
    with pytest.raises(imageio.InvalidImageError):
        imageio.validate_image_array(np.zeros((4, 4, 3), np.uint8))


def test_resize_preserves_mean():
    rng = np.random.default_rng(3)
    a = rng.random((100, 80, 3)).astype(np.float32)
    small = imageio.resize(a, 33, 41)
    assert small.shape == (41, 33, 3)
    assert abs(small.mean() - a.mean()) < 1e-3
    big = imageio.resize(a, 200, 160)
    assert abs(big.mean() - a.mean()) < 5e-3


# ---------------------------------------------------------------------------
# rasteriser / cameras
# ---------------------------------------------------------------------------

def test_uv_raster_area_matches_uv_area():
    m = uv_sphere(1.0, 24, 12)
    size = 256
    uvs = raster.rasterize_uv(m.tri_uvs, size)
    uv_area = meshdata.triangle_areas(m.tri_uvs.astype(np.float64)).sum()
    assert abs(uvs.coverage - uv_area) < 0.02


def test_uv_raster_barycentrics_reconstruct_uv():
    m = cube()
    size = 128
    uvs = raster.rasterize_uv(m.tri_uvs, size)
    uv = m.uv_points(uvs.tri, uvs.bary)
    ys, xs = np.divmod(uvs.index, size)
    assert np.allclose(uv[:, 0] * size, xs + 0.5, atol=1e-3)
    assert np.allclose((1 - uv[:, 1]) * size, ys + 0.5, atol=1e-3)


def test_zbuffer_keeps_nearest():
    xy = np.array([[[0, 0], [10, 0], [0, 10]], [[0, 0], [10, 0], [0, 10]]], np.float64)
    z = np.array([[0.5, 0.5, 0.5], [0.2, 0.2, 0.2]])
    r = raster.rasterize(xy, 10, 10, z=z)
    assert (r.tri[r.tri >= 0] == 1).all()


def test_large_triangle_strip_split():
    xy = np.array([[[-5, -5], [3000, -5], [-5, 3000]]], np.float64)
    r = raster.rasterize(xy, 1000, 1000, batch=10_000)
    assert (r.tri >= 0).sum() > 990_000


def test_uv_overlap_detection():
    m = cube()
    assert raster.uv_overlap_fraction(m.tri_uvs) < 0.01
    stacked = m.tri_uvs.copy()
    stacked[:] = stacked[0:2].repeat(6, axis=0)  # every face uses the same UV square
    assert raster.uv_overlap_fraction(stacked) > 0.5


def test_camera_fit_and_projection():
    m = uv_sphere(1.0, 16, 8, center=(3, 1, 2))
    for key, d in camera.VIEW_PRESETS.items():
        for ortho in (False, True):
            cam = camera.preset_camera(key, d, m.positions, 256, ortho=ortho)
            xy, _, w, depth = cam.project(m.positions)
            assert (w > 0).all() and (depth > 0).all()
            assert xy.min() >= 256 * 0.03 and xy.max() <= 256 * 0.97
    front = camera.preset_camera("FRONT", camera.VIEW_PRESETS["FRONT"], m.positions, 256)
    assert front.forward[1] > 0.99   # front camera looks along +Y


def test_viewport_camera_square_crop():
    view = camera.look_at((0, -5, 0), (0, 0, 0), (0, 0, 1))
    proj = camera.perspective(0.8, 2.0, 0.1, 100)
    cam = camera.viewport_camera("V", view, proj, 1000, 500, 256, True)
    # x scaled by aspect so a square frame covers the centre of the viewport
    assert np.isclose(cam.proj[0, 0], proj[0, 0] * 2)


# ---------------------------------------------------------------------------
# projection pipeline
# ---------------------------------------------------------------------------

def _run(mesh, keys, size=256, view=256, **kw):
    cams = [camera.preset_camera(k, camera.VIEW_PRESETS[k], mesh.positions, view) for k in keys]
    st = projection.TextureState(mesh, size)
    ps = projection.ProjectionSettings(texture_size=size, view_size=view, **kw)
    job = jobs.Job("t", "t")
    prov = providers.get("MOCK")
    spec = pipeline.GenerationSpec(prompt="iron", request_size=view)
    summ = pipeline.run_multiview(job, mesh, cams, st, prov, providers.ProviderConfig(model="mock"), spec, ps)
    return st, summ, job.drain()


def test_multiview_covers_convex_mesh_and_emits_layers():
    st, summ, events = _run(cube(), ["FRONT", "RIGHT", "LEFT", "BACK", "TOP", "BOTTOM"])
    kinds = [e[0] for e in events]
    assert kinds.count("view_done") == 6
    assert summ["unseen_fraction"] < 0.02
    # texels right at silhouette edges are not "claimed" but filled from the best-angle fallback
    assert st.textured_fraction() > 0.93
    assert st.textured_fraction() + summ["fallback_fraction"] > 0.98
    for k, p in events:
        if k == "view_done":
            assert p["layer"].shape == (256, 256, 4) and p["layer"].dtype == np.uint8
            assert set(p["debug"]) >= {"render", "mask", "generated", "aligned"}


def test_views_only_paint_untextured_and_skip_when_done():
    mesh = cube()
    st, summ, ev = _run(mesh, ["FRONT", "FRONT"])
    assert summ["calls"] == 1 and len(summ["skipped"]) == 1


def test_occlusion_leaves_hidden_area_unseen():
    outer = cube(2.0)
    inner = cube(0.5)
    m = combine("nested", outer, inner)
    st, summ, ev = _run(m, ["FRONT", "RIGHT", "LEFT", "BACK", "TOP", "BOTTOM"])
    assert 0.3 < summ["unseen_fraction"] < 0.7  # the inner cube's half of the atlas
    fill = [p for k, p in ev if k == "fill_done"]
    assert fill, "hidden texels must be filled (and labelled) rather than left black"


def test_min_facing_defers_grazing_texels_to_better_views():
    s = uv_sphere(1.0, 32, 16)
    st, summ, ev = _run(s, ["FRONT"], min_facing=0.6)
    first = [p for k, p in ev if k == "view_done"][0]
    frac_strict = first["stats"]["claimed_fraction"]
    st2, summ2, ev2 = _run(s, ["FRONT"], min_facing=0.05)
    frac_loose = [p for k, p in ev2 if k == "view_done"][0]["stats"]["claimed_fraction"]
    assert frac_strict < frac_loose
    assert summ["fallback_fraction"] > 0  # grazing texels kept as low-priority fallback


def test_estimate_shift_recovers_translation():
    rng = np.random.default_rng(5)
    img = masks.box_blur(rng.random((256, 256)).astype(np.float32), 3)
    img = np.repeat(img[..., None], 3, axis=2)
    moved = imageio.shift_image(img, 7, -5)
    dy, dx, conf = projection.estimate_shift(img, moved, 30)
    assert (dy, dx) == (-7, 5)
    assert conf > 8


def test_color_match():
    rng = np.random.default_rng(6)
    ref = rng.random((64, 64, 3)).astype(np.float32) * 0.5 + 0.2
    gen = np.clip(ref * 0.8 + 0.1, 0, 1)
    region = np.ones((64, 64), bool)
    out = projection.match_colors(gen, ref, region)
    assert np.abs(out - ref).mean() < np.abs(gen - ref).mean() * 0.3


def test_refine_uv_is_local():
    mesh = cube()
    st, _, _ = _run(mesh, ["FRONT", "RIGHT", "LEFT", "BACK", "TOP", "BOTTOM"])
    for _ in range(1):
        mask = np.zeros((256, 256), np.float32)
        mask[40:80, 20:60] = 1.0
        job = jobs.Job("r", "r")
        spec = pipeline.GenerationSpec(prompt="rust", request_size=256)
        pipeline.run_refine_uv(job, st, mask, providers.get("MOCK"), providers.ProviderConfig(), spec)
        (kind, p), = [e for e in job.drain() if e[0] == "refine_done"]
        a = p["layer"][..., 3]
        assert a[60, 40] > 200
        assert a[150:, 150:].max() == 0


def test_refine_view_projects_only_masked_visible_texels():
    mesh = cube()
    st, _, _ = _run(mesh, ["FRONT", "RIGHT", "LEFT", "BACK", "TOP", "BOTTOM"])
    uvs = raster.rasterize_uv(mesh.tri_uvs, 256, tri_subset=np.array([0, 1]))  # front face
    mask = np.zeros(256 * 256, np.float32)
    mask[uvs.index] = 1
    mask = mask.reshape(256, 256)
    cam = camera.preset_camera("FRONT", camera.VIEW_PRESETS["FRONT"], mesh.positions, 256)
    job = jobs.Job("r", "r")
    spec = pipeline.GenerationSpec(prompt="gold", request_size=256)
    stats = pipeline.run_refine_view(job, mesh, cam, st, mask, providers.get("MOCK"), providers.ProviderConfig(),
                                     spec)
    assert stats["visible_fraction"] > 0.9
    back = camera.preset_camera("BACK", camera.VIEW_PRESETS["BACK"], mesh.positions, 256)
    with pytest.raises(pipeline.PipelineError):
        pipeline.run_refine_view(jobs.Job("r", "r"), mesh, back, st, mask, providers.get("MOCK"),
                                 providers.ProviderConfig(), spec)


def test_tile_is_seamless():
    # A smooth but non-periodic image (a gradient): its raw edges do not match.
    yy, xx = np.mgrid[0:64, 0:64].astype(np.float32) / 63.0
    img = np.stack([xx, yy, xx * yy], axis=-1)
    assert np.abs(img[:, 0] - img[:, -1]).mean() > 0.4
    t = pipeline.make_seamless(img)
    # after the offset blend, wrapping edges are neighbours in the source -> continuous
    assert np.abs(t[:, 0] - t[:, -1]).mean() < 0.05
    assert np.abs(t[0] - t[-1]).mean() < 0.05


def test_cancellation_stops_between_views():
    job = jobs.Job("t", "t")
    job.cancel()
    mesh = cube()
    cams = [camera.preset_camera("FRONT", camera.VIEW_PRESETS["FRONT"], mesh.positions, 128)]
    st = projection.TextureState(mesh, 128)
    jobs.run_job(job, lambda j: pipeline.run_multiview(
        j, mesh, cams, st, providers.get("MOCK"), providers.ProviderConfig(),
        pipeline.GenerationSpec(prompt="x", request_size=128), projection.ProjectionSettings(128, 128)),
        synchronous=True)
    assert job.status == "CANCELLED"
    assert not [e for e in job.drain() if e[0] == "view_done"]


def test_job_failure_classification():
    def boom(j):
        raise providers.ProviderError("rate_limit", "slow down")
    job = jobs.Job("t", "t")
    jobs.run_job(job, boom, synchronous=True)
    assert job.status == "FAILED" and job.error_kind == "rate_limit" and job.retryable


# ---------------------------------------------------------------------------
# compositing, masks, pbr
# ---------------------------------------------------------------------------

def test_compositing_modes_and_opacity():
    base = np.zeros((4, 4, 4), np.float32)
    base[..., :3] = 0.5
    base[..., 3] = 1
    top = np.zeros((4, 4, 4), np.float32)
    top[..., :3] = 0.5
    top[..., 3] = 1
    rgb, a = compositing.composite([compositing.LayerInput(base), compositing.LayerInput(top, 1.0, "MULTIPLY")], 4)
    assert np.allclose(rgb, 0.25)
    rgb, a = compositing.composite([compositing.LayerInput(base), compositing.LayerInput(top * [2, 0, 0, 1], 0.5)], 4)
    assert np.allclose(rgb[..., 0], 0.75) and np.allclose(rgb[..., 1], 0.25)
    empty = np.zeros((4, 4, 4), np.float32)
    rgb, a = compositing.composite([compositing.LayerInput(empty)], 4)
    assert a.max() == 0
    for mode in compositing.BLEND_MODES:
        compositing.composite([compositing.LayerInput(base), compositing.LayerInput(top, 0.7, mode)], 4)


def test_gutter_and_push_pull():
    col = np.zeros((32, 32, 3), np.float32)
    valid = np.zeros((32, 32), bool)
    col[10:20, 10:20] = 1.0
    valid[10:20, 10:20] = True
    g, v = masks.fill_gutter(col, valid, 3)
    assert v.sum() > valid.sum() and np.allclose(g[v], 1.0)
    f = masks.push_pull_fill(col, valid)
    assert np.allclose(f, 1.0, atol=1e-4)


def test_normal_from_height_orientation():
    size = 64
    # height increases with V (towards the top of the image)
    v = np.linspace(1, 0, size)[:, None].repeat(size, axis=1)
    n = pbr.normal_from_height(v.astype(np.float32), 200.0)
    inner = n[5:-5, 5:-5]
    # slope up in +V -> normal tilts towards -V: green < 0.5 in OpenGL convention
    assert inner[..., 1].mean() < 0.45
    assert abs(inner[..., 0].mean() - 0.5) < 1e-3
    dx = pbr.normal_from_height(v.astype(np.float32), 200.0, directx=True)
    assert dx[5:-5, 5:-5, 1].mean() > 0.55


def test_pbr_maps_ranges():
    rng = np.random.default_rng(9)
    rgb = rng.random((64, 64, 3)).astype(np.float32)
    for preset in pbr.MATERIAL_PRESETS:
        r = pbr.roughness_map(rgb, preset)
        assert r.min() >= 0.02 and r.max() <= 1.0
    m = pbr.metallic_map(64, 1.0)
    assert m.mean() == 1.0
    orm = pbr.pack_orm(None, r, m)
    assert orm.shape == (64, 64, 3) and np.allclose(orm[..., 0], 1.0)
    px = pbr.pixelate(rgb, 8, 4)
    assert len(np.unique(px[..., 0])) <= 5


# ---------------------------------------------------------------------------
# prompts & secrets
# ---------------------------------------------------------------------------

def test_style_controls_change_request():
    base = prompting.StyleSettings(style="REALISTIC", detail=0.9, wear=0.0)
    a = prompting.compose("iron armor", "", base, purpose="first_view")
    b = prompting.compose("iron armor", "", prompting.StyleSettings(style="HAND_PAINTED", detail=0.1, wear=0.9),
                          purpose="first_view")
    assert a.prompt != b.prompt
    assert "hand-painted" in b.prompt.lower() and "photograph" in b.negative
    assert "pristine" in a.prompt and "heavily weathered" in b.prompt
    assert a.style_preset == "photographic" and b.style_preset == "fantasy-art"
    pal = prompting.compose("x", "", prompting.StyleSettings(use_palette=True, palette=[(1, 0, 0)]),
                            purpose="next_view")
    assert "#FF0000" in pal.prompt and "seamlessly" in pal.prompt
    lo = prompting.compose("x", "", prompting.StyleSettings(strength=0.1), purpose="first_view")
    assert "conservative" in lo.prompt and lo.strength == pytest.approx(0.1)


def test_redaction():
    secrets.register_secret("my-super-secret-value")
    s = secrets.redact("failed: my-super-secret-value and sk-abcdefghijklmnopqrstuv and AIzaSyA1234567890123456789012")
    assert "my-super-secret-value" not in s and "sk-abcdef" not in s and "AIzaSy" not in s
    assert secrets.redact("Authorization: Bearer abcdefghijklmnop1234").count("[REDACTED]") == 1
    cfg = providers.ProviderConfig(api_key="my-super-secret-value")
    assert "my-super-secret-value" not in repr(cfg)


def test_mesh_analysis():
    rep = meshdata.analyze(cube())
    assert rep.suitable and rep.uv_islands == 6 and rep.uv_overlap < 0.01
    m = cube()
    m.uv_name = ""
    assert not meshdata.analyze(m).suitable
    m = cube()
    m.tri_uvs = m.tri_uvs * 2.0
    rep = meshdata.analyze(m)
    assert any("outside 0-1" in t for _, t in rep.messages)


def test_flipped_normals_are_reoriented_and_textured():
    m = cube()
    m.tris = m.tris[:, ::-1].copy()          # reverse winding -> normals point inwards
    m.tri_uvs = m.tri_uvs[:, ::-1].copy()
    m.tri_normals = meshdata.compute_face_normals(m.positions, m.tris)
    m.corner_normals = np.repeat(m.tri_normals[:, None], 3, axis=1)
    assert meshdata.normals_inverted(m)
    assert meshdata.orient_normals(m) and m.normals_flipped
    assert any("flipped" in t for _, t in meshdata.analyze(m).messages)
    st, summ, _ = _run(m, ["FRONT", "RIGHT", "LEFT", "BACK", "TOP", "BOTTOM"])
    assert summ["unseen_fraction"] < 0.02


def test_thin_two_sided_object_keeps_front_and_back_separate():
    """A thin shield (3 mm thick, 1 m wide): front and back must not bleed into each other."""
    shield = cube(1.0)
    shield.positions = shield.positions * np.array([1.0, 0.003, 1.0])
    st, summ, ev = _run(shield, ["FRONT", "BACK"], size=256, view=256)
    views = {p["label"]: p["layer"] for k, p in ev if k == "view_done"}
    assert set(views) == {"Front", "Back"}
    # cube() puts the front (-Y) face in UV cell 0 and the back (+Y) face in cell 1
    front_cell = (slice(140, 240), slice(10, 75))
    back_cell = (slice(140, 240), slice(95, 160))
    assert views["Front"][front_cell][..., 3].mean() > 200 and views["Front"][back_cell][..., 3].mean() < 5
    assert views["Back"][back_cell][..., 3].mean() > 200
