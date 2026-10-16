"""Camera models for multi-view capture and projection (numpy only).

Conventions (OpenGL style, matching Blender's RegionView3D matrices):
* view matrix maps world -> camera space, camera looks down -Z, +Y up.
* projection matrix maps camera space -> clip space; NDC in [-1, 1].
* image pixel (0, 0) is the top-left corner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Direction from the object centre towards the camera, in the object's frame
# (Blender: -Y is "front", +Z is up).  Elevation for diagonal views ~25deg.
_E = math.radians(25.0)
_c, _s = math.cos(_E), math.sin(_E)
_d = math.sqrt(0.5)
VIEW_PRESETS: dict[str, tuple[float, float, float]] = {
    "FRONT": (0.0, -1.0, 0.0),
    "BACK": (0.0, 1.0, 0.0),
    "RIGHT": (1.0, 0.0, 0.0),
    "LEFT": (-1.0, 0.0, 0.0),
    "TOP": (0.0, 0.0, 1.0),
    "BOTTOM": (0.0, 0.0, -1.0),
    "FRONT_RIGHT": (_d * _c, -_d * _c, _s),
    "FRONT_LEFT": (-_d * _c, -_d * _c, _s),
    "BACK_RIGHT": (_d * _c, _d * _c, _s),
    "BACK_LEFT": (-_d * _c, _d * _c, _s),
}

VIEW_LABELS = {
    "FRONT": "Front", "BACK": "Back", "RIGHT": "Right", "LEFT": "Left",
    "TOP": "Top", "BOTTOM": "Bottom", "FRONT_RIGHT": "Front-Right 3/4",
    "FRONT_LEFT": "Front-Left 3/4", "BACK_RIGHT": "Back-Right 3/4",
    "BACK_LEFT": "Back-Left 3/4", "CUSTOM": "Custom",
}


def look_at(eye, target, up) -> np.ndarray:
    eye = np.asarray(eye, np.float64)
    target = np.asarray(target, np.float64)
    up = np.asarray(up, np.float64)
    f = target - eye
    f /= max(np.linalg.norm(f), 1e-12)
    s = np.cross(f, up)
    if np.linalg.norm(s) < 1e-6:
        alt = np.array([0.0, 1.0, 0.0]) if abs(f[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        s = np.cross(f, alt)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def perspective(fov_y: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(fov_y / 2.0)
    m = np.zeros((4, 4))
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def orthographic(half_w: float, half_h: float, near: float, far: float) -> np.ndarray:
    m = np.eye(4)
    m[0, 0] = 1.0 / half_w
    m[1, 1] = 1.0 / half_h
    m[2, 2] = -2.0 / (far - near)
    m[2, 3] = -(far + near) / (far - near)
    return m


@dataclass
class ViewCamera:
    name: str
    view: np.ndarray        # (4,4) world -> camera
    proj: np.ndarray        # (4,4) camera -> clip
    width: int
    height: int
    ortho: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def position(self) -> np.ndarray:
        inv = np.linalg.inv(self.view)
        return inv[:3, 3]

    @property
    def forward(self) -> np.ndarray:
        """World-space viewing direction (unit)."""
        return -self.view[2, :3] / max(np.linalg.norm(self.view[2, :3]), 1e-12)

    def to_clip(self, pts: np.ndarray) -> np.ndarray:
        p = np.asarray(pts, np.float64)
        h = np.concatenate([p, np.ones((len(p), 1))], axis=1)
        return h @ (self.proj @ self.view).T

    def project(self, pts: np.ndarray):
        """Project world points.

        Returns (xy pixels (N,2), ndc_z (N,), w (N,), view_depth (N,)).
        view_depth is the positive distance along the viewing axis.
        """
        p = np.asarray(pts, np.float64)
        h = np.concatenate([p, np.ones((len(p), 1))], axis=1)
        cam = h @ self.view.T
        clip = cam @ self.proj.T
        w = clip[:, 3]
        w_safe = np.where(np.abs(w) < 1e-12, 1e-12, w)
        ndc = clip[:, :3] / w_safe[:, None]
        xy = np.empty((len(p), 2))
        xy[:, 0] = (ndc[:, 0] * 0.5 + 0.5) * self.width
        xy[:, 1] = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * self.height
        return xy, ndc[:, 2], w, -cam[:, 2]

    def dir_to_camera(self, pts: np.ndarray) -> np.ndarray:
        if self.ortho:
            d = -self.forward
            return np.broadcast_to(d, (len(pts), 3)).astype(np.float64)
        v = self.position[None, :] - np.asarray(pts, np.float64)
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.maximum(n, 1e-12)

    def to_dict(self) -> dict:
        return {
            "name": self.name, "view": self.view.tolist(), "proj": self.proj.tolist(),
            "width": self.width, "height": self.height, "ortho": self.ortho, "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ViewCamera":
        return cls(d["name"], np.array(d["view"], np.float64), np.array(d["proj"], np.float64),
                   int(d["width"]), int(d["height"]), bool(d.get("ortho")), dict(d.get("meta", {})))


def _fit_ndc(cam: ViewCamera, points: np.ndarray, fill: float) -> ViewCamera:
    """Post-multiply an NDC affine so ``points`` fill ``fill`` of the frame."""
    xy, _, w, _ = cam.project(points)
    front = w > 1e-9
    if not front.any():
        return cam
    ndc_x = xy[front, 0] / cam.width * 2 - 1
    ndc_y = 1 - xy[front, 1] / cam.height * 2
    cx = (ndc_x.max() + ndc_x.min()) / 2
    cy = (ndc_y.max() + ndc_y.min()) / 2
    ext = max(ndc_x.max() - ndc_x.min(), ndc_y.max() - ndc_y.min()) / 2
    if ext < 1e-9:
        return cam
    s = fill / ext
    a = np.eye(4)
    a[0, 0] = s
    a[1, 1] = s
    a[0, 3] = -cx * s
    a[1, 3] = -cy * s
    cam.proj = a @ cam.proj
    return cam


def preset_camera(
    name: str,
    direction,
    points: np.ndarray,
    size: int,
    *,
    basis: np.ndarray | None = None,
    ortho: bool = False,
    fov_deg: float = 35.0,
    fill: float = 0.9,
) -> ViewCamera:
    """Camera looking at ``points`` from ``direction`` (object frame).

    ``basis`` is the object's 3x3 world rotation so "front" follows the
    object's own orientation.  The frame is fitted so the object fills
    ``fill`` of the image.
    """
    pts = np.asarray(points, np.float64)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    center = (lo + hi) / 2
    radius = max(float(np.linalg.norm(hi - lo)) / 2, 1e-6)
    R = np.eye(3) if basis is None else np.asarray(basis, np.float64)
    d = R @ np.asarray(direction, np.float64)
    d /= max(np.linalg.norm(d), 1e-12)
    up_local = np.array([0.0, 0.0, 1.0])
    if abs(direction[2]) > 0.99:  # top / bottom: image-up is object +Y / -Y
        up_local = np.array([0.0, 1.0 if direction[2] > 0 else -1.0, 0.0])
    up = R @ up_local
    fov = math.radians(fov_deg)
    if ortho:
        dist = radius * 3.0
        eye = center + d * dist
        view = look_at(eye, center, up)
        proj = orthographic(radius, radius, max(dist - radius * 1.5, 1e-4), dist + radius * 1.5)
    else:
        dist = radius / math.sin(fov / 2.0) * 1.05
        eye = center + d * dist
        view = look_at(eye, center, up)
        near = max(dist - radius * 1.5, dist * 0.01)
        proj = perspective(fov, 1.0, near, dist + radius * 1.5)
    cam = ViewCamera(name, view, proj, size, size, ortho, {"direction": list(map(float, direction))})
    return _fit_ndc(cam, pts, fill)


def viewport_camera(name: str, view_matrix, window_matrix, region_w: int, region_h: int,
                    size: int, is_perspective: bool) -> ViewCamera:
    """Square camera from a Blender viewport, cropping to the central square."""
    proj = np.array(window_matrix, np.float64).copy()
    a = np.eye(4)
    if region_w >= region_h:
        a[0, 0] = region_w / max(region_h, 1)
    else:
        a[1, 1] = region_h / max(region_w, 1)
    proj = a @ proj
    return ViewCamera(name, np.array(view_matrix, np.float64), proj, size, size, not is_perspective,
                      {"source": "viewport"})


def fit_camera_to_points(cam: ViewCamera, points: np.ndarray, fill: float = 0.9) -> ViewCamera:
    return _fit_ndc(ViewCamera(cam.name, cam.view.copy(), cam.proj.copy(), cam.width, cam.height,
                               cam.ortho, dict(cam.meta)), points, fill)
