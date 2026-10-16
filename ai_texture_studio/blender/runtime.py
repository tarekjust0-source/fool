"""Main-thread side of background jobs.

One job runs at a time.  A ``bpy.app.timers`` callback drains the job's
event queue and hands events to the job's handler on the main thread, where
creating images, layers and materials is allowed.  Set the environment
variable ``ATS_SYNC_JOBS=1`` (automated tests / background mode) to run jobs
inline.
"""

from __future__ import annotations

import os
import queue
import threading
import traceback

import bpy

from ..core import jobs
from ..core.secrets import redact
from . import storage

_state = {"job": None, "handler": None, "on_finish": None, "last": None}
_main_calls: "queue.Queue" = queue.Queue()
LAST_ARGS: dict = {}   # operator arguments of the last generate/refine, for Retry


def push_undo_step(message: str):
    """Make a finished job one Ctrl+Z step (no-op where unsupported)."""
    if bpy.app.background:
        return
    try:
        bpy.ops.ed.undo_push(message=message)
    except Exception:  # noqa: BLE001
        pass


def call_on_main(fn, *args, timeout: float = 60.0):
    """Run ``fn`` on Blender's main thread and wait for the result.

    Worker threads use this for the few operations that need bpy (e.g.
    decoding JPEG/WebP).  The job timer services the queue.
    """
    if threading.current_thread() is threading.main_thread():
        return fn(*args)
    done = threading.Event()
    box = {}
    _main_calls.put((fn, args, box, done))
    if not done.wait(timeout):
        raise TimeoutError("Blender's main thread did not respond")
    if "error" in box:
        raise box["error"]
    return box["value"]


def _service_main_calls():
    while True:
        try:
            fn, args, box, done = _main_calls.get_nowait()
        except queue.Empty:
            return
        try:
            box["value"] = fn(*args)
        except Exception as exc:  # noqa: BLE001
            box["error"] = exc
        done.set()


def preview_pixels(rgb, alpha, gutter: int = 8):
    """Display-ready base colour (worker thread): padded islands, no black."""
    from ..core import imageio, masks
    valid = alpha > 0.02
    col, v = masks.fill_gutter(rgb, valid, gutter)
    if v.any() and not v.all():
        col[~v] = col[v].mean(axis=0)
    return imageio.to_uint8(col)


class BridgeJob(jobs.Job):
    """Job whose emitted layers/debug images are written to disk in the worker."""

    def emit(self, kind, payload=None):
        if isinstance(payload, dict):
            out_dir = self.context.get("out_dirs", {}).get(payload.get("object", ""), self.context.get("out_dir"))
            if out_dir and payload.get("layer") is not None:
                tag = f"{kind}_{payload.get('view', '')}_{len(self.context.setdefault('written', []))}"
                path = os.path.join(out_dir, f"gen_{self.id}_{tag}.png")
                storage.write_png(path, payload["layer"], level=4)
                payload["layer_path"] = path
                self.context["written"].append(path)
            rgb = payload.pop("state_rgb", None)
            alpha = payload.pop("state_alpha", None)
            if rgb is not None and alpha is not None:
                payload["preview"] = preview_pixels(rgb, alpha)
            dbg = payload.pop("debug", None)
            if dbg and out_dir and self.context.get("keep_debug", True):
                ddir = os.path.join(out_dir, "views", f"{self.id}_{payload.get('view') or kind}")
                paths = {}
                for name, arr in dbg.items():
                    p = os.path.join(ddir, f"{name}.png")
                    storage.write_png(p, arr, level=3)
                    paths[name] = p
                payload["debug_paths"] = paths
                payload["debug_dir"] = ddir
        super().emit(kind, payload)


def active_job() -> jobs.Job | None:
    j = _state["job"]
    return j if j is not None and j.active else None


def last_job() -> jobs.Job | None:
    return _state["job"] or _state["last"]


def synchronous() -> bool:
    return os.environ.get("ATS_SYNC_JOBS") == "1"


def start(job: BridgeJob, worker, handler, on_finish=None):
    if active_job() is not None:
        raise RuntimeError("Another AI Texture Studio job is running. Cancel it or wait.")
    _state["job"] = job
    _state["handler"] = handler
    _state["on_finish"] = on_finish
    if synchronous():
        jobs.run_job(job, worker, synchronous=True)
        _drain()
        return job
    jobs.run_job(job, worker)
    if not bpy.app.timers.is_registered(_tick):
        bpy.app.timers.register(_tick, first_interval=0.1)
    return job


def cancel():
    j = active_job()
    if j is not None:
        j.cancel()
        j.message = "Cancelling… (waiting for the current request to return)"


def _drain() -> bool:
    """Process pending events. Returns True when the job has finished."""
    _service_main_calls()
    job = _state["job"]
    if job is None:
        return True
    finished = False
    for kind, payload in job.drain():
        if kind == "finished":
            finished = True
            continue
        try:
            if _state["handler"] is not None:
                _state["handler"](job, kind, payload)
        except Exception as exc:  # noqa: BLE001
            job.status = "FAILED"
            job.error = redact(f"Applying result failed: {exc}")
            job.error_kind = "internal"
            print(redact(traceback.format_exc()))
    if finished:
        cb = _state["on_finish"]
        _state["last"] = job
        _state["job"] = None
        _state["handler"] = None
        _state["on_finish"] = None
        if cb is not None:
            try:
                cb(job)
            except Exception:  # noqa: BLE001
                print(redact(traceback.format_exc()))
    _redraw()
    return finished


def _tick():
    try:
        done = _drain()
    except Exception:  # noqa: BLE001
        print(redact(traceback.format_exc()))
        done = True
    return None if done else 0.2


def _redraw():
    wm = getattr(bpy.context, "window_manager", None)
    if wm is None:
        return
    for win in wm.windows:
        for area in win.screen.areas:
            if area.type in ("VIEW_3D", "IMAGE_EDITOR", "PROPERTIES"):
                area.tag_redraw()


def shutdown():
    j = active_job()
    if j is not None:
        j.cancel()
    if bpy.app.timers.is_registered(_tick):
        bpy.app.timers.unregister(_tick)
