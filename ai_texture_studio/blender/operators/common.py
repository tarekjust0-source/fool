"""Shared operator helpers."""

import traceback

import bpy

from ...core.secrets import redact
from ...providers import ProviderError
from ..services import UserError


def active_mesh(context):
    obj = context.active_object
    return obj if obj is not None and obj.type == "MESH" else None


def run_safely(op, fn, *args, **kw):
    """Execute ``fn`` and convert expected failures into operator reports."""
    try:
        return fn(*args, **kw)
    except UserError as exc:
        op.report({"ERROR"}, redact(str(exc)))
    except ProviderError as exc:
        op.report({"ERROR"}, redact(exc.user_message()))
    except RuntimeError as exc:
        op.report({"ERROR"}, redact(str(exc)))
    except Exception as exc:  # noqa: BLE001
        print(redact(traceback.format_exc()))
        op.report({"ERROR"}, redact(f"Unexpected error: {type(exc).__name__}: {exc}"))
    return None


def show_image(context, img: bpy.types.Image) -> bool:
    """Show ``img`` in an Image Editor (existing one, else a new window)."""
    wm = context.window_manager
    for win in wm.windows:
        for area in win.screen.areas:
            if area.type == "IMAGE_EDITOR":
                area.spaces.active.image = img
                area.tag_redraw()
                return True
    try:
        bpy.ops.wm.window_new()
        win = wm.windows[-1]
        area = max(win.screen.areas, key=lambda a: a.width * a.height)
        area.type = "IMAGE_EDITOR"
        area.spaces.active.image = img
        return True
    except Exception:  # noqa: BLE001
        return False
