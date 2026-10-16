"""Class registration, handlers and timers."""

from __future__ import annotations

import bpy
from bpy.app.handlers import persistent

from .. import providers
from ..core.secrets import redact
from . import prefs, props, runtime, storage
from .operators import generate_ops, layer_ops, misc_ops, output_ops, refine_ops, view_ops
from .ui import panels

CLASSES = (
    prefs.ATS_ProviderSettings,
    prefs.ATS_AddonPreferences,
    *props.CLASSES,
    *generate_ops.CLASSES,
    *view_ops.CLASSES,
    *refine_ops.CLASSES,
    *layer_ops.CLASSES,
    *output_ops.CLASSES,
    *misc_ops.CLASSES,
    *panels.CLASSES,
)


@persistent
def _save_pre(filepath="", *_args):
    try:
        storage.prepare_save(filepath if isinstance(filepath, str) else "")
    except Exception as exc:  # noqa: BLE001
        print(redact(f"[AI Texture Studio] could not move texture files next to the .blend: {exc}"))


@persistent
def _save_post(*_args):
    try:
        storage.finish_save()
    except Exception as exc:  # noqa: BLE001
        print(redact(f"[AI Texture Studio] could not update texture paths: {exc}"))


@persistent
def _load_post(*_args):
    storage.clear_cache()
    runtime.reset_after_file_load()


def _decoder(data: bytes):
    """Fallback image decoder for providers (JPEG/WebP): runs on the main thread."""
    return runtime.call_on_main(storage.decode_with_blender, data)


def _init_prefs():
    try:
        prefs.get_prefs()
    except Exception:  # noqa: BLE001
        pass
    return None


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    props.register_props()
    bpy.app.handlers.save_pre.append(_save_pre)
    bpy.app.handlers.save_post.append(_save_post)
    bpy.app.handlers.load_post.append(_load_post)
    providers.base.FALLBACK_DECODER = _decoder
    try:
        storage.cleanup_stale_sessions()
    except OSError:
        pass
    if bpy.app.background:
        _init_prefs()
    else:
        bpy.app.timers.register(_init_prefs, first_interval=0.1)


def unregister():
    runtime.shutdown()
    providers.base.FALLBACK_DECODER = None
    for h, fn in ((bpy.app.handlers.save_pre, _save_pre), (bpy.app.handlers.save_post, _save_post),
                  (bpy.app.handlers.load_post, _load_post)):
        if fn in h:
            h.remove(fn)
    props.unregister_props()
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
    storage.clear_cache()
