"""Install the built extension ZIP into an isolated Blender user profile and enable it.

Used by the integration tests.  Works with the ``bpy`` PyPI module or a
real Blender (``blender -b --python ...``).  Set ATS_ZIP to override the zip.
"""

from __future__ import annotations

import glob
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Isolated user resources so the developer's real Blender profile is untouched.
if "BLENDER_USER_RESOURCES" not in os.environ:
    os.environ["BLENDER_USER_RESOURCES"] = tempfile.mkdtemp(prefix="ats_blender_user_")
os.environ.setdefault("ATS_ENABLE_TEST_PROVIDER", "1")
os.environ.setdefault("ATS_SYNC_JOBS", "1")

import bpy  # noqa: E402

MODULE = None


def built_zip() -> str:
    z = os.environ.get("ATS_ZIP")
    if z:
        return z
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import build  # noqa: WPS433
    return build.build(tempfile.mkdtemp(prefix="ats_dist_"))


def install_and_enable() -> str:
    """Returns the add-on module name (bl_ext.<repo>.ai_texture_studio)."""
    global MODULE
    if MODULE:
        return MODULE
    zip_path = built_zip()
    repos = bpy.context.preferences.extensions.repos
    repo = None
    for r in repos:
        if r.module == "user_default":
            repo = r
    if repo is None:
        repo = repos.new(name="User Default", module="user_default", custom_directory="", source="USER")
    result = bpy.ops.extensions.package_install_files(filepath=zip_path, repo=repo.module,
                                                      enable_on_install=True)
    if "FINISHED" not in result:
        raise RuntimeError(f"Extension install failed: {result}")
    MODULE = f"bl_ext.{repo.module}.ai_texture_studio"
    if MODULE not in bpy.context.preferences.addons:
        import addon_utils
        addon_utils.enable(MODULE, default_set=True)
    if MODULE not in bpy.context.preferences.addons:
        raise RuntimeError("Add-on did not enable")
    return MODULE


def package():
    """The installed add-on package (to reach internals in tests)."""
    import importlib
    return importlib.import_module(install_and_enable())


def installed_files() -> list[str]:
    mod = package()
    return sorted(glob.glob(os.path.join(os.path.dirname(mod.__file__), "**", "*.py"), recursive=True))
