# SPDX-License-Identifier: GPL-3.0-or-later
"""AI Texture Studio – AI-assisted, UV-aware texturing inside Blender.

The ``core`` and ``providers`` packages are pure Python + numpy (thread
safe, unit-testable outside Blender).  Everything touching ``bpy`` lives in
the ``blender`` package and is imported only at registration time.
"""


def register():
    from .blender import registration
    registration.register()


def unregister():
    from .blender import registration
    registration.unregister()
