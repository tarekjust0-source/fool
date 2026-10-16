"""Draw every panel / list / preferences page with a validating fake layout.

Headless Blender has no visible UI, so this checks the draw code: every
``prop()`` must name an existing RNA property and every ``operator()`` a
registered operator, in many scene states (empty, untextured, textured,
pending correction, job running, job failed).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bootstrap  # noqa: E402

import bpy  # noqa: E402

MOD = bootstrap.install_and_enable()
panels = __import__(f"{MOD}.blender.ui.panels", fromlist=["x"])
runtime = __import__(f"{MOD}.blender.runtime", fromlist=["x"])
jobs = __import__(f"{MOD}.core.jobs", fromlist=["x"])

CALLS = []


class OpProps:
    def __init__(self, idname):
        mod, name = idname.split(".")
        op = getattr(getattr(bpy.ops, mod), name)
        self._rna = op.get_rna_type()

    def __setattr__(self, k, v):
        if k.startswith("_"):
            return object.__setattr__(self, k, v)
        assert k in self._rna.properties, f"operator property {k} missing"


class FakeLayout:
    def __init__(self):
        self.enabled = True
        self.active = True
        self.alert = False
        self.scale_y = 1.0
        self.scale_x = 1.0
        self.use_property_split = False

    def _child(self, *a, **k):
        return FakeLayout()

    row = column = box = split = grid_flow = column_flow = _child

    def label(self, text="", icon="NONE", **k):
        assert isinstance(text, str)
        CALLS.append(("label", text))

    def separator(self, **k):
        pass

    def prop(self, data, prop, **k):
        assert data is not None, f"prop({prop}) on None"
        assert prop in data.bl_rna.properties, f"{type(data).__name__} has no property '{prop}'"
        CALLS.append(("prop", prop))

    def prop_search(self, data, prop, search_data, search_prop, **k):
        assert prop in data.bl_rna.properties, prop
        assert hasattr(search_data, search_prop), search_prop

    def operator(self, idname, **k):
        mod, name = idname.split(".")
        assert hasattr(getattr(bpy.ops, mod), name), f"operator {idname} not registered"
        CALLS.append(("op", idname))
        return OpProps(idname)

    def operator_menu_enum(self, idname, prop, **k):
        self.operator(idname)

    def template_list(self, list_type, list_id, data, propname, active_data, active_propname, **k):
        assert hasattr(bpy.types, list_type), list_type
        assert propname in data.bl_rna.properties and active_propname in active_data.bl_rna.properties

    def template_ID(self, data, prop, **k):
        assert prop in data.bl_rna.properties

    def template_icon(self, **k):
        pass

    def progress(self, **k):
        CALLS.append(("progress", k.get("text")))


class FakePanel:
    def __init__(self, cls):
        self.layout = FakeLayout()
        self._cls = cls

    def draw(self, context):
        self._cls.draw(self, context)


def draw_all(context):
    CALLS.clear()
    for cls in panels.CLASSES:
        if issubclass(cls, bpy.types.Panel):
            FakePanel(cls).draw(context)
    prefs = context.preferences.addons[MOD].preferences
    for pid in ("OPENAI", "STABILITY", "GEMINI", "COMFYUI", "MOCK"):
        prefs.provider = pid
        for mode in ("ENV", "SESSION", "KEYRING", "PREFS"):
            prefs.settings_for(pid).key_mode = mode
            fake = type("P", (), {})()
            fake.layout = FakeLayout()
            type(prefs).draw(_PrefsProxy(prefs, fake.layout), context)
    prefs.provider = "MOCK"
    return list(CALLS)


class _PrefsProxy:
    """Lets AddonPreferences.draw run against the fake layout."""

    def __init__(self, prefs, layout):
        object.__setattr__(self, "_p", prefs)
        object.__setattr__(self, "layout", layout)

    def __getattr__(self, k):
        return getattr(self._p, k)

    @property
    def bl_rna(self):
        return self._p.bl_rna


def _ui_list_draw(context, obj):
    sc = context.scene.ats
    for cls, data, coll in ((panels.ATS_UL_views, sc, "views"), (panels.ATS_UL_layers, obj.ats, "layers"),
                            (panels.ATS_UL_history, obj.ats, "history"), (panels.ATS_UL_prompts, sc, "prompt_history")):
        items = list(getattr(data, coll))
        assert items, f"{coll} should not be empty in this state"
        for i, item in enumerate(items):
            cls.draw_item(None, context, FakeLayout(), data, item, 0, data, "", i)


@pytest.fixture(autouse=True)
def setup():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    bpy.context.preferences.addons[MOD].preferences.provider = "MOCK"
    runtime._state["last"] = None
    yield


def test_draw_without_selection():
    calls = draw_all(bpy.context)
    assert ("label", "Select a mesh object") in calls


def test_draw_in_all_states():
    bpy.ops.mesh.primitive_cube_add()
    obj = bpy.context.active_object
    calls = draw_all(bpy.context)
    assert ("op", "ats.generate") in calls
    sc = bpy.context.scene.ats
    sc.prompt = "iron"
    sc.texture_size = sc.view_size = "512"
    sc.use_palette = True
    sc.style = "PIXEL_ART"
    sc.mode = "TILE"
    draw_all(bpy.context)
    sc.mode = "MULTIVIEW"
    assert bpy.ops.ats.generate() == {"FINISHED"}
    bpy.ops.ats.analyze()
    assert bpy.ops.ats.pbr_generate() == {"FINISHED"}
    calls = draw_all(bpy.context)
    _ui_list_draw(bpy.context, obj)
    assert ("op", "ats.layer_op") in calls and ("op", "ats.view_show") in calls
    # pending correction state
    obj.data.polygons[0].select = True
    sc.refine_source = "FACES"
    sc.refine_space = "UV"
    sc.refine_prompt = "rust"
    assert bpy.ops.ats.refine() == {"FINISHED"}
    calls = draw_all(bpy.context)
    assert ("op", "ats.refine_resolve") in calls
    # running job state
    j = jobs.Job("generate", "Busy")
    j.status = "RUNNING"
    runtime._state["job"] = j
    try:
        calls = draw_all(bpy.context)
        assert any(c[0] == "progress" for c in calls)
    finally:
        runtime._state["job"] = None
    # failed job state
    j.status = "FAILED"
    j.error = "Simulated failure for UI"
    runtime._state["last"] = j
    calls = draw_all(bpy.context)
    assert ("op", "ats.retry") in calls
