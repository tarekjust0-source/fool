"""Analysis, UV creation, reference loading, credentials and provider tests."""

import bpy
from bpy.props import EnumProperty, FloatProperty, StringProperty
from bpy_extras.io_utils import ImportHelper

from ... import providers
from .. import credentials, prefs, services
from .common import active_mesh, run_safely


class ATS_OT_analyze(bpy.types.Operator):
    """Inspect mesh, UVs, materials and how much surface the enabled views can see"""

    bl_idname = "ats.analyze"
    bl_label = "Analyze Mesh"

    @classmethod
    def poll(cls, context):
        return active_mesh(context) is not None

    def execute(self, context):
        info = run_safely(self, services.analyze, context, active_mesh(context))
        if info is None:
            return {"CANCELLED"}
        warn = [m for lvl, m in info["messages"] if lvl != "INFO"]
        self.report({"WARNING" if warn else "INFO"}, warn[0] if warn else "Mesh looks ready for texturing")
        return {"FINISHED"}


class ATS_OT_create_uv(bpy.types.Operator):
    """Add a NEW UV map by automatic unwrapping. Existing UV maps are kept untouched.
Automatic unwraps can create many seams; hand-made UVs give better results"""

    bl_idname = "ats.create_uv"
    bl_label = "Create UV Map"
    bl_options = {"REGISTER", "UNDO"}

    method: EnumProperty(name="Method", items=[
        ("SMART", "Smart UV Project", "Good general purpose unwrap for hard-surface props"),
        ("LIGHTMAP", "Lightmap Pack", "Every face gets its own island; many seams, no overlap"),
        ("CUBE", "Cube Projection", "Simple boxy objects"),
    ], default="SMART")
    angle: FloatProperty(name="Angle Limit", default=1.15192, min=0.0175, max=1.5533, subtype="ANGLE")
    margin: FloatProperty(name="Island Margin", default=0.01, min=0.0, max=0.2)
    set_render: bpy.props.BoolProperty(name="Make Active for Render/Export", default=False,
                                       description="Game engines usually read the first/active render UV map")
    uv_name: StringProperty(name="Name", default="ATS_UV")

    @classmethod
    def poll(cls, context):
        return active_mesh(context) is not None

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=380)

    def draw(self, context):
        col = self.layout.column()
        col.prop(self, "method")
        if self.method == "SMART":
            col.prop(self, "angle")
        col.prop(self, "margin")
        col.prop(self, "uv_name")
        col.prop(self, "set_render")
        box = col.box()
        box.label(text="A new UV map is added; existing UVs are not changed.", icon="INFO")
        box.label(text="Automatic unwraps may have many seams / uneven density.", icon="ERROR")

    def execute(self, context):
        obj = active_mesh(context)
        name = run_safely(self, services.create_uv_map, context, obj, self.method, self.angle, self.margin,
                          self.set_render, self.uv_name or "ATS_UV")
        if name is None:
            return {"CANCELLED"}
        self.report({"INFO"}, f"Created UV map '{name}' and selected it for texturing")
        return {"FINISHED"}


class ATS_OT_load_reference(bpy.types.Operator, ImportHelper):
    """Load a reference image from disk (it is only sent to the provider when you generate)"""

    bl_idname = "ats.load_reference"
    bl_label = "Load Reference Image"
    filter_glob: StringProperty(default="*.png;*.jpg;*.jpeg;*.webp;*.tga;*.bmp;*.tif;*.tiff;*.exr",
                                options={"HIDDEN"})

    def execute(self, context):
        try:
            img = bpy.data.images.load(self.filepath, check_existing=True)
        except RuntimeError as exc:
            self.report({"ERROR"}, f"Cannot load image: {exc}")
            return {"CANCELLED"}
        sc = context.scene.ats
        sc.reference = img
        sc.use_reference = True
        return {"FINISHED"}


class ATS_OT_set_model(bpy.types.Operator):
    """Use this model preset"""

    bl_idname = "ats.set_model"
    bl_label = "Set Model"
    provider: StringProperty()
    model: StringProperty()

    def execute(self, context):
        prefs.get_prefs(context).settings_for(self.provider).model = self.model
        return {"FINISHED"}


class ATS_OT_set_api_key(bpy.types.Operator):
    """Enter an API key for this session (or store it in the system keyring)"""

    bl_idname = "ats.set_api_key"
    bl_label = "Enter API Key"
    provider: StringProperty()
    key: StringProperty(name="API Key", subtype="PASSWORD", options={"SKIP_SAVE"})

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=420)

    def draw(self, context):
        self.layout.prop(self, "key")
        s = prefs.get_prefs(context).settings_for(self.provider)
        self.layout.label(text="Stored in: " + ("system keyring" if s.key_mode == "KEYRING" else
                                                 "memory only (this session)"))

    def execute(self, context):
        s = prefs.get_prefs(context).settings_for(self.provider)
        key = self.key.strip()
        self.key = ""
        if not key:
            return {"CANCELLED"}
        if s.key_mode == "KEYRING":
            try:
                credentials.set_keyring_key(self.provider, key)
            except Exception as exc:  # noqa: BLE001
                self.report({"ERROR"}, f"Keyring unavailable: {type(exc).__name__}")
                return {"CANCELLED"}
        else:
            credentials.set_session_key(self.provider, key)
        self.report({"INFO"}, "API key set")
        return {"FINISHED"}


class ATS_OT_clear_api_key(bpy.types.Operator):
    """Forget the session/keyring key for this provider"""

    bl_idname = "ats.clear_api_key"
    bl_label = "Clear API Key"
    provider: StringProperty()

    def execute(self, context):
        s = prefs.get_prefs(context).settings_for(self.provider)
        credentials.set_session_key(self.provider, "")
        if s.key_mode == "KEYRING":
            credentials.clear_keyring_key(self.provider)
        if s.key_mode == "PREFS":
            s.api_key = ""
        return {"FINISHED"}


class ATS_OT_reset_consent(bpy.types.Operator):
    """Ask again before sending data to this provider"""

    bl_idname = "ats.reset_consent"
    bl_label = "Revoke Acknowledgement"
    provider: StringProperty()

    def execute(self, context):
        prefs.get_prefs(context).settings_for(self.provider).consent = False
        return {"FINISHED"}


class ATS_OT_test_connection(bpy.types.Operator):
    """Check credentials and model availability (a lightweight request; no image is generated)"""

    bl_idname = "ats.test_connection"
    bl_label = "Test Connection"
    provider: StringProperty()

    def execute(self, context):
        prov, cfg = run_safely(self, prefs.provider_config, context, self.provider or None) or (None, None)
        if prov is None:
            return {"CANCELLED"}
        ok, msg = prefs.network_allowed(prov, cfg)
        if not ok:
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}
        try:
            text = prov.test_connection(cfg)
        except providers.ProviderError as exc:
            self.report({"ERROR"}, exc.user_message())
            return {"CANCELLED"}
        self.report({"INFO"}, text)
        return {"FINISHED"}


class ATS_OT_open_prefs(bpy.types.Operator):
    """Open the AI Texture Studio preferences"""

    bl_idname = "ats.open_prefs"
    bl_label = "Provider Settings"

    def execute(self, context):
        from .. import ADDON_ID
        bpy.ops.screen.userpref_show("INVOKE_DEFAULT")
        context.preferences.active_section = "ADDONS"
        try:
            bpy.ops.preferences.addon_show(module=ADDON_ID)
        except Exception:  # noqa: BLE001
            pass
        return {"FINISHED"}


CLASSES = (ATS_OT_analyze, ATS_OT_create_uv, ATS_OT_load_reference, ATS_OT_set_model, ATS_OT_set_api_key,
           ATS_OT_clear_api_key, ATS_OT_reset_consent, ATS_OT_test_connection, ATS_OT_open_prefs)
