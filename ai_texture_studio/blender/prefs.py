"""Add-on preferences: provider selection, models, credentials, consent."""

import json
import os

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty

from .. import providers
from . import ADDON_ID, credentials
from .props import PATH_OPTS

KEY_MODES = [
    ("ENV", "Environment Variable", "Read the key from an environment variable (recommended)"),
    ("SESSION", "Session Only", "Enter the key each session; kept in memory, never written to disk"),
    ("KEYRING", "System Keyring", "Store in the OS credential manager (requires the 'keyring' Python package)"),
    ("PREFS", "Blender Preferences", "Stored in userpref.blend as PLAIN TEXT when preferences are saved"),
]


def provider_items(self, context):
    items = []
    for i, p in enumerate(providers.all_providers()):
        items.append((p.id, p.name, p.description, i))
    return items


class ATS_ProviderSettings(bpy.types.PropertyGroup):
    model: StringProperty(name="Model", description="Model identifier sent to the provider")
    base_url: StringProperty(name="API URL", description="Leave empty for the official endpoint")
    key_mode: EnumProperty(name="Key Storage", items=KEY_MODES, default="ENV")
    env_var: StringProperty(name="Variable", description="Environment variable holding the API key")
    api_key: StringProperty(name="API Key", subtype="PASSWORD",
                            description="Stored in Blender's preferences as plain text")
    consent: BoolProperty(name="Data transfer acknowledged", default=False,
                          description="You acknowledged what this provider receives and that it may cost money")
    quality: EnumProperty(name="Quality", items=[("low", "Low", ""), ("medium", "Medium", ""),
                                                  ("high", "High", ""), ("auto", "Auto", "")], default="high")
    organization: StringProperty(name="Organization", description="Optional OpenAI organization id")
    grow_mask: IntProperty(name="Grow Mask", default=5, min=0, max=100,
                           description="Stability inpaint: pixels to grow the mask by")
    # ComfyUI
    controlnet: StringProperty(name="Depth ControlNet",
                               description="ControlNet model file name for depth (optional, recommended)")
    control_strength: FloatProperty(name="ControlNet Strength", default=0.8, min=0.0, max=2.0)
    steps: IntProperty(name="Steps", default=28, min=1, max=150)
    cfg: FloatProperty(name="CFG", default=6.5, min=1.0, max=30.0)
    sampler: StringProperty(name="Sampler", default="euler")
    scheduler: StringProperty(name="Scheduler", default="normal")
    workflow_path: StringProperty(name="Custom Workflow", subtype="FILE_PATH", options=PATH_OPTS,
                                  description="Optional ComfyUI API-format workflow JSON with {{PLACEHOLDERS}}")
    mock_fail_mode: StringProperty(name="Simulated failure", default="")


class ATS_AddonPreferences(bpy.types.AddonPreferences):
    bl_idname = ADDON_ID

    provider: EnumProperty(name="Provider", items=provider_items)
    openai: PointerProperty(type=ATS_ProviderSettings)
    stability: PointerProperty(type=ATS_ProviderSettings)
    gemini: PointerProperty(type=ATS_ProviderSettings)
    comfyui: PointerProperty(type=ATS_ProviderSettings)
    mock: PointerProperty(type=ATS_ProviderSettings)
    timeout: FloatProperty(name="Request Timeout (s)", default=180.0, min=10.0, max=1800.0)
    max_retries: IntProperty(name="Automatic Retries", default=2, min=0, max=6,
                             description="Retries for rate limits / server errors / dropped connections")
    keep_debug_views: BoolProperty(name="Keep Intermediate View Images", default=True,
                                   description="Save renders, masks and raw AI outputs for inspection")
    initialized: BoolProperty(default=False)

    def settings_for(self, provider_id: str) -> ATS_ProviderSettings:
        return getattr(self, provider_id.lower(), self.mock)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "provider")
        try:
            p = providers.get(self.provider)
        except providers.ProviderError:
            layout.label(text="Provider unavailable", icon="ERROR")
            return
        s = self.settings_for(p.id)
        box = layout.box()
        box.label(text=p.name, icon="WORLD" if not p.is_local else "DESKTOP")
        col = box.column(align=True)
        for line in _wrap(p.description, 90):
            col.label(text=line)
        notice = box.box()
        notice.label(text="Data handling & cost", icon="INFO")
        c = notice.column(align=True)
        for line in _wrap(p.data_notice, 90):
            c.label(text=line)
        row = notice.row()
        row.label(text="Acknowledged" if s.consent else "Not yet acknowledged (asked on first use)",
                  icon="CHECKMARK" if s.consent else "QUESTION")
        if s.consent:
            row.operator("ats.reset_consent", text="Revoke").provider = p.id

        if p.id != "MOCK":
            box.prop(s, "model")
            if p.model_presets:
                row = box.row(align=True)
                for m in p.model_presets:
                    op = row.operator("ats.set_model", text=m)
                    op.provider = p.id
                    op.model = m
            box.prop(s, "base_url", text="API URL" if not p.is_local else "Server URL")
        if p.id == "OPENAI":
            box.prop(s, "quality")
            box.prop(s, "organization")
        if p.id == "STABILITY":
            box.prop(s, "grow_mask")
        if p.id == "COMFYUI":
            box.label(text="Model above = checkpoint file name (as listed in ComfyUI)")
            box.prop(s, "controlnet")
            row = box.row(align=True)
            row.prop(s, "control_strength")
            row.prop(s, "steps")
            row.prop(s, "cfg")
            row = box.row(align=True)
            row.prop(s, "sampler")
            row.prop(s, "scheduler")
            box.prop(s, "workflow_path")
        if p.id == "MOCK":
            box.prop(s, "mock_fail_mode")

        if p.requires_key or p.id == "COMFYUI":
            kb = box.box()
            kb.label(text="Credentials" + (" (optional bearer token)" if p.id == "COMFYUI" else ""), icon="LOCKED")
            kb.prop(s, "key_mode")
            if s.key_mode == "ENV":
                kb.prop(s, "env_var")
            elif s.key_mode == "PREFS":
                kb.prop(s, "api_key")
                kb.label(text="Warning: saved as plain text in userpref.blend", icon="ERROR")
            elif s.key_mode in ("SESSION", "KEYRING"):
                if s.key_mode == "KEYRING" and not credentials.keyring_available():
                    kb.label(text="'keyring' package not found – choose another mode", icon="ERROR")
                row = kb.row(align=True)
                row.operator("ats.set_api_key", text="Enter Key…").provider = p.id
                row.operator("ats.clear_api_key", text="Clear").provider = p.id
            kb.label(text=credentials.key_status(p.id, s))
        box.operator("ats.test_connection", icon="URL").provider = p.id

        g = layout.box()
        g.label(text="General")
        g.prop(self, "timeout")
        g.prop(self, "max_retries")
        g.prop(self, "keep_debug_views")
        if not bpy.app.online_access:
            g.label(text="Blender 'Allow Online Access' is OFF: hosted providers are disabled.", icon="ERROR")


def _wrap(text: str, width: int):
    words = text.split()
    line = ""
    for w in words:
        if len(line) + len(w) + 1 > width:
            yield line
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        yield line


def get_prefs(context=None) -> ATS_AddonPreferences:
    ctx = context or bpy.context
    addon = ctx.preferences.addons.get(ADDON_ID)
    if addon is None:
        raise RuntimeError("AI Texture Studio preferences not found (is the add-on enabled?)")
    p = addon.preferences
    if not p.initialized:
        init_defaults(p)
    return p


def init_defaults(p: ATS_AddonPreferences):
    for prov in providers.all_providers():
        s = p.settings_for(prov.id)
        if not s.model:
            s.model = prov.default_model
        if not s.env_var:
            s.env_var = prov.key_env_var or f"{prov.id}_API_KEY"
    p.comfyui.key_mode = "SESSION"
    p.initialized = True


def provider_config(context, provider_id: str | None = None):
    """Resolve (provider, ProviderConfig) on the main thread."""
    p = get_prefs(context)
    pid = provider_id or p.provider
    prov = providers.get(pid)
    s = p.settings_for(pid)
    key = credentials.get_key(pid, s) if (prov.requires_key or pid == "COMFYUI") else ""
    options = {"quality": s.quality, "organization": s.organization, "grow_mask": s.grow_mask}
    if pid == "COMFYUI":
        options.update({"controlnet": s.controlnet.strip(), "control_strength": s.control_strength,
                        "steps": s.steps, "cfg": s.cfg, "sampler": s.sampler, "scheduler": s.scheduler})
        if s.workflow_path.strip():
            path = bpy.path.abspath(s.workflow_path)
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    options["workflow_json"] = fh.read()
            except OSError as exc:
                raise providers.ProviderError("config", f"Cannot read ComfyUI workflow: {exc}", retryable=False)
    if pid == "MOCK":
        options["fail_mode"] = s.mock_fail_mode or os.environ.get("ATS_MOCK_FAIL", "")
    cfg = providers.ProviderConfig(model=s.model.strip(), base_url=s.base_url.strip(), api_key=key,
                                   timeout=p.timeout, max_retries=p.max_retries, options=options)
    return prov, cfg


def network_allowed(prov, cfg) -> tuple[bool, str]:
    """Respect Blender's 'Allow Online Access' for anything not on localhost."""
    url = (cfg.base_url or prov.default_base_url).lower()
    if prov.is_local and not url:  # in-process provider, no network at all
        return True, ""
    local = any(h in url for h in ("://127.0.0.1", "://localhost", "://[::1]"))
    if local or bpy.app.online_access:
        return True, ""
    return False, ("Online access is disabled in Blender (Preferences > System > Network > Allow Online Access). "
                   f"{prov.name} needs it to send requests.")


def summary_json(context) -> str:
    """Non-secret summary of provider settings (for history records)."""
    p = get_prefs(context)
    s = p.settings_for(p.provider)
    return json.dumps({"provider": p.provider, "model": s.model, "base_url": s.base_url or "default"})
