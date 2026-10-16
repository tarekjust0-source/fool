"""Generate / cancel / retry / prompt preview operators."""

import bpy
from bpy.props import BoolProperty, IntProperty, StringProperty

from ... import providers
from .. import prefs, runtime, services
from .common import active_mesh, run_safely


def _targets(context, all_selected: bool):
    if all_selected:
        objs = [o for o in context.selected_objects if o.type == "MESH"]
        act = active_mesh(context)
        if act is not None and act in objs:  # active first: it sets the look for the others
            objs.remove(act)
            objs.insert(0, act)
        return objs
    obj = active_mesh(context)
    return [obj] if obj else []


class ATS_OT_generate(bpy.types.Operator):
    """Generate an AI texture for the selected mesh and apply it as new layers"""

    bl_idname = "ats.generate"
    bl_label = "Generate Texture"
    bl_options = {"REGISTER"}

    all_selected: BoolProperty(name="All Selected Objects", default=False)
    only_view: IntProperty(name="Only View", default=-1, description="Generate a single view (index)")
    confirmed: BoolProperty(default=False, options={"SKIP_SAVE", "HIDDEN"})

    @classmethod
    def poll(cls, context):
        return runtime.active_job() is None and active_mesh(context) is not None

    def invoke(self, context, event):
        need, notice = services.needs_consent(context)
        if need:
            self._notice = notice
            return context.window_manager.invoke_props_dialog(self, width=520)
        return self.execute(context)

    def draw(self, context):
        col = self.layout.column(align=True)
        col.label(text="Before the first request, please confirm:", icon="INFO")
        from ..prefs import _wrap
        for line in _wrap(getattr(self, "_notice", ""), 80):
            col.label(text=line)
        col.separator()
        col.label(text="Click OK to send this request. You can revoke this in the add-on preferences.")

    def execute(self, context):
        p = prefs.get_prefs(context)
        need, _ = services.needs_consent(context)
        if need:
            p.settings_for(p.provider).consent = True
        objs = _targets(context, self.all_selected)
        if not objs:
            self.report({"ERROR"}, "Select a mesh object")
            return {"CANCELLED"}
        only = self.only_view if self.only_view >= 0 else None
        runtime.LAST_ARGS["generate"] = {"all_selected": self.all_selected, "only_view": self.only_view}
        job = run_safely(self, services.start_generation, context, objs, only_view=only)
        if job is None:
            return {"CANCELLED"}
        if job.status == "FAILED":
            self.report({"ERROR"}, job.error)
            return {"CANCELLED"}
        self.report({"INFO"}, job.title)
        return {"FINISHED"}


class ATS_OT_cancel(bpy.types.Operator):
    """Cancel the running generation. Already finished views are kept"""

    bl_idname = "ats.cancel"
    bl_label = "Cancel"

    @classmethod
    def poll(cls, context):
        return runtime.active_job() is not None

    def execute(self, context):
        runtime.cancel()
        return {"FINISHED"}


class ATS_OT_retry(bpy.types.Operator):
    """Run the last request again. Multi-view only repaints areas that are still untextured"""

    bl_idname = "ats.retry"
    bl_label = "Retry"

    @classmethod
    def poll(cls, context):
        j = runtime.last_job()
        return runtime.active_job() is None and j is not None and j.status in ("FAILED", "CANCELLED")

    def execute(self, context):
        j = runtime.last_job()
        if j.kind == "refine":
            return bpy.ops.ats.refine("INVOKE_DEFAULT")
        return bpy.ops.ats.generate("INVOKE_DEFAULT", **runtime.LAST_ARGS.get("generate", {}))


class ATS_OT_preview_prompt(bpy.types.Operator):
    """Show exactly what text and parameters will be sent to the provider"""

    bl_idname = "ats.preview_prompt"
    bl_label = "Preview Prompt"

    text: StringProperty(options={"SKIP_SAVE"})

    def invoke(self, context, event):
        self.text = services.preview_prompt(context.scene.ats)
        return context.window_manager.invoke_popup(self, width=640)

    def draw(self, context):
        from ..prefs import _wrap
        col = self.layout.column(align=True)
        try:
            prov, cfg = prefs.provider_config(context)
            caps = prov.capabilities(cfg)
            col.label(text=f"Provider: {prov.name}  model: {cfg.model or prov.default_model}")
            sc = context.scene.ats
            notes = []
            if sc.negative and providers.base.NEGATIVE_PROMPT not in caps:
                notes.append("Negative prompt is appended as 'Avoid: …' text (no native field).")
            if sc.use_reference and sc.reference and providers.base.REFERENCE not in caps:
                notes.append("This provider ignores reference images.")
            if getattr(prov, "reference_purposes", None):
                notes.append("Reference image is used for the first view only.")
            if sc.use_seed and providers.base.SEED not in caps:
                notes.append("Seed is not supported by this provider.")
            for n in notes:
                col.label(text=n, icon="INFO")
        except Exception as exc:  # noqa: BLE001
            col.label(text=str(exc), icon="ERROR")
        col.separator()
        for para in self.text.split("\n\n"):
            for line in _wrap(para, 100):
                col.label(text=line)
            col.separator()

    def execute(self, context):
        return {"FINISHED"}


class ATS_OT_prompt_use(bpy.types.Operator):
    """Load this prompt from the history"""

    bl_idname = "ats.prompt_use"
    bl_label = "Use Prompt"
    index: IntProperty()

    def execute(self, context):
        sc = context.scene.ats
        if 0 <= self.index < len(sc.prompt_history):
            it = sc.prompt_history[self.index]
            sc.prompt = it.prompt
            sc.negative = it.negative
            if it.style:
                try:
                    sc.style = it.style
                except TypeError:
                    pass
        return {"FINISHED"}


class ATS_OT_prompt_remove(bpy.types.Operator):
    """Remove this prompt from the history"""

    bl_idname = "ats.prompt_remove"
    bl_label = "Remove Prompt"
    index: IntProperty()

    def execute(self, context):
        sc = context.scene.ats
        if 0 <= self.index < len(sc.prompt_history):
            sc.prompt_history.remove(self.index)
            sc.active_prompt = max(0, min(sc.active_prompt, len(sc.prompt_history) - 1))
        return {"FINISHED"}


CLASSES = (ATS_OT_generate, ATS_OT_cancel, ATS_OT_retry, ATS_OT_preview_prompt, ATS_OT_prompt_use,
           ATS_OT_prompt_remove)
