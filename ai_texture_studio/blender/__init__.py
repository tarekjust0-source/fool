"""Blender integration (everything that imports bpy lives in this package)."""

# The add-on's top-level package name (bl_ext.<repo>.ai_texture_studio when
# installed as an extension); used as the AddonPreferences bl_idname.
ADDON_ID = __package__.rpartition(".")[0]
