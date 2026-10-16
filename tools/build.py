"""Build the installable extension ZIP: python tools/build.py

Produces dist/ai_texture_studio-<version>.zip with blender_manifest.toml at
the archive root (Blender 4.2+ extension layout).
"""

import os
import re
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "ai_texture_studio")
EXCLUDE_DIRS = {"__pycache__", ".pytest_cache"}
EXCLUDE_EXT = {".pyc", ".pyo", ".tmp"}


def version() -> str:
    with open(os.path.join(PKG, "blender_manifest.toml"), encoding="utf-8") as fh:
        m = re.search(r'^version\s*=\s*"([^"]+)"', fh.read(), re.M)
    return m.group(1)


def build(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"ai_texture_studio-{version()}.zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for base, dirs, files in os.walk(PKG):
            dirs[:] = sorted(d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith("."))
            for f in sorted(files):
                if f.startswith(".") or os.path.splitext(f)[1] in EXCLUDE_EXT:
                    continue
                full = os.path.join(base, f)
                arc = os.path.relpath(full, PKG).replace(os.sep, "/")
                zf.write(full, arc)
    return path


if __name__ == "__main__":
    out = build(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "dist"))
    print(out)
