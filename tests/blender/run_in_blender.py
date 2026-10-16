"""Run the test suites inside a real Blender installation.

    blender --background --factory-startup --python tests/blender/run_in_blender.py
    blender -b --factory-startup --python tests/blender/run_in_blender.py -- -k layers   # pytest args after --

pytest is installed into a private folder (tests/.blender_site) the first
time, using Blender's own Python.  Nothing is installed into Blender itself.
The add-on is built and installed into a throw-away user profile, so your
real preferences and extensions are not touched.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SITE = os.path.join(ROOT, "tests", ".blender_site")

if SITE not in sys.path:
    sys.path.insert(0, SITE)
try:
    import pytest  # noqa: F401
except ImportError:
    py = sys.executable
    if subprocess.call([py, "-m", "pip", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL):
        subprocess.check_call([py, "-m", "ensurepip", "--upgrade"])
    subprocess.check_call([py, "-m", "pip", "install", "--target", SITE, "pytest"])
    import pytest  # noqa: F401

args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
os.chdir(ROOT)
code = pytest.main(["-q", "-p", "no:cacheprovider", os.path.join(ROOT, "tests")] + args)
sys.exit(int(code))
