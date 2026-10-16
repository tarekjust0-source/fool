"""The real-provider test script must work, and must never pass off the test provider as real."""

import json
import os
import subprocess
import sys
import tempfile

import bpy

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(ROOT, "tools", "real_provider_test.py")


def _run(args, env_extra=None):
    binary = bpy.app.binary_path
    if binary and os.path.basename(binary).lower().startswith("blender"):
        cmd = [binary, "-b", "--factory-startup", "--python", SCRIPT, "--"] + args
    else:
        cmd = [sys.executable, SCRIPT, "--"] + args
    env = {k: v for k, v in os.environ.items() if k not in ("ATS_ENABLE_TEST_PROVIDER", "ATS_SYNC_JOBS",
                                                             "OPENAI_API_KEY")}
    env.update(env_extra or {})
    return subprocess.run(cmd, capture_output=True, text=True, timeout=1200, env=env)


def test_refuses_test_provider_and_missing_key():
    p = _run(["--provider", "MOCK"])
    assert p.returncode == 2 and "Refusing" in p.stdout
    p = _run(["--provider", "OPENAI", "--yes"])
    assert p.returncode == 2 and "OPENAI_API_KEY" in p.stdout


def test_dry_run_sends_nothing():
    out = tempfile.mkdtemp()
    p = _run(["--provider", "COMFYUI", "--out", out])
    assert p.returncode == 0 and "Dry run only" in p.stdout
    assert not os.path.exists(os.path.join(out, "report.json"))


def test_self_check_with_test_provider_is_labelled_not_real():
    out = tempfile.mkdtemp()
    p = _run(["--provider", "MOCK", "--allow-test-provider", "--yes", "--views", "3", "--out", out])
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-2000:]
    report = json.load(open(os.path.join(out, "report.json")))
    assert report["real_provider_test"] is False and report["all_passed"]
    assert "NOT A REAL PROVIDER TEST" in open(os.path.join(out, "report.md")).read()
    for f in ("A_cube.png", "A_cube_reopened.png", "B_character.png", "B_character_corrected.png"):
        assert os.path.getsize(os.path.join(out, "renders", f)) > 10_000
