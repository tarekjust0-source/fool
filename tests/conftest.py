import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
TESTS = os.path.dirname(os.path.abspath(__file__))
if os.path.join(TESTS, "unit") not in sys.path:
    sys.path.insert(0, os.path.join(TESTS, "unit"))
os.environ.setdefault("ATS_ENABLE_TEST_PROVIDER", "1")
