"""Provider request-format and error-handling tests against a local fake server.

These verify what is actually put on the wire (endpoints, headers, multipart
fields, JSON structure) and how HTTP failures are classified.  They do not
contact any real service.
"""

from __future__ import annotations

import base64
import email.parser
import email.policy
import http.server
import json
import threading
import time

import numpy as np
import pytest

from ai_texture_studio import providers
from ai_texture_studio.core import http as h
from ai_texture_studio.core import imageio
from ai_texture_studio.providers import base as pb
from ai_texture_studio.providers.comfyui_provider import ComfyUIProvider, fill_placeholders
from ai_texture_studio.providers.gemini_provider import GeminiProvider
from ai_texture_studio.providers.openai_provider import OpenAIProvider
from ai_texture_studio.providers.stability_provider import StabilityProvider

PNG = imageio.encode_png(np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8))


class Recorder:
    def __init__(self):
        self.requests = []
        self.routes = {}

    def route(self, method, path, fn):
        self.routes[(method, path)] = fn


def parse_multipart(headers, body):
    msg = email.parser.BytesParser(policy=email.policy.default).parsebytes(
        b"Content-Type: " + headers["content-type"].encode() + b"\r\n\r\n" + body)
    out = []
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        fn = part.get_filename()
        out.append((name, fn, part.get_content_type(), part.get_payload(decode=True)))
    return out


@pytest.fixture
def server():
    rec = Recorder()

    class Handler(http.server.BaseHTTPRequestHandler):
        def _handle(self, method):
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n) if n else b""
            path = self.path.split("?")[0]
            hdrs = {k.lower(): v for k, v in self.headers.items()}
            rec.requests.append({"method": method, "path": self.path, "headers": hdrs, "body": body})
            fn = rec.routes.get((method, path))
            if fn is None:
                for (m, p), f in rec.routes.items():
                    if m == method and p.endswith("*") and path.startswith(p[:-1]):
                        fn = f
            if fn is None:
                self.send_response(404)
                self.end_headers()
                return
            status, hdrs, data = fn(self, body)
            self.send_response(status)
            for k, v in hdrs.items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            self._handle("POST")

        def do_GET(self):
            self._handle("GET")

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    rec.url = f"http://127.0.0.1:{srv.server_address[1]}"
    yield rec
    srv.shutdown()


def req(**kw):
    img = np.random.default_rng(1).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), bool)
    mask[10:30, 10:30] = True
    d = dict(prompt="paint it", negative_prompt="blurry", purpose="next_view", image=img, mask=mask,
             depth=np.zeros((64, 64), np.uint8), references=[img], width=1024, height=1024, seed=42,
             strength=0.4, style_preset="fantasy-art")
    d.update(kw)
    return pb.ImageRequest(**d)


def fast(cfg):
    cfg.max_retries = 0
    return cfg


# ---------------------------------------------------------------------------

def test_openai_edit_request_format(server):
    def ok(handler, body):
        return 200, {"Content-Type": "application/json"}, json.dumps(
            {"data": [{"b64_json": base64.b64encode(PNG).decode()}]}).encode()
    server.route("POST", "/v1/images/edits", ok)
    cfg = fast(providers.ProviderConfig(model="gpt-image-1.5", base_url=server.url + "/v1", api_key="sk-test12345678",
                                        options={"quality": "high"}))
    res = OpenAIProvider().generate(req(), cfg)
    assert res.image.shape == (64, 64, 3)
    r = server.requests[-1]
    assert r["headers"]["authorization"] == "Bearer sk-test12345678"
    parts = parse_multipart(r["headers"], r["body"])
    names = [p[0] for p in parts]
    assert names.count("image[]") == 2            # render + reference
    assert "mask" in names and "input_fidelity" in names
    vals = {p[0]: p[3] for p in parts if p[1] is None}
    assert vals["model"] == b"gpt-image-1.5" and vals["size"] == b"1024x1024" and vals["input_fidelity"] == b"high"
    assert b"Avoid: blurry" in vals["prompt"]
    mask_png = next(p[3] for p in parts if p[0] == "mask")
    m = imageio.decode_png(mask_png)
    assert m.shape == (64, 64, 4)
    assert m[20, 20, 3] == 0 and m[50, 50, 3] == 255   # transparent = edit


def test_openai_text_only_uses_generations(server):
    server.route("POST", "/v1/images/generations", lambda hd, b: (200, {}, json.dumps(
        {"data": [{"b64_json": base64.b64encode(PNG).decode()}]}).encode()))
    cfg = fast(providers.ProviderConfig(model="gpt-image-1", base_url=server.url + "/v1", api_key="sk-xxxxxxxxxxxx"))
    OpenAIProvider().generate(req(image=None, mask=None, references=[], purpose="tile"), cfg)
    payload = json.loads(server.requests[-1]["body"])
    assert payload["model"] == "gpt-image-1" and payload["n"] == 1 and "prompt" in payload


def test_openai_retries_without_unsupported_fidelity(server):
    calls = {"n": 0}

    def fn(handler, body):
        calls["n"] += 1
        if b"input_fidelity" in body:
            return 400, {}, json.dumps({"error": {"message": "Unknown parameter: 'input_fidelity'."}}).encode()
        return 200, {}, json.dumps({"data": [{"b64_json": base64.b64encode(PNG).decode()}]}).encode()
    server.route("POST", "/v1/images/edits", fn)
    cfg = fast(providers.ProviderConfig(model="gpt-image-1", base_url=server.url + "/v1", api_key="sk-xxxxxxxxxxxx"))
    OpenAIProvider().generate(req(), cfg)
    assert calls["n"] == 2


def test_stability_endpoints_and_fields(server):
    def img(handler, body):
        return 200, {"Content-Type": "image/png", "finish-reason": "SUCCESS", "seed": "42"}, PNG
    server.route("POST", "/v2beta/stable-image/*", img)
    cfg = fast(providers.ProviderConfig(base_url=server.url, api_key="sk-stab-123456789"))
    p = StabilityProvider()
    p.generate(req(purpose="first_view", references=[]), cfg)
    assert server.requests[-1]["path"].endswith("/control/structure")
    p.generate(req(purpose="first_view"), cfg)
    assert server.requests[-1]["path"].endswith("/control/style-transfer")
    parts = parse_multipart(server.requests[-1]["headers"], server.requests[-1]["body"])
    assert {"init_image", "style_image"} <= {x[0] for x in parts}
    p.generate(req(purpose="next_view"), cfg)
    r = server.requests[-1]
    assert r["path"].endswith("/edit/inpaint") and r["headers"]["accept"] == "image/*"
    parts = parse_multipart(r["headers"], r["body"])
    vals = {x[0]: x[3] for x in parts if x[1] is None}
    assert vals["negative_prompt"] == b"blurry" and vals["seed"] == b"42" and vals["style_preset"] == b"fantasy-art"
    mask = imageio.decode_png(next(x[3] for x in parts if x[0] == "mask"))
    assert mask[20, 20, 0] == 255 and mask[50, 50, 0] == 0   # white = repaint
    p.generate(req(image=None, mask=None, references=[], purpose="tile"), cfg)
    assert server.requests[-1]["path"].endswith("/generate/core")


def test_stability_content_filter(server):
    server.route("POST", "/v2beta/stable-image/*",
                 lambda hd, b: (200, {"Content-Type": "image/png", "finish-reason": "CONTENT_FILTERED"}, PNG))
    cfg = fast(providers.ProviderConfig(base_url=server.url, api_key="sk-stab-123456789"))
    with pytest.raises(pb.ProviderError) as e:
        StabilityProvider().generate(req(), cfg)
    assert e.value.kind == "content_policy"


def test_gemini_payload_and_parse(server):
    def fn(handler, body):
        return 200, {}, json.dumps({"candidates": [{"content": {"parts": [
            {"text": "here"}, {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(PNG).decode()}}]},
            "finishReason": "STOP"}]}).encode()
    server.route("POST", "/v1beta/models/gemini-2.5-flash-image:generateContent", fn)
    cfg = fast(providers.ProviderConfig(model="gemini-2.5-flash-image", base_url=server.url, api_key="AIza-test-123456"))
    res = GeminiProvider().generate(req(), cfg)
    assert res.image.shape == (64, 64, 3)
    r = server.requests[-1]
    assert r["headers"]["x-goog-api-key"] == "AIza-test-123456"
    assert "key=" not in r["path"]  # key never in the URL
    payload = json.loads(r["body"])
    parts = payload["contents"][0]["parts"]
    assert sum(1 for p in parts if "inline_data" in p) == 3   # render, mask, reference
    assert payload["generationConfig"]["imageConfig"]["aspectRatio"] == "1:1"


def test_gemini_block_and_no_image():
    g = GeminiProvider()
    with pytest.raises(pb.ProviderError) as e:
        g.parse({"promptFeedback": {"blockReason": "SAFETY"}}, "m")
    assert e.value.kind == "content_policy"
    with pytest.raises(pb.ProviderError) as e:
        g.parse({"candidates": [{"content": {"parts": [{"text": "I cannot"}]}, "finishReason": "STOP"}]}, "m")
    assert e.value.kind == "invalid_response" and "I cannot" in str(e.value)


def test_comfyui_flow(server):
    server.route("POST", "/upload/image", lambda hd, b: (200, {}, json.dumps({"name": "x.png", "subfolder": ""}).encode()))
    server.route("POST", "/prompt", lambda hd, b: (200, {}, json.dumps({"prompt_id": "abc", "node_errors": {}}).encode()))
    state = {"n": 0}

    def hist(handler, body):
        state["n"] += 1
        if state["n"] < 2:
            return 200, {}, b"{}"
        return 200, {}, json.dumps({"abc": {"status": {"status_str": "success", "completed": True},
                                            "outputs": {"10": {"images": [{"filename": "o.png", "subfolder": "",
                                                                           "type": "output"}]}}}}).encode()
    server.route("GET", "/history/abc", hist)
    server.route("GET", "/view", lambda hd, b: (200, {"Content-Type": "image/png"}, PNG))
    cfg = fast(providers.ProviderConfig(model="sdxl.safetensors", base_url=server.url,
                                        options={"controlnet": "depth.safetensors", "steps": 20, "cfg": 6.0}))
    res = ComfyUIProvider().generate(req(), cfg)
    assert res.image.shape == (64, 64, 3)
    prompt_req = next(r for r in server.requests if r["path"] == "/prompt")
    wf = json.loads(prompt_req["body"])["prompt"]
    types = {n["class_type"] for n in wf.values()}
    assert {"CheckpointLoaderSimple", "SetLatentNoiseMask", "ControlNetApplyAdvanced", "KSampler", "SaveImage"} <= types
    ks = next(n for n in wf.values() if n["class_type"] == "KSampler")["inputs"]
    assert ks["seed"] == 42 and ks["steps"] == 20 and 0.6 <= ks["denoise"] <= 1.0
    uploads = [r for r in server.requests if r["path"] == "/upload/image"]
    assert len(uploads) == 3   # image, mask, depth


def test_comfyui_node_errors(server):
    server.route("POST", "/upload/image", lambda hd, b: (200, {}, json.dumps({"name": "x.png"}).encode()))
    server.route("POST", "/prompt", lambda hd, b: (200, {}, json.dumps(
        {"prompt_id": "x", "node_errors": {"1": {"errors": [{"message": "ckpt not found"}]}}}).encode()))
    cfg = fast(providers.ProviderConfig(model="missing.safetensors", base_url=server.url))
    with pytest.raises(pb.ProviderError) as e:
        ComfyUIProvider().generate(req(), cfg)
    assert e.value.kind == "config" and "ckpt not found" in str(e.value)


def test_comfyui_placeholders_keep_types():
    wf = {"3": {"inputs": {"seed": "{{SEED}}", "text": "a {{PROMPT}} b", "denoise": "{{DENOISE}}"}}}
    out = fill_placeholders(wf, {"SEED": 7, "PROMPT": "rusty", "DENOISE": 0.5})
    assert out["3"]["inputs"] == {"seed": 7, "text": "a rusty b", "denoise": 0.5}


@pytest.mark.parametrize("status,body,kind,retryable", [
    (401, {"error": {"message": "Invalid key"}}, "auth", False),
    (402, {"error": "pay"}, "quota", False),
    (429, {"error": {"message": "Rate limit reached"}}, "rate_limit", True),
    (429, {"error": {"message": "You exceeded your current quota"}}, "quota", False),
    (400, {"error": {"message": "Your request was rejected by the safety system"}}, "content_policy", False),
    (400, {"error": {"message": "bad size"}}, "config", False),
    (500, {"error": "boom"}, "server", True),
    (503, {}, "server", True),
])
def test_http_error_classification(server, status, body, kind, retryable):
    server.route("POST", "/v1/images/edits", lambda hd, b: (status, {"Content-Type": "application/json"},
                                                             json.dumps(body).encode()))
    cfg = fast(providers.ProviderConfig(model="gpt-image-1", base_url=server.url + "/v1", api_key="sk-xxxxxxxxxxxx"))
    with pytest.raises(pb.ProviderError) as e:
        OpenAIProvider().generate(req(), cfg)
    assert e.value.kind == kind and e.value.retryable == retryable


def test_retry_after_and_eventual_success(server):
    calls = {"n": 0}

    def fn(handler, body):
        calls["n"] += 1
        if calls["n"] == 1:
            return 429, {"Retry-After": "0"}, b'{"error": {"message": "slow"}}'
        return 200, {}, json.dumps({"data": [{"b64_json": base64.b64encode(PNG).decode()}]}).encode()
    server.route("POST", "/v1/images/edits", fn)
    cfg = providers.ProviderConfig(model="gpt-image-1-mini", base_url=server.url + "/v1", api_key="sk-xxxxxxxxxxxx",
                                   max_retries=2)
    p = OpenAIProvider()
    p.retry_policy = lambda c: h.RetryPolicy(max_retries=2, base_delay=0.01)
    p.generate(req(), cfg)
    assert calls["n"] == 2


def test_invalid_image_response_rejected(server):
    server.route("POST", "/v1/images/edits", lambda hd, b: (200, {}, json.dumps(
        {"data": [{"b64_json": base64.b64encode(b"definitely not an image").decode()}]}).encode()))
    cfg = fast(providers.ProviderConfig(model="gpt-image-1", base_url=server.url + "/v1", api_key="sk-xxxxxxxxxxxx"))
    with pytest.raises(pb.ProviderError) as e:
        OpenAIProvider().generate(req(), cfg)
    assert e.value.kind == "invalid_response"
    server.route("POST", "/v1/images/edits", lambda hd, b: (200, {}, b"<html>gateway</html>"))
    with pytest.raises(pb.ProviderError) as e:
        OpenAIProvider().generate(req(), cfg)
    assert e.value.kind == "invalid_response"


def test_timeout_and_network_errors():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)  # accepts but never answers
    port = s.getsockname()[1]
    cfg = providers.ProviderConfig(model="gpt-image-1", base_url=f"http://127.0.0.1:{port}/v1",
                                   api_key="sk-xxxxxxxxxxxx", timeout=0.5, max_retries=0)
    t = time.time()
    with pytest.raises(pb.ProviderError) as e:
        OpenAIProvider().generate(req(), cfg)
    assert e.value.kind == "timeout" and time.time() - t < 5
    s.close()
    cfg.base_url = "http://127.0.0.1:1/v1"
    with pytest.raises(pb.ProviderError) as e:
        OpenAIProvider().generate(req(), cfg)
    assert e.value.kind == "network"


def test_missing_key_fails_before_network():
    cfg = providers.ProviderConfig(model="gpt-image-1", base_url="http://127.0.0.1:1/v1", api_key="")
    for p in (OpenAIProvider(), StabilityProvider(), GeminiProvider()):
        with pytest.raises(pb.ProviderError) as e:
            p.generate(req(), cfg)
        assert e.value.kind == "auth" and not e.value.retryable


def test_cancel_during_retry_wait(server):
    server.route("POST", "/v1/images/edits", lambda hd, b: (503, {"Retry-After": "30"}, b"{}"))
    cfg = providers.ProviderConfig(model="gpt-image-1", base_url=server.url + "/v1", api_key="sk-xxxxxxxxxxxx",
                                   max_retries=3)
    ev = threading.Event()
    threading.Timer(0.3, ev.set).start()
    t = time.time()
    with pytest.raises(pb.ProviderError) as e:
        OpenAIProvider().generate(req(), cfg, cancel=ev)
    assert e.value.kind == "cancelled" and time.time() - t < 5


def test_multipart_encoding_roundtrip():
    body, ctype = h.encode_multipart([h.MultipartField("a", "1"), h.MultipartField("f", b"\x00\x01", "x.png", "image/png")])
    parts = parse_multipart({"content-type": ctype}, body)
    assert parts[0][0] == "a" and parts[0][3] == b"1"
    assert parts[1][1] == "x.png" and parts[1][3] == b"\x00\x01"


def test_mock_provider_hidden_unless_enabled(monkeypatch):
    from ai_texture_studio.providers import _REGISTRY, all_providers
    monkeypatch.setenv("ATS_ENABLE_TEST_PROVIDER", "0")
    _REGISTRY.clear()
    assert "MOCK" not in [p.id for p in all_providers()]
    monkeypatch.setenv("ATS_ENABLE_TEST_PROVIDER", "1")
    _REGISTRY.clear()
    assert "MOCK" in [p.id for p in all_providers()]
