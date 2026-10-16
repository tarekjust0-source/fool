"""Minimal, dependency-free HTTP client for provider APIs.

* urllib only (ships with Blender's Python); honours HTTPS_PROXY etc.
* uses ``certifi`` CA bundle when available (Blender bundles it).
* multipart/form-data encoder.
* retries with exponential back-off for 429 / 5xx / connection errors,
  respecting ``Retry-After``.
* never logs request headers; error messages are redacted.
"""

from __future__ import annotations

import json
import mimetypes
import socket
import ssl
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field

from .secrets import redact

MAX_RESPONSE_BYTES = 96 * 1024 * 1024
USER_AGENT = "AI-Texture-Studio-Blender/1.0"


class HTTPError(Exception):
    def __init__(self, status: int | None, message: str, *, body: bytes = b"", headers=None, kind: str = "http"):
        super().__init__(redact(message))
        self.status = status
        self.body = body
        self.headers = dict(headers or {})
        self.kind = kind  # http | network | timeout | too_large | cancelled

    def json(self):
        try:
            return json.loads(self.body.decode("utf-8", "replace"))
        except Exception:
            return None

    def text(self, limit: int = 600) -> str:
        return redact(self.body[:limit].decode("utf-8", "replace"))


@dataclass
class Response:
    status: int
    headers: dict
    body: bytes

    def json(self):
        try:
            return json.loads(self.body.decode("utf-8"))
        except Exception as exc:
            raise HTTPError(self.status, f"Invalid JSON in response: {exc}", body=self.body[:2000],
                            kind="invalid") from None

    @property
    def content_type(self) -> str:
        return (self.headers.get("content-type") or "").split(";")[0].strip().lower()


@dataclass
class MultipartField:
    name: str
    value: bytes | str
    filename: str | None = None
    content_type: str | None = None


def encode_multipart(fields: list[MultipartField]):
    boundary = "----ATS" + uuid.uuid4().hex
    parts = []
    for f in fields:
        disp = f'form-data; name="{f.name}"'
        if f.filename is not None:
            disp += f'; filename="{f.filename}"'
        head = f"--{boundary}\r\nContent-Disposition: {disp}\r\n"
        if f.filename is not None:
            ct = f.content_type or mimetypes.guess_type(f.filename)[0] or "application/octet-stream"
            head += f"Content-Type: {ct}\r\n"
        head += "\r\n"
        val = f.value.encode("utf-8") if isinstance(f.value, str) else f.value
        parts.append(head.encode("utf-8") + val + b"\r\n")
    body = b"".join(parts) + f"--{boundary}--\r\n".encode("utf-8")
    return body, f"multipart/form-data; boundary={boundary}"


def _ssl_context():
    try:
        import certifi  # noqa: WPS433
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


@dataclass
class RetryPolicy:
    max_retries: int = 2
    base_delay: float = 2.0
    max_delay: float = 30.0
    retry_statuses: tuple = (408, 429, 500, 502, 503, 504)
    sleep: object = field(default=time.sleep, repr=False)


def request(method: str, url: str, *, headers: dict | None = None, body: bytes | None = None,
            timeout: float = 120.0, retry: RetryPolicy | None = None, cancel=None) -> Response:
    """Perform an HTTP request with retries. Raises HTTPError on failure."""
    retry = retry or RetryPolicy()
    hdrs = {"User-Agent": USER_AGENT}
    hdrs.update(headers or {})
    attempt = 0
    while True:
        if cancel is not None and cancel.is_set():
            raise HTTPError(None, "Cancelled", kind="cancelled")
        try:
            return _once(method, url, hdrs, body, timeout)
        except HTTPError as err:
            retryable = (err.kind in ("network", "timeout")) or (err.status in retry.retry_statuses)
            if not retryable or attempt >= retry.max_retries:
                raise
            delay = min(retry.max_delay, retry.base_delay * (2 ** attempt))
            ra = err.headers.get("retry-after")
            if ra:
                try:
                    delay = min(retry.max_delay, max(delay, float(ra)))
                except ValueError:
                    pass
            attempt += 1
            # Sleep in small steps so cancellation stays responsive.
            end = time.monotonic() + delay
            while time.monotonic() < end:
                if cancel is not None and cancel.is_set():
                    raise HTTPError(None, "Cancelled", kind="cancelled") from None
                retry.sleep(min(0.25, max(0.0, end - time.monotonic())))


def _once(method, url, headers, body, timeout) -> Response:
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    ctx = _ssl_context() if url.lower().startswith("https") else None
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            data = resp.read(MAX_RESPONSE_BYTES + 1)
            if len(data) > MAX_RESPONSE_BYTES:
                raise HTTPError(resp.status, "Response too large", kind="too_large")
            return Response(resp.status, {k.lower(): v for k, v in resp.headers.items()}, data)
    except urllib.error.HTTPError as e:
        try:
            data = e.read(1024 * 1024)
        except Exception:
            data = b""
        hdrs = {k.lower(): v for k, v in (e.headers.items() if e.headers else [])}
        raise HTTPError(e.code, f"HTTP {e.code} {e.reason}", body=data, headers=hdrs) from None
    except urllib.error.URLError as e:
        reason = e.reason
        if isinstance(reason, (socket.timeout, TimeoutError)):
            raise HTTPError(None, "Request timed out", kind="timeout") from None
        raise HTTPError(None, f"Network error: {reason}", kind="network") from None
    except (socket.timeout, TimeoutError):
        raise HTTPError(None, "Request timed out", kind="timeout") from None
    except (ConnectionError, OSError) as e:
        raise HTTPError(None, f"Network error: {e}", kind="network") from None


def post_json(url, payload, *, headers=None, **kw) -> Response:
    h = {"Content-Type": "application/json"}
    h.update(headers or {})
    return request("POST", url, headers=h, body=json.dumps(payload).encode("utf-8"), **kw)


def post_multipart(url, fields: list[MultipartField], *, headers=None, **kw) -> Response:
    body, ctype = encode_multipart(fields)
    h = {"Content-Type": ctype}
    h.update(headers or {})
    return request("POST", url, headers=h, body=body, **kw)


def get(url, *, headers=None, **kw) -> Response:
    return request("GET", url, headers=headers, **kw)
