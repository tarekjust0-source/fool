"""Background job runner (no bpy).

A job runs a worker function in a daemon thread.  The worker reports
progress and emits events (e.g. "a view finished") into a thread-safe queue;
the Blender bridge drains that queue on the main thread with a timer and
applies scene changes there, as Blender's API requires.

For automated tests ``synchronous=True`` runs the worker inline.
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field

from .secrets import redact


class Cancelled(Exception):
    pass


@dataclass
class Job:
    kind: str
    title: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    status: str = "QUEUED"   # QUEUED RUNNING DONE FAILED CANCELLED
    progress: float = 0.0
    message: str = ""
    error: str = ""
    error_kind: str = ""
    retryable: bool = False
    started: float = 0.0
    finished: float = 0.0
    events: "queue.Queue" = field(default_factory=queue.Queue, repr=False)
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    context: dict = field(default_factory=dict, repr=False)   # bridge-owned data
    result: object = None

    # -- worker side ---------------------------------------------------
    def set_progress(self, value: float, message: str | None = None):
        self.progress = max(0.0, min(1.0, float(value)))
        if message is not None:
            self.message = redact(message)

    def emit(self, kind: str, payload=None):
        self.events.put((kind, payload))

    def check_cancel(self):
        if self.cancel_event.is_set():
            raise Cancelled()

    # -- control side ----------------------------------------------------
    def cancel(self):
        self.cancel_event.set()

    @property
    def active(self) -> bool:
        return self.status in ("QUEUED", "RUNNING")

    def drain(self, limit: int = 100):
        out = []
        for _ in range(limit):
            try:
                out.append(self.events.get_nowait())
            except queue.Empty:
                break
        return out


def run_job(job: Job, worker, *, synchronous: bool = False):
    """Start ``worker(job)``; returns the thread (or None when synchronous)."""

    def target():
        job.status = "RUNNING"
        job.started = time.time()
        try:
            job.result = worker(job)
            if job.cancel_event.is_set():
                job.status = "CANCELLED"
                job.message = "Cancelled"
            else:
                job.status = "DONE"
                job.progress = 1.0
        except Cancelled:
            job.status = "CANCELLED"
            job.message = "Cancelled"
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            from ..providers.base import ProviderError
            job.status = "FAILED"
            if isinstance(exc, ProviderError):
                job.error = redact(exc.user_message())
                job.error_kind = exc.kind
                job.retryable = exc.retryable
                if exc.kind == "cancelled":
                    job.status = "CANCELLED"
                    job.message = "Cancelled"
            else:
                job.error = redact(f"{type(exc).__name__}: {exc}")
                job.error_kind = "internal"
                job.retryable = True
                job.context["traceback"] = redact(traceback.format_exc())
        finally:
            job.finished = time.time()
            job.emit("finished", None)

    if synchronous:
        target()
        return None
    t = threading.Thread(target=target, name=f"ATS-{job.kind}-{job.id}", daemon=True)
    t.start()
    return t
