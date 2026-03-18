from __future__ import annotations

import hashlib
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

StatusLiteral = Literal[
    "waiting_registration",
    "monitoring",
    "predicting",
    "disconnected",
    "fallback",
]


def utc_now() -> datetime:
    return datetime.now(UTC)


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return cleaned[:24] if cleaned else "child"


@dataclass
class ChildSession:
    child_id: str
    codename: str
    ip: str
    status: StatusLiteral = "waiting_registration"
    sample_buffer: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    latest_prediction: dict | None = None
    latest_rssi: float | None = None
    updated_at: datetime = field(default_factory=utc_now)

    def sample_count(self) -> int:
        return len(self.sample_buffer)


@dataclass
class MonitorSession:
    session_id: str = field(default_factory=lambda: f"run-{uuid4().hex[:8]}")
    teacher_active: bool = False
    mode: str = "idle"
    started_at: datetime | None = None
    stopped_at: datetime | None = None


class SessionStore:
    def __init__(self) -> None:
        self.monitor = MonitorSession()
        self.children_by_id: dict[str, ChildSession] = {}
        self.child_id_by_ip: dict[str, str] = {}

    def register_child(self, codename: str, ip: str) -> ChildSession:
        normalized_ip = ip.lower().strip()
        existing_id = self.child_id_by_ip.get(normalized_ip)
        if existing_id:
            child = self.children_by_id[existing_id]
            child.codename = codename.strip()
            child.updated_at = utc_now()
            return child

        base = _slug(codename)
        child_id = base
        idx = 2
        while child_id in self.children_by_id:
            child_id = f"{base}-{idx}"
            idx += 1

        child = ChildSession(child_id=child_id, codename=codename.strip(), ip=normalized_ip)
        self.children_by_id[child_id] = child
        self.child_id_by_ip[normalized_ip] = child_id
        return child

    def deterministic_replay_ip(self, codename: str) -> str:
        digest = hashlib.md5(codename.strip().lower().encode("utf-8")).digest()
        return f"10.255.{digest[0]}.{digest[1]}"

    def start_teacher_session(self, mode: str) -> None:
        self.monitor.teacher_active = True
        self.monitor.mode = mode
        self.monitor.started_at = utc_now()
        self.monitor.stopped_at = None
        for child in self.children_by_id.values():
            child.sample_buffer.clear()
            child.latest_prediction = None
            child.latest_rssi = None
            child.status = "monitoring"
            child.updated_at = utc_now()

    def stop_teacher_session(self) -> None:
        self.monitor.teacher_active = False
        self.monitor.stopped_at = utc_now()
        for child in self.children_by_id.values():
            child.status = "disconnected"
            child.updated_at = utc_now()

    def set_mode(self, mode: str) -> None:
        self.monitor.mode = mode

    def get_child_by_ip(self, ip: str) -> ChildSession | None:
        child_id = self.child_id_by_ip.get(ip.lower().strip())
        if not child_id:
            return None
        return self.children_by_id.get(child_id)

    def get_registered_ips(self) -> list[str]:
        return list(self.child_id_by_ip.keys())

    def ingest_rssi(self, ip: str, rssi: float) -> tuple[ChildSession | None, list[float] | None]:
        child = self.get_child_by_ip(ip)
        if child is None:
            return None, None
        if child.latest_prediction is not None:
            return None, None
        child.sample_buffer.append(float(rssi))
        child.latest_rssi = float(rssi)
        child.updated_at = utc_now()
        if len(child.sample_buffer) < 10:
            child.status = "monitoring"
            return child, None
        child.status = "predicting"
        return child, list(child.sample_buffer)

    def set_prediction(self, child_id: str, prediction: dict) -> None:
        child = self.children_by_id.get(child_id)
        if not child:
            return
        child.latest_prediction = prediction
        child.status = "predicting"
        child.updated_at = utc_now()

    def as_dict(self) -> dict:
        return {
            "monitor": {
                "session_id": self.monitor.session_id,
                "teacher_active": self.monitor.teacher_active,
                "mode": self.monitor.mode,
                "started_at": self.monitor.started_at,
                "stopped_at": self.monitor.stopped_at,
            },
            "children": [
                {
                    "child_id": child.child_id,
                    "codename": child.codename,
                    "ip": child.ip,
                    "status": child.status,
                    "sample_count": child.sample_count(),
                    "latest_rssi": child.latest_rssi,
                    "latest_prediction": child.latest_prediction,
                    "updated_at": child.updated_at,
                }
                for child in sorted(self.children_by_id.values(), key=lambda c: c.codename)
            ],
        }
