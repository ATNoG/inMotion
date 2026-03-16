from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

StatusLiteral = Literal[
    "waiting_registration",
    "monitoring",
    "predicting",
    "disconnected",
    "fallback",
]


class ChildRegisterRequest(BaseModel):
    codename: str = Field(min_length=1, max_length=32)
    mac: str = Field(min_length=5, max_length=32)


class ChildSessionView(BaseModel):
    child_id: str
    codename: str
    mac: str
    status: StatusLiteral
    sample_count: int
    latest_rssi: float | None
    latest_prediction: dict | None
    updated_at: datetime


class MonitorSessionView(BaseModel):
    session_id: str
    teacher_active: bool
    mode: str
    started_at: datetime | None
    stopped_at: datetime | None


class SessionStateResponse(BaseModel):
    monitor: MonitorSessionView
    children: list[ChildSessionView]


class TeacherControlResponse(BaseModel):
    monitor: MonitorSessionView


class DetectMacResponse(BaseModel):
    mac: str | None
    source: Literal["live", "replay", "none"]


class RSSIEvent(BaseModel):
    event_type: Literal["rssi"] = "rssi"
    timestamp: datetime
    session_id: str
    child_id: str
    codename: str
    mac: str
    rssi: float
    sample_count: int
    window_size: int
    source: Literal["live", "replay"]


class PredictionEvent(BaseModel):
    event_type: Literal["prediction"] = "prediction"
    timestamp: datetime
    session_id: str
    child_id: str
    codename: str
    mac: str
    predicted_class: Literal["AA", "AB", "BA", "BB"]
    confidence: float
    probabilities: dict[str, float]
    window: list[float]
    source: Literal["model", "degraded"]


class StatusEvent(BaseModel):
    event_type: Literal["status"] = "status"
    timestamp: datetime
    status: Literal["connect", "start", "error", "recover"]
    mode: Literal["live", "replay"]
    message: str
