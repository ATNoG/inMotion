from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from app.config import load_config
from app.contracts import (
    ChildRegisterRequest,
    ChildSessionView,
    DetectMacResponse,
    MonitorSessionView,
    PredictionEvent,
    RSSIEvent,
    SessionStateResponse,
    StatusEvent,
    TeacherControlResponse,
)
from app.event_hub import EventHub
from app.inference import InferenceService
from app.scanner_adapter import ScannerAdapter, ScannerEvent
from app.session_store import SessionStore
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    rssi_values: list[float] = Field(min_length=10, max_length=10)


config = load_config()
hub = EventHub()
store = SessionStore()
inference = InferenceService(config)
scanner = ScannerAdapter(config)
state_lock = asyncio.Lock()


def _monitor_view() -> MonitorSessionView:
    monitor = store.monitor
    return MonitorSessionView(
        session_id=monitor.session_id,
        teacher_active=monitor.teacher_active,
        mode=monitor.mode,
        started_at=monitor.started_at,
        stopped_at=monitor.stopped_at,
    )


def _child_view(child) -> ChildSessionView:
    return ChildSessionView(
        child_id=child.child_id,
        codename=child.codename,
        mac=child.mac,
        status=child.status,
        sample_count=child.sample_count(),
        latest_rssi=child.latest_rssi,
        latest_prediction=child.latest_prediction,
        updated_at=child.updated_at,
    )


def _session_state() -> SessionStateResponse:
    return SessionStateResponse(
        monitor=_monitor_view(),
        children=[_child_view(child) for child in store.children_by_id.values()],
    )


async def _publish(event: dict) -> None:
    await hub.publish(event)


async def _handle_status(payload: dict) -> None:
    async with state_lock:
        mode = payload.get("mode", "replay")
        store.set_mode(mode)
        if mode == "replay" and store.monitor.teacher_active:
            for child in store.children_by_id.values():
                if child.status in ("monitoring", "predicting"):
                    child.status = "fallback"
                    child.updated_at = datetime.now(UTC)

    status_event = StatusEvent.model_validate(payload)
    await _publish(status_event.model_dump(mode="json"))


async def _handle_rssi(event: ScannerEvent) -> None:
    async with state_lock:
        if not store.monitor.teacher_active:
            return

        child, window = store.ingest_rssi(event.mac, event.rssi)
        if child is None:
            return

        rssi_event = RSSIEvent(
            timestamp=event.timestamp,
            session_id=store.monitor.session_id,
            child_id=child.child_id,
            codename=child.codename,
            mac=child.mac,
            rssi=event.rssi,
            sample_count=child.sample_count(),
            window_size=config.window_size,
            source=event.source,
        )

    await _publish(rssi_event.model_dump(mode="json"))

    if window is None:
        return

    prediction = inference.predict(window)
    async with state_lock:
        store.set_prediction(child.child_id, prediction)

    pred_event = PredictionEvent(
        timestamp=prediction["timestamp"],
        session_id=store.monitor.session_id,
        child_id=child.child_id,
        codename=child.codename,
        mac=child.mac,
        predicted_class=prediction["predicted_class"],
        confidence=prediction["confidence"],
        probabilities=prediction["probabilities"],
        window=window,
        source=prediction["source"],
    )
    await _publish(pred_event.model_dump(mode="json"))


async def _lifespan_startup() -> None:
    inference.startup()
    await scanner.start(_handle_rssi, _handle_status, store.get_registered_macs)


async def _lifespan_shutdown() -> None:
    await scanner.stop()


app = FastAPI(
    title="inMotion RSSI Demo",
    version="2.0.0",
    on_startup=[_lifespan_startup],
    on_shutdown=[_lifespan_shutdown],
)
static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/teacher")


@app.get("/teacher")
def teacher_page() -> FileResponse:
    return FileResponse(static_dir / "teacher.html")


@app.get("/child")
def child_page() -> FileResponse:
    return FileResponse(static_dir / "child.html")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "session_id": store.monitor.session_id,
        "scanner_mode": scanner.mode,
        "model_degraded": inference.degraded,
        "classes": inference.classes,
        "window_size": config.window_size,
    }


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    prediction = inference.predict(req.rssi_values)
    return {
        "predicted_class": prediction["predicted_class"],
        "confidence": prediction["confidence"],
        "class_probabilities": prediction["probabilities"],
        "rssi_input": req.rssi_values,
        "source": prediction["source"],
        "timestamp": prediction["timestamp"],
    }


@app.get("/api/session/state", response_model=SessionStateResponse)
async def get_state() -> SessionStateResponse:
    async with state_lock:
        return _session_state()


@app.get("/api/child/detect-mac", response_model=DetectMacResponse)
async def detect_mac(codename: str = Query(default="child")) -> DetectMacResponse:
    async with state_lock:
        assigned = set(store.get_registered_macs())
        detected = scanner.detect_unassigned_mac(assigned)
        if detected:
            return DetectMacResponse(mac=detected, source="live")

        if scanner.mode == "replay":
            return DetectMacResponse(mac=store.deterministic_replay_mac(codename), source="replay")

        return DetectMacResponse(mac=None, source="none")


@app.post("/api/child/register", response_model=ChildSessionView)
async def register_child(req: ChildRegisterRequest) -> ChildSessionView:
    async with state_lock:
        if not req.mac.strip():
            raise HTTPException(status_code=400, detail="MAC is required")

        child = store.register_child(req.codename, req.mac)
        if store.monitor.teacher_active:
            child.status = "monitoring"
            child.updated_at = datetime.now(UTC)

        child_view = _child_view(child)

    await _publish(
        {
            "event_type": "status",
            "timestamp": datetime.now(UTC).isoformat(),
            "status": "recover",
            "mode": scanner.mode,
            "message": f"Child {child.codename} registered ({child.mac})",
        }
    )
    return child_view


@app.post("/api/teacher/start", response_model=TeacherControlResponse)
async def teacher_start() -> TeacherControlResponse:
    async with state_lock:
        store.start_teacher_session(scanner.mode)
        monitor = _monitor_view()

    await _publish(
        StatusEvent(
            timestamp=datetime.now(UTC),
            status="start",
            mode="live" if scanner.mode == "live" else "replay",
            message="Teacher started monitoring",
        ).model_dump(mode="json")
    )
    return TeacherControlResponse(monitor=monitor)


@app.post("/api/teacher/stop", response_model=TeacherControlResponse)
async def teacher_stop() -> TeacherControlResponse:
    async with state_lock:
        store.stop_teacher_session()
        monitor = _monitor_view()

    await _publish(
        StatusEvent(
            timestamp=datetime.now(UTC),
            status="recover",
            mode="live" if scanner.mode == "live" else "replay",
            message="Teacher stopped monitoring",
        ).model_dump(mode="json")
    )
    return TeacherControlResponse(monitor=monitor)


async def _send_stream_events(websocket: WebSocket, child_id: str | None) -> None:
    async for event in hub.subscribe():
        if (
            child_id
            and event.get("event_type") in {"rssi", "prediction"}
            and event.get("child_id") != child_id
        ):
            continue
        await websocket.send_json(event)


@app.websocket("/ws/teacher")
async def ws_teacher(websocket: WebSocket) -> None:
    await websocket.accept()
    await websocket.send_json(_session_state().model_dump(mode="json"))
    sender_task = asyncio.create_task(_send_stream_events(websocket, None))
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue

            action = payload.get("action")
            if action == "start":
                await teacher_start()
            elif action == "stop":
                await teacher_stop()
    except WebSocketDisconnect:
        pass
    finally:
        sender_task.cancel()


@app.websocket("/ws/child/{child_id}")
async def ws_child(websocket: WebSocket, child_id: str) -> None:
    await websocket.accept()
    sender_task = asyncio.create_task(_send_stream_events(websocket, child_id))
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        sender_task.cancel()


@app.get("/events")
async def sse_events() -> StreamingResponse:
    async def stream():
        async for event in hub.subscribe():
            payload = json.dumps(event, default=str)
            yield f"data: {payload}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
