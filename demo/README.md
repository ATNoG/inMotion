# inMotion Live RSSI Demo (Teacher + Child)

FastAPI-based local classroom demo with multi-child sessions, real-time RSSI streaming, and 10-sample sliding-window predictions.

## What this demo includes

- **Backend kept in FastAPI** with modular services:
  - `demo/app/session_store.py`: monitor and child session state
  - `demo/app/scanner_adapter.py`: live scanner + automatic replay fallback
  - `demo/app/inference.py`: `models/RandomForest.joblib` loading and prediction
  - `demo/app/event_hub.py`: event fan-out for WebSocket/SSE clients
- **Split local-only UI**:
  - `/teacher`: start/stop monitoring and roster view
  - `/child`: codename join flow, MAC detection, RSSI chart, latest prediction
- **Deterministic buffering**:
  - Per child rolling buffer of 10 RSSI samples
  - First prediction only at sample 10
  - Then prediction updates on each new sample (sliding window)
- **Offline resilience**:
  - Primary mode: live router scanner (`wavecom_files/wifi_scan.py`)
  - Fallback mode: replay from local dataset (`dataset_only_pure.csv`)

## Run with `uv`

From repository root:

```bash
uv run python demo/main.py
```

Open:

- Teacher view: `http://127.0.0.1:8000/teacher`
- Child view: `http://127.0.0.1:8000/child`

If port `8000` is busy:

```bash
uv run uvicorn main:app --app-dir demo --host 127.0.0.1 --port 8011
```

## Environment configuration

All scanner credentials are read from environment variables (no inline secrets):

- `INMOTION_ROUTER_IP`
- `INMOTION_ROUTER_USER` (default `root`)
- `INMOTION_ROUTER_PASS` (optional when key is used)
- `INMOTION_ROUTER_KEY_FILE` (optional)
- `INMOTION_ROUTER_PORT` (default `22`)
- `INMOTION_ROUTER_COMMAND_MODE` (`auto|ubus|system`, default `auto`)
- `INMOTION_SCAN_INTERVAL` (default `1.0` seconds)
- `INMOTION_MODEL_PATH` (default `models/RandomForest.joblib`)
- `INMOTION_FALLBACK_CSV` (default `dataset_only_pure.csv`)

Example:

```bash
export INMOTION_ROUTER_IP=10.20.0.1
export INMOTION_ROUTER_USER=admin
export INMOTION_ROUTER_PASS='***'
uv run python demo/main.py
```

## API contracts

### Child registration and identity

1. Detect MAC candidate:

```http
GET /api/child/detect-mac?codename=Alice
```

Response:

```json
{ "mac": "rp:63:84:e2:b2:18:4b", "source": "replay" }
```

2. Register child:

```http
POST /api/child/register
Content-Type: application/json

{"codename":"Alice","mac":"rp:63:84:e2:b2:18:4b"}
```

### Teacher controls

- `POST /api/teacher/start`
- `POST /api/teacher/stop`
- `GET /api/session/state`

### Compatibility endpoints

- `GET /health`
- `POST /predict` (single 10-value prediction request)
- `GET /events` (SSE event stream)

## Event contracts (canonical)

### RSSI event

```json
{
  "event_type": "rssi",
  "timestamp": "...",
  "session_id": "run-...",
  "child_id": "alice",
  "codename": "Alice",
  "mac": "...",
  "rssi": -53.0,
  "sample_count": 7,
  "window_size": 10,
  "source": "live"
}
```

### Prediction event

```json
{
  "event_type": "prediction",
  "timestamp": "...",
  "session_id": "run-...",
  "child_id": "alice",
  "codename": "Alice",
  "mac": "...",
  "predicted_class": "AB",
  "confidence": 0.71,
  "probabilities": { "AA": 0.0, "AB": 0.71, "BA": 0.01, "BB": 0.28 },
  "window": [-54, -55, -53, -52, -51, -50, -49, -50, -52, -53],
  "source": "model"
}
```

### Scanner status event

```json
{
  "event_type": "status",
  "timestamp": "...",
  "status": "connect|start|error|recover",
  "mode": "live|replay",
  "message": "..."
}
```

## Transport

- Teacher WebSocket: `/ws/teacher` (status + all children events + control messages)
- Child WebSocket: `/ws/child/{child_id}` (child-specific RSSI/prediction + global status)
- SSE fallback: `/events`

## Operator flow (demo runbook)

1. Start backend with `uv run python demo/main.py`.
2. Open teacher view and child view in separate tabs/devices.
3. In child view:
   - Enter codename
   - Wait MAC detection
   - Register
4. In teacher view, click **Start monitoring**.
5. Verify child RSSI chart starts moving.
6. After 10 samples, verify first prediction appears.
7. Keep monitoring to observe sliding-window prediction updates.
8. Click **Stop monitoring** when finished.

## Troubleshooting

- **Router unavailable / missing SSH tools**
  - Symptom: scanner mode changes to replay
  - Action: verify `ssh`/`sshpass`, credentials, and router reachability; replay mode still supports demo

- **No MAC detected in child view**
  - In live mode, ensure device is visible as a connected WiFi client
  - In fallback mode, MAC is generated deterministically from codename

- **Model load mismatch**
  - Symptom: `/health` returns `"model_degraded": true`
  - Action: ensure `models/RandomForest.joblib` exists and supports `predict_proba`

- **Port already in use**
  - Launch with an alternate port (`8011` example above)

## Notes

- Class output is constrained to `AA|AB|BA|BB`.
- Feature order is fixed as RSSI columns `1..10`.
- Frontend assets are fully local (`demo/static/*`), no CDN dependencies.
