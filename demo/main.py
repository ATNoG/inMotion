from __future__ import annotations

import asyncio
import csv
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field


# Config
# gen slop TODO
MODELS_DIR = Path("models")
_MODEL_CANDIDATES = ["model.joblib", "model.pkl", "classifier.joblib", "classifier.pkl"]
#
WIFI_SCAN_SCRIPT = Path.home() / "Desktop/inMotion/wavecom_files/wifi_scan.py"
EXPORT_SCRIPT    = Path.home() / "Desktop/inMotion/export.py"
ROUTER_IP        = "10.20.0.1"
ROUTER_USER      = "admin"
ROUTER_PASS      = "..."
SCAN_INTERVAL    = 1
DATA_DIR         = Path("demo_data")
ETHERNET_IFACE   = "eth0"       # ._.


# Model loading
MOCK_MODE = False
def _load_model():
    global MOCK_MODE
    for name in _MODEL_CANDIDATES:
        path = MODELS_DIR / name
        if path.exists():
            print(f"[inMotion] Loading model from {path}")
            return joblib.load(path)
    print("[inMotion] No model found — running in MOCK mode.")
    MOCK_MODE = True
    return None


model = _load_model()

# ._.

CLASSES: list[str] = (
    list(model.classes_) if (model and hasattr(model, "classes_")) else ["AA", "AB", "BA", "BB"]
)


# FastAPI
app = FastAPI(title="inMotion Demo", version="0.1.0")
DATA_DIR.mkdir(exist_ok=True)


class PredictRequest(BaseModel):
    rssi_values: list[float] = Field(..., min_length=10, max_length=10)


class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float
    class_probabilities: dict[str, float]
    rssi_input: list[float]


@app.get("/health")
def health():
    return {"status": "ok", "classes": CLASSES, "mock": MOCK_MODE}


# slop TODO
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if MOCK_MODE:
        mean_rssi = float(np.mean(req.rssi_values))
        rng = np.random.default_rng(seed=int(abs(mean_rssi) * 100) % (2**31))
        proba = rng.dirichlet(alpha=[4, 1, 1, 1])
    else:
        X = np.array(req.rssi_values).reshape(1, -1)
        proba = model.predict_proba(X)[0]

    predicted_idx = int(np.argmax(proba))
    predicted_class = CLASSES[predicted_idx]
    class_probs = {cls: round(float(p), 4) for cls, p in zip(CLASSES, proba)}
    return PredictResponse(
        predicted_class=predicted_class,
        confidence=round(float(proba[predicted_idx]), 4),
        class_probabilities=class_probs,
        rssi_input=req.rssi_values,
    )


# slop
# ARP keepalive helper
# async def _arping_keepalive(mac: str, stop_event: asyncio.Event):
#     while not stop_event.is_set():
#         try:
#             await asyncio.create_subprocess_exec(
#                 "arping", "-c", "1", "-I", ETHERNET_IFACE, mac,
#                 stdout=asyncio.subprocess.DEVNULL,
#                 stderr=asyncio.subprocess.DEVNULL,
#             )
#         except Exception:
#             pass
#         await asyncio.sleep(2)



@app.get("/capture")
async def capture(mac: str | None = None):
    async def run():
        prefix = str(DATA_DIR / "demo_capture")
        mac_info = f" [{mac}]" if mac else " [auto]"

# test
        if MOCK_MODE:
            yield "data: [MOCK] Simulating WiFi scan...\n\n"
            for i in range(1, 11):
                yield f"data: Reading {i}/10...\n\n"
                await asyncio.sleep(0.3)

            fake_csv = DATA_DIR / "demo_capture.csv"
            rng = np.random.default_rng()
            rssi = rng.uniform(-70, -30, 10).tolist()
            with open(fake_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["mac","1","2","3","4","5","6","7","8","9","10","label","noise","concurrent_noise_path"])
                writer.writerow(["00:00:00:00:00:00"] + rssi + ["", "False", "None"])
            yield "data: Scan complete.\n\n"

# slop TODO
        else:
            yield f"data: Starting WiFi scan{mac_info}...\n\n"
            scan_cmd = [
                "python3", str(WIFI_SCAN_SCRIPT),
                ROUTER_IP, "-u", ROUTER_USER, "-p", ROUTER_PASS,
                "-i", str(SCAN_INTERVAL), "-f", prefix,
            ]
            if mac:
                scan_cmd += ["--mac", mac]

            stop_arping = asyncio.Event()
            arping_task = None
            # if mac:
            #     arping_task = asyncio.create_task(_arping_keepalive(mac, stop_arping))

            try:
                proc = await asyncio.create_subprocess_exec(
                    *scan_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                async for line in proc.stdout:
                    text = line.decode().rstrip()
                    if text:
                        yield f"data: {text}\n\n"
                await proc.wait()
            finally:
                stop_arping.set()
                # if arping_task:
                #     await arping_task

            if proc.returncode != 0:
                yield "data: ERROR: wifi_scan.py failed.\n\n"
                yield "data: __ERROR__\n\n"
                return

            yield "data: Running export pipeline...\n\n"
            export_proc = await asyncio.create_subprocess_exec(
                "python3", str(EXPORT_SCRIPT), prefix,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            async for line in export_proc.stdout:
                text = line.decode().rstrip()
                if text:
                    yield f"data: {text}\n\n"
            await export_proc.wait()
            if export_proc.returncode != 0:
                yield "data: ERROR: export.py failed.\n\n"
                yield "data: __ERROR__\n\n"
                return

        # Read CSV
        csv_path = DATA_DIR / "demo_capture.csv"
        if not csv_path.exists():
            candidates = sorted(DATA_DIR.glob("*.csv"))
            if not candidates:
                yield "data: ERROR: No CSV found.\n\n"
                yield "data: __ERROR__\n\n"
                return
            csv_path = candidates[-1]

        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))

        if not rows:
            yield "data: ERROR: CSV is empty.\n\n"
            yield "data: __ERROR__\n\n"
            return

        if mac:
            filtered = [r for r in rows if r.get("mac", "").lower() == mac.lower()]
            if filtered:
                rows = filtered

        rssi_cols = ["1","2","3","4","5","6","7","8","9","10"]
        try:
            row = rows[0]
            rssi_values = [float(row[c]) for c in rssi_cols if c in row]
            if len(rssi_values) != 10:
                yield "data: ERROR: Expected 10 RSSI columns.\n\n"
                yield "data: __ERROR__\n\n"
                return
        except Exception as e:
            yield f"data: ERROR parsing CSV: {e}\n\n"
            yield "data: __ERROR__\n\n"
            return

        yield f"data: __RSSI__:{','.join(str(round(v, 2)) for v in rssi_values)}\n\n"
        yield "data: __DONE__\n\n"

    return StreamingResponse(
        run(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )



# UI 
HTML = """<!DOCTYPE html>
<html lang="pt">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>inMotion Demo</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;700&family=Syne:wght@700;800&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/qrcodejs/1.0.0/qrcode.min.js"></script>
<style>
  :root {
    --bg:       #f4f3ef;
    --surface:  #ffffff;
    --border:   #e2e0d8;
    --accent:   #1a6b4a;
    --accent-l: #e8f4ee;
    --danger:   #c0392b;
    --warn:     #d4860a;
    --text:     #1a1a18;
    --muted:    #8a8880;
    --AA: #1a6b4a; --AB: #1a5a8a; --BA: #8a6b00; --BB: #8a2020;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }

  .mock-banner {
    background: #fff8e6;
    border-bottom: 2px solid var(--warn);
    color: var(--warn);
    text-align: center;
    padding: 0.5rem;
    font-size: 0.75rem;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.1em;
    display: none;
  }

  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.25rem 2.5rem;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
  }
  .logo { font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 800; letter-spacing: -0.02em; color: var(--text); }
  .logo span { color: var(--accent); }
  .live-badge {
    display: flex; align-items: center; gap: 0.5rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem; color: var(--muted); letter-spacing: 0.12em;
  }
  .live-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-l);
    animation: pulse 2s ease-in-out infinite;
  }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

  main {
    flex: 1;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
    padding: 2rem 2.5rem;
    max-width: 1400px;
    margin: 0 auto;
    width: 100%;
    align-items: start;
  }

  /* ---- Left column ---- */
  .left-col { display: flex; flex-direction: column; gap: 1.5rem; }

  /* Result panel */
  .result-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 320px;
    position: relative;
    overflow: hidden;
  }
  .result-panel::after {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(26,107,74,0.06) 0%, transparent 70%);
    pointer-events: none;
  }
  .result-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem; letter-spacing: 0.2em; text-transform: uppercase;
    color: var(--muted); margin-bottom: 0.75rem;
  }
  .result-class {
    font-family: 'Syne', sans-serif;
    font-size: clamp(5rem, 12vw, 9rem);
    font-weight: 800;
    line-height: 1;
    letter-spacing: -0.04em;
    color: var(--accent);
    transition: color 0.3s;
  }
  .result-class.updated { animation: pop 0.45s cubic-bezier(0.16, 1, 0.3, 1); }
  @keyframes pop { 0%{transform:scale(0.75);opacity:0} 100%{transform:scale(1);opacity:1} }
  .result-confidence {
    margin-top: 1rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem; color: var(--muted); display: none;
  }
  .result-confidence strong { color: var(--text); }
  .idle-hint { font-size: 0.9rem; color: var(--muted); text-align: center; line-height: 2; }

  /* QR panel */
  .wifi-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.75rem;
  }
  .panel-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem; letter-spacing: 0.18em; text-transform: uppercase;
    color: var(--muted); margin-bottom: 1.25rem;
  }
  .wifi-inner { display: flex; align-items: center; gap: 1.5rem; }
  #qrcode canvas, #qrcode img { border-radius: 8px; display: block; border: 4px solid var(--bg); }
  .wifi-creds { flex: 1; }
  .wifi-row { margin-bottom: 0.75rem; }
  .wifi-key {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem; letter-spacing: 0.15em; text-transform: uppercase;
    color: var(--muted); display: block; margin-bottom: 0.2rem;
  }
  .wifi-val { font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 700; color: var(--text); }

  /* ---- Right column ---- */
  .right-col { display: flex; flex-direction: column; gap: 1.5rem; }

  /* Confidence bars */
  .bars-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.75rem;
  }
  .bar-row { display: flex; align-items: center; gap: 0.9rem; margin-bottom: 1rem; }
  .bar-class { font-family: 'Syne', sans-serif; font-weight: 700; font-size: 0.95rem; width: 2.5rem; flex-shrink: 0; }
  .bar-track { flex: 1; height: 10px; background: var(--bg); border-radius: 99px; overflow: hidden; border: 1px solid var(--border); }
  .bar-fill { height: 100%; border-radius: 99px; width: 0%; transition: width 0.8s cubic-bezier(0.16, 1, 0.3, 1); }
  .bar-pct { font-family: 'DM Mono', monospace; font-size: 0.8rem; color: var(--muted); width: 3.5rem; text-align: right; flex-shrink: 0; }

  /* Capture panel */
  .capture-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.75rem;
  }

  .mac-row { margin-bottom: 1rem; display: flex; flex-direction: column; gap: 0.35rem; }
  .mac-row label {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem; letter-spacing: 0.15em; text-transform: uppercase; color: var(--muted);
  }
  .mac-row .optional { text-transform: none; letter-spacing: 0; opacity: 0.7; }
  .mac-row input {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
    font-family: 'DM Mono', monospace;
    font-size: 0.9rem;
    padding: 0.6rem 0.8rem;
    width: 100%;
    transition: border-color 0.2s, box-shadow 0.2s;
  }
  .mac-row input:focus { outline: none; border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-l); }
  .mac-row input::placeholder { color: var(--muted); opacity: 0.5; }

  .btn-row { display: flex; gap: 0.75rem; }

  button {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem; font-weight: 700;
    border: none; border-radius: 8px;
    padding: 0.75rem 1.25rem;
    cursor: pointer;
    transition: all 0.15s;
  }

  .btn-capture {
    background: var(--accent); color: #fff;
    flex: 1;
  }
  .btn-capture:hover { filter: brightness(1.1); transform: translateY(-1px); box-shadow: 0 4px 12px rgba(26,107,74,0.25); }
  .btn-capture:active { transform: translateY(0); }
  .btn-capture:disabled { opacity: 0.45; cursor: not-allowed; transform: none; box-shadow: none; }

  .btn-predict {
    background: var(--text); color: #fff;
    flex: 1; display: none;
  }
  .btn-predict:hover { background: #333; transform: translateY(-1px); }
  .btn-predict:disabled { opacity: 0.45; cursor: not-allowed; transform: none; }

  /* RSSI values display */
  .rssi-display {
    display: none;
    margin: 1rem 0;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
  }
  .rssi-display-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem; letter-spacing: 0.15em; text-transform: uppercase;
    color: var(--muted); margin-bottom: 0.75rem;
  }
  .rssi-chips {
    display: flex; flex-wrap: wrap; gap: 0.4rem;
  }
  .rssi-chip {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.3rem 0.6rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: var(--text);
  }
  .rssi-chip span { color: var(--muted); font-size: 0.65rem; margin-right: 0.2rem; }

  /* Log box */
  .log-box {
    margin-top: 1rem;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.9rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    line-height: 1.7;
    color: var(--muted);
    max-height: 140px;
    overflow-y: auto;
    display: none;
  }
  .log-line.ok { color: var(--accent); }
  .log-line.err { color: var(--danger); }

  @media (max-width: 900px) {
    main { grid-template-columns: 1fr; }
    .result-panel { min-height: 220px; }
  }
</style>
</head>
<body>

<div id="mockBanner" class="mock-banner">⚠ MOCK MODE — no model loaded, results are simulated</div>

<header>
  <div class="logo">in<span>Motion</span></div>
  <div class="live-badge"><div class="live-dot"></div>LIVE</div>
</header>

<main>
  <!-- Left column -->
  <div class="left-col">

    <div class="result-panel">
      <div class="result-label">Predicted Position</div>
      <div class="result-class" id="resultClass">—</div>
      <div class="result-confidence" id="resultConf">Confidence: <strong id="confValue"></strong></div>
      <div class="idle-hint" id="idleHint">Capture RSSI data<br>then press Predict</div>
    </div>

    <div class="wifi-panel">
      <div class="panel-title">Connect to Device</div>
      <div class="wifi-inner">
        <div id="qrcode"></div>
        <div class="wifi-creds">
          <div class="wifi-row">
            <span class="wifi-key">Network</span>
            <span class="wifi-val" id="wifiSsid">inMotion-Demo</span>
          </div>
          <div class="wifi-row">
            <span class="wifi-key">Password</span>
            <span class="wifi-val" id="wifiPass">demo1234</span>
          </div>
        </div>
      </div>
    </div>

  </div>

  <!-- Right column -->
  <div class="right-col">

    <div class="bars-panel">
      <div class="panel-title">Class Probabilities</div>
      <div class="bar-row">
        <div class="bar-class" style="color:var(--AA)">AA</div>
        <div class="bar-track"><div class="bar-fill" id="bar-AA" style="background:var(--AA)"></div></div>
        <div class="bar-pct" id="pct-AA">—</div>
      </div>
      <div class="bar-row">
        <div class="bar-class" style="color:var(--AB)">AB</div>
        <div class="bar-track"><div class="bar-fill" id="bar-AB" style="background:var(--AB)"></div></div>
        <div class="bar-pct" id="pct-AB">—</div>
      </div>
      <div class="bar-row">
        <div class="bar-class" style="color:var(--BA)">BA</div>
        <div class="bar-track"><div class="bar-fill" id="bar-BA" style="background:var(--BA)"></div></div>
        <div class="bar-pct" id="pct-BA">—</div>
      </div>
      <div class="bar-row">
        <div class="bar-class" style="color:var(--BB)">BB</div>
        <div class="bar-track"><div class="bar-fill" id="bar-BB" style="background:var(--BB)"></div></div>
        <div class="bar-pct" id="pct-BB">—</div>
      </div>
    </div>

    <div class="capture-panel">
      <div class="panel-title">Data Capture</div>

      <div class="mac-row">
        <label>MAC Address <span class="optional">(optional — leave empty for auto)</span></label>
        <input type="text" id="macInput" placeholder="e.g. aa:bb:cc:dd:ee:ff" spellcheck="false" autocomplete="off">
      </div>

      <div class="rssi-display" id="rssiDisplay">
        <div class="rssi-display-title">Captured RSSI Values</div>
        <div class="rssi-chips" id="rssiChips"></div>
      </div>

      <div class="btn-row">
        <button class="btn-capture" id="btnCapture" onclick="startCapture()">Start Capture</button>
        <button class="btn-predict" id="btnPredict" onclick="sendPredict()">Predict</button>
      </div>

      <div class="log-box" id="logBox"></div>
    </div>

  </div>
</main>

<script>
const CLASS_COLORS = { AA:'#1a6b4a', AB:'#1a5a8a', BA:'#8a6b00', BB:'#8a2020' };
let capturedRssi = null;

// WiFi QR code
const WIFI_SSID = 'placeholder1';
const WIFI_PASS = 'placeholder2';
document.getElementById('wifiSsid').textContent = WIFI_SSID;
document.getElementById('wifiPass').textContent = WIFI_PASS;
new QRCode(document.getElementById('qrcode'), {
  text: `WIFI:T:WPA;S:${WIFI_SSID};P:${WIFI_PASS};;`,
  width: 110, height: 110,
  colorDark: '#1a1a18',
  colorLight: '#ffffff',
  correctLevel: QRCode.CorrectLevel.M,
});

fetch('/health').then(r=>r.json()).then(d=>{
  if(d.mock) document.getElementById('mockBanner').style.display='block';
});

function log(msg, type='') {
  const box = document.getElementById('logBox');
  box.style.display = 'block';
  const line = document.createElement('div');
  line.className = 'log-line ' + type;
  line.textContent = '› ' + msg;
  box.appendChild(line);
  box.scrollTop = box.scrollHeight;
}

function showRssi(vals) {
  capturedRssi = vals;
  const chips = document.getElementById('rssiChips');
  chips.innerHTML = vals.map((v, i) =>
    `<div class="rssi-chip"><span>R${i+1}</span>${v}</div>`
  ).join('');
  document.getElementById('rssiDisplay').style.display = 'block';
  document.getElementById('btnPredict').style.display = 'block';
}

function updateResult(data) {
  const cls = data.predicted_class;
  const color = CLASS_COLORS[cls] || '#1a1a18';
  const el = document.getElementById('resultClass');
  el.textContent = cls;
  el.style.color = color;
  el.classList.remove('updated');
  void el.offsetWidth;
  el.classList.add('updated');
  document.getElementById('idleHint').style.display = 'none';
  document.getElementById('resultConf').style.display = 'block';
  document.getElementById('confValue').textContent = (data.confidence*100).toFixed(1)+'%';
  for(const [c, p] of Object.entries(data.class_probabilities)) {
    const pct = (p*100).toFixed(1);
    document.getElementById(`bar-${c}`).style.width = pct+'%';
    document.getElementById(`pct-${c}`).textContent = pct+'%';
  }
}

async function startCapture() {
  const btn = document.getElementById('btnCapture');
  const mac = document.getElementById('macInput').value.trim();
  document.getElementById('logBox').innerHTML = '';
  document.getElementById('logBox').style.display = 'none';
  document.getElementById('rssiDisplay').style.display = 'none';
  document.getElementById('btnPredict').style.display = 'none';
  capturedRssi = null;

  btn.disabled = true;
  btn.textContent = 'Capturing...';

  const url = mac ? `/capture?mac=${encodeURIComponent(mac)}` : '/capture';
  const es = new EventSource(url);

  es.onmessage = (e) => {
    const msg = e.data;
    if (msg.startsWith('__RSSI__:')) {
      const vals = msg.split(':')[1].split(',').map(Number);
      showRssi(vals);
      log('RSSI values ready — press Predict to run the model.', 'ok');
    } else if (msg === '__DONE__') {
      es.close();
      btn.disabled = false;
      btn.textContent = 'Capture Again';
    } else if (msg === '__ERROR__') {
      es.close();
      btn.disabled = false;
      btn.textContent = 'Start Capture';
    } else if (msg.startsWith('ERROR')) {
      log(msg, 'err');
    } else {
      log(msg);
    }
  };

  es.onerror = () => {
    log('Connection lost.', 'err');
    es.close();
    btn.disabled = false;
    btn.textContent = 'Start Capture';
  };
}

async function sendPredict() {
  if (!capturedRssi) return;
  const btn = document.getElementById('btnPredict');
  btn.disabled = true;
  btn.textContent = 'Predicting...';
  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({rssi_values: capturedRssi})
    });
    const data = await res.json();
    updateResult(data);
    log(`Result: ${data.predicted_class} (${(data.confidence*100).toFixed(1)}% confidence)`, 'ok');
  } catch(err) {
    log('Prediction error: '+err.message, 'err');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Predict';
  }
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def ui():
    return HTML


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)