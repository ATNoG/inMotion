const role = document.querySelector(".page")?.dataset.role;

function fmtStatus(value) {
  return (value || "").replaceAll("_", " ");
}

async function fetchJSON(url, options) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `Request failed: ${response.status}`);
  }
  return response.json();
}

if (role === "teacher") {
  startTeacher();
} else if (role === "child") {
  startChild();
}

function startTeacher() {
  const rosterBody = document.getElementById("rosterBody");
  const stateLabel = document.getElementById("teacherState");
  const scannerBadge = document.getElementById("scannerBadge");
  const startBtn = document.getElementById("startBtn");
  const stopBtn = document.getElementById("stopBtn");

  const children = new Map();

  function render() {
    const rows = [...children.values()]
      .sort((a, b) => a.codename.localeCompare(b.codename))
      .map((child) => {
        const pred = child.latest_prediction?.predicted_class || "—";
        return `<tr>
          <td>${child.codename}</td>
          <td>${child.mac}</td>
          <td>${fmtStatus(child.status)}</td>
          <td>${child.sample_count}</td>
          <td>${pred}</td>
        </tr>`;
      });
    rosterBody.innerHTML =
      rows.join("") || '<tr><td colspan="5">No children registered</td></tr>';
  }

  async function refreshState() {
    const state = await fetchJSON("/api/session/state");
    stateLabel.textContent = state.monitor.teacher_active
      ? `monitoring (${state.monitor.mode})`
      : "waiting registration";
    scannerBadge.textContent = `scanner: ${state.monitor.mode}`;
    state.children.forEach((child) => children.set(child.child_id, child));
    render();
  }

  startBtn.addEventListener("click", async () => {
    await fetchJSON("/api/teacher/start", { method: "POST" });
    await refreshState();
  });

  stopBtn.addEventListener("click", async () => {
    await fetchJSON("/api/teacher/stop", { method: "POST" });
    await refreshState();
  });

  const ws = new WebSocket(
    `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/teacher`,
  );
  ws.onmessage = (event) => {
    const payload = JSON.parse(event.data);

    if (payload.monitor && payload.children) {
      payload.children.forEach((child) => children.set(child.child_id, child));
      stateLabel.textContent = payload.monitor.teacher_active
        ? `monitoring (${payload.monitor.mode})`
        : "waiting registration";
      scannerBadge.textContent = `scanner: ${payload.monitor.mode}`;
      render();
      return;
    }

    if (payload.event_type === "status") {
      stateLabel.textContent = `${fmtStatus(payload.status)} • ${payload.message}`;
      scannerBadge.textContent = `scanner: ${payload.mode}`;
      return;
    }

    if (payload.event_type === "rssi" || payload.event_type === "prediction") {
      const current = children.get(payload.child_id) || {
        child_id: payload.child_id,
        codename: payload.codename,
        mac: payload.mac,
      };
      if (payload.event_type === "rssi") {
        current.sample_count = payload.sample_count;
        current.status =
          payload.sample_count >= 10 ? "predicting" : "monitoring";
      } else {
        current.latest_prediction = {
          predicted_class: payload.predicted_class,
          confidence: payload.confidence,
        };
        current.status = "predicting";
      }
      current.codename = payload.codename;
      current.mac = payload.mac;
      children.set(payload.child_id, current);
      render();
    }
  };

  ws.onerror = () => {
    stateLabel.textContent = "websocket disconnected";
  };

  refreshState().catch((error) => {
    stateLabel.textContent = `failed to load state: ${error.message}`;
  });
}

function startChild() {
  const codenameInput = document.getElementById("codenameInput");
  const registerBtn = document.getElementById("registerBtn");
  const childInfo = document.getElementById("childInfo");
  const registerStatus = document.getElementById("registerStatus");
  const sampleStatus = document.getElementById("sampleStatus");
  const statusBadge = document.getElementById("childStatusBadge");
  const predMain = document.getElementById("predMain");
  const predConf = document.getElementById("predConf");
  const predProb = document.getElementById("predProb");
  const canvas = document.getElementById("rssiCanvas");
  const ctx = canvas.getContext("2d");

  let childId = localStorage.getItem("inmotion-child-id") || "";
  let detectedMac = localStorage.getItem("inmotion-child-mac") || "";
  const history = [];

  function drawChart() {
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    ctx.strokeStyle = "#cfd5dc";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i += 1) {
      const y = 15 + (i * (height - 30)) / 5;
      ctx.beginPath();
      ctx.moveTo(32, y);
      ctx.lineTo(width - 8, y);
      ctx.stroke();
    }

    ctx.fillStyle = "#6b7280";
    ctx.font = "12px sans-serif";
    [-30, -40, -50, -60, -70, -80].forEach((label, idx) => {
      const y = 15 + (idx * (height - 30)) / 5;
      ctx.fillText(String(label), 2, y + 4);
    });

    if (!history.length) {
      return;
    }

    const minRssi = -80;
    const maxRssi = -30;
    const xStep = (width - 50) / Math.max(history.length - 1, 1);

    ctx.strokeStyle = "#1f6f5f";
    ctx.lineWidth = 2;
    ctx.beginPath();
    history.forEach((value, index) => {
      const x = 32 + index * xStep;
      const clamped = Math.max(minRssi, Math.min(maxRssi, value));
      const ratio = (clamped - minRssi) / (maxRssi - minRssi);
      const y = height - 15 - ratio * (height - 30);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  async function detectMac(codename) {
    const state = await fetchJSON(
      `/api/child/detect-mac?codename=${encodeURIComponent(codename)}`,
    );
    if (state.mac) {
      detectedMac = state.mac;
      localStorage.setItem("inmotion-child-mac", detectedMac);
      registerStatus.textContent = `detected MAC: ${detectedMac} (${state.source})`;
    } else {
      registerStatus.textContent = "MAC detection pending";
    }
    return state;
  }

  function connectChildWs() {
    if (!childId) return;
    const ws = new WebSocket(
      `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/child/${childId}`,
    );

    ws.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      if (payload.event_type === "status") {
        statusBadge.textContent = `${fmtStatus(payload.status)} (${payload.mode})`;
        return;
      }

      if (payload.event_type === "rssi") {
        history.push(payload.rssi);
        while (history.length > 10) history.shift();
        drawChart();
        sampleStatus.textContent = `samples in window: ${payload.sample_count}/10`;
        statusBadge.textContent =
          payload.sample_count >= 10 ? "predicting" : "monitoring";
      }

      if (payload.event_type === "prediction") {
        predMain.textContent = payload.predicted_class;
        predConf.textContent = `confidence: ${(payload.confidence * 100).toFixed(1)}%`;
        sampleStatus.textContent = "prediction updated using sliding window";

        predProb.innerHTML = Object.entries(payload.probabilities)
          .map(
            ([key, value]) => `
            <div class="prob-row">
              <div class="prob-key">${key}</div>
              <div class="prob-bar-wrap"><div class="prob-bar" style="width:${Math.max(2, value * 100)}%"></div></div>
              <div class="small">${(value * 100).toFixed(1)}%</div>
            </div>
          `,
          )
          .join("");
      }
    };

    ws.onerror = () => {
      statusBadge.textContent = "disconnected";
    };
  }

  registerBtn.addEventListener("click", async () => {
    const codename =
      (codenameInput.value || "").trim() ||
      `Child-${Math.floor(Math.random() * 100)}`;
    try {
      if (!detectedMac) {
        await detectMac(codename);
      }

      if (!detectedMac) {
        registerStatus.textContent = "unable to detect MAC yet";
        return;
      }

      const child = await fetchJSON("/api/child/register", {
        method: "POST",
        body: JSON.stringify({ codename, mac: detectedMac }),
      });

      childId = child.child_id;
      localStorage.setItem("inmotion-child-id", childId);
      statusBadge.textContent = fmtStatus(child.status);
      childInfo.textContent = `child id: ${child.child_id} • MAC: ${child.mac}`;
      registerStatus.textContent = `registered as ${child.codename}`;
      connectChildWs();
    } catch (error) {
      registerStatus.textContent = `register failed: ${error.message}`;
    }
  });

  const initialCodename = codenameInput.value || "child";
  detectMac(initialCodename).catch(() => {
    registerStatus.textContent = "MAC detection pending";
  });

  if (childId) {
    childInfo.textContent = `child id: ${childId} • MAC: ${detectedMac || "unknown"}`;
    connectChildWs();
  }

  drawChart();
}
