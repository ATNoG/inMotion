const role = document.querySelector(".page")?.dataset.role;

const traducoesEstado = {
  waiting_registration: "a aguardar registo",
  monitoring: "a monitorizar",
  predicting: "a prever",
  disconnected: "desligado",
  fallback: "modo fallback",
  connect: "a ligar",
  start: "iniciado",
  error: "erro",
  recover: "recuperado",
};

const coresClasse = {
  AA: "#1f6f5f",
  AB: "#2f6ea5",
  BA: "#8a6c1a",
  BB: "#8a2f2f",
};

function traduzEstado(value) {
  return traducoesEstado[value] || value || "";
}

async function fetchJSON(url, options) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `Pedido falhou: ${response.status}`);
  }
  return response.json();
}

if (role === "teacher") {
  iniciarProfessor();
} else if (role === "child") {
  iniciarCrianca();
}

function iniciarProfessor() {
  const rosterBody = document.getElementById("rosterBody");
  const stateLabel = document.getElementById("teacherState");
  const scannerBadge = document.getElementById("scannerBadge");
  const startBtn = document.getElementById("startBtn");
  const stopBtn = document.getElementById("stopBtn");

  const children = new Map();

  function render() {
    const rows = [...children.values()]
      .sort((a, b) => (a.codename || "").localeCompare(b.codename || ""))
      .map((child) => {
        const pred = child.latest_prediction?.predicted_class || "—";
        return `<tr>
          <td>${child.codename}</td>
          <td>${child.mac}</td>
          <td>${traduzEstado(child.status)}</td>
          <td>${child.sample_count}</td>
          <td>${pred}</td>
        </tr>`;
      });

    rosterBody.innerHTML =
      rows.join("") || '<tr><td colspan="5">Sem crianças registadas</td></tr>';
  }

  async function refreshState() {
    const state = await fetchJSON("/api/session/state");
    stateLabel.textContent = state.monitor.teacher_active
      ? `monitorização ativa (${state.monitor.mode})`
      : "a aguardar registos";
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

  const testRouterStartBtn = document.getElementById("testRouterStartBtn");
  const testRouterStopBtn = document.getElementById("testRouterStopBtn");

  if (testRouterStartBtn) {
    testRouterStartBtn.addEventListener("click", async () => {
      await fetchJSON("/api/test-router/start", { method: "POST" });
      alert("Mock Router Ativado");
    });
  }

  if (testRouterStopBtn) {
    testRouterStopBtn.addEventListener("click", async () => {
      await fetchJSON("/api/test-router/stop", { method: "POST" });
      alert("Mock Router Desativado");
    });
  }

  const ws = new WebSocket(
    `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/teacher`,
  );

  ws.onmessage = (event) => {
    const payload = JSON.parse(event.data);

    if (payload.monitor && payload.children) {
      payload.children.forEach((child) => children.set(child.child_id, child));
      stateLabel.textContent = payload.monitor.teacher_active
        ? `monitorização ativa (${payload.monitor.mode})`
        : "a aguardar registos";
      scannerBadge.textContent = `scanner: ${payload.monitor.mode}`;
      render();
      return;
    }

    if (payload.event_type === "status") {
      stateLabel.textContent = `${traduzEstado(payload.status)} • ${payload.message}`;
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
    stateLabel.textContent = "ligação WebSocket interrompida";
  };

  refreshState().catch((error) => {
    stateLabel.textContent = `falha ao carregar estado: ${error.message}`;
  });
}

function iniciarCrianca() {
  const codenameInput = document.getElementById("codenameInput");
  const registerBtn = document.getElementById("registerBtn");
  const childInfo = document.getElementById("childInfo");
  const registerStatus = document.getElementById("registerStatus");
  const captureStatus = document.getElementById("captureStatus");
  const sampleStatus = document.getElementById("sampleStatus");
  const statusBadge = document.getElementById("childStatusBadge");
  const predMain = document.getElementById("predMain");
  const predConf = document.getElementById("predConf");
  const predProb = document.getElementById("predProb");

  let childId = localStorage.getItem("inmotion-child-id") || "";
  let detectedMac = localStorage.getItem("inmotion-child-mac") || "";

  const rssiHistory = [];
  const confidenceHistory = [];
  const predictionHistory = [];

  const captureDurationMs = 10_000;
  let captureStartedAt = null;
  let captureCompleted = false;
  let pendingPrediction = null;

  const rssiChart = criarGraficoRssi();
  const confidenceChart = criarGraficoConfianca();
  const predictionChart = criarGraficoPrevisoes();

  function atualizarGraficoRssi() {
    rssiChart.data.labels = rssiHistory.map((_, index) => `${index + 1}`);
    rssiChart.data.datasets[0].data = [...rssiHistory];
    rssiChart.update("none");
  }

  function atualizarGraficoConfianca() {
    confidenceChart.data.labels = confidenceHistory.map(
      (_, index) => `${index + 1}`,
    );
    confidenceChart.data.datasets[0].data = confidenceHistory.map(
      (v) => +(v * 100).toFixed(2),
    );
    confidenceChart.update("none");
  }

  function atualizarGraficoPrevisoes() {
    predictionChart.data.datasets[0].data = predictionHistory.map(
      (item, index) => ({
        x: index + 1,
        y: classeParaIndice(item.label),
      }),
    );
    predictionChart.data.datasets[0].pointBackgroundColor =
      predictionHistory.map((item) => coresClasse[item.label] || "#1f6f5f");
    predictionChart.update("none");
  }

  function classeParaIndice(label) {
    if (label === "AA") return 0;
    if (label === "AB") return 1;
    if (label === "BA") return 2;
    return 3;
  }

  function indiceParaClasse(index) {
    const i = Number(index);
    if (i === 0) return "AA";
    if (i === 1) return "AB";
    if (i === 2) return "BA";
    return "BB";
  }

  function resetCaptura() {
    captureStartedAt = Date.now();
    captureCompleted = false;
    pendingPrediction = null;
    rssiHistory.length = 0;
    confidenceHistory.length = 0;
    predictionHistory.length = 0;
    atualizarGraficoRssi();
    atualizarGraficoConfianca();
    atualizarGraficoPrevisoes();
    predMain.textContent = "—";
    predConf.textContent = "confiança: —";
    predProb.innerHTML = "";
    sampleStatus.textContent = "amostras na janela: 0/10";
  }

  function atualizarEstadoCaptura() {
    if (!captureStartedAt) {
      captureStatus.textContent =
        "a captura de 10 segundos começa quando o professor inicia a monitorização";
      return;
    }

    const elapsed = Date.now() - captureStartedAt;
    const remainingMs = Math.max(0, captureDurationMs - elapsed);

    if (remainingMs > 0) {
      captureStatus.textContent = `captura a decorrer: ${(remainingMs / 1000).toFixed(1)}s restantes`;
      return;
    }

    if (!captureCompleted) {
      captureCompleted = true;
      captureStatus.textContent = "captura concluída: previsões ativas";
      if (pendingPrediction) {
        aplicarPrevisao(pendingPrediction);
        pendingPrediction = null;
      }
    }
  }

  function aplicarPrevisao(payload) {
    predMain.textContent = payload.predicted_class;
    predConf.textContent = `confiança: ${(payload.confidence * 100).toFixed(1)}%`;
    sampleStatus.textContent = "previsão atualizada com janela deslizante";

    confidenceHistory.push(payload.confidence);
    while (confidenceHistory.length > 45) confidenceHistory.shift();
    atualizarGraficoConfianca();

    predictionHistory.push({
      label: payload.predicted_class,
      confidence: payload.confidence,
    });
    while (predictionHistory.length > 45) predictionHistory.shift();
    atualizarGraficoPrevisoes();

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

  async function detetarMac(codename) {
    const state = await fetchJSON(
      `/api/child/detect-mac?codename=${encodeURIComponent(codename)}`,
    );

    if (state.mac) {
      detectedMac = state.mac;
      localStorage.setItem("inmotion-child-mac", detectedMac);
      registerStatus.textContent = `MAC detetado: ${detectedMac} (${state.source})`;
    } else {
      registerStatus.textContent = "deteção de MAC pendente";
    }

    return state;
  }

  function ligarWebSocketCrianca() {
    if (!childId) return;

    const ws = new WebSocket(
      `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/child/${childId}`,
    );

    ws.onmessage = (event) => {
      const payload = JSON.parse(event.data);

      if (payload.event_type === "status") {
        statusBadge.textContent = `${traduzEstado(payload.status)} (${payload.mode})`;
        if (payload.status === "start") {
          resetCaptura();
        }
        return;
      }

      if (payload.event_type === "rssi") {
        if (!captureStartedAt) {
          captureStartedAt = Date.now();
        }

        rssiHistory.push(payload.rssi);
        while (rssiHistory.length > 10) rssiHistory.shift();
        atualizarGraficoRssi();
        atualizarEstadoCaptura();

        sampleStatus.textContent = `amostras na janela: ${payload.sample_count}/10`;
        statusBadge.textContent =
          payload.sample_count >= 10 ? "a prever" : "a monitorizar";
      }

      if (payload.event_type === "prediction") {
        if (!captureCompleted) {
          pendingPrediction = payload;
          sampleStatus.textContent =
            "a aguardar fim dos 10 segundos para mostrar a previsão";
        } else {
          aplicarPrevisao(payload);
        }
      }
    };

    ws.onerror = () => {
      statusBadge.textContent = "desligado";
    };
  }

  registerBtn.addEventListener("click", async () => {
    const codename =
      (codenameInput.value || "").trim() ||
      `Crianca-${Math.floor(Math.random() * 100)}`;

    try {
      if (!detectedMac) {
        await detetarMac(codename);
      }

      if (!detectedMac) {
        registerStatus.textContent = "não foi possível detetar o MAC ainda";
        return;
      }

      const child = await fetchJSON("/api/child/register", {
        method: "POST",
        body: JSON.stringify({ codename, mac: detectedMac }),
      });

      childId = child.child_id;
      localStorage.setItem("inmotion-child-id", childId);
      statusBadge.textContent = traduzEstado(child.status);
      childInfo.textContent = `ID criança: ${child.child_id} • MAC: ${child.mac}`;
      registerStatus.textContent = `registado como ${child.codename}`;
      ligarWebSocketCrianca();
    } catch (error) {
      registerStatus.textContent = `falha no registo: ${error.message}`;
    }
  });

  const initialCodename = codenameInput.value || "crianca";
  detetarMac(initialCodename).catch(() => {
    registerStatus.textContent = "deteção de MAC pendente";
  });

  if (childId) {
    childInfo.textContent = `ID criança: ${childId} • MAC: ${detectedMac || "desconhecido"}`;
    ligarWebSocketCrianca();
  }

  setInterval(atualizarEstadoCaptura, 250);

  function criarGraficoRssi() {
    return new Chart(document.getElementById("rssiCanvas"), {
      type: "line",
      data: {
        labels: [],
        datasets: [
          {
            label: "RSSI (dBm)",
            data: [],
            borderColor: "#1f6f5f",
            backgroundColor: "rgba(31, 111, 95, 0.15)",
            pointRadius: 2,
            tension: 0.25,
            fill: true,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            min: -80,
            max: -30,
            title: { display: true, text: "dBm" },
          },
          x: {
            title: { display: true, text: "Amostra" },
          },
        },
        plugins: {
          legend: { display: false },
        },
      },
    });
  }

  function criarGraficoConfianca() {
    return new Chart(document.getElementById("confidenceCanvas"), {
      type: "line",
      data: {
        labels: [],
        datasets: [
          {
            label: "Confiança (%)",
            data: [],
            borderColor: "#2f6ea5",
            backgroundColor: "rgba(47, 110, 165, 0.12)",
            pointRadius: 2,
            tension: 0.2,
            fill: true,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            min: 0,
            max: 100,
            title: { display: true, text: "%" },
          },
          x: {
            title: { display: true, text: "Janela" },
          },
        },
      },
    });
  }

  function criarGraficoPrevisoes() {
    return new Chart(document.getElementById("predictionTimelineCanvas"), {
      type: "scatter",
      data: {
        datasets: [
          {
            label: "Classe prevista",
            data: [],
            pointRadius: 4,
            showLine: false,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            title: { display: true, text: "Janela" },
            ticks: { precision: 0 },
          },
          y: {
            min: -0.5,
            max: 3.5,
            ticks: {
              stepSize: 1,
              callback: (value) => indiceParaClasse(value),
            },
            title: { display: true, text: "Classe" },
          },
        },
        plugins: {
          legend: { display: false },
        },
      },
    });
  }
}
