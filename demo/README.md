# Demo inMotion RSSI ao Vivo (Investigador + Participante)

Demo local baseada em FastAPI, com sessões multi-participante, streaming RSSI em tempo real e previsões com janela deslizante de 10 amostras.

## O que está incluído

- **Backend modular em FastAPI**:
  - `demo/app/session_store.py`: estado da sessão de monitorização e dos participantes
  - `demo/app/scanner_adapter.py`: scanner ao vivo + fallback automático por replay
  - `demo/app/inference.py`: carregamento de `models/RandomForest.joblib` e inferência
  - `demo/app/event_hub.py`: distribuição de eventos para WebSocket/SSE
- **UI local dividida**:
  - `/teacher`: controlo start/stop e lista de participantes
  - `/child`: registo, deteção de IP, gráfico RSSI e previsão
- **Regras de buffer determinísticas**:
  - buffer por participante com 10 RSSI
  - primeira previsão após 10 amostras
  - atualizações seguintes por janela deslizante
- **Resiliência offline**:
  - modo primário: scanner ao vivo (`demo/app/wifi_scan.py`)
  - fallback: replay de dados locais (`dataset_only_pure.csv`)

## Bibliotecas usadas (locais, descarregáveis)

- **Gráficos**: Chart.js (ficheiro local em `demo/static/vendor/chart.umd.min.js`)
- **Estilo base**: Pico CSS (ficheiro local em `demo/static/vendor/pico.min.css`)
- Ambas são servidas localmente pelo FastAPI, sem dependência de CDN em runtime.

## Executar com `uv`

Na raiz do repositório:

```bash
uv run python demo/main.py
```

Abrir:

- Vista investigador: `http://127.0.0.1:8000/teacher`
- Vista participante: `http://127.0.0.1:8000/child`

Se a porta `8000` estiver ocupada:

```bash
uv run uvicorn main:app --app-dir demo --host 127.0.0.1 --port 8011
```

## Configuração por variáveis de ambiente

Credenciais e parâmetros do scanner são lidos por ambiente:

- `INMOTION_ROUTER_IP`
- `INMOTION_ROUTER_USER` (default: `root`)
- `INMOTION_ROUTER_PASS` (opcional se usar chave)
- `INMOTION_ROUTER_KEY_FILE` (opcional)
- `INMOTION_ROUTER_PORT` (default: `22`)
- `INMOTION_ROUTER_COMMAND_MODE` (`auto|ubus|system`, default: `auto`)
- `INMOTION_SCAN_INTERVAL` (default: `1.0` segundo)
- `INMOTION_MODEL_PATH` (default: `models/RandomForest.joblib`)
- `INMOTION_FALLBACK_CSV` (default: `dataset_only_pure.csv`)

Exemplo:

```bash
export INMOTION_ROUTER_IP=10.20.0.1
export INMOTION_ROUTER_USER=admin
export INMOTION_ROUTER_PASS='***'
uv run python demo/main.py
```

## Contratos de API

### Registo do participante e identidade

1. Detetar IP:

```http
GET /api/child/detect-ip?codename=Alice
```

2. Registar participante:

```http
POST /api/child/register
Content-Type: application/json

{"codename":"Alice","ip":"10.255.99.132"}
```

### Controlo do investigador

- `POST /api/teacher/start`
- `POST /api/teacher/stop`
- `GET /api/session/state`

### Endpoints compatíveis

- `GET /health`
- `POST /predict` (previsão direta para 10 valores)
- `GET /events` (stream SSE)

## Transporte em tempo real

- WebSocket investigador: `/ws/teacher`
- WebSocket participante: `/ws/child/{child_id}`
- SSE compatível: `/events`

## Fluxo de operação (aula)

1. Iniciar backend (`uv run python demo/main.py`).
2. Abrir vistas de professor e criança.
3. No participante: registar codename + IP.
4. No investigador: clicar em **Iniciar monitorização**.
5. Pedir ao aluno para caminhar com tráfego ativo no telemóvel.
6. Aguardar 10 segundos para fim da captura.
7. Ver previsão e probabilidades na vista participante.

## Resolução de problemas

- **Router inacessível / sem SSH**
  - O sistema entra em modo replay automaticamente.
- **IP não detetado**
  - Em modo replay, IP é gerado de forma determinística pelo codename.
- **Modelo não carregado**
  - Verificar `/health` (`model_degraded`) e o ficheiro `models/RandomForest.joblib`.
- **Porta ocupada**
  - Executar numa porta alternativa (`8011`, por exemplo).
