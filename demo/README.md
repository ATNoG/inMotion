# inMotion Demo

Real-time position prediction demo using WiFi RSSI values.

## Overview

This demo provides a web interface and REST API for predicting room positions (AA, AB, BA, BB) based on 10 RSSI (WiFi signal strength) readings.

## Features

- 🎯 **Real-time Prediction**: Submit RSSI values and get instant position predictions
- 📊 **Visualizations**:
  - Signal strength graph showing RSSI over time
  - Confidence bar chart for all position classes
  - Architecture diagram
- 📡 **REST API**: Simple HTTP endpoint for programmatic access
- 🎨 **Interactive UI**: Built with Chart.js for real-time visualization

## Quick Start

### Using Python directly

```bash
# Install dependencies
uv pip install -e .

# Train the model (first time only)
python train_model.py

# Start the server
python main.py
```

### Using Docker

```bash
# Build and run
docker-compose up --build
```

## API Endpoints

### Health Check

```bash
GET /health
```

### Prediction

```bash
POST /predict
Content-Type: application/json

{
  "rssi_values": [-45.0, -47.0, -53.0, -47.0, -42.0, -48.0, -54.0, -46.0, -40.0, -38.0]
}
```

Response:

```json
{
  "predicted_class": "AA",
  "confidence": 0.95,
  "class_probabilities": {
    "AA": 0.95,
    "AB": 0.02,
    "BA": 0.02,
    "BB": 0.01
  },
  "rssi_input": [-45.0, -47.0, -53.0, -47.0, -42.0, -48.0, -54.0, -46.0, -40.0, -38.0],
  "features_importance": {
    "rssi_1": 0.12,
    "rssi_2": 0.08,
    ...
  }
}
```

## Accessing the Demo

- Web UI: http://localhost:8000/
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## Architecture

```
📱 Phone (WiFi Scan) → 🌐 REST API → 🤖 Random Forest Model → 📍 Position (AA/AB/BA/BB)
```

The model uses a RandomForest classifier trained on 10 RSSI readings to predict one of four room positions.
