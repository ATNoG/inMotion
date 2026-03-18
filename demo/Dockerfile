FROM astral/uv:python3.14-bookworm-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    openssh-client \
    sshpass \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY pyproject.toml .
RUN uv sync

# Copy application code
COPY . .

# Create models directory
RUN mkdir -p models

# Train model on build (optional - can also be done at runtime)
# RUN python train_model.py

# Expose port
EXPOSE 8000

# Run the application
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
