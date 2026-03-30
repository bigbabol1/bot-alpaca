FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download FinBERT model at build time — cached in Docker layer (~1.5GB)
# This avoids runtime internet dependency inside the container.
RUN python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='ProsusAI/finbert')"

# Copy source
COPY src/ ./src/
COPY config/ ./config/
COPY VERSION .

# Ensure volume mount points exist
RUN mkdir -p /app/data /app/logs

CMD ["python", "-m", "src.main"]
