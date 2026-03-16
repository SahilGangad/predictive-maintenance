# ── Base image ────────────────────────────────────────────
FROM python:3.10-slim

# ── Set working directory ─────────────────────────────────
WORKDIR /app

# ── Install system dependencies ───────────────────────────
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# ── Copy requirements first (layer caching) ───────────────
COPY requirements.txt .

# ── Install Python dependencies ───────────────────────────
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy rest of project ──────────────────────────────────
COPY . .

# ── Expose port ───────────────────────────────────────────
EXPOSE 8000

# ── Start FastAPI ─────────────────────────────────────────
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
