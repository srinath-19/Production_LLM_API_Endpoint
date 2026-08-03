FROM python:3.12-slim

WORKDIR /app

# Create non-root user early so we can own /app
RUN useradd --create-home appuser && chown appuser:appuser /app

# Install uv (fast Python package manager)
RUN pip install uv

# Copy dependency files first (Docker layer caching)
COPY --chown=appuser:appuser pyproject.toml .
COPY --chown=appuser:appuser uv.lock* .

# Switch to non-root user before installing deps (so .venv is owned by appuser)
USER appuser

# Install dependencies (production only)
RUN uv sync --frozen --no-dev

# Copy application code
COPY --chown=appuser:appuser app/ app/

# Expose port
EXPOSE 8000

# Health check.
# python:3.12-slim ships without curl, so probe with the stdlib instead of
# installing a package just to make the healthcheck pass. urlopen raises a
# non-zero exit on any 4xx/5xx or connection failure, which is what we want.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=5).status == 200 else 1)"]

# Run uvicorn directly from venv (avoids uv re-syncing at runtime)
CMD [".venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]