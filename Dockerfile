FROM python:3.10-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[api]"

# Mount or copy a model trained with `oncomap train` and point ONCOMAP_MODEL at it.
ENV ONCOMAP_MODEL=/models/model.joblib \
    ONCOMAP_CORS_ORIGINS=* \
    PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn oncomap.api:app --host 0.0.0.0 --port ${PORT}"]
