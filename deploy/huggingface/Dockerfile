# Hugging Face Docker Space: runs as uid 1000 and must listen on port 7860.
FROM python:3.10-slim

RUN useradd -m -u 1000 user
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[api]"
COPY --chown=user models ./models

USER user
ENV ONCOMAP_MODEL=/app/models/model.joblib \
    ONCOMAP_COMPOUNDS=/app/models/compounds.csv \
    ONCOMAP_CORS_ORIGINS=* \
    PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn oncomap.api:app --host 0.0.0.0 --port ${PORT}"]
