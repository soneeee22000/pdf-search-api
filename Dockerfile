FROM python:3.12-slim

# CPU-only torch. `pip install sentence-transformers` resolves the default
# torch wheel, which on Linux pulls the full NVIDIA CUDA stack -- roughly 2.8 GB
# of wheels into an image that will only ever run on CPU.
ARG TORCH_CPU_INDEX=https://download.pytorch.org/whl/cpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/models \
    PDF_SEARCH_STORAGE_DIR=/storage

# Tesseract and its French language data, for pages that have no text layer.
# Chosen over rapidocr-onnxruntime by measurement, not preference: see
# eval/OCR_DECISION.md. Adds 118 MB. OCR stays opt-in at the CLI -- installing
# the engine is not the same as trusting its output by default.
RUN apt-get update  && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-fra  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies before source, so editing code does not invalidate this layer.
COPY requirements.txt ./
RUN pip install --no-cache-dir --index-url ${TORCH_CPU_INDEX} \
        --extra-index-url https://pypi.org/simple torch \
 && pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image so the reviewer's first run needs no
# download and works offline.
ARG MODEL_NAME=intfloat/multilingual-e5-small
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${MODEL_NAME}')" \
 && chmod -R a+rX /opt/models

# The model is now in the image. Revalidating it against the Hub on every start
# would make the container need network access at runtime, so pin it offline.
# Override with -e HF_HUB_OFFLINE=0 when pointing PDF_SEARCH_MODEL at another model.
ENV HF_HUB_OFFLINE=1     TRANSFORMERS_OFFLINE=1     HF_HUB_DISABLE_TELEMETRY=1

COPY src/ ./src/
COPY docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh

ENV PYTHONPATH=/app/src

RUN useradd --create-home --uid 1000 appuser \
 && mkdir -p /storage /input \
 && chown -R appuser:appuser /storage /app
USER appuser

EXPOSE 8000

# The API loads the model and index during startup, which takes tens of seconds
# on a cold container. The healthcheck makes that state visible instead of
# looking like a hang.
HEALTHCHECK --interval=10s --timeout=5s --start-period=180s --retries=3   CMD python -c "import json,urllib.request,sys; sys.exit(0 if json.load(urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=4))['status']=='ok' else 1)"

ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["api"]
