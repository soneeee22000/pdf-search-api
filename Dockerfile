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

WORKDIR /app

# Dependencies before source, so editing code does not invalidate this layer.
COPY requirements.txt ./
RUN pip install --no-cache-dir --index-url ${TORCH_CPU_INDEX} \
        --extra-index-url https://pypi.org/simple torch \
 && pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image so the reviewer's first run needs no
# download and works offline.
ARG MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${MODEL_NAME}')" \
 && chmod -R a+rX /opt/models

COPY src/ ./src/
COPY docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh

ENV PYTHONPATH=/app/src

RUN useradd --create-home --uid 1000 appuser \
 && mkdir -p /storage /input \
 && chown -R appuser:appuser /storage /app
USER appuser

EXPOSE 8000

ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["api"]
