# The app on CPU, answering with models served by Ollama (see docker-compose.yml).
# Hugging Face models need an NVIDIA GPU and CUDA; for those, use environment.yml instead.
#
#   docker build -t llm-rag-local .
#   docker run -p 127.0.0.1:8501:8501 -v llm-rag-data:/data \
#     -e OLLAMA_HOST=http://host.docker.internal:11434 --add-host host.docker.internal:host-gateway llm-rag-local

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/huggingface \
    DATA_DIR=/data \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
# Reloading on code edits isn't needed in the image, and the file watcher's walk
# over loaded modules makes transformers try to import torchvision, logging errors
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

WORKDIR /app

# CPU-only PyTorch first: the default wheels bundle CUDA (several GB), which the
# embedding model doesn't need
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

# The app runs as an ordinary user, keeping its documents and chats in /data
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /data "$HF_HOME" \
    && chown app /data "$HF_HOME"
USER app

# Include the embedding model, so the app starts without downloading it and
# works offline. It must match RAG_CONFIG["embedding_model"] in config.py;
# build with --build-arg EMBEDDING_MODEL= to download it on first use instead.
ARG EMBEDDING_MODEL=all-MiniLM-L6-v2
RUN if [ -n "$EMBEDDING_MODEL" ]; then \
        python -c "import sys; from sentence_transformers import SentenceTransformer; SentenceTransformer(sys.argv[1])" "$EMBEDDING_MODEL"; \
    fi

COPY . .

VOLUME /data
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"
CMD ["streamlit", "run", "streamlit_app.py"]
