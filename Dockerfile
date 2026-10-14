FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, so editing source code doesn't reinstall them.
COPY pyproject.toml README.md ./
RUN mkdir -p src/akp && touch src/akp/__init__.py \
    && pip install . \
    && pip uninstall -y agentic-knowledge-pipeline

# Bake the embedding model into the image: containers never download it at start.
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL} \
    EMBEDDING_CACHE_DIR=/models
RUN python -c "import os; from fastembed import TextEmbedding; \
TextEmbedding(os.environ['EMBEDDING_MODEL'], cache_dir='/models')"
ENV HF_HUB_OFFLINE=1

COPY src ./src
RUN pip install --no-deps .
COPY corpus ./corpus

RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 8000 50051
CMD ["uvicorn", "akp.api:app", "--host", "0.0.0.0", "--port", "8000"]
