FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 \
    python3-pip \
    curl \
    git \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_HOME="/opt/poetry" \
    POETRY_VERSION=1.8.3 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry

WORKDIR /workspace

COPY src/ /workspace/

# For some reason pip doesn't respond, provide a mirror
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple .

ENTRYPOINT ["python3", "block_muon/block_muon_train.py"]

CMD ["--model", "qwen", \
     "--optimizer", "adamw", \
     "--lr", "1e-3", \
     "--wd", "0.1", \
     "--block_size", "256", \
     "--dataset", "openwebtext-100k", \
     "--split", "train", \
     "--hidden_size", "1024", \
     "--save_checkpoint_interval", "5000", \
     "--wandb_project", "muon-optimizer-experiments-debug", \
     "--upload_checkpoints", "False"]
