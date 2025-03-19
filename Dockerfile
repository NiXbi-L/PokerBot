# Stage 1: Сборка артефактов
FROM ubuntu:20.04 AS builder

# Настройка окружения для автоматического выбора региона
ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get clean && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libboost-test-dev \
    python3 \
    python3-pip \
    python3-dev \
    cython3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY tools/montecarlo_cython.pyx .

RUN cython3 -X language_level=3 --embed -o montecarlo_cython.c montecarlo_cython.pyx

RUN gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing \
    -I/usr/include/python3.8 -o montecarlo_cython.so montecarlo_cython.c

RUN mkdir -p /artifacts && \
    find /build -name "*.so" -exec cp {} /artifacts/ \; && \
    echo "Final artifacts:" && ls -la /artifacts


FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Кэширование для APT
RUN rm -f /etc/apt/apt.conf.d/docker-clean && \
    echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache

# 1. Базовые зависимости
RUN apt-get clean && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Установка NVIDIA Container Toolkit
RUN curl -sL https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add - \
    && curl -sL https://nvidia.github.io/nvidia-docker/ubuntu20.04/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list \
    && apt-get update \
    && apt-get install -y nvidia-container-toolkit

# 2. Репозитории
RUN add-apt-repository ppa:deadsnakes/ppa -y

# 3. Python-зависимости
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get clean && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-distutils \
    python3.11-dev && \
    rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# 4. CUDA-зависимости
RUN --mount=type=cache,target=/var/cuda-cache \
    apt-get clean && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    libcupti-dev \
    libcudnn8=8.5.0.*-1+cuda11.7 \
    libcudnn8-dev=8.5.0.*-1+cuda11.7 \
    && rm -rf /var/lib/apt/lists/*

# Настройка путей CUDA
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda-11.7

ENV CXXFLAGS="-std=c++17"

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Установка Poetry с кэшированием
COPY pyproject.toml poetry.lock ./
RUN pip install --upgrade pip && \
    pip cache purge && \
    pip install "poetry==2.1.1"

# Установка зависимостей с кэшированием
RUN poetry config virtualenvs.create false && \
    poetry install --no-root --no-interaction --no-ansi --only main

# Копирование артефактов и исходного кода
COPY --from=builder /artifacts/ ./tools/
COPY agents ./agents/
COPY tools ./tools/
COPY gym_env ./gym_env/
COPY main.py ./
COPY log ./log
COPY ./config.ini ./


CMD ["poetry", "run", "python", "main.py", "selfplay", "dqn_train", "--name=pro_8plr", "-c", "--episodes=1000", "--stack=5"]