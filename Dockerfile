# syntax=docker/dockerfile:1
ARG PYTHON_IMAGE=3.13-slim
ARG CUDA_DEVEL_IMAGE=12.6.2-devel-ubuntu24.04
ARG CUDA_BASE_IMAGE=12.6.2-base-ubuntu24.04

FROM public.ecr.aws/docker/library/python:${PYTHON_IMAGE} AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
ENV POETRY_HOME='/opt/poetry'
ENV POETRY_VIRTUALENVS_CREATE=false
ENV POETRY_NO_INTERACTION=true

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

RUN \
      rm -f /etc/apt/apt.conf.d/docker-clean \
      && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' \
        > /etc/apt/apt.conf.d/keep-cache

RUN \
      --mount=type=cache,target=/var/cache/apt,sharing=locked \
      --mount=type=cache,target=/var/lib/apt,sharing=locked \
      apt-get -y update \
      && apt-get -y upgrade \
      && apt-get -y install --no-install-recommends --no-install-suggests \
        g++

RUN \
      --mount=type=cache,target=/root/.cache \
      /usr/local/bin/python -m pip install --upgrade pip poetry

RUN \
      --mount=type=cache,target=/root/.cache \
      --mount=type=bind,source=.,target=/mnt/host \
      cp -a /mnt/host /tmp/sdeul \
      && /usr/local/bin/python -m poetry --directory=/tmp/sdeul build --format=wheel \
      && /usr/local/bin/python -m pip install /tmp/sdeul/dist/sdeul-*.whl


FROM public.ecr.aws/docker/library/python:${PYTHON_IMAGE} AS cli

ARG USER_NAME=sdeul
ARG USER_UID=1001
ARG USER_GID=1001

COPY --from=builder /usr/local /usr/local
COPY --from=builder /etc/apt/apt.conf.d/keep-cache /etc/apt/apt.conf.d/keep-cache

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

RUN \
      rm -f /etc/apt/apt.conf.d/docker-clean

RUN \
      --mount=type=cache,target=/var/cache/apt,sharing=locked \
      --mount=type=cache,target=/var/lib/apt,sharing=locked \
      apt-get -y update \
      && apt-get -y upgrade \
      && apt-get -y install --no-install-recommends --no-install-suggests \
        ca-certificates jq

RUN \
      groupadd --gid "${USER_GID}" "${USER_NAME}" \
      && useradd --uid "${USER_UID}" --gid "${USER_GID}" --shell /bin/bash --create-home "${USER_NAME}"

USER "${USER_NAME}"

HEALTHCHECK NONE

ENTRYPOINT ["/usr/local/bin/sdeul"]


FROM nvidia/cuda:${CUDA_DEVEL_IMAGE} AS cuda-builder

ARG PYTHON_VERSION=3.13
ARG CUDA_DOCKER_ARCH=all
ARG GGML_CUDA=1

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
ENV POETRY_HOME='/opt/poetry'
ENV POETRY_VIRTUALENVS_CREATE=false
ENV POETRY_NO_INTERACTION=true

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

RUN \
      rm -f /etc/apt/apt.conf.d/docker-clean \
      && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' \
        > /etc/apt/apt.conf.d/keep-cache

RUN \
      --mount=type=cache,target=/var/cache/apt,sharing=locked \
      --mount=type=cache,target=/var/lib/apt,sharing=locked \
      apt-get -y update \
      && apt-get -y install --no-install-recommends --no-install-suggests \
        gnupg gpg-agent software-properties-common \
      && add-apt-repository ppa:deadsnakes/ppa

RUN \
      --mount=type=cache,target=/var/cache/apt,sharing=locked \
      --mount=type=cache,target=/var/lib/apt,sharing=locked \
      apt-get -y update \
      && apt-get -y upgrade \
      && apt-get -y install --no-install-recommends --no-install-suggests \
        ca-certificates clinfo curl gcc g++ libclblast-dev libopenblas-dev \
        ocl-icd-opencl-dev opencl-headers "python${PYTHON_VERSION}-dev"

RUN \
      mkdir -p /etc/OpenCL/vendors \
      && echo 'libnvidia-opencl.so.1' > /etc/OpenCL/vendors/nvidia.icd

RUN \
      --mount=type=cache,target=/root/.cache \
      ln -s "python${PYTHON_VERSION}" /usr/bin/python \
      && curl -SL -o /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py \
      && /usr/bin/python /tmp/get-pip.py \
      && /usr/bin/python -m pip install --prefix /usr --upgrade pip \
      && rm -f /tmp/get-pip.py

RUN \
      --mount=type=cache,target=/root/.cache \
      --mount=type=bind,source=.,target=/mnt/host \
      cp -a /mnt/host /tmp/sdeul \
      && /usr/bin/python -m pip install --prefix /usr poetry \
      && /usr/bin/python -m poetry --directory=/tmp/sdeul build --format=wheel \
      && CMAKE_ARGS="-DGGML_CUDA=on" /usr/bin/python -m pip install --prefix /usr \
        /tmp/sdeul/dist/sdeul-*.whl


FROM nvidia/cuda:${CUDA_BASE_IMAGE} AS cuda-cli

ARG PYTHON_VERSION=3.13
ARG USER_NAME=sdeul
ARG USER_UID=1001
ARG USER_GID=1001

COPY --from=cuda-builder /usr/local /usr/local
COPY --from=cuda-builder /etc/apt/apt.conf.d/keep-cache /etc/apt/apt.conf.d/keep-cache

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

RUN \
      ln -s "python${PYTHON_VERSION}" /usr/bin/python \
      && rm -f /etc/apt/apt.conf.d/docker-clean

RUN \
      --mount=type=cache,target=/var/cache/apt,sharing=locked \
      --mount=type=cache,target=/var/lib/apt,sharing=locked \
      apt-get -y update \
      && apt-get -y install --no-install-recommends --no-install-suggests \
        gnupg gpg-agent software-properties-common \
      && add-apt-repository ppa:deadsnakes/ppa

RUN \
      --mount=type=cache,target=/var/cache/apt,sharing=locked \
      --mount=type=cache,target=/var/lib/apt,sharing=locked \
      apt-get -y update \
      && apt-get -y upgrade \
      && apt-get -y install --no-install-recommends --no-install-suggests \
        ca-certificates jq libopenblas0-openmp "python${PYTHON_VERSION}"

RUN \
      groupadd --gid "${USER_GID}" "${USER_NAME}" \
      && useradd --uid "${USER_UID}" --gid "${USER_GID}" --shell /bin/bash --create-home "${USER_NAME}"

USER "${USER_NAME}"

HEALTHCHECK NONE

ENTRYPOINT ["/usr/local/bin/sdeul"]
