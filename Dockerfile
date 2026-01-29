# syntax=docker/dockerfile:1
ARG UBUNTU_VERSION=24.04
ARG CUDA_DEVEL_IMAGE=13.1.1-devel-ubuntu24.04
ARG CUDA_BASE_IMAGE=13.1.1-base-ubuntu24.04

FROM public.ecr.aws/docker/library/ubuntu:${UBUNTU_VERSION} AS builder

ARG PYTHON_VERSION=3.13

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV CMAKE_ARGS=-DGGML_NATIVE=OFF

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

RUN \
      rm -f /etc/apt/apt.conf.d/docker-clean \
      && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' \
        > /etc/apt/apt.conf.d/keep-cache

# hadolint ignore=DL3008
RUN \
      --mount=type=cache,target=/var/cache/apt,sharing=locked \
      --mount=type=cache,target=/var/lib/apt,sharing=locked \
      apt-get -y update \
      && apt-get -y install --no-install-recommends --no-install-suggests \
        software-properties-common \
      && add-apt-repository ppa:deadsnakes/ppa

# hadolint ignore=DL3008
RUN \
      --mount=type=cache,target=/var/cache/apt,sharing=locked \
      --mount=type=cache,target=/var/lib/apt,sharing=locked \
      apt-get -y update \
      && apt-get -y upgrade \
      && apt-get -y install --no-install-recommends --no-install-suggests \
        ca-certificates cargo curl g++ gcc git libopenblas-dev \
        "python${PYTHON_VERSION}-dev"

RUN \
      --mount=type=cache,target=/root/.cache \
      ln -s "python${PYTHON_VERSION}" /usr/bin/python \
      && curl -SL -o /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py \
      && /usr/bin/python /tmp/get-pip.py \
      && /usr/bin/python -m pip install --prefix /usr --upgrade \
        pip uv \
      && rm -f /tmp/get-pip.py

RUN \
      --mount=type=cache,target=/root/.cache \
      --mount=type=bind,source=.,target=/mnt/host \
      cp -a /mnt/host /tmp/sdeul \
      && /usr/bin/python -m uv --directory=/tmp/sdeul build --wheel \
      && /usr/bin/python -m pip install --prefix /usr \
        /tmp/sdeul/dist/sdeul-*.whl


FROM public.ecr.aws/docker/library/ubuntu:${UBUNTU_VERSION} AS cli

ARG PYTHON_VERSION=3.13
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
      ln -s "python${PYTHON_VERSION}" /usr/bin/python \
      && rm -f /etc/apt/apt.conf.d/docker-clean

# hadolint ignore=DL3008
RUN \
      --mount=type=cache,target=/var/cache/apt,sharing=locked \
      --mount=type=cache,target=/var/lib/apt,sharing=locked \
      apt-get -y update \
      && apt-get -y install --no-install-recommends --no-install-suggests \
        software-properties-common \
      && add-apt-repository ppa:deadsnakes/ppa

# hadolint ignore=DL3008
RUN \
      --mount=type=cache,target=/var/cache/apt,sharing=locked \
      --mount=type=cache,target=/var/lib/apt,sharing=locked \
      apt-get -y update \
      && apt-get -y upgrade \
      && apt-get -y install --no-install-recommends --no-install-suggests \
        ca-certificates jq "python${PYTHON_VERSION}"

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
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

RUN \
      rm -f /etc/apt/apt.conf.d/docker-clean \
      && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' \
        > /etc/apt/apt.conf.d/keep-cache

# hadolint ignore=DL3008
RUN \
      --mount=type=cache,target=/var/cache/apt,sharing=locked \
      --mount=type=cache,target=/var/lib/apt,sharing=locked \
      apt-get -y update \
      && apt-get -y install --no-install-recommends --no-install-suggests \
        software-properties-common \
      && add-apt-repository ppa:deadsnakes/ppa

# hadolint ignore=DL3008
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
      && /usr/bin/python -m pip install --prefix /usr --upgrade \
        pip uv \
      && rm -f /tmp/get-pip.py

RUN \
      --mount=type=cache,target=/root/.cache \
      --mount=type=bind,source=.,target=/mnt/host \
      cp -a /mnt/host /tmp/sdeul \
      && /usr/bin/python -m uv --directory=/tmp/sdeul build --wheel \
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

# hadolint ignore=DL3008
RUN \
      --mount=type=cache,target=/var/cache/apt,sharing=locked \
      --mount=type=cache,target=/var/lib/apt,sharing=locked \
      apt-get -y update \
      && apt-get -y install --no-install-recommends --no-install-suggests \
        software-properties-common \
      && add-apt-repository ppa:deadsnakes/ppa

# hadolint ignore=DL3008
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
