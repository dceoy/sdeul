# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.13

FROM mcr.microsoft.com/devcontainers/python:${PYTHON_VERSION} AS devcontainer

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
ENV POETRY_HOME='/opt/poetry'
ENV POETRY_VIRTUALENVS_CREATE=false
ENV POETRY_NO_INTERACTION=true
ENV CMAKE_ARGS=-DGGML_NATIVE=OFF

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

# hadolint ignore=DL3008
RUN \
      --mount=type=cache,target=/var/cache/apt,sharing=locked \
      --mount=type=cache,target=/var/lib/apt,sharing=locked \
      apt-get -y update \
      && apt-get -y upgrade

RUN \
      --mount=type=cache,target=/root/.cache \
      /usr/local/bin/python -m pip install --prefix /usr/local --upgrade \
        pip poetry

RUN \
      chown -R vscode:vscode /usr/local

HEALTHCHECK NONE

USER vscode
