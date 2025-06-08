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
      && apt-get -y upgrade \
      && apt-get -yqq install --no-install-recommends --no-install-suggests \
        npm

RUN \
      --mount=type=cache,target=/root/.cache \
      curl -fsSL -o /tmp/awscliv2.zip \
        "https://awscli.amazonaws.com/awscli-exe-linux-$([ "$(uname -m)" = 'x86_64' ] && echo 'x86_64' || echo 'aarch64').zip" \
      && unzip /tmp/awscliv2.zip -d /tmp \
      && /tmp/aws/install \
      && rm -rf /tmp/awscliv2.zip /tmp/aws

# hadolint ignore=DL3016
RUN \
      --mount=type=cache,target=/root/.cache \
      /usr/local/bin/python -m pip install --prefix /usr/local --upgrade \
        pip poetry \
      && npm install -g @anthropic-ai/claude-code @openai/codex

RUN \
      chown -R vscode:vscode /usr/local

RUN \
      git config --global core.excludesfile "${HOME}/.gitignore" \
      && git config --global core.pager '' \
      && mkdir "${HOME}/.aws"

HEALTHCHECK NONE

USER vscode
