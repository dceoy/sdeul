# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.14

FROM mcr.microsoft.com/devcontainers/python:${PYTHON_VERSION} AS devcontainer

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
        pip uv \
      && npm install -g @anthropic-ai/claude-code @openai/codex

RUN \
      mkdir -p /home/vscode/.aws \
      && chown -R vscode:vscode /usr/local /home/vscode

USER vscode

RUN \
      git config --global core.excludesfile "${HOME}/.gitignore" \
      && git config --global core.pager ''

HEALTHCHECK NONE
