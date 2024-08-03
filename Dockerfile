# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.12

FROM public.ecr.aws/docker/library/python:${PYTHON_VERSION}-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

RUN rm -f /etc/apt/apt.conf.d/docker-clean \
      && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' \
        > /etc/apt/apt.conf.d/keep-cache

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
      --mount=type=cache,target=/var/lib/apt,sharing=locked \
      apt-get -y update \
      && apt-get -y install --no-install-recommends --no-install-suggests \
        g++

RUN --mount=type=cache,target=/root/.cache/pip \
      --mount=type=bind,source=.,target=/mnt/sdeul \
      cp -a /mnt/sdeul /tmp/sdeul \
      && /usr/local/bin/python -m pip install -U --no-cache-dir \
        pip /tmp/sdeul


FROM public.ecr.aws/docker/library/python:${PYTHON_VERSION}-slim AS cli

ARG USER_NAME=sdeul
ARG USER_UID=1000
ARG USER_GID=1000

COPY --from=builder /usr/local /usr/local
COPY --from=builder /etc/apt/apt.conf.d/keep-cache /etc/apt/apt.conf.d/keep-cache

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

RUN rm -f /etc/apt/apt.conf.d/docker-clean

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
      --mount=type=cache,target=/var/lib/apt,sharing=locked \
      apt-get -y update \
      && apt-get -y upgrade \
      && apt-get -y install --no-install-recommends --no-install-suggests \
        ca-certificates jq

RUN groupadd --gid "${USER_GID}" "${USER_NAME}" \
      && useradd --uid "${USER_UID}" --gid "${USER_GID}" --shell /bin/bash --create-home "${USER_NAME}"

USER "${USER_NAME}"

HEALTHCHECK NONE

ENTRYPOINT ["/usr/local/bin/sdeul"]
