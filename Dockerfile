FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND noninteractive

ADD https://bootstrap.pypa.io/get-pip.py /tmp/get-pip.py

RUN set -e \
      && apt-get -y update \
      && apt-get -y upgrade \
      && apt-get -y install --no-install-recommends --no-install-suggests \
        g++ python3 python3-distutils \
      && apt-get -y autoremove \
      && apt-get clean \
      && rm -rf /var/lib/apt/lists/*

RUN set -e \
      && /usr/bin/python3 /tmp/get-pip.py \
      && pip install -U --no-cache-dir pip \
      && pip install -U --no-cache-dir \
        docopt jsonschema langchain langchain-google-genai \
        llama-cpp-python openai

ADD . /tmp/sdeul

RUN set -e \
      && pip install -U --no-cache-dir /tmp/sdeul \
      && rm -rf /tmp/get-pip.py /tmp/sdeul


FROM ubuntu:24.04

ENV DEBIAN_FRONTEND noninteractive

COPY --from=builder /usr/local /usr/local

RUN set -e \
      && ln -sf bash /bin/sh \
      && ln -s python3 /usr/bin/python

RUN set -e \
      && apt-get -y update \
      && apt-get -y upgrade \
      && apt-get -y install --no-install-recommends --no-install-suggests \
        ca-certificates curl jq python3 python3-distutils \
      && apt-get -y autoremove \
      && apt-get clean \
      && rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["/usr/local/bin/sdeul"]
