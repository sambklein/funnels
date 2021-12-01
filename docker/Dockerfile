ARG TORCH_VERSION=latest

FROM pytorch/pytorch:${TORCH_VERSION}#this will get replaced in the ci though

USER root
RUN apt-get -qq -y update && \
    apt-get -qq -y upgrade && \
    apt-get -qq -y install \
        wget \
        curl \
        git \
        sudo \
        unzip \
        p7zip-full \
        fastjar

COPY docker/requirements.txt .
COPY . .

RUN python3 -m pip install -r requirements.txt

RUN bash -c 'echo -e $(pwd)'

ENV PYTHONPATH=/workspace
RUN cd /opt/conda/lib/python3.8/site-packages && \
    python3 /workspace/setup.py develop

EXPOSE 8888
