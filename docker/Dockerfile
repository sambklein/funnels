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

ENV PYTHONPATH=.
RUN cd /usr/local/lib/python3.7/site-packages && \
    python3 setup.py develop

RUN python3 -m pip install -r requirements.txt

EXPOSE 8888
