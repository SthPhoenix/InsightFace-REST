FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt-get update && \
    apt-get install -y software-properties-common build-essential &&\
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends libgl1-mesa-glx wget python3.7 python3-distutils-extra

RUN  wget -O ~/get-pip.py \
        https://bootstrap.pypa.io/get-pip.py && \
    python3.7 ~/get-pip.py && \
    ln -s /usr/bin/python3.7 /usr/local/bin/python3 && \
    ln -s /usr/bin/python3.7 /usr/local/bin/python

RUN pip install mxnet-cu100 torch torchvision numpy
RUN pip install insightface fastapi uvicorn ujson facenet-pytorch aiofiles

RUN apt-get install -y


WORKDIR /app
COPY api /app

ENTRYPOINT [ "uvicorn" ]

CMD [ "fapp:app", "--reload", "--port", "18080","--host","0.0.0.0","--workers","10"]
