FROM tensorflow/tensorflow:latest-py3

RUN apt-get update \
    && apt-get install -y \
        build-essential \
        cmake \
        wget \
        unzip \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavformat-dev \
        libpq-dev \
	libglib2.0-0 \
	libsm6 \
	libxext6 \ 
	libxrender-dev \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip --no-cache-dir install -r requirements.txt
