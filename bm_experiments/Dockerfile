FROM ubuntu:bionic

# This Dockerfile installs pytorch/xla 3.7 wheels. There are also 3.6 wheels available; see below.
ARG PYTHON_VERSION=3.7

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        $( [ ${PYTHON_VERSION%%.*} -ge 3 ] && echo "python${PYTHON_VERSION%%.*}-distutils" ) \
        build-essential \
        cmake \
        wget \
        unzip \
        ca-certificates \
    && \

# Install python dependencies
    wget https://bootstrap.pypa.io/get-pip.py --progress=bar:force:noscroll --no-check-certificate && \
    python${PYTHON_VERSION} get-pip.py && \
    rm get-pip.py && \

# Set the default python and install PIP packages
    update-alternatives --install /usr/bin/python${PYTHON_VERSION%%.*} python${PYTHON_VERSION%%.*} /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 && \

# Cleaning
    apt-get autoremove -y && \
    apt-get clean

RUN \
    # BIRL requirement
    apt-get update && apt-get install -y --no-install-recommends \
        openslide-tools \
    && \

# Download BIRL from master
    # TODO: use master...
    wget https://github.com/Borda/BIRL/archive/docker.zip --progress=bar:force:noscroll && \
    unzip -q docker.zip && \
    mv BIRL-docker BIRL && \

    # wget https://github.com/Borda/BIRL/archive/master.zip --progress=bar:force:noscroll && \
    # unzip -q master.zip && \
    # rm master.zip && \
    # mv BIRL-master BIRL && \

# Install BIRL
    # v46 crashes openslide-python install
    pip install "setuptools<46" -U && \
    pip install ./BIRL && \
    python -c "import birl ; print(birl.__version__)" && \
    mkdir BIRL/Applications


RUN \
    cd BIRL/Applications && \
    # add Fiji - needed for RVSS, bUnwarpJ
    wget https://downloads.imagej.net/fiji/latest/fiji-linux64.zip --progress=bar:force:noscroll && \
    unzip -q fiji-linux64.zip && \
    rm fiji-linux64.zip && \
    # https://imagej.nih.gov/ij/docs/guide/146-18.html
    Fiji.app/ImageJ-linux64 -eval "return getVersion();"

RUN \
    cd BIRL && \
    mkdir results && \

    python bm_experiments/bm_bUnwarpJ.py \
        -t ./data-images/pairs-imgs-lnds_histol.csv \
        -d ./data-images \
        -o ./results \
        -Fiji Applications/Fiji.app/ImageJ-linux64 \
        -cfg ./configs/ImageJ_bUnwarpJ_histol.yaml \
        --unique \
    && \

    python bm_experiments/bm_RVSS.py \
        -t ./data-images/pairs-imgs-lnds_histol.csv \
        -d ./data-images \
        -o ./results \
        -Fiji Applications/Fiji.app/ImageJ-linux64 \
        -cfg ./configs/ImageJ_RVSS_histol.yaml \
        --unique \

