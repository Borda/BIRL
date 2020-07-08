FROM ubuntu:bionic

# This Dockerfile installs pytorch/xla 3.7 wheels. There are also 3.6 wheels available; see below.
ARG PYTHON_VERSION=3.7

SHELL ["/bin/bash", "-c"]

# for installing tzdata see: https://stackoverflow.com/a/58264927/4521646
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        $( [ ${PYTHON_VERSION%%.*} -ge 3 ] && echo "python${PYTHON_VERSION%%.*}-distutils" ) \
        python${PYTHON_VERSION%%.*}-tk \
        build-essential \
        cmake \
        wget \
        unzip \
        git \
        ca-certificates \
    && \

# Install python dependencies
    wget https://bootstrap.pypa.io/get-pip.py --progress=bar:force:noscroll --no-check-certificate && \
    python${PYTHON_VERSION} get-pip.py && \
    rm get-pip.py && \

# Set the default python and install PIP packages
    update-alternatives --install /usr/bin/python${PYTHON_VERSION%%.*} python${PYTHON_VERSION%%.*} /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 && \

# Disable cache
    pip config set global.cache-dir false && \
    pip install "pip>20.1" -U  && \

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
    pip install ./BIRL --no-cache-dir && \
    python -c "import birl ; print(birl.__version__)" && \
    mkdir BIRL/Applications

ENV BIRL_APPs_PATH="/BIRL/Applications"


RUN \
    cd $BIRL_APPs_PATH && \
# add Fiji - needed for RVSS, bUnwarpJ
    # wget https://downloads.imagej.net/fiji/latest/fiji-linux64.zip --progress=bar:force:noscroll && \
    wget https://downloads.imagej.net/fiji/archive/20200625-1201/fiji-linux64.zip --progress=bar:force:noscroll && \
    unzip -q fiji-linux64.zip && \
    rm fiji-linux64.zip && \
    # https://imagej.nih.gov/ij/docs/guide/146-18.html
    Fiji.app/ImageJ-linux64 -eval "return getVersion();"

ENV ANTs_VERSION="2.3.4"

RUN \
    cd $BIRL_APPs_PATH && \
# Compile ANTs
    wget https://github.com/ANTsX/ANTs/archive/v$ANTs_VERSION.zip --progress=bar:force:noscroll && \
    unzip -q v$ANTs_VERSION.zip && \
    rm v$ANTs_VERSION.zip && \
    mv ANTs-$ANTs_VERSION ANTs && \
    mkdir antsbin && \
    cd antsbin && \
    cmake ../ANTs && \
    make -j$(nproc) && \
# remove source folder
    rm -rf $BIRL_APPs_PATH/ANTs && \
# clean some remaining in antsbin
    mkdir $BIRL_APPs_PATH/ANTs-regist && \
    cd $BIRL_APPs_PATH/antsbin/ANTS-build/Examples/ && \
    mv simpleSynRegistration $BIRL_APPs_PATH/ANTs-regist/ && \
    mv antsRegistration $BIRL_APPs_PATH/ANTs-regist/ && \
    mv antsApplyTransforms $BIRL_APPs_PATH/ANTs-regist/ && \
    mv antsApplyTransformsToPoints $BIRL_APPs_PATH/ANTs-regist/ && \
    rm -rf $BIRL_APPs_PATH/antsbin && \
# try to run ANTs
    $BIRL_APPs_PATH/ANTs-regist/antsRegistration --help

RUN \
# install ANTsPy
    pip install git+https://github.com/ANTsX/ANTsPy.git && \
    python -c "import ants ; print(ants.__version__)"

ENV ELASTIX_VERSION="5.0.0"

RUN \
# add Elastix
    cd $BIRL_APPs_PATH && \
    wget https://github.com/SuperElastix/elastix/releases/download/${ELASTIX_VERSION}/elastix-${ELASTIX_VERSION}-Linux.tar.bz2 --progress=bar:force:noscroll && \
    mkdir elastix && \
    tar xjf elastix-${ELASTIX_VERSION}-Linux.tar.bz2 --directory=elastix && \
    rm elastix-${ELASTIX_VERSION}-Linux.tar.bz2 && \
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$BIRL_APPs_PATH/elastix/lib/" && \
# Try run elastix
    $BIRL_APPs_PATH/elastix/bin/elastix --help

RUN \
# add required libs
    apt-get update && apt-get install -y --no-install-recommends \
        libtbb-dev \
        libboost-all-dev \
    && \
    cd $BIRL_APPs_PATH && \
# compile the DROP2
    git clone https://github.com/biomedia-mira/drop2.git && \
    cd drop2 && \
    bash build.sh && \
# clean up abter build
    apt-get remove -y \
        libtbb-dev \
        libboost-all-dev \
    && \
    apt-get clean && \
    mkdir cd $BIRL_APPs_PATH/DROP2 && \
    mv $BIRL_APPs_PATH/drop2/build/drop/apps/dropreg/* $BIRL_APPs_PATH/DROP2 && \
    rm -rf $BIRL_APPs_PATH/drop2 && \
# try to run DROP2
    $BIRL_APPs_PATH/DROP2/dropreg --help


RUN \
# show installed applications
    ls -l $BIRL_APPs_PATH && \
    cd BIRL && \
    mkdir results && \

# Run all experimenst with minimal dataset

    python bm_experiments/bm_bUnwarpJ.py \
        -t ./data-images/pairs-imgs-lnds_histol.csv \
        -d ./data-images \
        -o ./results \
        -Fiji $BIRL_APPs_PATH/Fiji.app/ImageJ-linux64 \
        -cfg ./configs/ImageJ_bUnwarpJ_histol.yaml \
        --unique \
    && \

    python bm_experiments/bm_RVSS.py \
        -t ./data-images/pairs-imgs-lnds_histol.csv \
        -d ./data-images \
        -o ./results \
        -Fiji $BIRL_APPs_PATH/Fiji.app/ImageJ-linux64 \
        -cfg ./configs/ImageJ_RVSS_histol.yaml \
        --unique \
    && \

    python bm_experiments/bm_ANTs.py \
        -t ./data-images/pairs-imgs-lnds_histol.csv \
        -d ./data-images \
        -o ./results \
        --path_ANTs $BIRL_APPs_PATH/ANTs-regist \
        --path_config ./configs/ANTs_SyN.txt \
        --unique \
    && \

    python bm_experiments/bm_ANTsPy.py \
        -t ./data-images/pairs-imgs-lnds_histol.csv \
        -d ./data-images \
        -o ./results \
        -py python3 \
        -script ./scripts/Python/run_ANTsPy.py \
        --unique \
    && \

    python bm_experiments/bm_elastix.py \
        -t ./data-images/pairs-imgs-lnds_histol.csv \
        -d ./data-images \
        -o ./results \
        -elastix $BIRL_APPs_PATH/elastix/bin \
        -cfg ./configs/elastix_affine.txt \
        --unique \
    && \

    python bm_experiments/bm_DROP2.py \
        -t ./data-images/pairs-imgs-lnds_histol.csv \
        -d ./data-images \
        -o ./results \
        -DROP $BIRL_APPs_PATH/DROP2/dropreg \
        --path_config ./configs/DROP2.txt \
        --unique \
    && \

# see and clean resuls
    ls -l ./results #&& \
    rm -rf ./results