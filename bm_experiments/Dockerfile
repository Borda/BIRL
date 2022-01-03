#FROM ubuntu:bionic
FROM ubuntu:20.04

ARG PYTHON_VERSION=3.7

LABEL maintainer="jiri.borovec@fel.cvut.cz"

SHELL ["/bin/bash", "-c"]

# for installing tzdata see: https://stackoverflow.com/a/58264927/4521646
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq --fix-missing && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update -qq --fix-missing && \
    apt-get install -y --no-install-recommends \
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
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV BIRL_APPs_PATH="/BIRL/Applications"

RUN mkdir -p $BIRL_APPs_PATH

RUN \
    cd $BIRL_APPs_PATH && \
# add Fiji - needed for RVSS, bUnwarpJ
    # wget https://downloads.imagej.net/fiji/latest/fiji-linux64.zip --progress=bar:force:noscroll && \
    wget https://downloads.imagej.net/fiji/archive/20200708-1553/fiji-linux64.zip --progress=bar:force:noscroll && \
    unzip -q fiji-linux64.zip && \
    rm fiji-linux64.zip && \
    # https://imagej.nih.gov/ij/docs/guide/146-18.html
    Fiji.app/ImageJ-linux64 -eval "return getVersion();"

ENV ANTs_VERSION="2.3.4"

RUN \
    apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        zlib1g-dev \
    && \
    cd $BIRL_APPs_PATH && \
# Compile ANTs
    wget https://github.com/ANTsX/ANTs/archive/v$ANTs_VERSION.zip --progress=bar:force:noscroll && \
    unzip -q v$ANTs_VERSION.zip && \
    rm v$ANTs_VERSION.zip && \
    mv ANTs-$ANTs_VERSION ANTs && \
    mkdir antsbin && \
    cd antsbin && \
    cmake ../ANTs \
        -D BUILD_ALL_ANTS_APPS=OFF \
        -D CMAKE_BUILD_TYPE=Release \
        -D BUILD_SHARED_LIBS=OFF \
        -D BUILD_TESTING=OFF \
        -D RUN_SHORT_TESTS=OFF \
        -D RUN_LONG_TESTS=OFF \
    && \
    make -j$(nproc) ANTS && \
# remove source folder
    rm -rf $BIRL_APPs_PATH/ANTs && \
# clean some remaining in antsbin
    mkdir $BIRL_APPs_PATH/ANTs-regist && \
    cd $BIRL_APPs_PATH/antsbin/ANTS-build/Examples/ && \
    # mv simpleSynRegistration $BIRL_APPs_PATH/ANTs-regist/ && \
    mv antsRegistration $BIRL_APPs_PATH/ANTs-regist/ && \
    mv antsApplyTransforms $BIRL_APPs_PATH/ANTs-regist/ && \
    mv antsApplyTransformsToPoints $BIRL_APPs_PATH/ANTs-regist/ && \
    rm -rf $BIRL_APPs_PATH/antsbin && \
# clean up after build
    apt-get remove -y \
        zlib1g-dev \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
# try to run ANTs
    $BIRL_APPs_PATH/ANTs-regist/antsRegistration --help

ENV ANTSPY_VERSION="0.2.5"

RUN \
    apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        libopenblas-dev \
        liblapack-dev \
        gfortran \
    && \
# install ANTsPy
    # requirements: https://github.com/ANTsX/ANTsPy/blob/v${ANTSPY_VERSION}/requirements.txt
    pip install --no-cache-dir https://github.com/ANTsX/ANTsPy/archive/v${ANTSPY_VERSION}.zip && \
# clean up after build
    apt-get remove -y \
        libopenblas-dev \
        liblapack-dev \
        gfortran \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
# test run
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

ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$BIRL_APPs_PATH/elastix/lib/"

RUN \
    apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        libtbb-dev \
        libboost-all-dev \
    && \
    cd $BIRL_APPs_PATH && \
# compile the DROP2
    git clone https://github.com/biomedia-mira/drop2.git && \
    cd drop2 && \
    # bash build.sh && \

# This is little editted internal script build.sh
    TEMP_PATH_DROP=$(pwd) && \
    # prepare 3-party libs
    export THIRD_PARTY_DIR=$TEMP_PATH_DROP/3rdParty && \
    mkdir $THIRD_PARTY_DIR && \
    DROP_EIGEN_VERSION="3.3.7" && \
    DROP_ITK_VERSION="5.0.0" && \

    # get Eigen
    cd $THIRD_PARTY_DIR && \
    wget https://gitlab.com/libeigen/eigen/-/archive/${DROP_EIGEN_VERSION}/eigen-${DROP_EIGEN_VERSION}.tar.gz --progress=bar:force:noscroll && \
    mkdir eigen && \
    tar xf eigen-${DROP_EIGEN_VERSION}.tar.gz -C eigen --strip-components=1 && \

    # download and install ITK
    cd $THIRD_PARTY_DIR && \
    wget https://sourceforge.net/projects/itk/files/itk/${DROP_ITK_VERSION%.*}/InsightToolkit-${DROP_ITK_VERSION}.tar.gz --progress=bar:force:noscroll && \
    tar xf InsightToolkit-${DROP_ITK_VERSION}.tar.gz && \
    cd InsightToolkit-${DROP_ITK_VERSION} && \
    mkdir build && \
    cd build && \
    cmake \
        -D BUILD_EXAMPLES:BOOL=OFF \
        -D BUILD_TESTING:BOOL=OFF \
        -D CMAKE_INSTALL_PREFIX=$THIRD_PARTY_DIR/itk \
       .. \
    && \
    make -j$(nproc) && \
    make install && \

    # building the
    cd $TEMP_PATH_DROP && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j$(nproc) && \
    unset THIRD_PARTY_DIR && \

# Cleaning the build folder
    mkdir $BIRL_APPs_PATH/DROP2 && \
    mv $BIRL_APPs_PATH/drop2/build/drop/apps/dropreg/* $BIRL_APPs_PATH/DROP2 && \
    rm -rf $TEMP_PATH_DROP && \
# clean up after build
    apt-get remove -y \
        libtbb-dev \
        libboost-all-dev \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
# try to run DROP2
    $BIRL_APPs_PATH/DROP2/dropreg --help

RUN \
    apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        r-base-core \
        r-base-dev \
        #libfftw3-dev \
        libtiff5-dev \
        libcurl4-gnutls-dev \
        libxml2-dev \
        libssl-dev \
        libgit2-dev \
    && \
# install RNiftyReg
    R -e 'install.packages(c("png", "jpeg", "OpenImageR", "devtools"))' && \
    R -e 'devtools::install_github("jonclayden/RNiftyReg")' && \
# clean up abter build
    apt-get remove -y \
        r-base-dev \
        #libfftw3-dev \
        libtiff5-dev \
        libcurl4-gnutls-dev \
        libxml2-dev \
        libssl-dev \
        libgit2-dev \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
# try to run RNiftyReg
    R -e 'library(RNiftyReg)'

COPY ./ /BIRL/

RUN \
    apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        openslide-tools \
    && \
# Install BIRL
    # v46 crashes openslide-python install
    pip install "setuptools<46" -U && \
    pip install ./BIRL --no-cache-dir && \
    python -c "import birl ; print(birl.__version__)" && \
# Cleaning
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


RUN \
    apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        tree \
    && \
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

    python bm_experiments/bm_rNiftyReg.py \
        -t ./data-images/pairs-imgs-lnds_histol.csv \
        -d ./data-images \
        -o ./results \
        -R Rscript \
        -script ./scripts/Rscript/RNiftyReg_linear.r \
        --unique \
    && \

# see and clean resuls
    tree -l ./results && \
    rm -rf ./results && \
    apt-get remove -y \
        tree \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
