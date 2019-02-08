#FROM ubuntu:16.04
FROM python:3.6-slim-stretch
#FROM borda/docker_python-opencv-ffmpeg:py3


RUN apt-get update && apt-get install -y --no-install-recommends \
        git>=2.0 \
        gcc>=5.1 \
        tk-dev>=8.5 \
#        python-dev \
#        python-tk \
#        pkg-config \
#        python-opencv \
#        openslide-tools \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \

    groupadd -r evaluator && \
    useradd -m --no-log-init -r -g evaluator evaluator && \
    mkdir -p /opt/evaluation /input /output && \
    chown evaluator:evaluator /opt/evaluation /input /output \
    && \

# for Ubuntu instalation
#    wget https://bootstrap.pypa.io/get-pip.py --progress=bar:force:noscroll && \
#    python get-pip.py && \
    pip install --upgrade --force-reinstall --no-cache-dir git+https://github.com/Borda/BIRL.git>=0.2 && \
#    pip install git+https://github.com/Borda/BIRL.git@anhi

    apt-get -y remove \
        gcc \
        git \
    && \
    apt-get autoremove -y && \
    apt-get clean


#USER evaluator
WORKDIR /opt/evaluation

ENV PATH="/home/evaluator/.local/bin:${PATH}"

COPY --chown=evaluator:evaluator ./evaluate_submission.py /opt/evaluation/
COPY --chown=evaluator:evaluator ./dataset_ANHIR/dataset_medium.csv /opt/evaluation/dataset.csv
COPY --chown=evaluator:evaluator ./dataset_ANHIR/computer-performances_cmpgrid-71.json /opt/evaluation/computer-performances.json
COPY --chown=evaluator:evaluator ./dataset_ANHIR/landmarks_extend /opt/evaluation/lnds_provided
COPY --chown=evaluator:evaluator ./dataset_ANHIR/landmarks_all /opt/evaluation/lnds_reference

ENTRYPOINT "python" "evaluate_submission.py" \
    "-e" "/input" \
    "-c" "/opt/evaluation/dataset.csv" \
    "-d" "/opt/evaluation/lnds_provided" \
    "-r" "/opt/evaluation/lnds_reference" \
    "-p" "/opt/evaluation/computer-performances.json" \
    "-o" "/output"