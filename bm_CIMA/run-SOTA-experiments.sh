#!/bin/bash

export DISPLAY=""
# DEFINE GLOBAL PARAMS
jobs=5
table="./bm_CIMA/dataset_CIMA.csv"
# this folder has to contain bland of images and landmarks
dataset="~/Medical-data/dataset_CIMA"
results="~/Medical-temp/experiments_CIMA/"


python ./bm_experiments/bm_bUnwarpJ.py \
     -t $table -d $dataset -o $results \
     --run_comp_benchmark \
     -Fiji ~/TEMP/Applications/Fiji.app/ImageJ-linux64 \
     -cfg ./configs/ImageJ_bUnwarpJ_histol.yaml \
     --visual --unique --nb_workers $jobs

python ./bm_experiments/bm_bUnwarpJ.py \
     -t $table -d $dataset -o $results \
     --run_comp_benchmark \
     -Fiji ~/TEMP/Applications/Fiji.app/ImageJ-linux64 \
     -cfg ./configs/ImageJ_bUnwarpJ-SIFT_histol.yaml \
     --visual --unique --nb_workers $jobs

python ./bm_experiments/bm_DROP2.py \
     -t $table -d $dataset -o $results \
     --run_comp_benchmark \
     -DROP ~/TEMP/Applications/DROP2/dropreg \
     -cfg ./configs/DROP2.txt \
     --visual --unique --nb_workers $jobs

python ./bm_experiments/bm_elastix.py \
     -t $table -d $dataset -o $results \
     --run_comp_benchmark \
     -elastix ~/Applications/elastix/bin \
     -cfg ./configs/elastix_bspline.txt \
     --visual --unique --nb_workers $jobs

python ./bm_experiments/bm_rNiftyReg.py \
     -t $table -d $dataset -o $results \
     --run_comp_benchmark \
     -R ~/TEMP/Applications/R-3.5.3/bin/Rscript \
     -script ./scripts/Rscript/RNiftyReg_linear.r \
     --visual --unique --nb_workers $jobs

python ./bm_experiments/bm_RVSS.py \
     -t $table -d $dataset -o $results \
     --run_comp_benchmark \
     -Fiji ~/TEMP/Applications/Fiji.app/ImageJ-linux64 \
     -cfg ./configs/ImageJ_RVSS_histol.yaml \
     --visual --unique --nb_workers $jobs

python ./bm_experiments/bm_ANTs.py \
     -t $table -d $dataset -o $results \
     --run_comp_benchmark \
     -ANTs ~/TEMP/Applications/antsbin/bin \
     -cfg ./configs/ANTs_SyN.txt \
     --visual --unique --nb_workers $jobs
