#!/bin/bash

export DISPLAY=""
# DEFINE GLOBAL PARAMS
jobs=5
table="./bm_CIMA/dataset_CIMA.csv"
# this folder has to contain bland of images and landmarks
dataset="~/Medical-data/dataset_CIMA"
results="~/Medical-temp/experiments_CIMA/"
apps="~/TEMP/Applications"


python ./bm_experiments/bm_bUnwarpJ.py \
     -t $table -d $dataset -o $results \
     --run_comp_benchmark \
     -Fiji $apps/Fiji.app/ImageJ-linux64 \
     -cfg ./configs/ImageJ_bUnwarpJ_histol.yaml \
     --nb_workers $jobs --unique

python ./bm_experiments/bm_bUnwarpJ.py \
     -t $table -d $dataset -o $results \
     --run_comp_benchmark \
     -Fiji $apps/Fiji.app/ImageJ-linux64 \
     -cfg ./configs/ImageJ_bUnwarpJ-SIFT_histol.yaml \
     --nb_workers $jobs --unique

python ./bm_experiments/bm_DROP2.py \
     -t $table -d $dataset -o $results \
     --run_comp_benchmark \
     -DROP ~/Applications/DROP2/dropreg \
     -cfg ./configs/DROP2.txt \
     --nb_workers $jobs --unique

export LD_LIBRARY_PATH=$apps/elastix/bin:$LD_LIBRARY_PATH
python ./bm_experiments/bm_elastix.py \
     -t $table -d $dataset -o $results \
     --run_comp_benchmark \
     -elastix $apps/elastix/bin \
     -cfg ./configs/elastix_bspline.txt \
     --nb_workers $jobs --unique

python ./bm_experiments/bm_rNiftyReg.py \
     -t $table -d $dataset -o $results \
     --run_comp_benchmark \
     -R $apps/R-3.5.3/bin/Rscript \
     -script ./scripts/Rscript/RNiftyReg_linear.r \
     --nb_workers $jobs --unique

python ./bm_experiments/bm_RVSS.py \
     -t $table -d $dataset -o $results \
     --run_comp_benchmark \
     -Fiji $apps/Fiji.app/ImageJ-linux64 \
     -cfg ./configs/ImageJ_RVSS_histol.yaml \
     --nb_workers $jobs --unique

python ./bm_experiments/bm_ANTs.py \
     -t $table -d $dataset -o $results \
     --run_comp_benchmark \
     -ANTs $apps/antsbin/bin \
     -cfg ./configs/ANTs_SyN.txt \
     --nb_workers $jobs --unique
