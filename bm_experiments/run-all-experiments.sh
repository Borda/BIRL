#!/bin/bash

export DISPLAY=""
# DEFINE GLOBAL PARAMS
jobs=3
table="~/Medical-data/dataset_ANHIR/images/dataset_medium.csv"
# this folder has to contain bland of images and landmarks
dataset="~/Medical-data/microscopy/TEMPORARY/borovec/dataset_ANHIR/images"
results="~/Medical-data/microscopy/TEMPORARY/borovec/experiments_ANHIR/"
apps="~/Applications"

preprocessings=("" \
                "--preprocessing gray" \
                "--preprocessing matching-rgb" \
                "--preprocessing gray matching-rgb" \
                "--preprocessing matching-rgb gray")

for pproc in "${preprocessings[@]}"
do

    python bm_experiments/bm_ANTs.py \
         -t $table \
         -d $dataset \
         -o $results \
         --run_comp_benchmark \
         -ANTs $apps/antsbin/bin \
         -cfg ./configs/ANTs_SyN.txt \
         $pproc \
         --visual --unique --nb_workers $jobs

    python bm_experiments/bm_ANTsPy.py \
         -t $table \
         -d $dataset \
         -o $results \
         --run_comp_benchmark \
         -py python3 \
         -script ./scripts/Python/run_ANTsPy.py \
         $pproc \
         --visual --unique --nb_workers $jobs

    python bm_experiments/bm_bUnwarpJ.py \
         -t $table \
         -d $dataset \
         -o $results \
         --run_comp_benchmark \
         -Fiji $apps/Fiji.app/ImageJ-linux64 \
         -cfg ./configs/ImageJ_bUnwarpJ_histol.yaml \
         $pproc \
         --visual --unique --nb_workers $jobs

    python bm_experiments/bm_bUnwarpJ.py \
         -t $table \
         -d $dataset \
         -o $results \
         --run_comp_benchmark \
         -Fiji $apps/Fiji.app/ImageJ-linux64 \
         -cfg ./configs/ImageJ_bUnwarpJ-SIFT_histol.yaml \
         $pproc \
         --visual --unique --nb_workers $jobs

    python bm_experiments/bm_DROP2.py \
         -t $table \
         -d $dataset \
         -o $results \
         --run_comp_benchmark \
         -DROP $apps/DROP2/dropreg \
         -cfg ./configs/DROP2.txt \
         $pproc \
         --visual --unique --nb_workers $jobs

    python bm_experiments/bm_elastix.py \
         -t $table \
         -d $dataset \
         -o $results \
         --run_comp_benchmark \
         -elastix $apps/elastix/bin \
         -cfg ./configs/elastix_bspline.txt \
         $pproc \
         --visual --unique --nb_workers $jobs

    python bm_experiments/bm_rNiftyReg.py \
         -t $table \
         -d $dataset \
         -o $results \
         --run_comp_benchmark \
         -R $apps/R-3.5.3/bin/Rscript \
         -script ./scripts/Rscript/RNiftyReg_linear.r \
         $pproc \
         --visual --unique --nb_workers $jobs

    python bm_experiments/bm_RVSS.py \
         -t $table \
         -d $dataset \
         -o $results \
         --run_comp_benchmark \
         -Fiji $apps/Fiji.app/ImageJ-linux64 \
         -cfg ./configs/ImageJ_RVSS_histol.yaml \
         $pproc \
         --visual --unique --nb_workers $jobs

done