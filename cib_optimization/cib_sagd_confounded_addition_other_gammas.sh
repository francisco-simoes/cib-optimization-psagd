#!/bin/bash
# Description: This script runs an ensemble of pSAGD optimizers for each
# value of gamma in a chosen range, for the Confounded Addition experiment.
#
# Usage (from root directory of the repository): bash cib-optimization/cib_sagd_confounded_addition_other_gammas.sh

EXPERIMENT="confounded-addition"
OPTIMIZER_ALGO="pSAGD"
EXPERIMENT_NAME="Confounded Addition - other gammas"
TEMP=10.0
COOL_RATE=0.99
MAX_ITER=1000
HALF_ENSEMBLE_SIZE=3 # The full ensemble will have size HALF_ENSEMBLE_SIZE * length of lr_values


ry_values=(0.1 0.5 0.9)
gammas=$(seq 0 0.2 1)
lr_values=(1e0 1e1) # 1e0 can be better, especially for lower gammas.

for ry in ${ry_values[@]}; do
    for gamma in ${gammas[@]}; do
        for lr in ${lr_values[@]}; do
            for i in $(seq 1 $HALF_ENSEMBLE_SIZE); do
                echo "=== Run number $i for ry=$ry, gamma=$gamma, lr=$lr ==="
                pipenv run python cib-optimization/optimize_cib.py \
                    --experiment="${EXPERIMENT}" \
                    --optimizer_algo="${OPTIMIZER_ALGO}" \
                    --experiment_name="${EXPERIMENT_NAME}" \
                    --lr=${lr} \
                    --temperature=${TEMP} \
                    --cooling_rate=${COOL_RATE} \
                    --gamma=${gamma} \
                    --use-penalty \
                    --r_y=${ry} \
                    --max_iter=${MAX_ITER}
            done
        done
    done
done
