#!/bin/bash
# Description: This script runs an ensemble of pGD optimizers for each
# value of gamma in a chosen range, for the Odd and Even experiment.
#
# Usage (from root directory of the repository): bash tests/cib_gd_odd_and_even_grid_search.sh

EXPERIMENT="odd-and-even"
OPTIMIZER_ALGO="pGD"
EXPERIMENT_NAME="Odd and Even"
MAX_ITER=1000
HALF_ENSEMBLE_SIZE=2 # The full ensemble will have size HALF_ENSEMBLE_SIZE * length of lr_values


uncertainty_y_values=$(seq 0 0.1 0.5)
gammas=$(seq 0 0.2 1)
lr_values=(1e-1 1e0)

for uncertainty_y in ${uncertainty_y_values}; do
    for gamma in ${gammas}; do
        for lr in ${lr_values[@]}; do
            for i in $(seq 1 $HALF_ENSEMBLE_SIZE); do
                echo "=== Running number $i for gamma=${gamma}, uncertainty_Y=${uncertainty_y}, lr=$lr"
                pipenv run python tests/optimize_cib.py \
                    --experiment="${EXPERIMENT}" \
                    --optimizer_algo="${OPTIMIZER_ALGO}" \
                    --experiment_name="${EXPERIMENT_NAME}" \
                    --lr=${lr} \
                    --gamma=${gamma} \
                    --uncertainty_y=${uncertainty_y} \
                    --max_iter=${MAX_ITER}
            done
        done
    done
done
