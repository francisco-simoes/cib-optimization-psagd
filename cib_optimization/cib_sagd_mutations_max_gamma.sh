#!/bin/bash
# Description: This script runs a hyperparameter search for pSAGD applied to
# the CIB loss with gamma=1, for the Genetic Mutations experiment.
#
# Usage (from root directory of the repository): bash tests/cib_sagd_mutations_max_gamma.sh.


EXPERIMENT="mutations"
OPTIMIZER_ALGO="pSAGD"
EXPERIMENT_NAME="Mutations - max gamma"
COOL_RATE=0.99
GAMMA=1.0
MAX_ITER=1000
RUN_NUMBER=100

lr_values=(1e1 1e2 1e3 1e4 1e5 1e6 1e7)
temp_values=(1e1)

for lr in ${lr_values[@]}; do
        for temp in ${temp_values[@]}; do
                for i in $(seq 1 $RUN_NUMBER); do
                        echo "=== Run number $i for lr=$lr and temp=$temp ==="
                        pipenv run python tests/optimize_cib.py \
                                --experiment="${EXPERIMENT}" \
                                --optimizer_algo="${OPTIMIZER_ALGO}" \
                                --experiment_name="${EXPERIMENT_NAME}" \
                                --lr=${lr} \
                                --temperature=${temp} \
                                --cooling_rate=${COOL_RATE} \
                                --gamma=${GAMMA} \
                                --use-penalty \
                                --max_iter=${MAX_ITER}
                done
        done
done
