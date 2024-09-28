#!/bin/bash
# Description: This script runs a hyperparameter search for pSAGD applied to
# the CIB loss with gamma=1, for the Confounded Addition experiment.
#
# Usage (from root directory of the repository): bash tests/cib_sagd_confounded_addition_max_gamma.sh.

EXPERIMENT="confounded-addition"
OPTIMIZER_ALGO="pSAGD"
EXPERIMENT_NAME="Confounded Addition - max gamma"
COOL_RATE=0.99
GAMMA=1.0
R_Y=0.1
MAX_ITER=1000
RUN_NUMBER=100

lr_values=(1e0 1e1)
temp_values=(1e1 1e2)

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
                                --r_y=${R_Y} \
                                --max_iter=${MAX_ITER}
                done
        done
done


# Same as above, but using non-surjectivity penalty
lr_values=(1e0)
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
                                --r_y=${R_Y} \
                                --max_iter=${MAX_ITER}
                done
        done
done
