#!/bin/bash
# Description: Run standard IB with different gamma values, for all experiments.

METHOD="IB"
EXPERIMENTS=("odd-and-even" "confounded-addition" "mutations")
OPTIMIZER_ALGO="pGD"
EXPERIMENT_NAME="IB tests"
MAX_ITER=1000
LR=1.0

gammas=$(seq 0 0.2 1)

for experiment in ${EXPERIMENTS[@]}; do
    for gamma in ${gammas[@]}; do
        echo "=== Experiment $experiment for gamma=$gamma ==="
        pipenv run python cib_optimization/optimize_cib.py \
            --method="${METHOD}" \
            --experiment="${experiment}" \
            --optimizer_algo="${OPTIMIZER_ALGO}" \
            --experiment_name="${EXPERIMENT_NAME}" \
            --lr=${LR} \
            --gamma=${gamma} \
            --max_iter=${MAX_ITER}
    done
done
