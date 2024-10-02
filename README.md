# Learning Optimal Causal Representations by Minimizing the CIB

*Code for the paper: Optimal Causal Representations and the Causal Information Bottleneck*

We apply (simplex) projected gradient descent (pGD) and (simplex) projected simulated annealing gradient descent (pSAGD) to find the minima of the Causal Information Bottleneck (CIB) Lagrangian, for three experiments of increasing complexity, called "Odd and Even", "Confounded Addition", and "Genetic Mutations".
The learned minima are encoders for the optimal causal representations.

This repository consists of two main parts: a Python package `pgd_optim_pytorch` containing implementations of the optimizers to be used, and a directory `cib_optimization` containing the CIB lagrangian, the optimization script, and the experiments, which make use of the package `pgd_optim_pytorch`.

Note that `pgd_optim_pytorch` is not specific to the CIB: it can be used for any minimization problem where the search space is the space of encoders for a representation $T$, that is, any constrained minimization problem whose constraint space corresponds to the probability simplices for the conditional distributions $q_{T\mid X=x}$.

Furthermore, the optimization script `cib_optimization/optimize_cib.py` can easily be adapted to other experiments/causal models.


## Table of Contents
- [Citation](#citation)
- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Usage](#usage)
- [Results](#results)

## Citation
If you use this code, please cite the following paper:

```bibtex
@article{simoes2024optimal,
      title={Optimal Causal Representations and the Causal Information Bottleneck}, 
      author={Francisco N. F. Q. Simoes and Mehdi Dastani and Thijs van Ommen},
      year={2024},
      eprint={2410.00535},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.00535}, 
}
```

## Installation
All commands are to be run from the root directory of the repository.

### Using Pipenv

``` bash
pipenv install
```

### Using Pip
```bash
pip install .
```
or
```bash
pip install -r requirements.txt
```

## Repository Structure
Main directories and files:

``` sh
cib-optimization-psagd
├── cib_optimization                # CIB optimization and experiments
│   ├── optimize_cib.py             # CIB optimization script
│   ├── _cib_lagrangian.py          # Lagrangian implementation
│   ├── *.sh                        # Scripts for all experiments
│   ├── data                        # Experiments' data
│   │   ├── ConfoundedAddition
│   │   │   └── *.py                # Generate data 
│   │   ├── Mutations
│   │   │   ├── *.ipynb             # Generate joint table
│   │   │   └── *.py                # Generate data from joint table
│   │   └── OddAndEven
│   │       └── *.py                # Generate data 
│   └── Results                     # Experimental results
│       ├── ConfoundedAddition 
│       │   ├── *.ipynb             # Analysis of experiment results
│       │   └── *.html              # Interactive parallel plot with main results
│       ├── Mutations                 
│       │   ├── *.ipynb             # Analysis of experiment results
│       │   └── *.html              # Interactive parallel plot with main results
│       └── OddEven                    
│           ├── *.ipynb             # Analysis of experiment results
│           └── *.html              # Interactive parallel plot with main results
├── pgd_optim_pytorch               # Optimizers package 
├── Pipfile                         # Package requirements (Pipenv)
├── requirements.txt                # Package dependencies (pip)
├── setup.py                        # Package setup script (pip)
└── mlruns                          # MLflow experiment tracking
```

## Usage

### Set-up
MLflow will track the runs.
It will need to run in the background whenever one wants to run an optimizer using `optimize_cib.py`.
To run MLflow:

### If you use pipenv
``` sh
pipenv run mlflow ui
```

### If you use pip
``` sh
mlflow ui
```

This command will output the URL you can use to interact with the MLflow UI.

### Learning an encoder for one of the experiments' models
You can execute one optimization run using `cib_optimization/optimize_cib.py`.
For example, to optimize the CIB with $\gamma=1.0$ for the Confounded Addition experiment using the SAGD optimizer:
``` sh
pipenv run python tests/optimize_cib.py --experiment="confounded-addition" --optimizer_algo="pSAGD" --experiment_name "Default" --gamma=1.0 --r_y=0.5 --lr=1.0 --temperature=10.0 --max_iter=1000 
```
Run `cib_optimization/optimize_cib.py --help` to see all the arguments accepted by the script.

### Running an entire experiment
Alternatively, if one wants to execute an experiment, one can use one of the bash scripts.
For the Confounded Addition and Genetic Mutations experiments, there are two types experiments: one exploring different learning rates for the case $\gamma = 1.0$, and another one learning representations for different $\gamma$ values using ensembles of optimizers.
The later is the one described in the paper, while the former simply informed us what learning rates to use for the ensembles of the latter.

For example, running
``` sh
bash cib_optimization/cib_sagd_confounded_addition_other_gammas.sh
```
will replicate the experiment "Confounded Addition - other gammas" in MLflow.

Note that, if all you want is to check the results of the experiments, there is no need to run them yourself, since they are already stored in the `mlruns` (see [Results](#results)).


## Results
The results of our experiments were stored in `mlruns`, and can be analyzed using the MLflow UI or with the help of the IPython notebooks in the `cib_optimization/Results` directory, which were used to generate the parallel plots that can be seen in the paper, and also interactive (and more complete) versions of those in `html` format, which you can also find in the subdirectories of `cib_optimization/Results`.
