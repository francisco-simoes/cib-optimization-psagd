"""Set up the command-line interface (CLI) for the optimization script."""


import argparse
from typing import Any


def create_parser(defaults: dict[str, Any]):
    parser = argparse.ArgumentParser(description="Minimize the CIB Lagrangian.")

    # Add required command-line arguments to select experiment and optimizer algorithm
    possible_experiments = ["odd-and-even", "confounded-addition", "mutations"]
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=possible_experiments,
        help="The experiments to run. Choices are: "
        + ", ".join(possible_experiments),  # Help description including choices
    )
    possible_optimizer_algorithms = ["pGD", "pSAGD", "pAdam"]
    parser.add_argument(
        "--optimizer_algo",
        type=str,
        required=True,
        choices=possible_optimizer_algorithms,
        help="Optimizer algorithm to use. Choices are: "
        + ", ".join(
            possible_optimizer_algorithms
        ),  # Help description including choices
    )

    # Add optional command-line arguments to override constants
    parser.add_argument(
        "--beta",
        type=float,
        default=defaults["beta"],
        help=f"Value of beta (default: {defaults['beta']}).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=defaults["gamma"],
        help=f"Value of gamma (default: {defaults['gamma']}).",
    )
    parser.add_argument(
        "--use-penalty",
        dest="use_penalty",
        action="store_true",
        help="Enable the non-surjectivity penalty, which is inactive by default.",
    )
    parser.set_defaults(use_penalty=False)
    parser.add_argument(
        "--r_y",
        type=float,
        default=defaults["r_y"],
        help=f"For Conf. Addition experiment (default: {defaults['r_y']})",
    )
    parser.add_argument(
        "--uncertainty_y",
        type=float,
        default=defaults["uncertainty_y"],
        help=f"For Odd and Even experiment (default: {defaults['uncertainty_y']})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=defaults["lr"],
        help=f"Learning rate (default: {defaults['lr']})",
    )
    parser.add_argument(
        "--end_lr_factor",
        type=float,
        default=defaults["end_factor"],
        help=f"End factor for Learning rate schedule (default: {defaults['end_factor']})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=defaults["temp"],
        help=f"Initial temperature for simulated annealing (default: {defaults['temp']})",
    )
    parser.add_argument(
        "--cooling_rate",
        type=float,
        default=defaults["cool_rate"],
        help=f"(Additive) cooling rate for simulated annealing (default: {defaults['cool_rate']})",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=defaults["max_iter"],
        help=f"Value of max_iter (default: {defaults['max_iter']})",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=defaults["eps"],
        help=f"Value of eps (default: {defaults['eps']})",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=defaults["experiment_name"],
        help=f"Suffix for run name (default: '{defaults['experiment_name']}')",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Suffix for run name (default: empty string)",
    )

    return parser
