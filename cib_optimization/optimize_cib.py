"""Subspace PGD used on the #TODO

"""
from itertools import product
from typing import Any, Callable

import mlflow
import torch
from pgd_optim_pytorch import (SimplexProjectedAdam,
                               SimplexProjectedGradientDescent,
                               SimplexProjectedSAGD)
from torch import Tensor
from torch.optim.lr_scheduler import LinearLR

from _cib_lagrangian import CIBLagrangian
from _cli import create_parser
from _tests_utils import (detect_cycling, log_metrics, monitor_iteration,
                          permute_first_indices, simplex_uniform_sampling,
                          update_history,
                          variation_of_information_of_abstractions)
from data.ConfoundedAddition.generate_confounded_addition_constants import \
    get_counfounded_addition_constants
from data.Mutations.generate_mutations_constants import get_mutations_constants
from data.OddAndEven.generate_odd_and_even_constants import \
    get_odd_and_even_constants

# === Defaults ===
# Stopping conditions and learning rate and max norm
EPS = 1e-5
MAX_ITER = 1000
LR = 1e-3
END_FACTOR = 1.0
MAX_GRAD_NORM = 1.0

# Temperature and cooling rate for SA
TEMP = 100
COOL_RATE = 0.9

# Choose weight of Hc and whether to use penalty term
BETA = 5.5
GAMMA = None
USE_PENALTY = False

# Experiments' parameters
R_Y = 0.1
UNCERTAINTY_Y = 0.1
# BXi = 0.5
BXi = 0.3
BY = 0.1
BS = 0.5

# MLflow
EXPERIMENT_NAME = "Default"


def run_optimizer_on_confounded_addition_cib(
    q: Tensor,
    eps: float,
    max_iter: int,
    loss_func: Callable,
    use_penalty: bool,
    temperature: float,
    cooling_rate: float,
    Xcardinal: int,
    Tcardinal: int,
):
    """q is the initial encoder."""
    if eps < 0.0:
        raise ValueError(f"Invalid eps: {eps} - should be >= 0.0")

    # q is the initial guess for the optimal encoder (before running optimizer)

    # Create custom optimizer
    if OPTIMIZER_ALGO.lower() == "psagd":
        optimizer = SimplexProjectedSAGD(
            [q], Xcardinal, Tcardinal, loss_func, temperature, cooling_rate, lr=LR
        )
    elif OPTIMIZER_ALGO.lower() == "pgd":
        optimizer = SimplexProjectedGradientDescent([q], Xcardinal, Tcardinal, lr=LR)
    elif OPTIMIZER_ALGO.lower() == "padam":
        optimizer = SimplexProjectedAdam([q], Xcardinal, Tcardinal, lr=LR)
    else:
        raise ValueError(f"Invalid optimizer algorithm {OPTIMIZER_ALGO}.")

    # lr schedule
    scheduler = LinearLR(
        # optimizer, start_factor=1.0, end_factor=10e-4, total_iters=max_iter
        optimizer,
        start_factor=1.0,
        end_factor=END_FACTOR,
        total_iters=max_iter,
    )

    # ======= Optimization loop ======
    iteration = 0
    diff_q: float = 2 * eps  # Initializy larger than eps
    diff_loss: float = 2 * eps  # Initializy larger than eps
    # old_loss: float = 10e10

    # Track last two values of q and their losses
    q_last1 = None
    q_last2 = None
    loss_last1 = float("inf")
    loss_last2 = float("inf")

    print("=" * 4 + "Training started" + "=" * 4)
    if GAMMA is None:
        print(f"Beta: {BETA}")
    else:
        print(f"Gamma: {GAMMA}")

    print(f"Initial q: {q}")

    # Initialize loss and components before the loop
    components: dict[str, float]
    loss: Tensor
    loss, components = loss_func(q)  # Initial loss

    # Log and monitor iteration 0
    log_metrics(loss, diff_q, diff_loss, components, iteration)
    monitor_iteration(iteration, optimizer, loss, diff_q, diff_loss, components)
    iteration += 1

    while iteration < MAX_ITER and (
        (hasattr(optimizer, "sa_worked") and optimizer.sa_worked) or diff_q > eps
    ):
        # For pSAGD, stop if non-SA step is small OR max iter reached

        print("-" * 20 + "\n" + f"Iteration [{iteration}]")

        # Cache the current loss and q to compute the differences after the update
        old_loss = loss.item()
        old_tensor: Tensor = q.clone().detach()

        # Backward pass and optimization using the cached loss from the previous iter
        optimizer.zero_grad()  # Clear gradients
        loss.backward()  # Use gradients from the previous loss
        torch.nn.utils.clip_grad_norm_(q, max_norm=MAX_GRAD_NORM)  # Clip gradient

        # Update parameters
        optimizer.step()

        # Compute the difference in q after the update
        diff_q = torch.norm(q - old_tensor)

        # Compute the loss and components after the update
        loss, components = cib_lagrangian.compute_lagrangian(q)

        # Update diff_loss
        new_loss = loss.item()  # Use the last computed loss
        diff_loss = new_loss - old_loss

        # Logging and monitoring
        log_metrics(loss, diff_q, diff_loss, components, iteration)
        monitor_iteration(iteration, optimizer, loss, diff_q, diff_loss, components)

        # Step the scheduler
        scheduler.step()

        # Check if cycling between two values of q
        best_q, cycling_detected = detect_cycling(
            q, q_last1, q_last2, loss_last1, loss_last2, new_loss
        )
        if cycling_detected:
            print("Cycle detected. Stopping loop.")
            q = best_q
            break  # Stop the loop if cycling is detected

        # Update history for the next iteration
        q_last1, q_last2, loss_last1, loss_last2 = update_history(
            q, loss, q_last1, q_last2, loss_last1, loss_last2
        )

        iteration += 1

    q = torch.round(q, decimals=3)

    converged: bool
    if iteration == MAX_ITER:
        converged = False
    else:
        converged = True

    return q, loss, converged, components


if __name__ == "__main__":

    # === CLI ===
    # Create argument parser
    defaults = {
        "eps": EPS,
        "max_iter": MAX_ITER,
        "lr": LR,
        "end_factor": END_FACTOR,
        "temp": TEMP,
        "cool_rate": COOL_RATE,
        "beta": BETA,
        "gamma": GAMMA,
        "r_y": R_Y,
        "uncertainty_y": UNCERTAINTY_Y,
        "experiment_name": EXPERIMENT_NAME,
    }

    parser = create_parser(defaults)

    # Update constants with command-line argument values
    args = parser.parse_args()
    EXPERIMENT = args.experiment
    OPTIMIZER_ALGO = args.optimizer_algo
    BETA = args.beta
    GAMMA = args.gamma
    USE_PENALTY = args.use_penalty
    R_Y = args.r_y
    UNCERTAINTY_Y = args.uncertainty_y
    MAX_ITER = args.max_iter
    EPS = args.eps
    LR = args.lr
    END_FACTOR = args.end_lr_factor
    TEMP = args.temperature
    COOL_RATE = args.cooling_rate
    EXPERIMENT_NAME = args.experiment_name
    run_name_suffix = args.suffix

    # === Start MLflow run ===
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    mlflow.set_experiment(EXPERIMENT_NAME)

    run_name_prefix: str
    if GAMMA is None:
        run_name_prefix = f"beta={BETA}-"
    else:
        run_name_prefix = f"gamma={GAMMA}-"
    run_name_params: str
    if EXPERIMENT.lower() == "odd-and-even":
        mlflow.start_run(run_name=run_name_prefix + f"-uncertainty_y={UNCERTAINTY_Y}")
    elif EXPERIMENT.lower() == "confounded-addition":
        mlflow.start_run(run_name=run_name_prefix + f"-ry={R_Y}")
    elif EXPERIMENT.lower() == "mutations":
        mlflow.start_run(run_name=run_name_prefix + f"-bxi={BXi},by={BY},bs={BS}")

    # === Example set-up ===
    experiment_constants: dict[str, Any]
    experiment_specific_params: dict[str, Any]  # For mlflow logging
    if EXPERIMENT.lower() == "odd-and-even":
        experiment_constants = get_odd_and_even_constants(UNCERTAINTY_Y)
        experiment_specific_params = {"uncertainty_y": UNCERTAINTY_Y}
    elif EXPERIMENT.lower() == "confounded-addition":
        experiment_constants = get_counfounded_addition_constants(R_Y)
        experiment_specific_params = {"r_y": R_Y}
    elif EXPERIMENT.lower() == "mutations":
        experiment_constants = get_mutations_constants(BXi, BY, BS)
        experiment_specific_params = {"bxi": BXi, "by": BY, "bs": BS}

    Tcardinal = experiment_constants["Tcardinal"]
    Xcardinal = experiment_constants["Xcardinal"]
    pX = experiment_constants["pX"]
    pZ = experiment_constants["pZ"]
    pXcondYZ = experiment_constants["pXcondYZ"]
    pXcondZ = experiment_constants["pXcondZ"]
    pYcondZ = experiment_constants["pYcondZ"]
    pYcondXZ = experiment_constants["pYcondXZ"]
    NTs = experiment_constants["NTs"]
    NXs = experiment_constants["NXs"]
    NYs = experiment_constants["NYs"]
    NZs = experiment_constants["NZs"]
    SOL_Q = experiment_constants["SOL_Q"]

    # === Initialization ===

    # T is a single variable in all of our experiments.
    # We use this to simplify the initialization code.
    n = NTs[0]
    qTcondX_0 = torch.zeros((n,) + NXs)
    ranges = tuple(list(range(NXs[n])) for n in range(len(NXs)))
    for indices in product(*ranges):
        qTcondX_0[(slice(None),) + indices] = simplex_uniform_sampling(NTs[0])

    # qTcondX needs to be flatenned into a vector before being fed to the optimizer.
    # Furthermore, if one first permutes the T and X indices, the flattened version
    # is easier to interpret when monitoring or debugging the learning loop.
    # Specifically, permuting before flattening makes it so that
    # all q(t|X=x) (for each x) are together after flattening.
    permuted_qTcondX_0 = permute_first_indices(qTcondX_0, len(NTs))
    permuted_qTcondX_0_flat = (
        permuted_qTcondX_0.clone().detach().contiguous().view(-1).requires_grad_(True)
    )  # This is the leaf tensor we start with

    unflattened_shape = permuted_qTcondX_0.shape

    # Define loss
    cib_lagrangian = CIBLagrangian(
        pX,
        pZ,
        pXcondYZ,
        pXcondZ,
        pYcondZ,
        pYcondXZ,
        NTs,
        NXs,
        NYs,
        NZs,
        beta=BETA,
        gamma=GAMMA,
        use_penalty=USE_PENALTY,
        unflattened_shape=unflattened_shape,
    )

    # === Learn q ===
    (
        q_permuted_flat,
        loss,
        converged,
        components,
    ) = run_optimizer_on_confounded_addition_cib(
        permuted_qTcondX_0_flat,
        EPS,
        MAX_ITER,
        loss_func=cib_lagrangian.compute_lagrangian,
        use_penalty=USE_PENALTY,
        temperature=TEMP,
        cooling_rate=COOL_RATE,
        Xcardinal=Xcardinal,
        Tcardinal=Tcardinal,
    )
    permuted_q = q_permuted_flat.view(unflattened_shape)
    q = permute_first_indices(permuted_q, len(NXs))  # reverse permutation

    # === Print training results ===
    print(
        "=" * 10
        + "\nTraining results:"
        + (f"Beta: {BETA}" if GAMMA is None else f"Gamma: {GAMMA}")
        + f"\n\tLearned q: \n\t{q}"
        + f"\n\n\t HT value: {components['HT']}"
        + f"\n\t HTcondX value: {components['HTcondX']}"
        + f"\n\t HcYdoT value: {components['HcYdoT']}"
        + f"\n\tCIB value: {loss.item()}"
    )

    (
        theoretically_optimal_cib,
        theoretically_optimal_components,
    ) = cib_lagrangian.compute_lagrangian(
        permute_first_indices(SOL_Q, len(NTs)).reshape(-1),
    )
    print("\nGround truth for gamma=1 case:")
    print(
        f"\n\tOptimal q: \n\t{SOL_Q}"
        + f"\n\n\t Optimal HT value: {theoretically_optimal_components['HT']}"
        + f"\n\t Optimal HTcondX value: {theoretically_optimal_components['HTcondX']}"
        + f"\n\t Optimal HcYdoT value: {theoretically_optimal_components['HcYdoT']}"
        + f"\n\t Optimal CIB value: {theoretically_optimal_cib}"
    )

    VI: float = variation_of_information_of_abstractions(
        q.detach(), SOL_Q, pX, NTs, NTs, NXs
    )
    print(f"\tVariation of Information between T* and the ground truth T_: {VI}")

    # Compute beta from gamma and vice-versa, for logging
    if GAMMA is None:
        GAMMA = BETA / (1 + BETA)
        loss_name = "CIB"
    else:
        # Division of tensors to allow for infty
        BETA = float(torch.tensor(GAMMA) / torch.tensor(1 - GAMMA))
        loss_name = "wCIB"

    # === MLflow logs ===
    mlflow.set_tag("optimizer", OPTIMIZER_ALGO)
    # If non-surjectivity penalty is used, that should be stated in the loss tag
    loss_name += "+penalty" if USE_PENALTY else ""
    mlflow.set_tag("loss", loss_name)

    # Log Parameters
    generic_params: dict[str, Any] = {
        "beta": BETA,
        "gamma": GAMMA,
        "eps": EPS,
        "lr": LR,
        "end_lr_factor": END_FACTOR,
        "max iter": MAX_ITER,
    }
    algo_specific_params: dict[str, Any]
    if OPTIMIZER_ALGO.lower() == "psagd":
        algo_specific_params = {
            "temperature": TEMP,
            "cooling rate": COOL_RATE,
        }
    elif OPTIMIZER_ALGO.lower() == "pgd":
        algo_specific_params = {}
    elif OPTIMIZER_ALGO.lower() == "padam":
        algo_specific_params = {}

    all_params: dict[str, Any] = {
        **generic_params,
        **experiment_specific_params,
        **algo_specific_params,
    }

    mlflow.log_params(all_params)

    # Log metrics
    mlflow.log_metric("converged", converged)
    mlflow.log_metric("Final CIB loss", loss.item())
    mlflow.log_metric("Expected optimal CIB loss", theoretically_optimal_cib)
    mlflow.log_metric("VI of T and T_", VI)
    mlflow.log_metric("Expected optimal HT", theoretically_optimal_components["HT"])
    mlflow.log_metric(
        "Expected optimal HTcondX", theoretically_optimal_components["HTcondX"]
    )
    mlflow.log_metric(
        "Expected optimal HcYdoT", theoretically_optimal_components["HcYdoT"]
    )
    mlflow.log_params(
        {
            "Learned q": torch.round(q, decimals=2),
            "Expected optimal q": SOL_Q,
        }
    )
    mlflow.end_run()
