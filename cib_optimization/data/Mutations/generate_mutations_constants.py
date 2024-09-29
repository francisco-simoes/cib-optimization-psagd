# flake8: noqa: E402

import os
import pickle
import sys

# Add the tests path to the Python path, to make use of _probability_utils
tests_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(tests_dir_path)

from math import prod
from typing import Any

import torch
from _probability_utils import conditionals, marginal, rvs_einsum
from torch import Tensor


def get_mutations_constants(bxi, by, bs) -> dict[str, Any]:

    # === Choose abstraction shape ===
    NTs = (4,)
    Tcardinal = prod(NTs)

    # === Example set-up ===

    # Cardinalities
    NXs = (2, 2, 2, 2)  # X1 through X4 are binary
    NYs = (7,)
    NSs = (2,)

    Xcardinal = prod(NXs)
    Ycardinal = prod(NYs)
    Scardinal = prod(NSs)

    Xs_indices = list(range(len(NXs)))
    Y_index = len(NXs)
    S_index = len(NXs) + 1

    # Joint can be extracted computed using the BN
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(
        script_dir, f"mutations_joint_distribution_table_{bxi}-{by}-{bs}.pkl"
    )
    with open(file_path, "rb") as handle:
        pXYS: Tensor = torch.tensor(pickle.load(handle), dtype=torch.float32)
    # print(pXYS.shape)

    pXcondYS = conditionals(pXYS, xs=Xs_indices, cond_set=[Y_index, S_index])

    pX = marginal(pXYS, xs=Xs_indices)
    pS = marginal(pXYS, xs=[S_index])
    pXcondS = conditionals(pXYS, xs=Xs_indices, cond_set=[S_index])
    pYcondS = conditionals(pXYS, xs=[Y_index], cond_set=[S_index])
    pYcondXS = conditionals(pXYS, xs=[Y_index], cond_set=Xs_indices + [S_index])

    pX = torch.tensor(pX, dtype=torch.float32)
    pS = torch.tensor(pS, dtype=torch.float32)
    pXcondYS = torch.tensor(pXcondYS, dtype=torch.float32)
    pXcondS = torch.tensor(pXcondS, dtype=torch.float32)
    pYcondS = torch.tensor(pYcondS, dtype=torch.float32)
    pYcondXS = torch.tensor(pYcondXS, dtype=torch.float32)

    # Ground-truth for gamma=1 case
    file_path = os.path.join(script_dir, "mutations_sol_q.pkl")
    with open(file_path, "rb") as handle:
        SOL_Q: Tensor = torch.tensor(pickle.load(handle), dtype=torch.float32)

    constants = {
        "Tcardinal": Tcardinal,
        "Xcardinal": Xcardinal,
        "pX": pX,
        "pZ": pS,
        "pXcondYZ": pXcondYS,
        "pXcondZ": pXcondS,
        "pYcondZ": pYcondS,
        "pYcondXZ": pYcondXS,
        "NTs": NTs,
        "NXs": NXs,
        "NYs": NYs,
        "NZs": NSs,
        "SOL_Q": SOL_Q,
    }
    return constants
