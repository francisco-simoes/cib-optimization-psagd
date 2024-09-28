# flake8: noqa: E402

import os
import sys

# Add the tests path to the Python path, to make use of _conversion_utils
tests_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(tests_dir_path)

from math import prod
from typing import Any

import numpy as np
import torch
from _conversion_utils import conditionals, rvs_einsum
from torch import Tensor


def get_odd_and_even_constants(uncertainty_y) -> dict[str, Any]:

    # === Choose abstraction shape ===
    NTs = (2,)
    Tcardinal = prod(NTs)

    # === Example set-up ===
    # pX
    Xcardinal = 5
    pX = 1 / Xcardinal * np.ones(Xcardinal)
    NXs = (Xcardinal,)

    # pY
    pYcondX = np.zeros((2, Xcardinal))
    even_prob = (Xcardinal // 2 + 1) / Xcardinal
    odd_prob = (Xcardinal // 2) / Xcardinal
    for i in range(pYcondX.shape[1]):
        if i % 2 == 0:
            pYcondX[0, i] = 1.0 - uncertainty_y  # prob of Y=0
            pYcondX[1, i] = uncertainty_y
        if not i % 2 == 0:
            pYcondX[0, i] = uncertainty_y
            pYcondX[1, i] = 1.0 - uncertainty_y
    pY = pYcondX @ pX
    NYs = (2,)

    # pZ
    # There's no Z. Same as having a Z with a single value 0.
    pZ = np.ones(1)
    NZs = (1,)

    pXY = np.einsum("yx,x->xy", pYcondX, pX)
    # same as: pXYZ = np.einsum("yx,x,z->xyz", pYcondX, pX, pZ)
    pXYZ = np.expand_dims(pXY, -1)

    pXcondYZ = np.expand_dims(conditionals(pXYZ, xs=[0], cond_set=[1, 2]), -1)
    pXcondZ = np.expand_dims(conditionals(pXYZ, xs=[0], cond_set=[2]), -1)
    pYcondZ = np.expand_dims(conditionals(pXYZ, xs=[1], cond_set=[2]), -1)
    pYcondXZ = np.expand_dims(conditionals(pXYZ, xs=[1], cond_set=[0, 2]), -1)

    # Convert all to tensors
    pX = torch.tensor(pX, dtype=torch.float32)
    pZ = torch.tensor(pZ, dtype=torch.float32)
    pXcondYZ = torch.tensor(pXcondYZ, dtype=torch.float32)
    pXcondZ = torch.tensor(pXcondZ, dtype=torch.float32)
    pYcondZ = torch.tensor(pYcondZ, dtype=torch.float32)
    pYcondXZ = torch.tensor(pYcondXZ, dtype=torch.float32)

    # Ground-truth for gamma=1 case
    SOL_Q: Tensor = torch.tensor(
        [[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]], dtype=torch.float32
    )

    constants = {
        "Tcardinal": Tcardinal,
        "Xcardinal": Xcardinal,
        "pX": pX,
        "pZ": pZ,
        "pXcondYZ": pXcondYZ,
        "pXcondZ": pXcondZ,
        "pYcondZ": pYcondZ,
        "pYcondXZ": pYcondXZ,
        "NTs": NTs,
        "NXs": NXs,
        "NYs": NYs,
        "NZs": NZs,
        "SOL_Q": SOL_Q,
    }
    return constants
