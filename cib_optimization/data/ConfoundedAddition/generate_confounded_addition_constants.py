# flake8: noqa: E402

import os
import sys

# Add the tests path to the Python path, to make use of _probability_utils
tests_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(tests_dir_path)

from math import prod
from typing import Any

import torch
from _probability_utils import conditionals, rvs_einsum
from torch import Tensor


def get_counfounded_addition_constants(r_y) -> dict[str, Any]:

    # === Choose abstraction shape ===
    NTs = (3,)
    Tcardinal = prod(NTs)

    # === Example set-up ===
    # pNY
    NNYs = (2,)
    # NYcardinal = prod(NNYs)
    pNY = torch.zeros(NNYs)
    pNY[0] = 1 - r_y
    pNY[1] = r_y

    # pNX2
    NNX2s = (2,)
    # NNX2cardinal = prod(NNX2s)
    pNX2 = torch.zeros(NNX2s)
    pNX2[0] = 0.5
    pNX2[1] = 0.5

    # pNX1
    NNX1s = (2,)
    # NNX1cardinal = prod(NNX1s)
    pNX1 = torch.zeros(NNX1s)
    pNX1[0] = 0.5
    pNX1[1] = 0.5

    # # pY
    NYs = (6,)
    # Ycardinal = prod(NYs)
    # pY = torch.zeros(NYs)

    # Will define pycondxy, pxcondw and pw. Can compute joint pXYW from them.

    # pW
    NWs = (2,)
    Wcardinal = prod(NWs)
    pW = 1 / Wcardinal * torch.ones(Wcardinal)

    # pX
    NXs = (2, 2)  # Two binary Xis
    Xcardinal = prod(NXs)
    pX = torch.zeros(NXs)
    # (p_1,2(x1,x2) = p_1(x1|x2) p_2(x2))
    # X1=0, X2=0 iff NX2=0, NX1=0 (W does not matter)
    # (p_1,2(0,0) = p_1(0|0) p_2(0))
    pX[0, 0] = pNX2[0] * pNX1[0]
    # X1=0, X2=1 iff NX2=1, NX1=0, W=0
    pX[0, 1] = pNX2[1] * pNX1[0] * pW[0]
    # X1=1, X2=0 iff NX2=0, NX1=1
    pX[1, 0] = pNX2[0] * pNX1[1]
    # X1=1, X2=1 iff NX2=1, NX1=1 OR NX2=1, NX1=0, W=3
    pX[1, 1] = pNX2[1] * (pNX1[1] + pNX1[0] * pW[1])

    # pYcondXW
    pYcondXW = torch.zeros(NYs + NXs + NWs)
    # idxs: Y, X1, X2, W
    # Y=0 iff W=0,NY=0 OR W=0, Xi=0
    pYcondXW[0, 0, 1, 0] = pNY[0]
    pYcondXW[0, 1, 0, 0] = pNY[0]
    pYcondXW[0, 1, 1, 0] = pNY[0]
    pYcondXW[0, 0, 0, 0] = 1.0
    # Y=1 iff W=0,NY=1,X1=1,X2=0 OR W=0,NY=1,X1=0,X2=1
    pYcondXW[1, 0, 1, 0] = pNY[1]
    pYcondXW[1, 1, 0, 0] = pNY[1]
    # Y=2 iff W=0,NY=1,Xi=1
    pYcondXW[2, 1, 1, 0] = pNY[1]
    # Y=3 iff W=3,NY=0 OR W=3,Xi=0
    pYcondXW[3, 0, 1, 1] = pNY[0]
    pYcondXW[3, 1, 0, 1] = pNY[0]
    pYcondXW[3, 1, 1, 1] = pNY[0]
    pYcondXW[3, 0, 0, 1] = 1.0
    # Y=4 iff W=3,NY=1,X1=1,X2=0 OR W=3,NY=1,X1=0,X2=1
    pYcondXW[4, 0, 1, 1] = pNY[1]
    pYcondXW[4, 1, 0, 1] = pNY[1]
    # Y=5 iff W=3,NY=1,Xi=1
    pYcondXW[5, 1, 1, 1] = pNY[1]

    # pXcondW #NOTE: this is giving 1 summed prob...
    pXcondW = torch.zeros(NXs + NWs)
    # W=0 => X1=0
    pXcondW[0, 0, 0] = pNX2[0]
    pXcondW[0, 1, 0] = pNX2[1]
    # W=1
    pXcondW[0, 0, 1] = pNX2[0] * pNX1[0]
    pXcondW[0, 1, 1] = pNX2[1] * pNX1[0]
    pXcondW[1, 0, 1] = pNX2[0] * pNX1[1]
    pXcondW[1, 1, 1] = pNX2[1] * pNX1[1]

    # Test whether pYcondXW is appropriate transition matrix
    for x1 in (0, 1):
        for x2 in (0, 1):
            for w in (0, 1):
                assert (
                    float(torch.sum(pYcondXW[:, x1, x2, w])) == 1.0
                ), f"p(y|x,w) not a distribution for x1={x1}, x2={x2}, w={w}"

    # Test whether pXcondW is appropriate transition matrix
    for w in (0, 1):
        assert (
            float(torch.sum(pXcondW[:, :, w])) == 1.0
        ), f"p(x|w) not a distribution for w={w}"

    # Compute distributions needed by CIB
    var_numbers = {"x": len(NXs), "y": len(NYs), "w": len(NWs)}
    #
    #
    pXYW = rvs_einsum(
        (pYcondXW, pXcondW, pW), ("yxw", "xw", "w"), "xyw", var_numbers=var_numbers
    )

    pXcondYW = conditionals(pXYW, xs=[0, 1], cond_set=[2, 3])
    # pXcondW = # already computed!
    pYcondW = conditionals(pXYW, xs=[2], cond_set=[3])

    # Convert all to torch #TODO make all directly in torch...
    # pX = torch.tensor(pX, dtype=torch.float32)
    # pY = torch.tensor(pX)
    # pZ = torch.tensor(pZ, dtype=torch.float32)
    pXcondYW = torch.tensor(pXcondYW, dtype=torch.float32)
    pXcondW = torch.tensor(pXcondW, dtype=torch.float32)
    pYcondW = torch.tensor(pYcondW, dtype=torch.float32)

    # Ground-truth for gamma=1 case
    # fmt:off
    SOL_Q: Tensor = torch.tensor(
        [[[1, 0], [0, 0]],
         [[0, 1], [1, 0]],
         [[0, 0], [0, 1]]], dtype=torch.float32
    )
    # fmt:on

    constants = {
        "Tcardinal": Tcardinal,
        "Xcardinal": Xcardinal,
        "pX": pX,
        "pZ": pW,
        "pXcondYZ": pXcondYW,
        "pXcondZ": pXcondW,
        "pYcondZ": pYcondW,
        "pYcondXZ": pYcondXW,
        "NTs": NTs,
        "NXs": NXs,
        "NYs": NYs,
        "NZs": NWs,
        "SOL_Q": SOL_Q,
    }
    return constants
