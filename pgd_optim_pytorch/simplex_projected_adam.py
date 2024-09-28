"""Module containing the SimplexProjectedAdam optimizer class.

This module defines a custom optimizer for performing constrained optimization with
the probability simplex as the constraint space.
Each step of the optimizer consists of an Adam step followed by a projection
onto the probability simplex.
The projection is an implementation of the algorithm proposed in Duchi et
 al (2008) - "Efficient projections onto the l 1-ball for learning in high dimensions."
There must be a single parameter (of arbitrary dimension) for the step() method to work.
"""
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.optim.optimizer import required


class SimplexProjectedAdam(Adam):
    """"""

    # TODO: understand each argument of Adam's init; delete unnecessary ones
    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        Xcardinal: int,
        Tcardinal: int,
        lr: Union[float, Tensor] = required,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        *,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ):
        """TODO"""
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            foreach=foreach,
            maximize=maximize,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
        )
        self.Xcardinal = Xcardinal
        self.Tcardinal = Tcardinal

    @torch.no_grad()
    def step(self):
        """Perform a single optimization step.

        Updates model parameters by first performing a Adam step, and
        then projecting the resulting parameters onto the probability simplex.
        """
        # Gradient descent step
        super().step()

        # Projection onto simplex
        # A single group; a single parameter
        qTcondX = self.param_groups[0]["params"][0]
        V = qTcondX.clone().detach().reshape(self.Xcardinal, self.Tcardinal)
        # og_shape = q.shape
        # q = q.view((Xcardinal, Tcardinal))
        V_projected = self._project_onto_prob_simplex(V).to(dtype=torch.float32)
        # q = (
        #     project_onto_prob_simplex(q)
        #     .reshape(og_shape)
        #     .to(dtype=torch.float32)
        # )
        qTcondX.data = V_projected.reshape(qTcondX.shape)

    def _project_onto_prob_simplex(self, V):
        """V is matrix with one vector to project on each row."""
        N, D = V.shape

        MU = np.sort(V, axis=1)[:, ::-1]  # Sort Y in descending order along axis=1

        MUcum = (np.cumsum(MU, axis=1) - 1) @ np.diag(1.0 / np.arange(1, D + 1))
        mask = MU > MUcum
        col_idxs = np.arange(1, MU.shape[1] + 1)
        masked_col_idxs_array = np.tile(col_idxs, (MU.shape[0], 1)) * mask
        rho = np.max(masked_col_idxs_array, axis=1)

        theta = np.expand_dims(1 / rho * (np.sum(MU * mask, axis=1) - 1.0), axis=-1)

        W = np.maximum(V - theta, 0)

        return W
