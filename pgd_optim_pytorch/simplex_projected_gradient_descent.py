"""Module containing the SimplexProjectedGradientDescent optimizer class.

This module defines a custom optimizer for performing constrained optimization with
the probability simplex as the constraint space.
Each step of the optimizer consists of a gradient descent step followed by a projection
onto the probability simplex.
The projection is an implementation of the algorithm proposed in Duchi et
 al (2008) - "Efficient projections onto the l 1-ball for learning in high dimensions."
There must be a single parameter (of arbitrary dimension) for the step() method to work.
"""
from typing import Iterable

import numpy as np
import torch
from torch import Tensor
from torch.optim import SGD
from torch.optim.optimizer import required


class SimplexProjectedGradientDescent(SGD):
    """pGD optimizer for simplex constraint space.

    Performs a gradient descent step followed by projection onto
    the simplex constraint space.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        Xcardinal: int,
        Tcardinal: int,
        lr: float = required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
    ):
        """Initialize the pGD optimizer.

        Parameters:
        ----------
        params : Iterable[Tensor]
            An iterable *of length 1* containing a tensor representing
            the parameter to optimize.
        Xcardinal : int
            The cardinality of the range of the random vector X.
        Tcardinal : int
            The cardinality of the range of the random vector T.
        lr : float
            The learning rate for the optimizer. Must be explicitly provided.
        momentum : float, optional
            Momentum factor for GD (default is 0).
        dampening : float, optional
            Dampening for momentum (default is 0).
        weight_decay : float, optional
            Weight decay for GD (default is 0).
        nesterov : bool, optional
            Enables Nesterov momentum (default is False).
        """
        super().__init__(
            params,
            lr,
            momentum,
            dampening,
            weight_decay,
            nesterov,
            maximize=False,
        )
        self.Xcardinal = Xcardinal
        self.Tcardinal = Tcardinal

    @torch.no_grad()
    def step(self):
        """Perform a single optimization step.

        Updates model parameters by first performing a gradient descent step, and
        then projecting the resulting parameters onto the probability simplex.
        """
        # Gradient descent step
        super().step()

        # Projection onto simplex
        # A single group; a single parameter
        qTcondX = self.param_groups[0]["params"][0]
        V = qTcondX.clone().detach().reshape(self.Xcardinal, self.Tcardinal)
        # q = q.view((Xcardinal, Tcardinal))
        V_projected = self._project_onto_prob_simplex(V).to(dtype=torch.float32)
        qTcondX.data = V_projected.reshape(qTcondX.shape)

    def _project_onto_prob_simplex(self, V):
        """V is matrix with one vector to project on each row."""
        V_precise = V.to(torch.float64)
        # NOTE: Due to finite precision, it can happen (and it happens at
        #   non-surjective encoders when using the penalty) that elements of
        #   MU have such large values (order 1e9) that it is blind to subtraction of 1,
        #   leading to Mu=MUcum in those entries, which is wrong. This can lead to
        #   all-False masks (which are impossible). Using higher precision here for this reason.

        # N, D = V.shape
        N, D = V_precise.shape

        # MU = np.sort(V, axis=1)[:, ::-1]  # Sort V in descending order along axis=1
        MU = np.sort(V_precise, axis=1)[
            :, ::-1
        ]  # Sort V in descending order along axis=1

        MUcum = (np.cumsum(MU, axis=1) - 1) @ np.diag(1.0 / np.arange(1, D + 1))
        mask = MU > MUcum
        col_idxs = np.arange(1, MU.shape[1] + 1)
        masked_col_idxs_array = np.tile(col_idxs, (MU.shape[0], 1)) * mask
        rho = np.max(masked_col_idxs_array, axis=1)

        theta = np.expand_dims(1 / rho * (np.sum(MU * mask, axis=1) - 1.0), axis=-1)

        # W = np.maximum(V - theta, 0)
        W = np.maximum(V_precise - theta, 0)

        return W
