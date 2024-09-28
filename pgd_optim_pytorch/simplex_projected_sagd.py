"""Module containing the SimplexProjectedSAGD optimizer class.

This module defines a custom optimizer for performing constrained optimization with
the probability simplex as the constraint space.
The optimizer tries to perform a simulated annealing step at each step. If it fails,
a gradient descent step is done instead. This is followed by a projection onto the
probability simplex.
The projection is an implementation of the algorithm proposed in Duchi et
 al (2008) - "Efficient projections onto the l 1-ball for learning in high dimensions."
The tensor to be learned must consist of a single parameter (of arbitrary dimension)
 for the step() method to work.
"""
from typing import Callable, Iterable, Optional

import numpy as np
import torch
from torch import Tensor
from torch.optim import SGD
from torch.optim.optimizer import required


class SimplexProjectedSAGD(SGD):
    """pSAGD optimizer for simplex constraint space.

    Attempts to perform a simulated annealing step at each step, while remaining in the
    simplex constraint space. If the simulated annealing step fails, a projected
    gradient descent step is performed instead.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        Xcardinal: int,
        Tcardinal: int,
        loss_func: Callable,
        temperature: float,
        cooling_rate: float,
        lr: float = required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
    ):
        """Initialize the pSAGD optimizer.

        Parameters:
        ----------
        params : Iterable[Tensor]
            An iterable *of length 1* containing a tensor representing
            the parameter to optimize.
        Xcardinal : int
            The cardinality of the range of the random vector X.
        Tcardinal : int
            The cardinality of the range of the random vector T.
        loss_func : Callable
            The loss function to minimize.
        temperature : float
            The initial temperature for the simulated annealing process.
        cooling_rate : float
            The rate at which the temperature decreases during simulated annealing.
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
        self.loss_func = loss_func
        self.temperature = temperature
        self.cooling_rate = cooling_rate
        self.sa_worked = False

    @torch.no_grad()
    def step(self):
        """Perform a single optimization step.

        Updates model parameters by first performing a projected SA or GD step, where
        the projection is onto the onto the appropriate probability simplex constraint
        space.
        """
        # A single group; a single parameter
        qTcondX = self.param_groups[0]["params"][0]

        # delta is random point in sphere of radius lr
        dim = qTcondX.numel()
        proposal_delta = self._random_sphere_points(
            dimension=dim, radius=self.param_groups[0]["lr"]
        ).reshape(qTcondX.shape)
        # proposal is q + delta
        proposal = (qTcondX + proposal_delta).reshape(self.Xcardinal, self.Tcardinal)
        # Project it onto simplex constraint space
        proposal_projected = (
            self._project_onto_prob_simplex(proposal)
            .to(dtype=torch.float32)
            .reshape(qTcondX.shape)
        )

        # Accept proposal if improves loss or stochastically
        proposal_loss = self.loss_func(proposal_projected)[0].data
        current_loss = self.loss_func(qTcondX)[0]
        change = proposal_loss - current_loss
        accept_prob = torch.exp(-change / self.temperature)
        if change < 0 or torch.rand(1) < accept_prob:
            qTcondX.copy_(proposal_projected)
            self.sa_worked = True
            print("SA worked")
        else:
            # GD step
            super().step()
            self.sa_worked = False

            # Project onto simplex
            V = qTcondX.clone().detach().reshape(self.Xcardinal, self.Tcardinal)
            # q = q.view((Xcardinal, Tcardinal))
            V_projected = self._project_onto_prob_simplex(V).to(dtype=torch.float32)
            qTcondX.data = V_projected.reshape(qTcondX.shape)

        print(
            "change, temp, prob",
            change,
            self.temperature,
            accept_prob,
        )

        # Update temperature
        self.temperature *= self.cooling_rate

    def _project_onto_prob_simplex(self, V):
        """V is matrix with one vector to project on each row."""
        N, D = V.shape

        MU = np.sort(V, axis=1)[:, ::-1]  # Sort V in descending order along axis=1

        MUcum = (np.cumsum(MU, axis=1) - 1) @ np.diag(1.0 / np.arange(1, D + 1))
        mask = MU > MUcum
        col_idxs = np.arange(1, MU.shape[1] + 1)
        masked_col_idxs_array = np.tile(col_idxs, (MU.shape[0], 1)) * mask
        rho = np.max(masked_col_idxs_array, axis=1)

        theta = np.expand_dims(1 / rho * (np.sum(MU * mask, axis=1) - 1.0), axis=-1)

        W = np.maximum(V - theta, 0)

        return W

    def _random_sphere_points(self, dimension, radius=1, num_points=1):
        """Generate unformly distributed points on the sphere of chosen dimension and radius.

        Random directions are generated by normalizing Gaussian vectors.
        These are then multiplied by the chosen radius.
        """
        random_ball_points = torch.randn(dimension, num_points)
        random_directions = random_ball_points / torch.norm(random_ball_points, dim=0)
        return radius * random_directions
