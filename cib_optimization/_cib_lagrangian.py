from typing import Optional

import torch
from torch import Tensor

from _probability_utils import rvs_einsum
from _tests_utils import permute_first_indices


class CIBLagrangian:
    """A class to compute the CIB Lagrangian.

    This class encapsulates the necessary parameters and methods to calculate various
    components of the CIB Lagrangian based on input probability distributions.
    """

    def __init__(
        self,
        pX: Tensor,
        pZ: Tensor,
        pXcondYZ: Tensor,
        pXcondZ: Tensor,
        pYcondZ: Tensor,
        pYcondXZ: Tensor,
        NTs: tuple[int, ...],
        NXs: tuple[int, ...],
        NYs: tuple[int, ...],
        NZs: tuple[int, ...],
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        use_penalty: bool = False,
        track_terms: bool = False,
        unflattened_shape: tuple[int, ...] = (0,),
    ):
        """Initialize a CIBLagrangian object.

        Parameters:
        -----------
        pX : Tensor
            Probability distribution of X.

        pZ : Tensor
            Probability distribution of Z.

        pXcondYZ : Tensor
            Conditional probability distribution of X given Y and Z.

        pXcondZ : Tensor
            Conditional probability distribution of X given Z.

        pYcondZ : Tensor
            Conditional probability distribution of Y given Z.

        pYcondXZ : Tensor
            Conditional probability distribution of Y given X and Z.

        NTs : tuple[int, ...]
            Dimensions of the T variable. E.g., NTs=(3,) if T is a 3-value
            random variable.

        NXs : tuple[int, ...]
            Dimensions of the X variable. E.g., NXs=(2,2,2) if X consists of
            3 binary random variables.

        NYs : tuple[int, ...]
            Dimensions of the Y variable.

        NZs : tuple[int, ...]
            Dimensions of the Z variable.

        beta : Optional[float], default=None
            Trade-off parameter for the CIB with the original parameterization.

        gamma : Optional[float], default=None
            Trade-off parameter for the wCIB (the reparameterized CIB).
            If not None, the wCIB will be used instead of the original CIB.

        use_penalty : bool, default=False
            Flag indicating whether to use a non-surjectivity penalty.

        track_terms : bool, default=False
            Flag indicating whether to print the individual terms of the Lagrangian
            when computing it.

        unflattened_shape : tuple[int, ...], default=(0,)
            Shape of the version of qTcondX with X indices and T indices permuted
            (i.e., of permuted_qTcondX in optimize_cib.py)

        Raises:
        -------
        ValueError
            If both 'beta' and 'gamma' are None.
        """
        if beta is None and gamma is None:
            raise ValueError("At least one of 'beta' or 'gamma' must be non-None")
        self.pX = pX
        self.pZ = pZ
        self.pXcondYZ = pXcondYZ
        self.pXcondZ = pXcondZ
        self.pYcondZ = pYcondZ
        self.pYcondXZ = pYcondXZ
        self.NTs = NTs
        self.NXs = NXs
        self.NYs = NYs
        self.NZs = NZs
        self.beta = beta
        self.gamma = gamma
        self.use_penalty = use_penalty
        self.track_terms = track_terms
        self.unflattened_shape = unflattened_shape
        self.var_numbers: dict[str, int] = {
            "t": len(NTs),
            "x": len(NXs),
            "y": len(NYs),
            "z": len(NZs),
        }

    def _compute_HT(self, qTcondX):
        qts = rvs_einsum((qTcondX, self.pX), ("tx", "x"), "t", self.var_numbers)
        return -rvs_einsum((qts, log_ext(qts)), ("t", "t"), "", self.var_numbers)

    def _compute_HY(self, qTcondX):
        pys = rvs_einsum((self.pYcondZ, self.pZ), ("yz", "z"), "y", self.var_numbers)
        return -rvs_einsum((pys, log_ext(pys)), ("y", "y"), "", self.var_numbers)

    def _compute_HTcondX(self, qTcondX):
        return -rvs_einsum(
            (self.pX, qTcondX, log_ext(qTcondX)),
            ("x", "tx", "tx"),
            "",
            self.var_numbers,
        )

    def _compute_HcYdoT(self, qTcondX):
        denum = rvs_einsum((qTcondX,), ("tx",), "t", self.var_numbers)
        qydots = rvs_einsum(
            (self.pZ, self.pYcondXZ, qTcondX, 1 / (denum + 1e-6)),
            ("z", "yxz", "tx", "t"),
            "ty",
            self.var_numbers,
        )
        qT = (
            1 / self.pX.numel() * rvs_einsum((qTcondX,), ("tx",), "t", self.var_numbers)
        )
        HcYdoT = -1 * rvs_einsum(
            (qT, qydots, log_ext(qydots)), ("t", "ty", "ty"), "", self.var_numbers
        )
        return HcYdoT

    @staticmethod
    def _log_penalty(x, epsilon=1e-8):
        # Apply penalty only when x < 1
        penalty = torch.where(x < 1, -torch.log(x + epsilon), torch.zeros_like(x))
        return penalty

    def _compute_penalty(self, qTcondX):
        """Compute non-surjectivity penalty.

        The value of the penalty increases as the total probability
        of (some of) the values of T approach zero.
        """
        penalty: Tensor = torch.tensor(0.0)
        if self.use_penalty:
            num_dims_to_sum = len(self.NXs)
            dims_to_sum_over = tuple(
                range(-num_dims_to_sum, 0)
            )  # Tuple of last `num_dims_to_sum` dimensions
            total_T_probs = torch.sum(qTcondX, dim=dims_to_sum_over)
            penalties = self._log_penalty(total_T_probs)
            if self.gamma is not None:  # weighted CIB
                penalty = penalties.mean()
            else:  # need rescaling
                penalty = (1 + self.beta) * penalties.mean()  # ensure same scale as CIB
        return penalty

    def compute_lagrangian(
        self,
        qTcondX: Tensor,
        permute: bool = True,
    ) -> tuple[float, dict[str, float]]:
        """Compute the CIB."""
        if permute:  # In case q has been permuted and flattened.
            assert self.unflattened_shape != (0,), "Invalid unflattened shape."
            permuted_qTcondX = qTcondX.view(self.unflattened_shape)  # Undo flattening
            qTcondX = permute_first_indices(
                permuted_qTcondX, len(self.NXs)
            )  # reverse permutation

        HT = self._compute_HT(qTcondX)
        HTcondX = self._compute_HTcondX(qTcondX)
        HY = self._compute_HY(qTcondX)
        HcYdoT = self._compute_HcYdoT(qTcondX)

        if self.track_terms:
            print(
                f"\tHT = {HT}\n\tHTcondX = {HTcondX}\n\tHY = {HY}\n\tHcYdoT = {HcYdoT}"
            )

        penalty = self._compute_penalty(qTcondX)

        # All terms
        if self.gamma is not None:  # weighted CIB
            result = (
                (1 - self.gamma) * (HT - HTcondX) - self.gamma * (HY - HcYdoT) + penalty
            )
        else:
            result = HT - HTcondX - self.beta * (HY - HcYdoT) + penalty

        components: dict[str, float] = {
            "HT": HT,
            "HTcondX": HTcondX,
            "HY": HY,
            "HcYdoT": HcYdoT,
            "penalty": penalty,
        }

        return result, components


def log_ext(x, eps=1e-6):
    """Extension of log2 (to domain [0, +infty)) often used in information theory."""
    x = torch.where(x == 0.0, torch.tensor(eps), x)
    logext = torch.log2(x)
    return logext
