from typing import Optional

import torch
from torch import Tensor

from _cib_lagrangian import CIBLagrangian, log_ext
from _probability_utils import rvs_einsum
from _tests_utils import permute_first_indices


class IBLagrangian(CIBLagrangian):
    """A class to compute the IB Lagrangian.

    This class encapsulates the necessary parameters and methods to calculate various
    components of the IB Lagrangian based on input probability distributions.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """Initialize a IBLagrangian object.

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

        pYcondX: Tensor,
            Conditional probability distribution of Y given X.

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
        super().__init__(*args, **kwargs)

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
        HYcondT = self._compute_HYcondT(qTcondX)

        if self.track_terms:
            print(
                f"\tHT = {HT}\n\tHTcondX = {HTcondX}\n\tHY = {HY}\n\tHYcondT = {HYcondT}"
            )

        penalty = self._compute_penalty(qTcondX)

        # All terms
        if self.gamma is not None:  # weighted CIB
            result = (
                (1 - self.gamma) * (HT - HTcondX)
                - self.gamma * (HY - HYcondT)
                + penalty
            )
        else:
            result = HT - HTcondX - self.beta * (HY - HYcondT) + penalty

        components: dict[str, float] = {
            "HT": HT,
            "HTcondX": HTcondX,
            "HY": HY,
            "HcYdoT": self._compute_HcYdoT(qTcondX),
            "HYcondT": HYcondT,
            "penalty": penalty,
        }

        return result, components
