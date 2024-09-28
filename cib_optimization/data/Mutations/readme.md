Given the complexity of the mutations example, we use a Bayesian Network to compute the joint distribution.
We need only supply it with the CPDs, which are easier to manually compute.

This is done in the IPython notebook in this directory.

Note that the parameters for the Mutations experiment cannot be set using the CLI for the optimization script.
The generation of the joint that they determine, which is computed by a Bayesian Network in the IPython notebook, was not automatized.
The parameters should be changed first in the notebook, and then also in the optimization script so that the correct joint is imported.
