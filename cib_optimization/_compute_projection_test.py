"""Test of the compute_projection method."""

import torch
from pgd_optim_pytorch import SubspaceProjectedGradientDescent

e1 = 1 / 2 * torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.float32)
e2 = 1 / 2 * torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float32)

pgd = SubspaceProjectedGradientDescent(
    [torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)], lr=0.01, normals=(e1, e2)
)

P = pgd.projection_matrix
print(f"Projection matrix: \n{P}")
ground_truth = (
    1
    / 4
    * (4 * torch.eye(8) - torch.block_diag(torch.ones((4, 4)), torch.ones((4, 4))))
)  # Computed by hand

print(f"Expected: \n{ground_truth}")

if torch.all(torch.eq(P, ground_truth)).item():
    print("PASS")
else:
    print("FAIL")
