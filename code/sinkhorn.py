"""
Implementation taken from:
    Sinkformers: Transformers with Doubly Stochastic Attention
    https://arxiv.org/abs/2110.11773
    https://github.com/michaelsdr/sinkformers

"""


import torch
import torch.nn as nn


class SinkhornDistance(nn.Module):
    def __init__(self, eps, max_iter, reduction="none"):
        super().__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, c):
        C = -c
        x_points = C.shape[-2]
        y_points = C.shape[-1]
        batch_size = C.shape[0]

        # both marginals are fixed with equal weights
        mu = (
            torch.empty(
                batch_size,
                x_points,
                dtype=torch.float,
                requires_grad=False,
                device=C.device,
            )
            .fill_(1.0 / x_points)
            .squeeze()
        )
        nu = (
            torch.empty(
                batch_size,
                y_points,
                dtype=torch.float,
                requires_grad=False,
                device=C.device,
            )
            .fill_(1.0 / y_points)
            .squeeze()
        )

        if mu.dim() < 2:
            mu = mu.view(-1, 1)

        if nu.dim() < 2:
            nu = nu.view(-1, 1)

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)

        # Stopping criterion
        thresh = 1e-12
        err = torch.tensor(1e6)

        # Sinkhorn iterations
        for i in range(self.max_iter):
            if i % 2 == 0:
                u1 = u  # useful to check the update
                u = (
                    self.eps
                    * (torch.log(mu) - torch.logsumexp(self.M(C, u, v), dim=-1))
                    + u
                )
                err = (u - u1).abs().sum(-1).mean()
            else:
                v = (
                    self.eps
                    * (
                        torch.log(nu)
                        - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)
                    )
                    + v
                )
                v = v.detach().requires_grad_(False)
                v[v > 9 * 1e8] = 0.0
                v = v.detach().requires_grad_(True)

            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))

        # Sinkhorn distance
        return pi, C, U, V

    def M(self, C, u, v):
        r"""
        Modified cost for logarithmic updates
        $M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$
        """
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1
