"""
Implementation follows the code and paper from:
    Deep Deterministic Uncertainty: A Simple Baseline (2021)
    Mukhoti J., Kirsch A., van Amersfoort J., Torr H.S. P., Gal, Y
    https://arxiv.org/abs/2102.11582
    https://github.com/omegafragger/DDU/blob/main/utils/gmm_utils.py
"""


import sys
from tqdm import tqdm

import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
DOUBLE_INFO = torch.finfo(torch.double)
JITTERS = [0, DOUBLE_INFO.tiny] + [10**i for i in range(-308, 0, 1)]


def centered_cov_torch(x):
    n = x.shape[0]
    return 1 / (n - 1) * torch.mm(x.T, x)


@torch.no_grad()
def fit_gmm(embeds, lbls, num_classes):
    classwise_mean_features = torch.stack(
        [embeds[lbls == c].mean(dim=0) for c in range(num_classes)]
    )
    classwise_cov_features = torch.stack(
        [
            centered_cov_torch(embeds[lbls == c] - classwise_mean_features[c])
            for c in range(num_classes)
        ]
    )

    for j in JITTERS:
        try:
            jitter = j * torch.eye(classwise_cov_features.shape[1]).unsqueeze(0).to(
                classwise_cov_features.device
            )
            gmm = torch.distributions.MultivariateNormal(
                loc=classwise_mean_features,
                covariance_matrix=classwise_cov_features + jitter,
            )
            return gmm
        except RuntimeError as e:
            if "cholesky" in str(e):
                continue
        except ValueError as e:
            if "The parameter convariance_matrix has invalid values" in str(e):
                continue
    raise RuntimeError("Could not fit GMM.")


@torch.no_grad()
def evaluate_gmm(model, gmm, loader):
    gmm_logits, lbls = [], []

    for batch in tqdm(loader, desc="GMM Partial: ", file=sys.stdout, unit="batches"):
        seq, lbl = batch.seq, batch.lbl
        seq, lbl = seq.to(device), lbl.to(device)

        embeds = model(seq, return_embeddings=True)

        gmm_log_probs = gmm.log_prob(embeds.unsqueeze(1))
        gmm_logits.append(gmm_log_probs)
        lbls.append(lbl)

    return torch.cat(gmm_logits), torch.cat(lbls)
