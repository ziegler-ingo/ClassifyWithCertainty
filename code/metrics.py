"""
Expected calibration error implementation is based on code from the paper:
    Deep Deterministic Uncertainty: A Simple Baseline (2021)
    Mukhoti J., Kirsch A., van Amersfoort J., Torr H.S. P., Gal, Y
    https://arxiv.org/abs/2102.11582
    https://github.com/omegafragger/DDU/blob/main/utils/gmm_utils.py

Threshold adaptive calibration error is based on:
    Nixon, J., Dusenberry, M. W., Zhang, L., Jerfel, G., & Tran, D. (2019, June).
    Measuring Calibration in Deep Learning. In CVPR workshops (Vol. 2, No. 7).
and anonyomus Github Repository at:
    https://anonymous.4open.science/r/Uncertainty_aware_SSL-95D7/utils/metrics.py

Code for false positive rate at k recall is taken from:
    Yibo Hu and Latifur Khan. 2021. Uncertainty-Aware Reliable Text 
    Classification. In Proceedings of the 27th ACM SIGKDD Conference 
    on Knowledge Discovery and Data Mining (KDD ’21), August 14–18, 
    2021, Virtual Event, Singapore. ACM, New York, NY, USA, 9 pages. 
    https://doi.org/10.1145/3447548. 3467382
    https://github.com/snowood1/BERT-ENN/blob/main/utils.py#L311
"""


import numpy as np

import torch
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, average_precision_score


class ECELoss:
    def __init__(self, num_bins):
        bin_boundaries = torch.linspace(0, 1, num_bins + 1)
        self.lower_bins = bin_boundaries[:-1]
        self.upper_bins = bin_boundaries[1:]

    @torch.no_grad()
    def __call__(self, logits, lbls):
        softmaxes = F.softmax(logits, dim=1)
        confs, preds = softmaxes.max(dim=1)
        accs = preds == lbls

        ece = torch.zeros(1).to(confs.device)
        for lb, ub in zip(self.lower_bins, self.upper_bins):
            in_bin = (confs.gt(lb.item()) * confs.le(ub.item())).to("cpu")
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                ece += (
                    torch.abs(confs[in_bin].mean() - accs[in_bin].float().mean())
                    * prop_in_bin
                )
        return ece.item()


class TACELoss:
    def __init__(self, num_bins):
        self.num_bins = num_bins

    def binary_results_per_class(self, logits, lbls, threshold):
        self.n_samples, self.n_classes = logits.shape
        idx = torch.arange(self.n_samples).to(torch.long)
        lbls = lbls.to(torch.long)
        pred_matrix = torch.zeros((self.n_samples, self.n_classes))
        lbl_matrix = torch.zeros((self.n_samples, self.n_classes))

        self.softmaxes = F.softmax(logits, dim=1)
        confs, preds = self.softmaxes.max(dim=1)
        self.softmaxes[self.softmaxes < threshold] = 0

        pred_matrix[idx, preds] = 1
        lbl_matrix[idx, lbls] = 1
        self.accuracies = (pred_matrix == lbl_matrix).float()

    def bin_boundaries_per_class(self, class_idx):
        sorted_confs = self.softmaxes[:, class_idx].sort()[0]
        bin_boundaries = sorted_confs[:: self.n_samples // self.num_bins]
        bin_boundaries[-1] = 1
        return bin_boundaries[:-1], bin_boundaries[1:]

    @torch.no_grad()
    def __call__(self, logits, lbls, threshold=0.01):
        self.binary_results_per_class(logits, lbls, threshold)

        tace_total = 0.0
        for i in range(self.n_classes):
            lower_bins, upper_bins = self.bin_boundaries_per_class(class_idx=i)
            confs = self.softmaxes[:, i]
            accs = self.accuracies[:, i]

            tace = torch.zeros(1).to(self.softmaxes.device)
            for lb, ub in zip(lower_bins, upper_bins):
                in_bin = (confs.gt(lb.item()) * confs.le(ub.item())).to("cpu")
                prop_in_bin = in_bin.float().mean()
                if prop_in_bin.item() > 0:
                    tace += (
                        torch.abs(confs[in_bin].mean() - accs[in_bin].mean())
                        * prop_in_bin
                    )
            tace_total += tace.item()

        return tace_total / self.n_classes


@torch.no_grad()
def entropy_per_sample(logits):
    p = F.softmax(logits, dim=1)
    logp = F.log_softmax(logits, dim=1)
    return -torch.sum(p * logp, dim=1)


@torch.no_grad()
def get_auroc_auprc(id_logits, ood_logits, measure="logsumexp", confidence=None):
    assert (
        measure in ['logsumexp', 'entropy']
    ), "uncertainty measure must be logsumexp or entropy"

    if measure == "logsumexp":
        id_scores = torch.logsumexp(id_logits, dim=1)
        ood_scores = torch.logsumexp(ood_logits, dim=1)
    else:
        id_scores = entropy_per_sample(id_logits)
        ood_scores = entropy_per_sample(ood_logits)
    scores = torch.cat([id_scores, ood_scores])

    bin_labels = torch.cat(
        [torch.zeros(id_logits.shape[0]), torch.ones(ood_logits.shape[0])]
    )

    if confidence is not None:
        bin_labels = 1 - bin_labels

    auroc = roc_auc_score(bin_labels.tolist(), scores.tolist())
    auprc = average_precision_score(bin_labels.tolist(), scores.tolist())
    return auroc, auprc 


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out

def fpr_at_k_recall(y_true, y_score, recall_level=0.9, pos_label=1):
    classes = np.unique(y_true)
    assert len(classes) <= 2, "Data needs to be binary"

    y_true = y_true == pos_label

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score)[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing
    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    recall, fps = np.r_[recall[last_ind::-1], 1], np.r_[fps[last_ind::-1], 0]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))