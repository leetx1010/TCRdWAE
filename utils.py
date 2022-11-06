import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import jaccard_score, matthews_corrcoef, f1_score, precision_score, recall_score, hamming_loss, coverage_error, average_precision_score, label_ranking_loss, label_ranking_average_precision_score, roc_curve, auc, roc_auc_score, precision_recall_curve
import torch
import numpy as np

### Metrics
def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_f1_mcc_auc_aupr_pre_rec(preds, labels, probs):
    acc = simple_accuracy(preds, labels)
    precision = precision_score(y_true=labels, y_pred=preds)
    recall = recall_score(y_true=labels, y_pred=preds)
    f1 = f1_score(y_true=labels, y_pred=preds)
    mcc = matthews_corrcoef(labels, preds)
    auroc = roc_auc_score(labels, probs)
    
    p, r, thr = precision_recall_curve(labels, probs)
    aupr = auc(r,p)
    return {
        "acc": acc,
        "f1": f1,
        "mcc": mcc,
        "auc": auroc,
        "aupr": aupr,
        "precision": precision,
        "recall": recall,
    } 


### Calculating mutual information
def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        raise ValueError('Must specify the dimension.')

def log_density(sample, mu, logsigma):
    mu = mu.type_as(sample)
    logsigma = logsigma.type_as(sample)
    c = torch.Tensor([np.log(2 * np.pi)]).type_as(sample.data)

    inv_sigma = torch.exp(-logsigma)
    tmp = (sample - mu) * inv_sigma
    return -0.5 * (tmp * tmp + 2 * logsigma + c)

def log_importance_weight_matrix(batch_size, dataset_size):
    N = dataset_size
    M = batch_size - 1
    strat_weight = (N - M) / (N * M)
    W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
    W.view(-1)[::M+1] = 1 / N
    W.view(-1)[1::M+1] = strat_weight
    W[M-1, 0] = strat_weight
    return W.log()