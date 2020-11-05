import torch
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CCC metric: Concordance Correlation Coefficient
# attention: cannot compute for pred/lbl with batchsize = 1
def ccc(pred, lbl):
    mean_cent_prod = ((pred - pred.mean()) * (lbl - lbl.mean())).mean()
    return (2 * mean_cent_prod) / (pred.var() + lbl.var() + (pred.mean() - lbl.mean()) ** 2)