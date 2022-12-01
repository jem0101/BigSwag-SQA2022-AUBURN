import torch.nn.functional as F
from resource.mmd import mix_rbf_mmd2

def nll_loss(output, target):
    return F.nll_loss(output, target)

def mmd_gen_loss(x,y):
    return mix_rbf_mmd2(x, y, sigma_list=0)
