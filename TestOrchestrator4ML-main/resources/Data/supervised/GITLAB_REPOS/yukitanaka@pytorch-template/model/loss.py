import torch.nn.functional as F


def my_loss(y_input, y_target):
    return F.nll_loss(F.log_softmax(y_input, dim=1), y_target)
