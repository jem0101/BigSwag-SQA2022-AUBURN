from __future__ import absolute_import, print_function, unicode_literals, division

from torch import nn


class L1Loss(nn.L1Loss):
    """
    Wrapper for `torch.nn.L1Loss` that accepts dictionaries as input.

    :param output_key:
        key in given sample dict which maps to the output tensor
    :param target_key:
        key in given sample dict which maps to the target tensor
    :param reduction:
        Specifies the reduction to apply to the output:
        ``'none'`` | ``'mean'`` | ``'sum'``.
        ``'none'``: no reduction will be applied,
        ``'mean'``: the sum of the output will be divided by the number of
        elements in the output, ``'sum'``: the output will be summed.
        Default: ``'mean'``
    """
    def __init__(self,
                 output_key='output', target_key='target_image',
                 reduction='mean'):
        nn.L1Loss.__init__(self, reduction=reduction)
        self.output_key = output_key
        self.target_key = target_key

    def forward(self, sample):
        return nn.L1Loss.forward(
            self, sample[self.output_key], sample[self.target_key]
        )


class MSELoss(nn.MSELoss):
    """
    Wrapper for `torch.nn.MSELoss` that accepts dictionaries as input.

    :param output_key:
        key in given sample dict which maps to the output tensor
    :param target_key:
        key in given sample dict which maps to the target tensor
    :param reduction:
        Specifies the reduction to apply to the output:
        ``'none'`` | ``'mean'`` | ``'sum'``.
        ``'none'``: no reduction will be applied,
        ``'mean'``: the sum of the output will be divided by the number of
        elements in the output, ``'sum'``: the output will be summed.
        Default: ``'mean'``
    """
    def __init__(self,
                 output_key='output', target_key='target_image',
                 reduction='mean'):
        nn.MSELoss.__init__(self, reduction=reduction)
        self.output_key = output_key
        self.target_key = target_key

    def forward(self, sample):
        return nn.MSELoss.forward(
            self, sample[self.output_key], sample[self.target_key]
        )


class NLLLoss(nn.NLLLoss):
    """
    Wrapper for `torch.nn.NLLLoss` that accepts dictionaries as input.

    :param output_key:
        key in given sample dict which maps to the output tensor
    :param target_key:
        key in given sample dict which maps to the target tensor
    :param weight:
        a manual rescaling weight given to each
        class. If given, it has to be a Tensor of size `C`. Otherwise, it is
        treated as if having all ones.
    :param reduction:
        Specifies the reduction to apply to the output:
        ``'none'`` | ``'mean'`` | ``'sum'``.
        ``'none'``: no reduction will be applied,
        ``'mean'``: the sum of the output will be divided by the number of
        elements in the output, ``'sum'``: the output will be summed.
        Default: ``'mean'``
    :param ignore_index: Specifies a target value that is ignored
        and does not contribute to the input gradient. When
        :attr:`size_average` is ``True``, the loss is averaged over
        non-ignored targets.
    """
    def __init__(self,
                 output_key='output', target_key='label',
                 weight=None, reduction='mean', ignore_index=-100):
        nn.NLLLoss.__init__(self, weight=weight,
                            ignore_index=ignore_index,
                            reduction=reduction)
        self.output_key = output_key
        self.target_key = target_key

    def forward(self, sample):
        return nn.NLLLoss.forward(
            self, sample[self.output_key], sample[self.target_key].squeeze()
        )


class CrossEntropyLoss(nn.CrossEntropyLoss):
    """
    Wrapper for `torch.nn.CrossEntropyLoss` that accepts dictionaries as input.

    :param output_key:
        key in given sample dict which maps to the output tensor
    :param target_key:
        key in given sample dict which maps to the target tensor
    :param weight: a manual rescaling weight given to each
        class. If given, it has to be a Tensor of size `C`. Otherwise, it is
        treated as if having all ones.
    :param reduction:
        Specifies the reduction to apply to the output:
        ``'none'`` | ``'mean'`` | ``'sum'``.
        ``'none'``: no reduction will be applied,
        ``'mean'``: the sum of the output will be divided by the number of
        elements in the output, ``'sum'``: the output will be summed.
        Default: ``'mean'``
    :param ignore_index: Specifies a target value that is ignored
        and does not contribute to the input gradient. When
        :attr:`size_average` is ``True``, the loss is averaged over
        non-ignored targets.
    """
    def __init__(self, output_key='output', target_key='label',
                 weight=None, reduction='mean', ignore_index=-100):
        nn.CrossEntropyLoss.__init__(self,
                                     weight=weight,
                                     reduction=reduction,
                                     ignore_index=ignore_index)
        self.output_key = output_key
        self.target_key = target_key

    def forward(self, sample):
        return nn.CrossEntropyLoss.forward(
             self, sample[self.output_key], sample[self.target_key].squeeze()
        )


class NSSLoss(nn.Module):
    """
    Loss for saliency applications that optimizes the
    normalized scanpath saliency (NSS) metric.

    The output of the network is normalized to zero-mean
    and unit standard deviation.
    Then the values at gaze locations given by the target image
    tensor are maximized.

    Since with NSS higher values are better and it does not have
    an upper bound, the output is simply negated.
    This means the loss will become negative at some point
    if your network is learning.

    :param output_key:
        key in given sample dict which maps to the output tensor
    :param target_key:
        key in given sample dict which maps to the target tensor
    """
    def __init__(self, output_key='output', target_key='target_image'):
        nn.Module.__init__(self)
        self.output_key = output_key
        self.target_key = target_key

    def forward(self, sample):
        output = sample[self.output_key]
        target = sample[self.target_key]
        n, c, h, w = output.size()
        vectorized = output.view(n, c*h*w)
        mean = vectorized.mean(1).view(n, c, 1, 1)
        std = vectorized.std(1).view(n, c, 1, 1)
        loss = output - mean
        loss /= std + 1e-5
        loss = loss.mul(target).sum() / target.sum()
        return loss / -n


class LabelSmoothing(nn.Module):
    """
    Loss for LabelSmoothing based on NLL-Loss

    :param smoothing:
        label smoothing factor
    :param output_key:
        key in given sample dict which maps to the output tensor
    :param target_key:
        key in given sample dict which maps to the target tensor
    """
    def __init__(self, smoothing=0.0, output_key='output', target_key='target_image'):
        nn.Module.__init__(self)
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.output_key = output_key
        self.target_key = target_key

    def forward(self, sample):
        logprobs = nn.functional.log_softmax(sample[self.output_key], dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=sample[self.target_key])
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


