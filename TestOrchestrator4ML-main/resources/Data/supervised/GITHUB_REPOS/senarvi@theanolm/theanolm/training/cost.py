#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A module that implements the BasicOptimizer class, a base class for
optimizers.
"""

from abc import abstractmethod, ABCMeta

import theano
import theano.tensor as tensor

from theanolm.backend import IncompatibleStateError
from theanolm.backend import test_value
from theanolm.backend import l1_norm, sum_of_squares

class Cost(object, metaclass=ABCMeta):
    """Base Class for Cost Functions
    """

    def __init__(self, network, exclude_id=None, l1_regularization=None,
                 l2_regularization=None, epsilon=1e-6):
        """Constructs a cost function.

        :type network: Network
        :param network: the cost function will be defined with regard to the
                        outputs of this network

        :type exclude_id: int
        :param exclude_id: if other than ``None``, exclude these class IDs from
                           the cost (useful for excluding <unk> tokens)

        :type l1_regularization: float
        :param l1_regularization: if other than ``None``, the cost will include
            a term that is the L1 norm weighted by this value

        :type l2_regularization: float
        :param l2_regularization: if other than ``None``, the cost will include
            a term that is the L2 norm weighted by this value

        :type epsilon: float
        :param epsilon: numerical stability term, added to probabilities before
                        taking the logarithm
        """

        self._network = network
        self._exclude_id = exclude_id
        self._l1_regularization = l1_regularization
        self._l2_regularization = l2_regularization
        self._epsilon = epsilon

    @abstractmethod
    def _get_logprobs(self):
        """Returns a symbolic variable that represents the log probabilities of
        the target outputs, computed either exactly or using some approximation.

        :rtype: symbolic 2D tensor
        :returns: the log probability of each time step of each sequence
        """

        assert False

    def get_tensor(self):
        """Returns a symbolic variable that represents the mini-batch cost.

        The cost is the negative log probability normalized by the number of
        tokens in the mini-batch, so that the gradients will also be normalized
        by the number of tokens. Tokens that are outside the mask will be
        excluded from the cost. If ``exclude_id`` is given to the constructor,
        that class ID will be excluded too.

        If ``l1_regularization`` or ``l2_regularization`` is given to the
        constructor, a term will be added to the cost that penalizes for large
        norm accordingly.

        :rtype: a tuple of a symbolic 2D tensor and a symbolic integer
        :returns: the cost value for each time step of each sequence, and the
                  number of words in the mini-batch
        """

        mask = self._network.mask
        if self._exclude_id is not None:
            mask *= tensor.neq(self._network.target_word_ids, self._exclude_id)
        # The mask has to be cast to floatX, otherwise the result will be
        # float64 and pulled out from the GPU earlier than necessary.
        logprobs = self._get_logprobs() * tensor.cast(mask, theano.config.floatX)
        num_words = mask.sum()
        cost = -logprobs.sum() / num_words

        model_params = self._network.get_variables().values()
        l1_weight = self._l1_regularization
        l2_weight = self._l2_regularization
        if l1_weight is not None:
            cost += l1_weight * l1_norm(model_params)
        if l2_weight is not None:
            cost += l2_weight * sum_of_squares(model_params)

        return cost, num_words

class CrossEntropyCost(Cost):
    """Cross-Entropy Cost
    """

    def __init__(self, *args, **kwargs):
        """Constructs a cost function.

        :type network: Network
        :param network: the cost function will be defined with regard to the
                        outputs of this network

        :type exclude_id: int
        :param exclude_id: if other than ``None``, exclude these class IDs from
                           the cost (useful for excluding <unk> tokens)

        :type epsilon: float
        :param epsilon: numerical stability term, added to probabilities before
                        taking the logarithm
        """

        super().__init__(*args, **kwargs)

    def _get_logprobs(self):
        """Returns a symbolic variable that represents the log probabilities of
        the target outputs.

        :rtype: Variable
        :returns: a symbolic 2-dimensional matrix that contains the log
                  probability of each time step of each sequence
        """

        return tensor.log(self._network.target_probs())

class NCECost(Cost):
    """Noise-Contrastive Estimation Cost

    M. U. Gutmann (2012)
    Noise-Contrastive Estimation of Unnormalized Statistical Models, with
    Applications to Natural Image Statistics
    http://www.jmlr.org/papers/v13/gutmann12a.html
    """

    def __init__(self, *args, **kwargs):
        """Constructs a cost function.

        :type network: Network
        :param network: the cost function will be defined with regard to the
                        outputs of this network

        :type exclude_id: int
        :param exclude_id: if other than ``None``, exclude these class IDs from
                           the cost (useful for excluding <unk> tokens)

        :type epsilon: float
        :param epsilon: numerical stability term, added to probabilities before
                        taking the logarithm
        """

        super().__init__(*args, **kwargs)

    def _get_logprobs(self):
        """Returns a symbolic variable that represents the log probabilities of
        the target outputs.

        :rtype: Variable
        :returns: a symbolic 2-dimensional matrix that contains the log
                  probability of each time step of each sequence
        """

        target_logprobs = self._network.unnormalized_logprobs()
        target_class_ids = self._network.target_class_ids
        noise_distribution = self._network.noise_distribution
        # If a single value is retured, it will be broadcasted to the mini-batch
        # shape.
        target_prior_probs = noise_distribution.probs(target_class_ids)
        target_prior_logprobs = tensor.log(target_prior_probs + self._epsilon)
        # In the article, h = 1 / (1 + e^-G). log(h) can be expressed using the
        # softplus function: log(h) = -log(1 + e^-G) = -softplus(-G)
        G = target_logprobs - target_prior_logprobs
        target_log_h = -tensor.nnet.softplus(-G)

        sample, sample_logprobs = self._network.noise_sample()
        # sample_prior_logprobs will be a one-dimensional array (or a scalar in
        # case of uniform noise), but it will be broadcasted when subtracted
        # from sample_logprobs.
        sample_prior_probs = noise_distribution.probs(sample)
        sample_prior_logprobs = tensor.log(sample_prior_probs + self._epsilon)
        # log(1 - h) = log(1 - e^G / (e^G + 1))
        #            = log((e^G + 1 - e^G) / (e^G + 1))
        #            = log(1) - log(e^G + 1)
        #            = -softplus(G)
        G = sample_logprobs - sample_prior_logprobs
        sample_log_one_minus_h = -tensor.nnet.softplus(G)
        return target_log_h + sample_log_one_minus_h.sum(2)

class BlackoutCost(Cost):
    """BlackOut Cost

    S. Ji (2016)
    BlackOut: Speeding up Recurrent Neural Network Language Models With Very
    Large Vocabularies
    https://arxiv.org/abs/1511.06909
    """

    def __init__(self, *args, **kwargs):
        """Constructs a cost function.

        :type network: Network
        :param network: the cost function will be defined with regard to the
                        outputs of this network

        :type exclude_id: int
        :param exclude_id: if other than ``None``, exclude these class IDs from
                           the cost (useful for excluding <unk> tokens)

        :type epsilon: float
        :param epsilon: numerical stability term, added to probabilities before
                        taking the logarithm
        """

        super().__init__(*args, **kwargs)

    def _get_logprobs(self):
        """Returns a symbolic variable that represents the log probabilities of
        the target outputs.

        :rtype: Variable
        :returns: a symbolic 2-dimensional matrix that contains the log
                  probability of each time step of each sequence
        """

        target_logprobs = self._network.unnormalized_logprobs()
        target_probs = tensor.exp(target_logprobs)
        target_class_ids = self._network.target_class_ids
        noise_distribution = self._network.noise_distribution
        # If a single value is retured, it will be broadcasted to the mini-batch
        # shape.
        target_prior_probs = noise_distribution.probs(target_class_ids)
        target_weighted_probs = target_probs / target_prior_probs

        sample, sample_logprobs = self._network.noise_sample()
        # sample_prior_probs will be a one-dimensional array (or a scalar in
        # case of uniform noise), but it will be broadcasted when used to divide
        # sample_logprobs.
        sample_probs = tensor.exp(sample_logprobs)
        sample_prior_probs = noise_distribution.probs(sample)
        sample_weighted_probs = sample_probs / sample_prior_probs

        denominators = target_weighted_probs + \
                       sample_weighted_probs.sum(2)
        target_costs = target_weighted_probs / denominators
        sample_costs = sample_weighted_probs / denominators[:, :, None]
        sample_costs = 1.0 - sample_costs
        result = tensor.log(target_costs + self._epsilon)
        result += tensor.log(sample_costs + self._epsilon).sum(2)
        return result
