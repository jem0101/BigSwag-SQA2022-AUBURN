"""MXNet modules for graph convolutions."""
# pylint: disable= no-member, arguments-differ
import mxnet as mx
from mxnet import gluon

from ... import function as fn
from ...utils import get_ndata_name

__all__ = ['GraphConv']

class GraphConv(gluon.Block):
    r"""Apply graph convolution over an input signal.

    Graph convolution is introduced in `GCN <https://arxiv.org/abs/1609.02907>`__
    and can be described as below:

    .. math::
      h_i^{(l+1)} = \sigma(b^{(l)} + \sum_{j\in\mathcal{N}(i)}\frac{1}{c_{ij}}h_j^{(l)}W^{(l)})

    where :math:`\mathcal{N}(i)` is the neighbor set of node :math:`i`. :math:`c_{ij}` is equal
    to the product of the square root of node degrees:
    :math:`\sqrt{|\mathcal{N}(i)|}\sqrt{|\mathcal{N}(j)|}`. :math:`\sigma` is an activation
    function.

    The model parameters are initialized as in the
    `original implementation <https://github.com/tkipf/gcn/blob/master/gcn/layers.py>`__ where
    the weight :math:`W^{(l)}` is initialized using Glorot uniform initialization
    and the bias is initialized to be zero.

    Notes
    -----
    Zero in degree nodes could lead to invalid normalizer. A common practice
    to avoid this is to add a self-loop for each node in the graph, which
    can be achieved by:

    >>> g = ... # some DGLGraph
    >>> g.add_edges(g.nodes(), g.nodes())


    Parameters
    ----------
    in_feats : int
        Number of input features.
    out_feats : int
        Number of output features.
    norm : bool, optional
        If True, the normalizer :math:`c_{ij}` is applied. Default: ``True``.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.
    activation: callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.

    Attributes
    ----------
    weight : mxnet.gluon.parameter.Parameter
        The learnable weight tensor.
    bias : mxnet.gluon.parameter.Parameter
        The learnable bias tensor.
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 norm=True,
                 bias=True,
                 activation=None):
        super(GraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._feat_name = "_gconv_feat"
        self._msg_name = "_gconv_msg"

        with self.name_scope():
            self.weight = self.params.get('weight', shape=(in_feats, out_feats),
                                          init=mx.init.Xavier())
            if bias:
                self.bias = self.params.get('bias', shape=(out_feats,),
                                            init=mx.init.Zero())
            else:
                self.bias = None

        self._activation = activation

    def forward(self, feat, graph):
        r"""Compute graph convolution.

        Notes
        -----
            * Input shape: :math:`(N, *, \text{in_feats})` where * means any number of additional
              dimensions, :math:`N` is the number of nodes.
            * Output shape: :math:`(N, *, \text{out_feats})` where all but the last dimension are
              the same shape as the input.

        Parameters
        ----------
        feat : mxnet.NDArray
            The input feature
        graph : DGLGraph
            The graph.

        Returns
        -------
        mxnet.NDArray
            The output feature
        """
        self._feat_name = get_ndata_name(graph, self._feat_name)

        if self._norm:
            degs = graph.in_degrees().astype('float32')
            norm = mx.nd.power(degs, -0.5)
            shp = norm.shape + (1,) * (feat.ndim - 1)
            norm = norm.reshape(shp).as_in_context(feat.context)
            feat = feat * norm

        if self._in_feats > self._out_feats:
            # mult W first to reduce the feature size for aggregation.
            feat = mx.nd.dot(feat, self.weight.data(feat.context))
            graph.ndata[self._feat_name] = feat
            graph.update_all(fn.copy_src(src=self._feat_name, out=self._msg_name),
                             fn.sum(msg=self._msg_name, out=self._feat_name))
            rst = graph.ndata.pop(self._feat_name)
        else:
            # aggregate first then mult W
            graph.ndata[self._feat_name] = feat
            graph.update_all(fn.copy_src(src=self._feat_name, out=self._msg_name),
                             fn.sum(msg=self._msg_name, out=self._feat_name))
            rst = graph.ndata.pop(self._feat_name)
            rst = mx.nd.dot(rst, self.weight.data(feat.context))

        if self._norm:
            rst = rst * norm

        if self.bias is not None:
            rst = rst + self.bias.data(rst.context)

        if self._activation is not None:
            rst = self._activation(rst)

        return rst

    def __repr__(self):
        summary = 'GraphConv('
        summary += 'in={:d}, out={:d}, normalization={}, activation={}'.format(
            self._in_feats, self._out_feats,
            self._norm, self._activation)
        summary += '\n)'
        return summary
