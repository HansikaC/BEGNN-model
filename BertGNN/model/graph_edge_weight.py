from dgl.nn.pytorch import GatedGraphConv as GraphN
from dgl import function as fn
from dgl.base import DGLError
from dgl.utils import expand_as_pair
import torch as th

'''
Here we are calculating the weights 
Gated Graph Convolution layer from `Gated Graph Sequence Neural Networks <https://arxiv.org/pdf/1511.05493.pdf>
'''

class GraphEdgeWeight(GraphN):
    '''Compute Gated Graph Convolution layer.'''
    def forward(self, graph, feat,  weight=None, edge_weights=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('The DGLError will be raised if the input graph contains 0-in-degree nodes because no message will be sent to them. Invalid output will result from this. Setting the "allow zero in degree" argument to "True" will cause the error to be ignored.')

            feature_source, feat_dst = expand_as_pair(feat, graph)
            if self._norm == 'both':
                degrees = graph.out_degrees().float().clamp(min=1)
                norm = th.pow(degrees, -0.5)
                shape_ = norm.shape + (1,) * (feature_source.dim() - 1)
                norm = th.reshape(norm, shape_)
                feature_source = feature_source * norm

            if weight is not None:
                if self.weight is not None:
                    raise DGLError('Despite the fact that the module has its own specified weight parameter, external weight is still given. Please set weight=False in the module flags.')
            else:
                weight = self.weight

            if self._in_feats > self._out_feats:
                # For aggregation, multiplt Weights first to reduce the feature size.
                if weight is not None:
                    feature_source = th.matmul(feature_source, weight)
                graph.srcdata['h'] = feature_source
                if edge_weights is None:
                    graph.update_all(fn.copy_src(src='h', out='m'),fn.sum(msg='m', out='h'))
                else:
                    graph.edata['a'] = edge_weights
                    graph.update_all(fn.u_mul_e('h', 'a', 'm'),fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
            else:
                # aggregate first then mult W
                graph.srcdata['h'] = feature_source
                if edge_weights is None:
                    graph.update_all(fn.copy_src(src='h', out='m'),fn.sum(msg='m', out='h'))
                else:
                    graph.edata['a'] = edge_weights
                    graph.update_all(fn.u_mul_e('h', 'a', 'm'),fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
                if weight is not None:
                    rst = th.matmul(rst, weight)

            if self._norm != 'none':
                degrees = graph.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = th.pow(degrees, -0.5)
                else:
                    norm = 1.0 / degrees
                shape_ = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = th.reshape(norm, shape_)
                rst = rst * norm

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst