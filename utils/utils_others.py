import logging
import logging.config
from typing import Any
import inspect
import numpy as np
#from torch_sparse import SparseTensor
# PyTorch related imports
import torch
import torch_scatter
from torch.nn import Parameter
from torch.nn.init import xavier_normal_
from torch_scatter import scatter_add, scatter_max
import torch.nn.functional as F
from torch_geometric.utils import degree, unbatch, to_dense_batch
import math

np.set_printoptions(precision=4)


def maybe_num_nodes(index, num_nodes=None):
    return index.max().item() + 1 if num_nodes is None else num_nodes

'''
def softmax(src, index, num_nodes=None):
    r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """

    num_nodes = maybe_num_nodes(index, num_nodes)

    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()
    out = out / (
        scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)

    return out
'''

def get_param(shape, norm=False):
    param = Parameter(torch.Tensor(*shape))
    xavier_normal_(param.data)
    if norm:
        param.data -= torch.mean(param.data, 1, keepdim=True)
        param.data /= (torch.var(param.data, 1, keepdim=True) + 1e-5) ** 0.5
    return param


def com_mult(a, b):
    r1, i1 = a[..., 0], a[..., 1]
    r2, i2 = b[..., 0], b[..., 1]
    return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)


def conj(a):
    a[..., 1] = -a[..., 1]
    return a


def cconv(a, b):
    return torch.irfft(com_mult(torch.rfft(a, 1), torch.rfft(b, 1)), 1,
                       signal_sizes=(a.shape[-1],))


def ccorr(a, b):
    return torch.irfft(com_mult(conj(torch.rfft(a, 1)), torch.rfft(b, 1)), 1,
                       signal_sizes=(a.shape[-1],))

def rotate(h, r):
    # re: first half, im: second half
    # assume embedding dim is the last dimension
    d = h.shape[-1]
    h_re, h_im = torch.split(h, d // 2, -1)
    r_re, r_im = torch.split(r, d // 2, -1)
    return torch.cat([h_re * r_re - h_im * r_im,
                        h_re * r_im + h_im * r_re], dim=-1)


def scatter_(name, src, index, dim_size=None):
    r"""Aggregates all values from the :attr:`src` tensor at the indices
    specified in the :attr:`index` tensor along the first dimension.
    If multiple indices reference the same location, their contributions
    are aggregated according to :attr:`name` (either :obj:`"add"`,
    :obj:`"mean"` or :obj:`"max"`).

    Args:
        name (string): The aggregation to use (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"max"`).
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements to scatter.
        dim_size (int, optional): Automatically create output tensor with size
            :attr:`dim_size` in the first dimension. If set to :attr:`None`, a
            minimal sized output tensor is returned. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """

    assert name in ['add', 'mean', 'max']

    op = getattr(torch_scatter, 'scatter_{}'.format(name))
    fill_value = -1e38 if name == 'max' else 0
    out = op(src, index, 0, None, dim_size, fill_value)
    if isinstance(out, tuple):
        out = out[0]

    if name == 'max':
        out[out == fill_value] = 0

    return out


#@profile
def topk_per_group(group_tensor, score_tensor, k):
    '''
    select topk score nodes in each group, return their scores and indices of the selected nodes(faster version)

    group_tensor: 1D tensor, each element is the group id of the corresponding node
    score_tensor: 1D tensor, each element is the score of the corresponding node
    k: int, number of nodes to be selected in each group
    '''
    # Reindex group_tensor starting from 0 to num_groups-1
    _, group_tensor = group_tensor.unique(sorted=True, return_inverse=True)
    group_sizes = degree(group_tensor, dtype=torch.long)
    score_tensor_matrix, mark = to_dense_batch(score_tensor, group_tensor, fill_value=float('-inf')) # [num_groups, max_group_size]
    #score_tensor_matrix, mark = to_dense_batch(score_tensor, group_tensor, fill_value=float('-9999')) # [num_groups, max_group_size]

    # Use torch.topk() to find the top k scores and their indices in each group
    topk_scores, topk_indices = score_tensor_matrix.topk(min(k, score_tensor_matrix.shape[-1]), dim=1) #[group, k]
    mark_k = mark[:, :k] #[group, k]

    #flatten, only select where mark_k is True
    topk_scores = topk_scores[mark_k] 
    group_base = group_sizes.cumsum(0) #each group's start index
    topk_indices[1:] = topk_indices[1:] + group_base[:-1].unsqueeze(1) #[group, k] + [group, 1]
    topk_indices = topk_indices[mark_k]
    
    return topk_scores, topk_indices


def topk_per_group_old(group_tensor, score_tensor, k):
    '''
    !!! require group_tensor is in order
    select topk score nodes in each group, return their scores and indices of the selected nodes

    group_tensor: 1D tensor, each element is the group id of the corresponding node
    score_tensor: 1D tensor, each element is the score of the corresponding node
    k: int, number of nodes to be selected in each group
    '''
    '''
    sorted_indices = torch.argsort(group_tensor)
    group_tensor = group_tensor[sorted_indices]
    score_tensor = score_tensor[sorted_indices]
    '''
    #reindex group_tensor starting from 0 to num_groups-1
    _, group_tensor = group_tensor.unique(sorted=True, return_inverse=True)
    sizes = degree(group_tensor, dtype=torch.long)
    size_list = sizes.tolist()
    group_scores = score_tensor.split(size_list, dim = 0)
    #group_scores = unbatch(score_tensor, group_tensor, dim= 0)

    topk_scores = []
    topk_indices = []
    
    group_base = 0
    for group_score in group_scores:
        topk_group_scores, topk_group_indices = torch.topk(group_score, min(k, group_score.shape[0]))
        topk_scores.append(topk_group_scores)
        topk_group_indices += group_base
        topk_indices.append(topk_group_indices)
        group_base += group_score.size(0)

    topk_scores = torch.cat(topk_scores)
    topk_indices = torch.cat(topk_indices)
        
    return topk_scores, topk_indices


def glorot(value: Any):
    if isinstance(value, torch.Tensor):
        stdv = math.sqrt(6.0 / (value.size(-2) + value.size(-1)))
        value.data.uniform_(-stdv, stdv)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            glorot(v)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            glorot(v)