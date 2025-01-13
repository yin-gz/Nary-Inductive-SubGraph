import torch
import math
from torch import nn
from torch import Tensor
from torch.nn import Parameter
from torch.nn.init import xavier_normal_
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.nn import TransformerConv,MessagePassing
from typing import Optional, Tuple, Union
from torch_geometric.typing import Adj, OptTensor, PairTensor
import torch.nn.functional as F
#from torch_sparse import SparseTensor
from torch_geometric.utils import softmax
from torch_geometric.loader import DataLoader as GDataLoader
from torch_geometric.data import Data
from itertools import combinations


class GRANConv(MessagePassing):
    _alpha: OptTensor
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = False,
        beta: bool = False,
        dropout: float = 0.,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super(GRANConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_query = nn.Linear(in_channels[1], heads * out_channels)
        self.lin_value = nn.Linear(in_channels[0], heads * out_channels)
        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = nn.Linear(in_channels[1], heads * out_channels,
                                   bias=bias)
            if self.beta:
                self.lin_beta = nn.Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = nn.Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = nn.Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, return_attention_weights=None):
        r"""
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """

        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)

        # propagate_type: (query: Tensor, key:Tensor, value: Tensor, edge_attr: OptTensor) # noqa
        out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=edge_attr, size=None)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out += x_r

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            #elif isinstance(edge_index, SparseTensor):
                #return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
                                                      self.out_channels)
            key_j += edge_attr

        # q*(k+edge_attr)
        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j
        if edge_attr is not None:
            out += edge_attr

        out *= alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
    

class GRAN(nn.Module):
    def __init__(self,param = None):
        super(GRAN, self).__init__()
        self.p = param
        self.n_gcn_layer = self.p.n_gcn_layer

        self.GraphLayer = GRANConv(in_channels=self.p.embed_dim, out_channels=self.p.embed_dim, bias=True, heads=self.p.num_heads, dropout=self.p.drop_gcn_in, edge_dim = self.p.embed_dim, concat =False)

        #graph layers(conv+drop)
        self.gcn_comp_layers = nn.ModuleList()
        self.gcn_out_drops = nn.ModuleList()
        self.layer_norm = nn.ModuleList()
        self.FFN = nn.ModuleList()
        for i in range(self.n_gcn_layer):
            self.gcn_comp_layers.append(self.GraphLayer)
            self.gcn_out_drops.append(torch.nn.Dropout(self.p.drop_gcn_in))
            self.layer_norm.append(torch.nn.LayerNorm([self.p.embed_dim], eps=1e-6))
            self.FFN.append(nn.Sequential(nn.Linear(self.p.embed_dim, self.p.intermediate_size), nn.Linear(self.p.intermediate_size, self.p.embed_dim)))
        #decoder
        self.f_out_rel = nn.Sequential(
            nn.Linear(self.p.embed_dim, self.p.embed_dim),
            nn.Linear(self.p.embed_dim, self.p.num_rel)
        )
        self.f_out_ent = nn.Sequential(
            nn.Linear(self.p.embed_dim, self.p.embed_dim),
            nn.Linear(self.p.embed_dim, self.p.num_ent)
        )
        self.embedding_hetro_layer = nn.Embedding(10, self.p.embed_dim)
        self.embedding_hetro_layer.reset_parameters()
        

    def forward(self, batch_data, unity_embeddings, mode = 'train'):
        #batch_input_ids
        #batch_input_ids, batch_input_mask, predict_type, batch_mask_position
        
        batch_input_ids = batch_data['batch_input_seqs']
        batch_input_mask = (batch_input_ids != 0)
        batch_mask_position = batch_data['batch_mask_position']

        #consturct graph one by one
        data_list = []
        for k in range(batch_input_ids.size(0)):
            bool_mask = torch.tensor(batch_input_mask[k], dtype=torch.bool).to(self.p.device)
            node_index = batch_input_ids[k][bool_mask]
            length = node_index.size(0)
            ent_index = node_index[[i for i in range(0, length, 2)]]
            rel_index = node_index[[i for i in range(1, length, 2)]]
            e = unity_embeddings[ent_index]
            r = unity_embeddings[rel_index]
            x = torch.cat((e,r),0)
            qual_num = int((x.size(0)-3)/2)
            r_begin = qual_num + 2

            edge_index = [[0,r_begin],[1,r_begin]] #h-r / t- r
            edge_type = [1,2]

            for i in range(1,qual_num+1):
                edge_index.append([r_begin+i,r_begin]) #q_r-r / q_v - q_r
                edge_type.append(3)
                edge_index.append([2+i,r_begin+i])
                edge_type.append(4)

            for i,j in combinations([i for i in range(length)], 2):
                if ([i,j] not in edge_index) and ([j,i] not in edge_index):
                    edge_index.append([i,j])
                    edge_type.append(0)

            edge_index = torch.LongTensor(edge_index).t().to(self.p.device)
            edge_type = torch.LongTensor(edge_type).to(self.p.device)

            #add reverse
            edge_index_inverse = torch.stack((edge_index[1], edge_index[0]), 0)
            edge_type_inverse = edge_type + 5
            edge_index = torch.cat((edge_index, edge_index_inverse), 1)
            edge_type = torch.cat((edge_type, edge_type_inverse), 0)

            batch_graph = Data(x = x, edge_index=edge_index, edge_type = edge_type, edge_attr = None)
            data_list.append(batch_graph)

        loader = GDataLoader(data_list, batch_size=batch_input_ids.size(0))
        batch_mask_position = (batch_mask_position/2).long()
        for data in loader:
            data.edge_attr = self.embedding_hetro_layer(data.edge_type)

        pre_out = data.x
        for i in range(self.n_gcn_layer):
            data.x = self.gcn_comp_layers[i](data.x, data.edge_index, data.edge_attr)
            data.x = self.gcn_out_drops[i](data.x)
            data.x += pre_out
            data.x = self.layer_norm[i](data.x) #add layer normalization
            data.x = self.FFN[i](data.x)
            pre_out = data.x
        batch_mask_emb = []
        for i in range(batch_input_ids.size(0)): # batch_size sugraphs
            each_sub = data[i]
            mask_emb = each_sub.x[batch_mask_position[i]]
            batch_mask_emb.append(mask_emb)
        batch_mask_emb = torch.stack(batch_mask_emb,0).to(self.p.device)

        return batch_mask_emb
