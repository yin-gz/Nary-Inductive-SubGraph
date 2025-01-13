'''
Codes from Message Passing for Hyper-Relational Knowledge Graphs
'''
import torch
import numpy as np
from typing import Dict
from torch_geometric.nn import MessagePassing
from torch.nn import Parameter
import torch.nn as nn
from torch.nn.init import xavier_normal_
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_max, scatter_mean, scatter
from argparse import Namespace
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from model.transfer_model import TransLoss
from torch_geometric.utils import softmax, coalesce, degree

def get_param(shape):
    param = Parameter(torch.Tensor(*shape))
    xavier_normal_(param.data)
    return param

def conj(a):
    a[..., 1] = -a[..., 1]
    return a

def ccorr(a, b):
    return torch.irfft(com_mult(conj(torch.rfft(a, 1)), torch.rfft(b, 1)), 1,
                       signal_sizes=(a.shape[-1],))

def com_mult(a, b):
    r1, i1 = a[..., 0], a[..., 1]
    r2, i2 = b[..., 0], b[..., 1]
    return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)

def rotate(h, r):
    # re: first half, im: second half
    # assume embedding dim is the last dimension
    d = h.shape[-1]
    h_re, h_im = torch.split(h, d // 2, -1)
    r_re, r_im = torch.split(r, d // 2, -1)
    return torch.cat([h_re * r_re - h_im * r_im,
                        h_re * r_im + h_im * r_re], dim=-1)

def maybe_num_nodes(index, num_nodes=None):
    return index.max().item() + 1 if num_nodes is None else num_nodes

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

class StarEConvLayer(MessagePassing):
    """ The important stuff. """

    def __init__(self, in_channels, out_channels, num_rels, act=lambda x: x,
                 config=None):
        super(self.__class__, self).__init__(flow='target_to_source',
                                             aggr='add')

        self.p = config
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_rels = num_rels
        self.act = act
        self.use_att = False

        self.w_loop = get_param((in_channels, out_channels))  # (100,200)
        self.w_in = get_param((in_channels, out_channels))  # (100,200)
        self.w_out = get_param((in_channels, out_channels))  # (100,200)
        self.w_rel = get_param((in_channels, out_channels))  # (100,200)
        self.w_q = get_param((in_channels, in_channels))  # new for quals setup


        self.loop_rel = get_param((1, in_channels))  # (1,100)
        self.loop_ent = get_param((1, in_channels))  # new

        self.drop = torch.nn.Dropout(self.p.drop_gcn_in)
        self.bn = torch.nn.BatchNorm1d(out_channels)


        self.heads = self.p.num_heads
        self.attn_dim = self.out_channels // self.heads
        self.negative_slope = 0.01
        self.attn_drop = 0.3
        self.att = get_param((1, self.heads, 2 * self.attn_dim))

        self.register_parameter('bias', Parameter(
            torch.zeros(out_channels)))

    def forward(self, x, edge_index, edge_type, rel_embed,
                qualifier_ent=None, qualifier_rel=None, quals=None):

        """

        See end of doc string for explaining.

        :param x: all entities*dim_of_entities (for jf17k -> 28646*200)
        :param edge_index: COO matrix (2 list each having nodes with index
        [1,2,3,4,5]
        [3,4,2,5,4]

        Here node 1 and node 3 are connected with edge.
        And the type of edge can be found using edge_type.

        Note that there are twice the number of edges as each edge is also reversed.
        )
        :param edge_type: The type of edge connecting the COO matrix
        :param rel_embed: 2 Times Total relation * embed_dim (200 in our case and 2 Times because of inverse relations)
        :param qualifier_ent:
        :param qualifier_rel:
        :param quals: Another sparse matrix

        where
            quals[0] --> qualifier relations type
            quals[1] --> qualifier entity
            quals[2] --> index of the original COO matrix that states for which edge this qualifier exists ()


        For argument sake if a knowledge graph has following statements

        [e1,p1,e4,qr1,qe1,qr2,qe2]
        [e1,p1,e2,qr1,qe1,qr2,qe3]
        [e1,p2,e3,qr3,qe3,qr2,qe2]
        [e1,p2,e5,qr1,qe1]
        [e2,p1,e4]
        [e4,p3,e3,qr4,qe1,qr2,qe4]
        [e1,p1,e5]
                                                 (incoming)         (outgoing)
                                            <----(regular)------><---(inverse)------->
        Edge index would be             :   [e1,e1,e1,e1,e2,e4,e1,e4,e2,e3,e5,e4,e3,e5]
                                            [e4,e2,e3,e5,e4,e3,e5,e1,e1,e1,e1,e2,e4,e1]

        Edge Type would be              :   [p1,p1,p2,p2,p1,p3,p1,p1_inv,p1_inv,p2_inv,p2_inv,p1_inv,p3_inv,p1_inv]

                                            <-------on incoming-----------------><---------on outgoing-------------->
        quals would be                  :   [qr1,qr2,qr1,qr2,qr3,qr2,qr1,qr4,qr2,qr1,qr2,qr1,qr2,qr3,qr2,qr1,qr4,qr2]
                                            [qe1,qe2,qe1,qe3,qe3,qe2,qe1,qe1,qe4,qe1,qe2,qe1,qe3,qe3,qe2,qe1,qe1,qe4]
                                            [0,0,1,1,2,2,3,5,5,0,0,1,1,2,2,3,5,5]
                                            <--on incoming---><--outgoing------->

        Note that qr1,qr2... and qe1, qe2, ... all belong to the same space
        :return:
        """

        rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)
        num_edges = edge_index.size(1) // 2
        num_ent = x.size(0)

        #in_index 和 out_index是被分开的
        self.in_index, self.out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]
        self.in_type, self.out_type = edge_type[:num_edges], edge_type[num_edges:]

        
        if self.p.binary == False and quals is not None:
            num_quals = quals.size(1) // 2
            self.in_index_qual_ent, self.out_index_qual_ent = quals[1, :num_quals], quals[1, num_quals:]
            self.in_index_qual_rel, self.out_index_qual_rel = quals[0, :num_quals], quals[0, num_quals:]
            self.quals_index_in, self.quals_index_out = quals[2, :num_quals], quals[2, num_quals:]

        '''
            Adding self loop by creating a COO matrix. Thus \
             loop index [1,2,3,4,5]
                        [1,2,3,4,5]
             loop type [10,10,10,10,10] --> assuming there are 9 relations


        '''
        # Self edges between all the nodes
        self.loop_index = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(self.p.device)
        self.loop_type = torch.full((num_ent,), rel_embed.size(0) - 1,
                                    dtype=torch.long).to(self.p.device)  # if rel meb is 500, the index of the self emb is
        # 499 .. which is just added here

        self.in_norm = self.compute_norm(self.in_index, num_ent)
        self.out_norm = self.compute_norm(self.out_index, num_ent)


        #if self.p['STATEMENT_LEN'] != 3: # type: ignore
        if self.p.binary or quals is None: # type: ignore
            in_res = self.propagate(self.in_index, x=x, edge_type=self.in_type,
                                    rel_embed=rel_embed, edge_norm=self.in_norm, mode='in',
                                    ent_embed=None, qualifier_ent=None, qualifier_rel=None,
                                    qual_index=None, source_index=None)

            loop_res = self.propagate(self.loop_index, x=x, edge_type=self.loop_type,
                                      rel_embed=rel_embed, edge_norm=None, mode='loop',
                                      ent_embed=None, qualifier_ent=None, qualifier_rel=None,
                                      qual_index=None, source_index=None)

            out_res = self.propagate(self.out_index, x=x, edge_type=self.out_type,
                                     rel_embed=rel_embed, edge_norm=self.out_norm, mode='out',
                                     ent_embed=None, qualifier_ent=None, qualifier_rel=None,
                                     qual_index=None, source_index=None)
        else:
            in_res = self.propagate(self.in_index, x=x, edge_type=self.in_type,
                                    rel_embed=rel_embed, edge_norm=self.in_norm, mode='in',
                                    ent_embed=x, qualifier_ent=self.in_index_qual_ent,
                                    qualifier_rel=self.in_index_qual_rel,
                                    qual_index=self.quals_index_in,
                                    source_index=self.in_index[0])
            loop_res = self.propagate(self.loop_index, x=x, edge_type=self.loop_type,
                                      rel_embed=rel_embed, edge_norm=None, mode='loop',
                                      ent_embed=None, qualifier_ent=None, qualifier_rel=None,
                                      qual_index=None, source_index=None)
            out_res = self.propagate(self.out_index, x=x, edge_type=self.out_type,
                                     rel_embed=rel_embed, edge_norm=self.out_norm, mode='out',
                                     ent_embed=x, qualifier_ent=self.out_index_qual_ent,
                                     qualifier_rel=self.out_index_qual_rel,
                                     qual_index=self.quals_index_out,
                                     source_index=self.out_index[0])



        out = self.drop(in_res) * (1 / 3) + self.drop(out_res) * (1 / 3) + loop_res * (1 / 3)
        out = self.bn(out)

        # Ignoring the self loop inserted, return.
        return self.act(out), torch.matmul(rel_embed, self.w_rel)[:-1]

    def rel_transform(self, ent_embed, rel_embed):
        trans_embed = rotate(ent_embed, rel_embed)
        '''
        if self.p['STAREARGS']['OPN'] == 'corr':
            trans_embed = ccorr(ent_embed, rel_embed)
        elif self.p['STAREARGS']['OPN'] == 'sub':
            trans_embed = ent_embed - rel_embed
        elif self.p['STAREARGS']['OPN'] == 'mult':
            trans_embed = ent_embed * rel_embed
        elif self.p['STAREARGS']['OPN'] == 'rotate':
            trans_embed = rotate(ent_embed, rel_embed)
        else:
            raise NotImplementedError
        '''

        return trans_embed

    def qual_transform(self, qualifier_ent, qualifier_rel):
        trans_embed = rotate(qualifier_ent, qualifier_rel)
        '''
        if self.p['STAREARGS']['QUAL_OPN'] == 'corr':
            trans_embed = ccorr(qualifier_ent, qualifier_rel)
        elif self.p['STAREARGS']['QUAL_OPN'] == 'sub':
            trans_embed = qualifier_ent - qualifier_rel
        elif self.p['STAREARGS']['QUAL_OPN'] == 'mult':
            trans_embed = qualifier_ent * qualifier_rel
        elif self.p['STAREARGS']['QUAL_OPN'] == 'rotate':
            trans_embed = rotate(qualifier_ent, qualifier_rel)
        else:
            raise NotImplementedError
        '''
        return trans_embed

    def qualifier_aggregate(self, qualifier_emb, rel_part_emb, alpha=0.5, qual_index=None):
        """
            sum aggregation
            qualifier_emb      :   [a,b,c,d,e,f,g,......]               (here a,b,c ... are of 200 dim)
            qual_index         :   [1,1,2,1,2,3,2,......]               (here 1,2,3 .. are edge index of Main COO)
            edge_type          :   [q,w,e',r,t,y,u,i,o,p, .....]        (here q,w,e' .. are of 200 dim each)
            rel_part_emb = rel_embed[edge_type]
        """


        qualifier_emb = torch.einsum('ij,jk -> ik',
                                        self.coalesce_quals(qualifier_emb, qual_index, rel_part_emb.shape[0]),
                                        self.w_q)
        return alpha * rel_part_emb + (1 - alpha) * qualifier_emb      # [N_EDGES / 2 x embed_dim]


    def update_rel_emb_with_qualifier(self, ent_embed, rel_embed,
                                      qualifier_ent, qualifier_rel, edge_type, qual_index=None):
        """
        The update_rel_emb_with_qualifier method performs following functions:

        Input is the secondary COO matrix (QE (qualifier entity), QR (qualifier relation), edge index (Connection to the primary COO))

        Step1 : Embed all the input
            Step1a : Embed the qualifier entity via ent_embed (So QE shape is 33k,1 -> 33k,200)
            Step1b : Embed the qualifier relation via rel_embed (So QR shape is 33k,1 -> 33k,200)
            Step1c : Embed the main statement edge_type via rel_embed (So edge_type shape is 61k,1 -> 61k,200)

        Step2 : Combine qualifier entity emb and qualifier relation emb to create qualifier emb (See self.qual_transform).
            This is generally just summing up. But can be more any pair-wise function that returns one vector for a (qe,qr) vector

        Step3 : Update the edge_type embedding with qualifier information. This uses scatter_add/scatter_mean.


        before:
            qualifier_emb      :   [a,b,c,d,e,f,g,......]               (here a,b,c ... are of 200 dim)
            qual_index         :   [1,1,2,1,2,3,2,......]               (here 1,2,3 .. are edge index of Main COO)
            edge_type          :   [q,w,e',r,t,y,u,i,o,p, .....]        (here q,w,e' .. are of 200 dim each)

        After:
            edge_type          :   [q+(a+b+d),w+(c+e+g),e'+f,......]        (here each element in the list is of 200 dim)


        :param ent_embed: essentially x (28k*200 in case of Jf17k)
        :param rel_embed: essentially relation embedding matrix

        For secondary COO matrix (QE, QR, edge index)
        :param qualifier_ent:  QE
        :param qualifier_rel: QR
        edge_type:
        :return:

        index select from embedding
        phi operation between qual_ent, qual_rel
        """

        # Step 1: embedding
        qualifier_emb_rel = rel_embed[qualifier_rel]
        qualifier_emb_ent = ent_embed[qualifier_ent]

        rel_part_emb = rel_embed[edge_type]

        # Step 2: pass it through qual_transform
        qualifier_emb = self.qual_transform(qualifier_ent=qualifier_emb_ent,
                                            qualifier_rel=qualifier_emb_rel)

        # Pass it through a aggregate layer
        return self.qualifier_aggregate(qualifier_emb, rel_part_emb, alpha=0.8,
                                        qual_index=qual_index)

    # return qualifier_emb
    def message(self, x_j, x_i, edge_type, rel_embed, edge_norm, mode, ent_embed=None, qualifier_ent=None,
                qualifier_rel=None, qual_index=None, source_index=None):

        """

        The message method performs following functions

        Step1 : get updated relation representation (rel_embed) [edge_type] by aggregating qualifier information (self.update_rel_emb_with_qualifier).
        Step2 : Obtain edge message by transforming the node embedding with updated relation embedding (self.rel_transform).
        Step3 : Multiply edge embeddings (transform) by weight
        Step4 : Return the messages. They will be sent to subjects (1st line in the edge index COO)
        Over here the node embedding [the first list in COO matrix] is representing the message which will be sent on each edge


        More information about updating relation representation please refer to self.update_rel_emb_with_qualifier

        :param x_j: objects of the statements (2nd line in the COO)
        :param x_i: subjects of the statements (1st line in the COO)
        :param edge_type: relation types
        :param rel_embed: embedding matrix of all relations
        :param edge_norm:
        :param mode: in (direct) / out (inverse) / loop
        :param ent_embed: embedding matrix of all entities
        :param qualifier_ent:
        :param qualifier_rel:
        :param qual_index:
        :param source_index:
        :return:
        """
        weight = getattr(self, 'w_{}'.format(mode))
        if self.p.binary == False and qual_index is not None:
            # add code here
            if mode != 'loop':
                rel_emb = self.update_rel_emb_with_qualifier(ent_embed, rel_embed, qualifier_ent,
                                                                 qualifier_rel, edge_type, qual_index)
            else:
                rel_emb = torch.index_select(rel_embed, 0, edge_type)
        else:
            rel_emb = torch.index_select(rel_embed, 0, edge_type)

        xj_rel = self.rel_transform(x_j, rel_emb)
        out = torch.einsum('ij,jk->ik', xj_rel, weight)

        if self.use_att and mode != 'loop':
            out = out.view(-1, self.heads, self.attn_dim)
            x_i = x_i.view(-1, self.heads, self.attn_dim)

            alpha = torch.einsum('bij,kij -> bi', [torch.cat([x_i, out], dim=-1), self.att])
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = softmax(alpha, source_index, ent_embed.size(0))
            alpha = F.dropout(alpha, p=self.attn_drop)
            return out * alpha.view(-1, self.heads, 1)
        else:
            return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out, mode):
        if self.use_att and mode != 'loop':
            aggr_out = aggr_out.view(-1, self.heads * self.attn_dim)

        return aggr_out

    @staticmethod
    def compute_norm(edge_index, num_ent):
        """
        Re-normalization trick used by GCN-based architectures without attention.

        Yet another torch scatter functionality. See coalesce_quals for a rough idea.

        row         :      [1,1,2,3,3,4,4,4,4, .....]        (about 61k for Jf17k)
        edge_weight :      [1,1,1,1,1,1,1,1,1,  ....] (same as row. So about 61k for Jf17k)
        deg         :      [2,1,2,4,.....]            (same as num_ent about 28k in case of Jf17k)

        :param edge_index:
        :param num_ent:
        :return:
        """
        row, col = edge_index
        edge_weight = torch.ones_like(
            row).float()  # Identity matrix where we know all entities are there
        deg = scatter_add(edge_weight, row, dim=0,
                          dim_size=num_ent)  # Summing number of weights of
        # the edges, D = A + I
        deg_inv = deg.pow(-0.5)  # D^{-0.5}
        deg_inv[deg_inv == float('inf')] = 0  # for numerical stability
        norm = deg_inv[row] * edge_weight * deg_inv[
            col]  # Norm parameter D^{-0.5} *

        return norm

    def coalesce_quals(self, qual_embeddings, qual_index, num_edges, fill=0):
        """

        before:
            qualifier_emb      :   [a,b,c,d,e,f,g,......]               (here a,b,c ... are of 200 dim)
            qual_index         :   [1,1,2,1,2,3,2,......]               (here 1,2,3 .. are edge index of Main COO)
            edge_type          :   [0,0,0,0,0,0,0, .....]               (empty array of size num_edges)

        After:
            edge_type          :   [a+b+d,c+e+g,f ......]        (here each element in the list is of 200 dim)

        :param qual_embeddings: shape of [1, N_QUALS]
        :param qual_index: shape of [1, N_QUALS] which states which quals belong to which main relation from the index,
            that is, all qual_embeddings that have the same index have to be summed up
        :param num_edges: num_edges to return the appropriate tensor
        :param fill: fill value for the output matrix - should be 0 for sum/concat and 1 for mul qual aggregation strat
        :return: [1, N_EDGES]
        """
        #TODO:  check what this function do!!!!
        output = scatter_add(qual_embeddings, qual_index, dim=0, dim_size=num_edges)

        if fill != 0:
            # by default scatter_ functions assign zeros to the output, so we assign them 1's for correct mult
            mask = output.sum(dim=-1) == 0
            output[mask] = fill

        return output

    def __repr__(self):
        return '{}({}, {}, num_rels={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_rels)


class StarEBase(torch.nn.Module):
    def __init__(self, param):
        super(StarEBase, self).__init__()
        """ Not saving the config dict bc model saving can get a little hairy. """

        self.act = torch.tanh
        self.bceloss = torch.nn.BCELoss()
        self.p = param

        self.embed_dim = self.p.embed_dim
        self.num_ent = self.p.num_ent + 2 #include pad, mask
        self.num_rel = self.p.num_rel + 2 #include pad, mask
        self.n_layer = 3
        self.gcn_dim = self.p.embed_dim
        self.hid_drop = self.p.drop_gcn_in
        self.triple_mode = self.p.binary
        self.qual_mode = "sparse"
        self.myloss = TransLoss(self.p)

    def loss(self, pred, batch_data):
        batch_mask_label = batch_data['batch_mask_label']
        batch_label_all = batch_data['batch_target_all']
        return self.myloss(pred, batch_mask_label, batch_label_all)
    


class StarEEncoder(StarEBase):
    def __init__(self, ent_feature, param: dict):
        super().__init__(param)

        self.p = param
        self.gcn_dim = self.embed_dim if self.n_layer == 1 else self.gcn_dim
        self.dis_embedding = get_param(((2*self.p.K_hop+3)*(2*self.p.K_hop+3), self.p.embed_dim))
        
        if ent_feature is not None:
            self.ent_feature = ent_feature.to(self.p.device)
            self.projection = nn.Linear(ent_feature.size(-1), self.embed_dim, bias=False).to(self.p.device)
            self.other_emb = get_param((self.p.num_ent + 2 - self.ent_feature.size(0), self.p.embed_dim)) #add ['pad'] and ['mask']

        #* FOR rotateE
        phases = 2 * np.pi * torch.rand(self.num_rel, self.embed_dim // 2)
        self.init_rel = nn.Parameter(torch.cat([
            torch.cat([torch.cos(phases), torch.sin(phases)], dim=-1),
            torch.cat([torch.cos(phases), -torch.sin(phases)], dim=-1)
        ], dim=0))
        self.init_rel.data[0] = 0 # padding

        #* Conv Model
        self.conv1 = StarEConvLayer(self.embed_dim, self.gcn_dim, self.num_rel, act=self.act,
                                       config=param)
        self.conv2 = StarEConvLayer(self.gcn_dim, self.embed_dim, self.num_rel, act=self.act,
                                       config=param) if self.n_layer == 2 else None

        self.register_parameter('bias', Parameter(torch.zeros(self.num_ent)))

    def load_ent_emb(self, bigraph):
        #* Init embeddings
        #* If timestamps is None, initialize the embeddings
        #if self.p.ent_feature == 'learn':
            #ent_emb = get_param((self.num_ent, self.embed_dim))
            #ent_emb.data[0] = 0  # padding
            #ent_emb.data[-1] = 0  # mask
        if self.p.task == 'TR-EF':
            ent_emb_file = self.projection(self.ent_feature)
            ent_emb = torch.cat([self.other_emb[0].unsqueeze(0), ent_emb_file, self.other_emb[1:]], 0)
        else:
            bi_edge, bi_edge_type = bigraph['edge_index'], bigraph['edge_type']
            type_to_ent = torch.stack([bi_edge_type, bi_edge[0]], dim=0)
            type_to_ent = coalesce(type_to_ent, sort_by_row = False)
            type_emb = self.init_rel[type_to_ent[0]]
            ent_emb = scatter(type_emb, type_to_ent[1], dim=0, reduce='mean', dim_size=self.p.num_ent + 1) #average rel emb as ent emd [pad + ent_emb]
            ent_emb = torch.cat([ent_emb, torch.zeros(1, self.embed_dim).to(self.p.device)], dim=0) #add mask
        return ent_emb
    

    def forward_base(self, sub, rel, obj, drop1, drop2,
                     quals=None, embed_qualifiers: bool = False, return_mask: bool = False, base_graph: Dict = {}, bi_graph: Dict = {}):
        """"
        :param sub: sub_idx in batch
        :param rel: rel_idx in batch
        :param obj: obj_idx in batch
        :param drop1:
        :param drop2:
        :param quals: (optional) (bs, maxqpairs*2) Each row is [qp, qe, qp, qe, ...]
        :param embed_qualifiers: if True, we also indexselect qualifier information
        :param return_mask: if True, returns a True/False mask of [bs, total_len] that says which positions were padded
        :return:
        """
        #* x: all ent emb
        self.init_embed = self.load_ent_emb(bi_graph)
        # Storing the KG
        self.edge_index = base_graph['edge_index'] #type: ignore
        self.edge_type = base_graph['edge_type'] #type: ignore
        


        if quals is not None:
            if not self.triple_mode:
                if self.qual_mode == "sparse":
                    self.quals = torch.tensor(base_graph['quals'], dtype=torch.long, device=self.p.device)
                else:
                    raise NotImplementedError


        r = self.init_rel

        if quals is not None and (not self.triple_mode):
            #! In each batch, first, conv the graph to get repsentations
            if self.qual_mode == "full":
                # 全图进入卷积层进行计算
                # x, edge_index, edge_type, rel_embed, qual_ent, qual_rel
                x, r = self.conv1(x=self.init_embed, edge_index=self.edge_index,
                                  edge_type=self.edge_type, rel_embed=r,
                                  qualifier_ent=self.qual_ent,
                                  qualifier_rel=self.qual_rel,
                                  quals=None)

                x = drop1(x)
                x, r = self.conv2(x=x, edge_index=self.edge_index,
                                  edge_type=self.edge_type, rel_embed=r,
                                  qualifier_ent=self.qual_ent,
                                  qualifier_rel=self.qual_rel,
                                  quals=None) if self.n_layer == 2 else (x, r)
            elif self.qual_mode == "sparse":
                # x, edge_index, edge_type, rel_embed, qual_ent, qual_rel
                x, r = self.conv1(x=self.init_embed, edge_index=self.edge_index,
                                  edge_type=self.edge_type, rel_embed=r,
                                  qualifier_ent=None,
                                  qualifier_rel=None,
                                  quals=self.quals)

                x = drop1(x)
                x, r = self.conv2(x=x, edge_index=self.edge_index,
                                  edge_type=self.edge_type, rel_embed=r,
                                  qualifier_ent=None,
                                  qualifier_rel=None,
                                  quals=self.quals) if self.n_layer == 2 else (x, r)

        else:
            x, r = self.conv1(x=self.init_embed, edge_index=self.edge_index,
                              edge_type=self.edge_type, rel_embed=r)

            x = drop1(x)
            x, r = self.conv2(x=x, edge_index=self.edge_index,
                              edge_type=self.edge_type, rel_embed=r) \
                if self.n_layer == 2 else \
                (x, r)

        x = drop2(x) if self.n_layer == 2 else x

        sub_emb = torch.index_select(x, 0, sub)
        rel_emb = torch.index_select(r, 0, rel)
        obj_emb = torch.index_select(x, 0, obj)

        if embed_qualifiers:
            assert quals is not None, "Expected a tensor as quals."
            # flatten quals
            #quals_ents = quals[:, 1::2].contiguous().view(1,-1).squeeze(0)
            #quals_rels = quals[:, 0::2].contiguous().view(1,-1).squeeze(0)
            quals_ents = quals[:, 0::2].contiguous().view(1,-1).squeeze(0)
            quals_rels = quals[:, 1::2].contiguous().view(1,-1).squeeze(0)
            qual_obj_emb = torch.index_select(x, 0, quals_ents)
            # qual_obj_emb = torch.index_select(x, 0, quals[:, 1::2])
            qual_rel_emb = torch.index_select(r, 0, quals_rels)
            qual_obj_emb = qual_obj_emb.view(sub_emb.shape[0], -1 ,sub_emb.shape[1])
            qual_rel_emb = qual_rel_emb.view(rel_emb.shape[0], -1, rel_emb.shape[1])
            if not return_mask:
                return sub_emb, rel_emb, qual_obj_emb, qual_rel_emb, x
            else:
                # mask which shows which entities were padded - for future purposes, True means to mask (in transformer)
                # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py : 3770
                # so we first initialize with False
                mask = torch.zeros((sub.shape[0], quals.shape[1] + 2)).bool().to(self.p.device)
                # and put True where qual entities and relations are actually padding index 0
                mask[:, 2:] = quals == 0
                return sub_emb, rel_emb, obj_emb, qual_obj_emb, qual_rel_emb, x, mask
        
        #if self.p.model_name == 'StarESubg':
            #x = self.init_embed
        else:
            return sub_emb, rel_emb, obj_emb, x
    

class StarEModel(StarEEncoder):
    def __init__(self,  ent_feature = None, param: Namespace = Namespace()):
        super(self.__class__, self).__init__(ent_feature, param)
        self.p = param
        self.hid_drop2 = self.p.drop_gcn_in
        self.feat_drop = self.p.feature_drop
        self.num_transformer_layers = self.p.trans_layer
        self.num_heads = self.p.num_heads
        self.num_hidden = self.p.hidden_dim
        self.embed_dim = self.p.embed_dim
        self.positional = (self.p.position_mode != None)
        self.hidden_drop = torch.nn.Dropout(self.p.drop_gcn_in)
        self.hidden_drop2 = torch.nn.Dropout(self.p.drop_gcn_in)
        self.feature_drop = torch.nn.Dropout(self.p.feature_drop)

        encoder_layers = TransformerEncoderLayer(self.p.embed_dim, self.p.num_heads, self.p.hidden_dim, self.p.drop_decoder)
        self.encoder = TransformerEncoder(encoder_layers, self.p.trans_layer)
        self.position_embeddings = nn.Embedding(self.p.max_seq_length, self.p.embed_dim)
        self.layer_norm = torch.nn.LayerNorm(self.embed_dim)
        self.fc = torch.nn.Linear(self.embed_dim, self.embed_dim)

    def concat(self, e1_embed, rel_embed,e2_embed, qual_rel_embed, qual_obj_embed):
        e1_embed = e1_embed.view(-1, 1, self.embed_dim)
        rel_embed = rel_embed.view(-1, 1, self.embed_dim)
        e2_embed = e2_embed.view(-1, 1, self.embed_dim)
        """
            arrange quals in the conve format with shape [bs, num_qual_pairs, embed_dim]
            num_qual_pairs is 2 * (any qual tensor shape[1])
            for each datum in bs the order will be 
                rel1, emb
                en1, emb
                rel2, emb
                en2, emb
        """
        quals = torch.cat((qual_rel_embed, qual_obj_embed), 2).view(-1, 2 * qual_rel_embed.shape[1],
                                                                    qual_rel_embed.shape[2])
        stack_inp = torch.cat([e1_embed, rel_embed, e2_embed, quals], 1).transpose(1, 0)  # [2 + num_qual_pairs, bs, embed_dim]
        return stack_inp
    
    #@profile
    #@torch.no_grad()
    def turn_heteroG_to_starEG(self, batch_data):
        '''
        edge index:
        [ [s1, s2],
            [o1, o2] ]

        edge type:
        [ p1, p2 ]

        quals will looks like
        [ [qr1, qr2, qr3],
            [qe1, qe2, qe3],
            [0  , 0  , 1  ]       <- obtained from the edge index columns
        '''
        result_edge_index = []
        result_edge_type = []
        edge_type_inverse = []
        full_quals = []



        G = batch_data
        subg_node_id = G['node'].x #subg_node_id may have dupliacated node
        ori_subg_hedge_id = G['Hedge'].x
        V2E_edge_index = G['node', 'to', 'Hedge'].edge_index
        V2E_edge_type = G['node', 'to', 'Hedge'].edge_type
        V2E_edge_attr = G['node', 'to', 'Hedge'].edge_attr

        #turn to original index
        V2E_edge_index[1] = ori_subg_hedge_id[V2E_edge_index[1]]
        V2E_edge_index[0] = subg_node_id[V2E_edge_index[0]]

        #delete predicting edge because it has no tail entity
        if 'batch_predict_hedge' in batch_data:
            predict_edge_mask = torch.isin(V2E_edge_index[1], batch_data['batch_predict_hedge'])
        else:
            predict_edge_mask = G['Hedge'].mark_source.bool()[V2E_edge_index[1]]
        
        V2E_edge_index = V2E_edge_index[:, ~predict_edge_mask]
        V2E_edge_type = V2E_edge_type[~predict_edge_mask]
        V2E_edge_attr = V2E_edge_attr[~predict_edge_mask]

        #sort according to hedge id
        sort_index = torch.argsort(V2E_edge_index[1])
        V2E_edge_index = V2E_edge_index[:, sort_index]
        V2E_edge_type = V2E_edge_type[sort_index]
        V2E_edge_attr = V2E_edge_attr[sort_index]
        V2E_edge_index[1] = torch.unique(V2E_edge_index[1], return_inverse=True)[1]

        #delete duplicate edge
        full_edge = torch.cat([V2E_edge_index, V2E_edge_type.unsqueeze(0), V2E_edge_attr.unsqueeze(0)], dim=0)
        full_edge_t = full_edge.t()
        unique_tensor = torch.unique(full_edge_t, dim=0).t()
        V2E_edge_index = unique_tensor[:2]
        V2E_edge_type = unique_tensor[2]
        V2E_edge_attr = unique_tensor[3]

        qual_index_bool = (V2E_edge_type == 2)
        head_index_bool = (V2E_edge_type == 0)
        tail_index_bool = (V2E_edge_type == 1)
        qual_index = qual_index_bool.nonzero().squeeze(1)
        head_index = head_index_bool.nonzero().squeeze(1)
        tail_index = tail_index_bool.nonzero().squeeze(1)

        #select head and tail edge, and then concat them
        #keep all hyperedges, each one with a head-tail pair
        head_edge_index = V2E_edge_index[:, head_index]
        tail_edge_index = V2E_edge_index[:, tail_index]
        hedge_ori_id = V2E_edge_index[1, head_index]
        result_edge_index = torch.stack([head_edge_index[0], tail_edge_index[0]], dim=0).contiguous()
        result_edge_type = V2E_edge_attr[head_index]
        edge_type_inverse = V2E_edge_attr[tail_index]

        #set the remain as qual
        if qual_index.size(0) != 0:
            full_quals = torch.cat([V2E_edge_attr[qual_index].unsqueeze(0), V2E_edge_index[:, qual_index]], dim = 0)
        
        #Add reverse
        result_edge_index = torch.cat([result_edge_index, result_edge_index[[1, 0]]], dim=1).contiguous()
        result_edge_type = torch.cat([result_edge_type, edge_type_inverse])
        if len(full_quals) != 0:
            full_quals = torch.cat([full_quals, full_quals], dim=1)
            return {'edge_index': result_edge_index,
                    'edge_type': result_edge_type,
                    'quals': full_quals}
        else:
            return {'edge_index': result_edge_index,
                    'edge_type': result_edge_type}


    #@profile
    def forward(self, predict_type = '', batch_data = None, target_ent_index = None, bigraph: Dict = {}, base_graph: Dict = {}, mode = 'train'):
        '''
        :param sub: bs
        :param rel: bs
        :param quals: bs*(sl-2) # bs*14
        :return:
        '''
        if self.p.task == "PSR":
            batch_data['batch_input_seqs'] = batch_data['batch_input_seqs'].reshape(-1, self.p.max_seq_length)
        sub = batch_data['batch_input_seqs'][:, 0]
        rel = batch_data['batch_input_seqs'][:, 1]
        obj = batch_data['batch_input_seqs'][:, 2]

        #turn heterograph to starEgraph
        if self.p.task == "PSR":
            base_graph = self.turn_heteroG_to_starEG(batch_data)
        elif self.p.use_subg:
            base_graph = self.turn_heteroG_to_starEG(batch_data['batch_graph'])
        
        if self.p.binary or 'quals' not in base_graph:
            quals = None
            sub_emb, rel_emb, obj_emb, all_ent = self.forward_base(sub, rel, obj, self.hidden_drop, self.feature_drop, quals, False, True, base_graph = bigraph, bi_graph = bigraph)
            sub_emb = sub_emb.view(-1, 1, self.embed_dim)
            rel_emb = rel_emb.view(-1, 1, self.embed_dim)
            obj_emb = obj_emb.view(-1, 1, self.embed_dim)
            stk_inp = torch.cat([sub_emb, rel_emb, obj_emb], 1).transpose(1, 0)  # [3, bs, emb_dim]
        else:
            quals = batch_data['batch_input_seqs'][:, 4:]
            sub_emb, rel_emb, obj_emb, qual_obj_emb, qual_rel_emb, all_ent, mask= self.forward_base(sub, rel, obj, self.hidden_drop, self.feature_drop, quals, True, True, base_graph = base_graph, bi_graph = bigraph)
            stk_inp = self.concat(sub_emb, rel_emb, obj_emb, qual_rel_emb, qual_obj_emb)

        #delete_mask_ent
        batch_mask_pos = batch_data['batch_mask_position'][0].item()
        stk_inp = torch.cat([stk_inp[:batch_mask_pos,:,:], stk_inp[batch_mask_pos+1:,:,:]], dim=0)

        if self.positional:
            positions = torch.arange(stk_inp.shape[0], dtype=torch.long, device=self.p.device).repeat(stk_inp.shape[1], 1)
            pos_embeddings = self.position_embeddings(positions).transpose(1, 0)
            stk_inp = stk_inp + pos_embeddings

        if self.p.binary or 'quals' not in base_graph:
            x = self.encoder(stk_inp)
        else:
            x = self.encoder(stk_inp, src_key_padding_mask=mask)
        x = torch.mean(x, dim=0)
        x = self.fc(x)


        #*　get target emb
        if predict_type == 'ent':
            E_embed = all_ent[:self.p.num_ent + 1] #include pad
        else:
            E_embed = rel_emb[:self.p.num_rel + 1] #include pad

        if self.p.use_neg and 'neg_target_index' in batch_data:
            batch_target = batch_data['neg_target_index']
            target_emb = E_embed[batch_target]
        elif target_ent_index is not None:
            if type(target_ent_index) != torch.Tensor:
                target_ent_index = torch.LongTensor(target_ent_index)
            target_emb = E_embed[target_ent_index]
        elif self.p.unit_encode:
            target_emb = E_embed
        else:
            target_emb = E_embed[:-1]

        # For Negative Sampling, [batch, 1, dim] * [batch, dim, num_neg] →[batch, 1, num_neg]→[batch, num_neg]
        if len(target_emb.size()) == 3:
            score = torch.bmm(x.unsqueeze(1), target_emb.transpose(1, 2)).squeeze(1)
        else:
            score = torch.mm(x, target_emb.transpose(1, 0))

        #score = torch.sigmoid(x)
        return score