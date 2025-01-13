from utils.utils_others import get_param
import torch
from torch import nn
from torch_scatter import scatter_add
from torch_geometric.nn import MessagePassing
from torch.nn import Parameter
import numpy as np
from typing import Dict
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from argparse import Namespace

class CompGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, act=lambda x: x, bias=True, drop_rate=0., opn='corr', reverse=True):
        super(self.__class__, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act
        self.device = None
        self.rel = None
        self.opn = opn
        self.reverse = reverse

        self.w_loop = get_param((in_channels, out_channels))
        self.w_in = get_param((in_channels, out_channels))
        self.w_out = get_param((in_channels, out_channels))
        self.w_rel = get_param((in_channels, out_channels))
        self.loop_rel = get_param((1, in_channels))

        self.drop = nn.Dropout(drop_rate)
        self.bn = nn.BatchNorm1d(out_channels)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x, edge_index, edge_type, rel_embed):
        if self.device is None:
            self.device = edge_index.device

        rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0) #rel_embed里包含了self.loop_rel，所以最后要去掉
        num_edges = edge_index.size(1) // 2
        num_ent = x.size(0)

        self.in_index, self.out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]
        self.in_type, self.out_type = edge_type[:num_edges], edge_type[num_edges:]

        self.loop_index = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(self.device)
        self.loop_type = torch.full((num_ent,), rel_embed.size(0) - 1, dtype=torch.long).to(self.device)

        self.in_norm = self.compute_norm(self.in_index, num_ent)
        self.out_norm = self.compute_norm(self.out_index, num_ent)

        in_res = self.propagate(self.in_index, x=x, edge_type=self.in_type, rel_embed=rel_embed,
                                edge_norm=self.in_norm, mode='in')
        loop_res = self.propagate(self.loop_index, x=x, edge_type=self.loop_type, rel_embed=rel_embed,
                                  edge_norm=None, mode='loop')
        out_res = self.propagate(self.out_index, x=x, edge_type=self.out_type, rel_embed=rel_embed,
                                 edge_norm=self.out_norm, mode='out')
        out = self.drop(in_res) * (1 / 3) + self.drop(out_res) * (1 / 3) + loop_res * (1 / 3)

        if self.bias is not None: out = out + self.bias
        out = self.bn(out)

        return self.act(out), torch.matmul(rel_embed, self.w_rel)[:-1]  # Ignoring the self loop inserted


    def rel_transform(self, ent_embed, rel_embed):
        def com_mult(a, b):
            r1, i1 = a[..., 0], a[..., 1]
            r2, i2 = b[..., 0], b[..., 1]
            return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)
        def conj(a):
            a[..., 1] = -a[..., 1]
            return a
        def ccorr(a, b):
            return torch.fft.irfft(com_mult(conj(torch.fft.rfft(a, 1)), torch.fft.rfft(b, 1)), 1,
                                   signal_sizes=(a.shape[-1],))
        def rotate(h, r):  # 实际用的这一种
            d = h.shape[-1]
            h_re, h_im = torch.split(h, d // 2, -1)
            r_re, r_im = torch.split(r, d // 2, -1)
            return torch.cat([h_re * r_re - h_im * r_im, h_re * r_im + h_im * r_re], dim=-1)

        if self.opn == 'corr':
            trans_embed = ccorr(ent_embed, rel_embed)
        elif self.opn == 'sub':
            trans_embed = ent_embed - rel_embed
        elif self.opn == 'mult':
            trans_embed = ent_embed * rel_embed
        elif self.opn == 'rotate':
            trans_embed = rotate(ent_embed, rel_embed.expand_as(ent_embed))
        else:
            raise NotImplementedError

        return trans_embed

    def message(self, x_j, edge_type, rel_embed, edge_norm, mode):
        weight = getattr(self, 'w_{}'.format(mode))
        rel_emb = torch.index_select(rel_embed, 0, edge_type)
        xj_rel = self.rel_transform(x_j, rel_emb)
        out = torch.mm(xj_rel, weight)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out):
        return aggr_out

    def compute_norm(self, edge_index, num_ent):
        row, col = edge_index
        edge_weight = torch.ones_like(row).float()
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_ent)  # Summing number of weights of the edges
        deg_inv = deg.pow(-0.5)  # D^{-0.5}
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[row] * edge_weight * deg_inv[col]  # D^{-0.5}

        return norm
    
class CompGCNBase(torch.nn.Module):
    def __init__(self, param):
        super(CompGCNBase, self).__init__()
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
        # self.bias = config['STAREARGS']['BIAS']
        self.triple_mode = False
        self.qual_mode = "sparse"
        self.myloss = MyLoss(self.p)

    def loss(self, pred, batch_data):
        batch_mask_label = batch_data['batch_mask_label']
        batch_label_all = batch_data['batch_target_all']
        return self.myloss(pred, batch_mask_label, batch_label_all)
    
class CompGCNEncoder(torch.nn.Module):
    def __init__(self, ent_feature, param: dict):
        super().__init__(param)

        self.p = param
        self.device = self.p.device
        self.gcn_dim = self.embed_dim if self.n_layer == 1 else self.gcn_dim
        self.ent_feature = ent_feature.to(self.device)

        if ent_feature is not None:
            self.projection = nn.Linear(ent_feature.size(-1), self.embed_dim, bias=False).to(self.device)

        #* FOR rotateE
        phases = 2 * np.pi * torch.rand(self.num_rel, self.embed_dim // 2)
        self.init_rel = nn.Parameter(torch.cat([
            torch.cat([torch.cos(phases), torch.sin(phases)], dim=-1),
            torch.cat([torch.cos(phases), -torch.sin(phases)], dim=-1)
        ], dim=0))
        self.init_rel.data[0] = 0 # padding

        #* Conv Model
        self.conv1 = CompGCNConv(self.embed_dim, self.gcn_dim, self.num_rel, act=self.act,
                                       config=param)
        self.conv2 = CompGCNConv(self.gcn_dim, self.embed_dim, self.num_rel, act=self.act,
                                       config=param) if self.n_layer == 2 else None

        self.register_parameter('bias', Parameter(torch.zeros(self.num_ent)))

    def load_ent_emb(self, ent_feature):
        #* Init embeddings
        #* If timestamps is None, initialize the embeddings
        if ent_feature is None:
            ent_emb = get_param((self.num_ent, self.embed_dim))
            ent_emb.data[0] = 0  # padding
            ent_emb.data[-1] = 0  # mask
        else:
            ent_emb = self.projection(ent_feature)
            ent_emb = torch.cat([torch.zeros(1, self.embed_dim).to(self.p.device), ent_emb, torch.zeros(1, self.embed_dim).to(self.p.device)], dim=0)
        return ent_emb

    def forward_base(self, sub, rel, drop1, drop2,
                     quals=None, embed_qualifiers: bool = False, return_mask: bool = False, base_graph = Dict[str, np.ndarray]):
        """"
        :param sub: sub_idx in batch
        :param rel: rel_idx in batch
        :param drop1:
        :param drop2:
        :param quals: (optional) (bs, maxqpairs*2) Each row is [qp, qe, qp, qe, ...]
        :param embed_qualifiers: if True, we also indexselect qualifier information
        :param return_mask: if True, returns a True/False mask of [bs, total_len] that says which positions were padded
        :return:
        """
        #* x: all ent emb
        self.init_embed = self.load_ent_emb(self.ent_feature)

        # Storing the KG
        self.edge_index = base_graph['edge_index'] #type: ignore
        self.edge_type = base_graph['edge_type'] #type: ignore


        r = self.init_rel

        if not self.triple_mode:
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
                mask = torch.zeros((sub.shape[0], quals.shape[1] + 2)).bool().to(self.device)
                # and put True where qual entities and relations are actually padding index 0
                mask[:, 2:] = quals == 0
                return sub_emb, rel_emb, qual_obj_emb, qual_rel_emb, x, mask

        return sub_emb, rel_emb, x

class CompGCNModel(CompGCNEncoder):
    def __init__(self,  ent_feature = None, param: Namespace = Namespace()):
        super(self.__class__, self).__init__(ent_feature, param)
        self.p = param
        self.hid_drop2 = self.p.drop_gcn_in
        self.feat_drop = self.p.feature_drop
        self.num_transformer_layers = self.p.trans_layer
        self.num_heads = self.p.num_heads
        self.num_hidden = self.p.hidden_dim
        self.embed_dim = self.p.embed_dim
        self.positional = (self.p.position_mode is not None)
        self.hidden_drop = torch.nn.Dropout(self.p.drop_gcn_in)
        self.hidden_drop2 = torch.nn.Dropout(self.p.drop_gcn_in)
        self.feature_drop = torch.nn.Dropout(self.p.feature_drop)

        encoder_layers = TransformerEncoderLayer(self.p.embed_dim, self.p.num_heads, self.p.hidden_dim, self.p.drop_decoder)
        self.encoder = TransformerEncoder(encoder_layers, self.p.trans_layer)
        self.position_embeddings = nn.Embedding(self.p.max_seq_length, self.p.embed_dim)
        self.layer_norm = torch.nn.LayerNorm(self.embed_dim)
        self.fc = torch.nn.Linear(self.embed_dim, self.embed_dim)

    def concat(self, e1_embed, rel_embed, qual_rel_embed, qual_obj_embed):
        e1_embed = e1_embed.view(-1, 1, self.embed_dim)
        rel_embed = rel_embed.view(-1, 1, self.embed_dim)
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
        stack_inp = torch.cat([e1_embed, rel_embed, quals], 1).transpose(1, 0)  # [2 + num_qual_pairs, bs, embed_dim]
        return stack_inp

    def forward(self, predict_type, batch_data, target_ent_index, bigraph: Dict = {}, base_graph: Dict = {}):
        '''
        :param sub: bs
        :param rel: bs
        :param quals: bs*(sl-2) # bs*14
        :return:
        '''
        sub = batch_data['batch_input_seqs'][:, 0]
        rel = batch_data['batch_input_seqs'][:, 1]
        quals = batch_data['batch_input_seqs'][:, 4:]


        sub_emb, rel_emb = self.forward_base(sub, rel, self.hidden_drop, self.feature_drop, quals, True, True, bigraph)
        # bs*embed_dim , ......, bs*6*embed_dim

        stk_inp = self.concat(sub_emb, rel_emb, qual_rel_emb, qual_obj_emb)

        if self.positional:
            positions = torch.arange(stk_inp.shape[0], dtype=torch.long, device=self.device).repeat(stk_inp.shape[1], 1)
            pos_embeddings = self.position_embeddings(positions).transpose(1, 0)
            stk_inp = stk_inp + pos_embeddings

        x = self.encoder(stk_inp, src_key_padding_mask=mask)
        x = torch.mean(x, dim=0)
        x = self.fc(x)


        #*　get target emb
        if predict_type == 'ent':
            E_embed = all_ent[:self.p.num_ent + 1] #include pad
        else:
            E_embed = rel_emb[:self.p.num_rel + 1] #include pad

        #calculate scores
        if target_ent_index is not None:
            #for inductive setting, choose target entity embedding
            target_ent_index = torch.LongTensor(target_ent_index)
            target_emb = E_embed[target_ent_index]
        else:
            target_emb = E_embed

        score = torch.mm(x, target_emb.transpose(1, 0))

        #score = torch.sigmoid(x)
        return score