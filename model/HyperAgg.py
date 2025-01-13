import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Parameter
#from performer_pytorch import PerformerLM, Performer, SelfAttention
from torch_geometric.utils import softmax, coalesce
from argparse import Namespace
from torch_geometric.nn.dense.linear import Linear
from typing import Any
import math
from torch_scatter import scatter_add,scatter, scatter_mean
from utils.utils_others import get_param, glorot
from model.AllSetTrans import HalfNLHconv


class HyperAggLayer(nn.Module):
    def __init__(self, params):
        '''
        For att, qr_att, qr_trans
        '''
        super(HyperAggLayer, self).__init__()
        self.p = params

        if self.p.hagg_method == 'sage':
            self.W = Linear(self.p.embed_dim, self.p.embed_dim, False, weight_initializer='glorot')

        elif self.p.hagg_method == 'att':
            self.heads = self.p.num_heads
            self.att_src = Parameter(torch.Tensor(1, self.heads, self.p.embed_dim))
            self.att_dst = Parameter(torch.Tensor(1, self.heads, self.p.embed_dim))
            self.lin_src = Linear(self.p.embed_dim, self.heads * self.p.embed_dim, False, weight_initializer='glorot')
            self.lin_dst = Linear(self.p.embed_dim, self.heads * self.p.embed_dim, False, weight_initializer='glorot')
            self.lin_src.reset_parameters()
            self.lin_dst.reset_parameters()
            glorot(self.att_src)
            glorot(self.att_dst)

        elif self.p.hagg_method == 'settrans':
            self.Hyconvs = HalfNLHconv(in_dim=self.p.embed_dim,
                                    hid_dim=self.p.embed_dim,
                                    out_dim=self.p.embed_dim,
                                    num_layers=2,
                                    dropout=self.p.drop_gcn_in,
                                    Normalization='ln',
                                    InputNorm=True,
                                    heads=4,
                                    attention=True)
            self.bn = nn.BatchNorm1d(self.p.embed_dim)
            

    def SageAgg(self, source_emb, target_emb, edge_index):
        x_i = self.W(source_emb)
        if target_emb is not None:
            x_j = self.W(target_emb)
        else:
            j_emb = scatter(source_emb, edge_index[1], dim=0, reduce='mean') #, [n_edge, dim]
            x_j = j_emb[edge_index[1]]
            x_j = self.W(x_j)
        v_out = scatter(x_i, edge_index[1], dim=0, reduce='mean')
        #v_out = F.relu(v_out)
        #v_out = scatter_add(x_i, edge_index[1], dim=0)
        return v_out


    def GATAgg(self, source_emb, target_emb, edge_index):
        x_i = self.lin_src(source_emb).view(-1, self.heads, self.p.embed_dim)  # Wq, [n_edge, self.heads, dim]
        if target_emb is not None:
            x_j = self.lin_dst(target_emb).view(-1, self.heads, self.p.embed_dim) #Wj
        else:
            j_emb = scatter(source_emb, edge_index[1], dim=0, reduce='mean') #, [n_edge, dim]
            x_j = j_emb[edge_index[1]]
            x_j = self.lin_dst(x_j).view(-1, self.heads, self.p.embed_dim) #Wj, [n_edge, self.heads, dim]
        alpha_src = (x_i * self.att_src).sum(-1) #a * Wq, [n_edge, self.heads]
        alpha_dst = (x_j * self.att_dst).sum(-1) #a * Wj, [n_edge, self.heads]
        alpha = alpha_src + alpha_dst #a * Wq + a * Wj
        alpha = F.leaky_relu(alpha, 0.2)  # relu(a*W*source + a*W*target): (n_edge,self.heads)
        alpha = softmax(alpha, edge_index[1], dim=0) # [n_edge, self.heads]
        x_i = x_i * alpha.unsqueeze(-1)  # (n_edge, self.heads, dim) * (n_edge, self.heads) : (n_edge, self.heads, dim)
        x_i = x_i.mean(dim=1)  # (n_edge, dim)
        
        v_out = scatter(x_i, edge_index[1], dim=0, reduce='sum')
        return v_out, alpha
    
    def SetTransAgg(self, x_source, x_target, edge_index):
        #calculate x_target
        if x_target is None:
            source_emb = x_source[edge_index[0]]
            x_target = scatter(source_emb, edge_index[1], dim=0, reduce='mean') #[n_edge, dim]

        norm = torch.ones_like(edge_index[0])
        all_emb = torch.cat([x_source, x_target], dim=0)
        new_edge_index = torch.stack([edge_index[0], edge_index[1] + x_source.size(0)], dim=0)  #edge_index max > x_target size        
        v_out = self.Hyconvs(all_emb, new_edge_index, norm)
        v_out = F.relu(v_out)
        target_out = v_out[-x_target.size(0):]
        return target_out
    

    def MMAN(self, e_emb, edge_index):
        x_i = e_emb
        x_i_K = self.lin_src_K(x_i).view(-1, self.heads, self.p.embed_dim)
        x_i_V = self.lin_src_V(x_i).view(-1, self.heads, self.p.embed_dim)
        alpha = (self.Q * x_i_K).sum(dim=-1)  # (n_edge, heads)
        alpha = alpha / math.sqrt(x_i_K.size(-1))
        alpha = softmax(alpha, edge_index[0], dim=0)  # softmax for source nodes to the same target, (n_edge, view_num, heads)
        out = (x_i_V * (alpha.unsqueeze(-1))).mean(dim=2)  # (n_edge, view_num, heads, dim)  * (n_edge, view_num, heads,1 ) → (n_edge, view_num, heads, dim) → (n_edge, view_num, dim)
        v_out = scatter(out, edge_index[0], dim=0, reduce='sum')  # (n_group, view_num, dim)
        return v_out, alpha


    def forward(self, edge_source_emb, edge_target_emb, edge_index):
        '''
        edge_source_emb: [n_edge, dim]
        edge_target_emb: [n_edge, dim]
        edge_index: [2, n_edge]
        return: [n_unique_target, dim]
        '''
        if self.p.hagg_method == 'sage':
            target_node_emb = self.SageAgg(edge_source_emb, edge_target_emb, edge_index)
        elif self.p.hagg_method == 'att':
            target_node_emb, alpha = self.GATAgg(edge_source_emb, edge_target_emb, edge_index)
        elif self.p.hagg_method == 'settrans':
            target_node_emb = self.SetTransAgg(x_source = edge_source_emb, x_target = edge_target_emb, edge_index = edge_index)
        return target_node_emb


class HyperAggModel(torch.nn.Module):
    def __init__(self, param: Namespace = Namespace()):
        super(HyperAggModel, self).__init__()
        self.p = param
        self.cls_edge_emb = get_param((1, self.p.embed_dim))
        self.pad_edge_emb = get_param((1, self.p.embed_dim))

        #self.hagg_method = E2VAgg(self.p, self.p.hagg_method)
        #Transformer
        self.lin_rel = Linear(self.p.embed_dim, self.p.embed_dim, False, weight_initializer='glorot')
        self.agg_layer = HyperAggLayer(self.p)

        self.V2Econvs = torch.nn.ModuleList()
        self.E2Vconvs = torch.nn.ModuleList()
        for k in range(self.p.K_hop + 1):
            self.V2Econvs.append(HyperAggLayer(self.p))
            self.E2Vconvs.append(HyperAggLayer(self.p))
    
    #@profile
    def forward(self, batch_data, unity_embeddings, mode = 'train'):
        with torch.no_grad():
            h_out_list = []
            G = batch_data['batch_graph']
            subg_node_id = G['node'].x #subg_node_id may have dupliacated node
            ori_subg_hedge_id = G['Hedge'].x
            V2E_edge_index = G['node', 'to', 'Hedge'].edge_index
            V2E_edge_type = G['node', 'to', 'Hedge'].edge_type
            V2E_edge_attr = G['node', 'to', 'Hedge'].edge_attr
            query_hypere = batch_data['batch_predict_hedge']
            batch_query_rel = batch_data['batch_query_rel']
            batch_mask_etype = batch_data['batch_mask_etype']

            #* turn to original index
            V2E_edge_index[1] = ori_subg_hedge_id[V2E_edge_index[1]]
            V2E_edge_index[0] = subg_node_id[V2E_edge_index[0]]


            # *****************************Deal with E2V conv********************************
            # delete duplicated subg_hedge_id, subg_node_id and get V2E_edge_reindex for E2V aggragation
            subg_node_id, V2E_edge_row_reindex = V2E_edge_index[0].unique(sorted=True, return_inverse=True)
            uni_subg_hedge, V2E_edge_col_reindex = V2E_edge_index[1].unique(sorted=True, return_inverse=True)
            E2V_edge_index = torch.stack([V2E_edge_col_reindex, V2E_edge_row_reindex],dim=0)
            E2V_edge_index, E2V_edge_attr = coalesce(E2V_edge_index, V2E_edge_attr, sort_by_row=False, reduce = 'min')
            predict_edge_index = torch.where(torch.eq(uni_subg_hedge.unsqueeze(1), batch_data['batch_predict_hedge'].unsqueeze(0)))[0]


            # *****************************Deal with V2E conv********************************
            #* add relation(edge_attr) to V2E_edge_index
            if self.p.V2E_with_rel:
                rel_edge_index = torch.stack([V2E_edge_attr, V2E_edge_index[1]], dim=0) #type: ignore , the same length as original_edge
                V2E_edge_index = torch.cat([V2E_edge_index, rel_edge_index], dim=1)

            #* add query relation to V2E_edge_index
            if self.p.mark_query is not None:
                #batch_query_rel = batch_query_rel.unsqueeze(1).expand(-1, self.p.neg_num+1).reshape(-1) #batch * [neg_num + 1]
                query_edge = torch.stack([batch_query_rel, query_hypere], dim=0) 
                V2E_edge_index = torch.cat([V2E_edge_index, query_edge], dim=1)

            #* delete duplicate hedge, pad node set to the same length for each hyperedge
            V2E_edge_index = coalesce(V2E_edge_index, sort_by_row=False, reduce = 'max')
            V2E_edge_index[1] = V2E_edge_index[1].unique(sorted=True, return_inverse=True)[1] # turn to continuous index
            V2E_edge_index[0] = V2E_edge_index[0].unique(sorted=True, return_inverse=True)[1] # turn to continuous index
            subg_rel_id = torch.cat([V2E_edge_attr, batch_query_rel],dim = 0).unique(sorted=True)

        
        subg_hedge_emb = None
        subg_node_emb = unity_embeddings[subg_node_id]
        subg_rel_emb = unity_embeddings[subg_rel_id]
        subg_all_emb = torch.cat([subg_node_emb, subg_rel_emb], dim=0)

        for k in range(self.p.K_hop + 1):
            #--------------------------1: V2E Conv, update hedge embeddings--------------------------#
            if self.p.hagg_method == 'settrans':
                subg_hedge_emb = self.V2Econvs[k](subg_all_emb, subg_hedge_emb, V2E_edge_index)
            else:
                V2E_source_emb = subg_all_emb[V2E_edge_index[0]]
                V2E_target_emb = None if subg_hedge_emb is None else subg_hedge_emb[V2E_edge_index[1]]
                subg_hedge_emb = self.V2Econvs[k](V2E_source_emb, V2E_target_emb, V2E_edge_index)
            h_out_list.append(subg_hedge_emb[predict_edge_index])

            #--------------------------2: E2V Conv, update vertex embeddings--------------------------#
            if k < self.p.K_hop:
                if self.p.hagg_method == 'settrans':
                    subg_node_emb = self.E2Vconvs[k](subg_hedge_emb, subg_node_emb, E2V_edge_index)
                else:
                    E2V_source_emb = subg_hedge_emb[E2V_edge_index[0]]
                    E2V_target_emb = subg_node_emb[E2V_edge_index[1]]
                    subg_node_emb = self.E2Vconvs[k](E2V_source_emb, E2V_target_emb, E2V_edge_index)
                subg_node_emb = subg_node_emb.to(subg_rel_emb.dtype)
                subg_all_emb = torch.cat([subg_node_emb, subg_rel_emb], dim=0)
                unity_embeddings[subg_node_id] = subg_node_emb
        h_out = h_out_list[-1]
        return h_out

