import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Parameter
#from performer_pytorch import PerformerLM, Performer, SelfAttention
from torch_geometric.utils import to_dense_adj, to_dense_batch, unbatch, sort_edge_index, softmax, coalesce
from argparse import Namespace
from torch_geometric.nn import Set2Set, SAGPooling, GraphMultisetTransformer
from torch_geometric.nn.dense.linear import Linear
from typing import Any
import math
from torch_scatter import scatter_add,scatter, scatter_mean
from utils.utils_others import get_param
from torch_geometric.data import HeteroData
from model.flashatt.FlashAtt import FalshAttEncoder

class HARTBinary(nn.Module):
    def __init__(self, param: Namespace = Namespace()):
        super(HARTBinary, self).__init__()
        self.p = param
        self.cls_edge_emb = get_param((1, self.p.embed_dim))
        self.pad_edge_emb = get_param((1, self.p.embed_dim))

        #self.hagg_method = E2VAgg(self.p, self.p.hagg_method)
        #Transformer
        self.position_embeddings = nn.Embedding(2*self.p.max_arity + 3, self.p.embed_dim)
        self.lin_rel = Linear(self.p.embed_dim, self.p.embed_dim, False, weight_initializer='glorot')
        if self.p.use_flash:
            self.encoder  = FalshAttEncoder(param)
        else:
            encoder_layers = TransformerEncoderLayer(self.p.embed_dim, self.p.num_heads, self.p.hidden_dim, self.p.drop_decoder)
            self.encoder = TransformerEncoder(encoder_layers, self.p.trans_layer)

    def add_cls_emb(self, x_emb_dense, mask, mask_id):
        x_emb_dense = torch.cat([torch.zeros(x_emb_dense.size(0), 1,
                                 self.p.embed_dim, device=self.p.device), x_emb_dense], dim=1)
        if mask.size(-1) != x_emb_dense.size(1):
            if mask_id ==0:
                mask = torch.cat([torch.zeros(mask.size(0), 1, device=self.p.device).bool(), mask], dim=1)
            else:
                mask = torch.cat([torch.ones(mask.size(0), 1, device=self.p.device).bool(), mask], dim=1)
        return x_emb_dense, mask
    
    def add_cls_id(self, x_id_dense, mask, cls_id):
        x_id_dense = torch.cat([cls_id * torch.ones(x_id_dense.size(0), 1, device=self.p.device).long(), x_id_dense], dim=1)
        mask = torch.cat([torch.ones(mask.size(0), 1, device=self.p.device).bool(), mask], dim=1)
        return x_id_dense, mask
    
    def format_V2E_matrix(self, V2E_edge_index, V2E_edge_attr, V2E_edge_type, query_hypere = None, batch_query_rel = None, batch_mask_position = None):
            #* add relation(edge_attr) to V2E_edge_index
            if self.p.V2E_with_rel:
                rel_edge_index = torch.stack([V2E_edge_attr, V2E_edge_index[1]], dim=0) #type: ignore , the same length as original_edge
                V2E_edge_index = torch.cat([V2E_edge_index, rel_edge_index], dim=1)
                #generate rel_edge's type according to thier edge_type3
                rel_edge_type = V2E_edge_type + 3 #rel: 3\4\5
                V2E_edge_type = torch.cat([V2E_edge_type, rel_edge_type], dim=0)

            #* add query relation to V2E_edge_index
            if self.p.with_query:
                #batch_query_rel = batch_query_rel.unsqueeze(1).expand(-1, self.p.neg_num+1).reshape(-1) #batch * [neg_num + 1]
                query_edge = torch.stack([batch_query_rel, query_hypere], dim=0) 
                V2E_edge_index = torch.cat([V2E_edge_index, query_edge], dim=1)
                # calculate query edge type based on the query pos(assume that all targets are ents now)
                query_edge_type = (batch_mask_position//2 + 3).unsqueeze(1).expand(-1, self.p.neg_num+1).reshape(-1)
                V2E_edge_type = torch.cat([V2E_edge_type, query_edge_type], dim=0)

            #* delete duplicate hedge, pad node set to the same length for each hyperedge
            V2E_edge_index, V2E_edge_type = coalesce(V2E_edge_index, edge_attr=V2E_edge_type, sort_by_row=False, reduce = 'max')
            V2E_edge_index[1] = V2E_edge_index[1].unique(sorted=True, return_inverse=True)[1] # turn to continuous index
            xid_dense, mask = to_dense_batch(V2E_edge_index[0], V2E_edge_index[1], fill_value = 0) # [n_hyper_edge, maxn_in_edge]\

            if self.p.position_mode is not None:
                x_pos_dense, pos_mask = to_dense_batch(V2E_edge_type, V2E_edge_index[1], fill_value = 7) # [n_hyper_edge, maxn_in_edge]
                x_pos_dense, _ = self.add_cls_id(x_pos_dense, pos_mask, 7)
                return xid_dense, mask, x_pos_dense, pos_mask
            else:
                return xid_dense, mask, None, None
    
    #@profile
    def forward(self, batch_data, unity_embeddings):
        with torch.no_grad():
            h_out_list = []
            G = batch_data['batch_graph']
            subg_ent_id = G['node'].x #subg_ent_id may have dupliacated node
            ori_subg_hedge_id = G['Hedge'].x
            V2E_edge_index = G['node', 'to', 'Hedge'].edge_index
            V2E_edge_type = G['node', 'to', 'Hedge'].edge_type
            V2E_edge_attr = G['node', 'to', 'Hedge'].edge_attr
            query_hypere = batch_data['batch_predict_hedge']
            batch_mask_position = batch_data['batch_mask_position']
            batch_query_rel = batch_data['batch_query_rel']

            #* turn to original index
            V2E_edge_index[1] = ori_subg_hedge_id[V2E_edge_index[1]]
            V2E_edge_index[0] = subg_ent_id[V2E_edge_index[0]]


           
            #* Modify to biary graph
            #find [(h,t), [e1,e1]] and format new edge_index, edge_attr, edge_type
            #turn [(h,t,v1,v2), [e1,e1,e1,e1]] to [[h,t],[e1,e1]]\ [[h,v1],[e2,e2]]\ [[h,v2],[e3,e3]] and add corresponding edge_attr\edge_type
            main_edge_mark = torch.isin(V2E_edge_type, torch.tensor([0,1]))
            V2E_edge_index = V2E_edge_index[:, main_edge_mark]
            V2E_edge_attr = V2E_edge_attr[main_edge_mark]
            V2E_edge_type = V2E_edge_type[main_edge_mark]
            #add qulifier hyper edge
            head_edge_mark = torch.where(torch.eq(V2E_edge_type, 0))[0]
            qual_edge_mark = torch.where(torch.eq(V2E_edge_type, 1))[0]  




            # *****************************Deal with E2V conv********************************
            # delete duplicated subg_hedge_id, subg_ent_id and get V2E_edge_reindex for E2V aggragation
            subg_ent_id, V2E_edge_row_reindex = V2E_edge_index[0].unique(sorted=True, return_inverse=True)
            uni_subg_hedge, V2E_edge_col_reindex = V2E_edge_index[1].unique(sorted=True, return_inverse=True)
            E2V_edge_index = torch.stack([V2E_edge_col_reindex, V2E_edge_row_reindex],dim=0)
            E2V_edge_index, E2V_edge_attr = coalesce(E2V_edge_index, V2E_edge_attr, sort_by_row=False, reduce = 'min')
            predict_edge_index = torch.where(torch.eq(uni_subg_hedge.unsqueeze(1), batch_data['batch_predict_hedge'].unsqueeze(0)))[0]

            # *****************************Deal with V2E conv********************************
            xid_dense, mask, x_pos_dense, pos_mask = self.format_V2E_matrix(V2E_edge_index, V2E_edge_attr, V2E_edge_type, 
                                                                            query_hypere, batch_query_rel, batch_mask_position)

        
        for k in range(self.p.K_hop + 1):
            #--------------------------1: V2E Conv, update hedge embeddings--------------------------#
            # emb xid_dense
            dense_matrix_emb = unity_embeddings[xid_dense]
            dense_matrix_emb, mask = self.add_cls_emb(dense_matrix_emb, mask, 1)
            if self.p.position_mode is not None:
                pos_embeddings = self.position_embeddings(x_pos_dense.long()) #[n_hyper_edge, maxn_in_edge, embed_dim]
                dense_matrix_emb = dense_matrix_emb + pos_embeddings
            #use transformer encoder
            if self.p.use_flash:
                trans_result = self.encoder(dense_matrix_emb, mask)
                trans_result = trans_result.transpose(0, 1)
            else:
                stk_inp = dense_matrix_emb.transpose(0, 1)
                trans_result, v2e_weights = self.encoder(stk_inp, src_key_padding_mask=~mask)
            subg_hedge_emb = trans_result[0]
            h_out_list.append(subg_hedge_emb[predict_edge_index])

            #--------------------------2: E2V Conv, update vertex embeddings--------------------------#
            if k < self.p.K_hop:
                #Transformer Attention
                #pad_id: subg_hedge_emb.size(0)
                #cls_id: subg_hedge_emb.size(0)+1
                if k==0:
                    hid_dense, h_mask = self.add_cls_id(hid_dense, h_mask, subg_hedge_emb.size(0)+1)
                subg_hedge_emb = torch.cat([subg_hedge_emb, self.pad_edge_emb, self.cls_edge_emb], dim=0) #concat cls edge emb
                dense_matrix_emb = subg_hedge_emb[hid_dense]
                if self.p.use_flash:
                    trans_result = self.encoder(dense_matrix_emb, h_mask)
                    trans_result = trans_result.transpose(0, 1)
                else:
                    stk_inp = dense_matrix_emb.transpose(0, 1)
                    trans_result, e2v_weights = self.encoder(stk_inp, src_key_padding_mask=~h_mask)
                    #print(self.encoder.layers[-1].self_attn.out_proj.weight)
                subg_ent_emb = trans_result[0] #update ent_emb in subgraph
                unity_embeddings[subg_ent_id] = subg_ent_emb #update ent_emb of subgraph in unity_embeddings

        h_out = h_out_list[-1]
        return h_out