import torch
import torch.nn as nn
from typing import Dict
from argparse import Namespace
from model.HyperAgg import HyperAggLayer
from torch_geometric.nn.dense.linear import Linear
from utils.utils_others import get_param
from torch_geometric.utils import to_dense_adj, to_dense_batch, unbatch, sort_edge_index, softmax, coalesce
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from model.flashatt.FlashAtt import FalshAttEncoder

'''
Only used for HART or HyperAgg
'''

class PairLoss(torch.nn.Module):
    def __init__(self,  param: Namespace = Namespace()):
        super(PairLoss, self).__init__()
        self.bceloss = torch.nn.BCEWithLogitsLoss()
        self.p = param

    def forward(self, pos_score, neg_score):
        pos_score = pos_score.unsqueeze(1)
        neg_score = neg_score.unsqueeze(1)
        scores = torch.cat([pos_score, neg_score], dim=1) #[batch, 2]
        target_label = torch.cat([torch.ones_like(pos_score), 0*torch.ones_like(neg_score)], dim=1)
        loss = self.bceloss(scores, target_label)

        return loss


class NPairBase(nn.Module):
    '''
    Transformer-based models for Nary data
    Include: HART/QBLP/SubgTrans
    '''
    def __init__(self, param: Namespace = Namespace()):
        super(NPairBase, self).__init__()
        self.p = param
        self.dis_embedding = get_param((2*self.p.K_hop+3, self.p.embed_dim), norm=False)
        self.rel_embedding = get_param((self.p.num_rel, self.p.embed_dim), norm=False)
        self.out_model = globals()[self.p.model_name](self.p)
        self.myloss = PairLoss(self.p)

    def forward(self, batch_data):
        #use other targeted entities in the batch as negative samples
        hedge_out, pos_ent_emb, neg_ent_emb = self.out_model(batch_data, self.dis_embedding, self.rel_embedding) #[batch, dim]
        
        pos_score = torch.sum(pos_ent_emb * hedge_out, dim=-1) #[batch,dim] * [batch,dim] -> [batch]
        neg_score = torch.sum(neg_ent_emb * hedge_out, dim=-1)
        return pos_score, neg_score

    def loss(self, pos_score, neg_score):
        return self.myloss(pos_score, neg_score)
    
    
class HART(nn.Module):
    '''
    HART for SLR tasks (the input is a batch of paired subgraphs)
    '''
    def __init__(self, param: Namespace = Namespace()):
        super(HART, self).__init__()
        self.p = param
        self.cls_edge_emb = get_param((1, self.p.embed_dim))
        self.pad_edge_emb = get_param((1, self.p.embed_dim))

        #Transformer
        #max: 2*self.p.max_arity - 1, unique: 2*self.p.max_arity, pad: 2*self.p.max_arity+1, cls: 2*self.p.max_arity+2
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
    
    def format_matrix(self, edge_index, edge_attr, edge_type, with_rel = True, with_query = False, fill_value =0, pad_length = 16, \
                          query_hypere = None, batch_query_rel = None, position_mode = None, rel_id_base = 0):
        '''
        Format edge_index to dense matrix to put into Transformer, add relation and cls when V2E
        '''
        #* add relation(edge_attr) to edge_index
        if with_rel:
            rel_edge_index = torch.stack([edge_attr + rel_id_base, edge_index[1]], dim=0)
            edge_index = torch.cat([edge_index, rel_edge_index], dim=1)
            #generate rel_edge's type according to thier edge_type3
            rel_edge_type = edge_type + self.p.max_arity
            edge_type = torch.cat([edge_type, rel_edge_type], dim=0)

        #* add query relation to edge_index
        if with_query:
            query_edge = torch.stack([batch_query_rel + rel_id_base, query_hypere], dim=0) 
            edge_index = torch.cat([edge_index, query_edge], dim=1)
            query_edge_type = (2*self.p.max_arity)*torch.ones_like(batch_query_rel, device = self.p.device)
            edge_type = torch.cat([edge_type, query_edge_type], dim=0)

        
        if rel_id_base == 2*self.p.K_hop+3:
            #sort edge by edge_index[1]
            edge_index, edge_type = sort_edge_index(edge_index, edge_attr=edge_type, sort_by_row=False)
        else:
            #* delete duplicate hedge
            edge_index, edge_type = coalesce(edge_index, edge_attr=edge_type, sort_by_row=False, reduce = 'max')
        edge_index[1] = edge_index[1].unique(sorted=True, return_inverse=True)[1] # turn to continuous index
        xid_dense, mask = to_dense_batch(edge_index[0], edge_index[1], fill_value = fill_value, max_num_nodes = pad_length)

        if position_mode is not None:
            pos_fill_value = 2*self.p.max_arity+1
            x_pos_dense, pos_mask = to_dense_batch(edge_type, edge_index[1], fill_value = pos_fill_value, max_num_nodes = pad_length) # [n_hyper_edge, maxn_in_edge]
            x_pos_dense, _ = self.add_cls_id(x_pos_dense, pos_mask, pos_fill_value + 1)
            return xid_dense, mask, x_pos_dense, pos_mask
        else:
            return xid_dense, mask, None, None
    
    #@profile
    def forward(self, batch_data, dis_embedding, rel_embedding):
        '''
        output updated embeddings of unity_embeddings
        '''
        with torch.no_grad():
            h_out_list = [] #store embedding for predicting hyperedges
            G = batch_data
            V2E_edge_index = G['node', 'to', 'Hedge'].edge_index
            V2E_edge_type = G['node', 'to', 'Hedge'].edge_type
            V2E_edge_attr = G['node', 'to', 'Hedge'].edge_attr
            v_dis = G['node'].dis
            source_hedge_mark = G['Hedge'].mark_source
            pos_target_mark = G['node'].mark_target
            neg_target_mark = G['node'].mark_neg
            batch_query_rel = G.query_rel
            subg_ent_num = G['node'].x.unique().size(0)

            # construct edge index using distance mapping
            dis_edge_index = V2E_edge_index.clone()
            dis_edge_index[0] = v_dis[dis_edge_index[0]]
            
            #find id of query_hypere
            #1. first, mark edge_index[1] is source or not
            #2. then, find the index of predict hyperedge in edge_index
            query_hypere = V2E_edge_index[1][source_hedge_mark.bool()[V2E_edge_index[1]]].unique()
            

            # ***************************** Format E2V matrix [subg_ent, max_vinE] for Transformer ********************************
            E2V_edge_index = torch.stack([V2E_edge_index[1], V2E_edge_index[0]], dim=0)
            hid_dense, h_mask, _, _ = self.format_matrix(E2V_edge_index, None, None, 
                                                         False, False, E2V_edge_index[0].max()+1, self.p.max_VinE,
                                                         query_hypere, batch_query_rel, None, 0)
            
            # ***************************** Format V2E matrix ( vertex as distance id ) for Transformer ********************************
            with_query = (self.p.mark_query is not None)
            # xdis_id_dense: [subg_hedge, max_seq]
            xdis_id_dense, xdis_mask, xdis_pos_dense, _ = self.format_matrix(dis_edge_index, V2E_edge_attr, V2E_edge_type, 
                                                                        self.p.V2E_with_rel, with_query, 0, None,
                                                                        query_hypere, batch_query_rel, self.p.position_mode, 
                                                                        rel_id_base = 2*self.p.K_hop+3)
            
            
            # ***************************** Format V2E matrix ( vertex as idx in subg ) for Transformer (add semantic relations) ********************************
            # xid_dense: [subg_hedge, max_seq]
            xid_dense, xid_mask, x_pos_dense, _ = self.format_matrix(V2E_edge_index, V2E_edge_attr, V2E_edge_type, 
                                                                        self.p.V2E_with_rel, with_query, 0, None,
                                                                        query_hypere, batch_query_rel, self.p.position_mode, 
                                                                        rel_id_base = subg_ent_num)


        v2e_weights_list = []        
        for k in range(self.p.K_hop + 1):
            #* --------------------------1: V2E Conv, update hedge embeddings--------------------------#
            if k==0:
                # emb xid_dense
                all_embeddings = torch.cat([dis_embedding, rel_embedding], dim=0)
                dense_matrix_emb = all_embeddings[xdis_id_dense]
                pos_embeddings = self.position_embeddings(xdis_pos_dense.long()) #[n_hyper_edge, maxn_in_edge, embed_dim]
                dense_matrix_emb, mask = self.add_cls_emb(dense_matrix_emb, xdis_mask, 1)
            else:
                all_embeddings = torch.cat([subg_ent_emb, rel_embedding], dim=0)
                dense_matrix_emb = all_embeddings[xid_dense]
                pos_embeddings = self.position_embeddings(x_pos_dense.long()) #[n_hyper_edge, maxn_in_edge, embed_dim]
                dense_matrix_emb, mask = self.add_cls_emb(dense_matrix_emb, xid_mask, 1)
            if self.p.position_mode is not None:
                dense_matrix_emb = dense_matrix_emb + pos_embeddings
            if self.p.use_flash:
                trans_result = self.encoder(dense_matrix_emb, mask)
                trans_result = trans_result.transpose(0, 1)
            else:
                stk_inp = dense_matrix_emb.transpose(0, 1)
                trans_result, v2e_weights = self.encoder(stk_inp, src_key_padding_mask=~mask)
                v2e_weights_list.append(v2e_weights)
            subg_hedge_emb = trans_result[0]
            h_out_list.append(subg_hedge_emb[source_hedge_mark.bool()])

            #* --------------------------2: E2V Conv, update vertex embeddings--------------------------#
            if k < self.p.K_hop:
                # Transformer Attention
                # pad_id: subg_hedge_emb.size(0)
                # cls_id: subg_hedge_emb.size(0)+1
                if k == 0:
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

        # h_out = torch.mean(torch.stack(h_out_list, dim=0), dim=0)
        h_out = h_out_list[-1]
        pos_subg_id = V2E_edge_index[0][pos_target_mark.bool()[V2E_edge_index[0]]].unique()
        pos_ent_emb = subg_ent_emb[pos_subg_id]
        neg_subg_id = V2E_edge_index[0][neg_target_mark.bool()[V2E_edge_index[0]]].unique()
        neg_ent_emb = subg_ent_emb[neg_subg_id]
        return h_out, pos_ent_emb, neg_ent_emb
        
        
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
    def forward(self, batch_data, dis_embedding, rel_embedding):
        with torch.no_grad():
            h_out_list = [] #store embedding for predicting hyperedge
            G = batch_data
            V2E_edge_index = G['node', 'to', 'Hedge'].edge_index
            #V2E_edge_type = G['node', 'to', 'Hedge'].edge_type
            V2E_edge_attr = G['node', 'to', 'Hedge'].edge_attr
            v_dis = G['node'].dis
            source_hedge_mark = G['Hedge'].mark_source
            pos_target_mark = G['node'].mark_target
            neg_target_mark = G['node'].mark_neg
            batch_query_rel = G.query_rel
            subg_ent_num = G['node'].x.unique().size(0)

            # turn to dis index in V2E_edge_index
            dis_edge_index = V2E_edge_index.clone()
            dis_edge_index[0] = v_dis[dis_edge_index[0]]
            
            #format E2V matrix [subg_ent, max_vinE] for Transformer
            E2V_edge_index = torch.stack([V2E_edge_index[1], V2E_edge_index[0]], dim=0)
                        
            #* add relation(edge_attr) to edge_index
            if self.p.V2E_with_rel:
                rel_edge_index = torch.stack([V2E_edge_attr + subg_ent_num, V2E_edge_index[1]], dim=0)
                drel_edge_index = torch.stack([V2E_edge_attr + 2*self.p.K_hop+3, dis_edge_index[1]], dim=0)
                V2E_edge_index = torch.cat([V2E_edge_index, rel_edge_index], dim=1)
                dis_edge_index = torch.cat([dis_edge_index, drel_edge_index], dim=1)
                #rel_edge_type = edge_type + self.p.max_arity
                #edge_type = torch.cat([edge_type, rel_edge_type], dim=0)

            #* add query relation to edge_index
            query_hypere = V2E_edge_index[1][source_hedge_mark.bool()[V2E_edge_index[1]]].unique()
            if self.p.mark_query is not None:
                query_edge = torch.stack([batch_query_rel + subg_ent_num, query_hypere], dim=0)
                dquery_edge = torch.stack([batch_query_rel + 2*self.p.K_hop+3, query_hypere], dim=0)
                V2E_edge_index = torch.cat([V2E_edge_index, query_edge], dim=1)
                dis_edge_index = torch.cat([dis_edge_index, dquery_edge], dim=1)
                #query_edge_type = (2*self.p.max_arity)*torch.ones_like(batch_query_rel, device = self.p.device)
                #edge_type = torch.cat([edge_type, query_edge_type], dim=0)
                
            V2E_edge_index = coalesce(V2E_edge_index, sort_by_row=False) #V2E_edge_index \ dis_edge_index contain rel idx
            E2V_edge_index = coalesce(E2V_edge_index, sort_by_row=False)


        for k in range(self.p.K_hop + 1):
            #--------------------------1: V2E Conv, update hedge embeddings--------------------------#
            if k==0:
                edge_index = dis_edge_index
                subg_ent_emb = dis_embedding
                subg_all_emb = torch.cat([dis_embedding, rel_embedding], dim=0)
                subg_hedge_emb = None
            else:
                edge_index = V2E_edge_index
                subg_all_emb = torch.cat([subg_ent_emb, rel_embedding], dim=0)
            if self.p.hagg_method == 'settrans':
                subg_hedge_emb = self.V2Econvs[k](subg_all_emb, subg_hedge_emb, edge_index)
            else:
                V2E_source_emb = subg_all_emb[edge_index[0]]
                V2E_target_emb = None if subg_hedge_emb is None else subg_hedge_emb[edge_index[1]]
                subg_hedge_emb = self.V2Econvs[k](V2E_source_emb, V2E_target_emb, edge_index)
            h_out_list.append(subg_hedge_emb[source_hedge_mark.bool()])

            #--------------------------2: E2V Conv, update vertex embeddings--------------------------#
            if k < self.p.K_hop:
                if self.p.hagg_method == 'settrans':
                    if k == 0:
                        subg_ent_emb = self.E2Vconvs[k](subg_hedge_emb, None, E2V_edge_index)
                    else:
                        subg_ent_emb = self.E2Vconvs[k](subg_hedge_emb, subg_ent_emb, E2V_edge_index)
                else:
                    E2V_source_emb = subg_hedge_emb[E2V_edge_index[0]]
                    if k == 0:
                        E2V_target_emb = None
                    else:
                        E2V_target_emb = subg_ent_emb[E2V_edge_index[1]]
                    subg_ent_emb = self.E2Vconvs[k](E2V_source_emb, E2V_target_emb, E2V_edge_index)
                subg_ent_emb = subg_ent_emb.to(rel_embedding.dtype)
                subg_all_emb = torch.cat([subg_ent_emb, rel_embedding], dim=0)
                
        h_out = h_out_list[-1]
        pos_subg_id = E2V_edge_index[1][pos_target_mark.bool()[E2V_edge_index[1]]].unique()
        pos_ent_emb = subg_ent_emb[pos_subg_id]
        neg_subg_id = E2V_edge_index[1][neg_target_mark.bool()[E2V_edge_index[1]]].unique()
        neg_ent_emb = subg_ent_emb[neg_subg_id]
        
        return h_out, pos_ent_emb, neg_ent_emb

