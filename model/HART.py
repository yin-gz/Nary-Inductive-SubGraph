import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch_geometric.utils import to_dense_adj, to_dense_batch, unbatch, sort_edge_index, softmax, coalesce
from argparse import Namespace
from torch_geometric.nn import Set2Set, SAGPooling, GraphMultisetTransformer
from torch_geometric.nn.dense.linear import Linear
import math
from torch_scatter import scatter_add,scatter, scatter_mean
from utils.utils_others import get_param, topk_per_group
from torch_geometric.data import HeteroData
from model.flashatt.FlashAtt import FalshAttEncoder
from utils.utils_others import glorot

class HART(nn.Module):
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
                          query_hypere = None, batch_query_rel = None, batch_mask_etype = None, position_mode = None):
            #* add relation(edge_attr) to edge_index
            if with_rel:
                rel_edge_index = torch.stack([edge_attr, edge_index[1]], dim=0) #type: ignore , the same length as original_edge
                edge_index = torch.cat([edge_index, rel_edge_index], dim=1)
                #generate rel_edge's type according to thier edge_type3
                if self.p.position_mode != 'same':
                    rel_edge_type = edge_type + self.p.max_arity
                else:
                    rel_edge_type = edge_type
                edge_type = torch.cat([edge_type, rel_edge_type], dim=0)

            #* add query relation to edge_index
            if with_query:
                query_edge = torch.stack([batch_query_rel, query_hypere], dim=0) 
                edge_index = torch.cat([edge_index, query_edge], dim=1)

                if self.p.mark_query == 'unique':
                    query_edge_type = (2*self.p.max_arity)*torch.ones_like(batch_mask_etype, device = self.p.device)
                elif self.p.mark_query == 'same':
                    if self.p.position_mode != 'same':
                        query_edge_type = batch_mask_etype + self.p.max_arity 
                    else:
                        query_edge_type = batch_mask_etype
                query_edge_type = query_edge_type.unsqueeze(1).expand(-1, self.p.neg_num+1).reshape(-1)
                edge_type = torch.cat([edge_type, query_edge_type], dim=0)

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
    def forward(self, batch_data, unity_embeddings, mode = 'train'):
        '''
        output updated embeddings of unity_embeddings
        '''
        with torch.no_grad():
            h_out_list = [] #store embedding for predicting hyperedge
            
            G = batch_data['batch_graph']
            V2E_edge_index = G['node', 'to', 'Hedge'].edge_index
            
            if self.p.task != "full-trans":
                ori_subg_ent_id = G['node'].x #subg_ent_id may have dupliacated node
                ori_subg_hedge_id = G['Hedge'].x
                #* turn to original index in the whole graph
                V2E_edge_index[1] = ori_subg_hedge_id[V2E_edge_index[1]]
                V2E_edge_index[0] = ori_subg_ent_id[V2E_edge_index[0]]
            
            V2E_edge_type = G['node', 'to', 'Hedge'].edge_type
            V2E_edge_attr = G['node', 'to', 'Hedge'].edge_attr
            query_hypere = batch_data['batch_predict_hedge']
            batch_mask_etype = batch_data['batch_mask_etype']
            batch_query_rel = batch_data['batch_query_rel']

            # ***************************** Format E2V matrix [subg_ent, max_vinE] for Transformer ********************************
            # *** Reindex entity and hyperedge id in the subgraph to get continuous index (start from 0)
            # delete duplicated subg_hedge_id, subg_ent_id and get V2E_edge_reindex for E2V aggragation
            subg_ent_id, V2E_edge_row_reindex = V2E_edge_index[0].unique(sorted=True, return_inverse=True)
            subg_hedge_id, V2E_edge_col_reindex = V2E_edge_index[1].unique(sorted=True, return_inverse=True)
            E2V_edge_index = torch.stack([V2E_edge_col_reindex, V2E_edge_row_reindex],dim=0)
            predict_edge_index = torch.where(torch.eq(subg_hedge_id.unsqueeze(1), batch_data['batch_predict_hedge'].unsqueeze(0)))[0]
            hid_dense, h_mask, _, _ = self.format_matrix(E2V_edge_index, None, None, 
                                                         False, False, E2V_edge_index[0].max()+1, self.p.max_VinE,
                                                         query_hypere, batch_query_rel, batch_mask_etype, None)
            
            # ***************************** Format V2E matrix [subg_hedge, max_seq] for Transformer (add semantic relations)********************************
            with_query = (self.p.mark_query is not None)
            xid_dense, mask, x_pos_dense, pos_mask = self.format_matrix(V2E_edge_index, V2E_edge_attr, V2E_edge_type, 
                                                                        self.p.V2E_with_rel, with_query, 0, None,
                                                                        query_hypere, batch_query_rel, batch_mask_etype, self.p.position_mode)


        subg_ent_emb = unity_embeddings[subg_ent_id]
        v2e_attmatrix_list = []
        e2v_attmatrix_list = []
        for k in range(self.p.K_hop + 1):
            #--------------------------1: V2E Conv, update hedge embeddings--------------------------#
            # emb xid_dense
            dense_matrix_emb = unity_embeddings[xid_dense]
            dense_matrix_emb, mask = self.add_cls_emb(dense_matrix_emb, mask, 1)
            if self.p.position_mode is not None:
                pos_embeddings = self.position_embeddings(x_pos_dense.long()) #[n_hyper_edge, maxn_in_edge, embed_dim]
                dense_matrix_emb = dense_matrix_emb + pos_embeddings
            # use transformer encoder
            if self.p.use_flash: 
                trans_result = self.encoder(dense_matrix_emb, mask)
                trans_result = trans_result.transpose(0, 1)
            else:
                stk_inp = dense_matrix_emb.transpose(0, 1)
                trans_result, v2e_weights = self.encoder(stk_inp, src_key_padding_mask=~mask)
                v2e_weights = torch.mean(torch.stack(v2e_weights), dim=0) #average the weights of layers
                #! trans_result: [max_seq, n_subg_hedge, embed]
                #! v2e_weights: [n_subg_hedge, max_seq, max_seq], select the cls one's attention scores
                v2e_matrix = self.turn_matrix(v2e_weights[:,0,1:], xid_dense, mask[:,1:], self.p.num_ent + self.p.num_rel + 1)  #include [PAD]
                #v2e_matrix = v2e_matrix[:, :self.p.num_ent+1]
                v2e_attmatrix_list.append(v2e_matrix) #only save the weights of ent
            subg_hedge_emb = trans_result[0]
            h_out_list.append(subg_hedge_emb[predict_edge_index])

            #--------------------------2: E2V Conv, update vertex embeddings--------------------------#
            if k < self.p.K_hop:
                # Transformer Attention
                # pad_id: subg_hedge_emb.size(0)
                # cls_id: subg_hedge_emb.size(0)+1
                if k==0:
                    hid_dense, h_mask = self.add_cls_id(hid_dense, h_mask, subg_hedge_emb.size(0)+1)
                subg_hedge_emb = torch.cat([subg_hedge_emb, self.pad_edge_emb, self.cls_edge_emb], dim=0) #concat cls edge emb
                dense_matrix_emb = subg_hedge_emb[hid_dense]
                if self.p.use_flash:
                    trans_result = self.encoder(dense_matrix_emb, h_mask)
                    trans_result = trans_result.transpose(0, 1)
                else:
                    stk_inp = dense_matrix_emb.transpose(0, 1)
                    #! e2v_weights: [subg_ent, max_vinE]
                    trans_result, e2v_weights = self.encoder(stk_inp, src_key_padding_mask=~h_mask)
                    e2v_weights = torch.mean(torch.stack(e2v_weights), dim=0) #average the weights of layers
                    e2v_matrix = self.turn_matrix(e2v_weights[:,0,1:], hid_dense[:,1:], h_mask[:,1:], subg_hedge_id.size(0)) #include [PAD]
                    #[n_ent+1,subg_hedge]
                    e2v_attmatrix = torch.zeros((self.p.num_ent+self.p.num_rel+1, subg_hedge_id.size(0)), dtype=torch.float, device=e2v_matrix.device)
                    e2v_attmatrix[subg_ent_id] = e2v_matrix
                    e2v_attmatrix_list.append(e2v_attmatrix) # not save the weights of cls and pad
                subg_ent_emb = trans_result[0] #update ent_emb in subgraph
                unity_embeddings[subg_ent_id] = subg_ent_emb #update ent_emb of subgraph in unity_embeddings

        h_out = h_out_list[-1]
        if self.p.explain_fact:
            #return: x_id_vector[predict_edge_index], e_id_vector[main_ent], v2E_att_out [query_length], E2V_att_mainent [n_edge_main], v2E_att_inner[all_edge, length], 
            predictE_node = xid_dense[predict_edge_index]
            node_weights = self.cal_node_weights(v2e_attmatrix_list, e2v_attmatrix_list, predict_edge_index)
            node_weights[node_weights == 0] = float('-inf')
            node_weights = F.softmax(node_weights, dim=0)
            # main_ent_neighborE = hid_dense[0][1:][1:4]
            outE_ent_att = v2e_weights[predict_edge_index.item()][0][1:] #output the last layer's self-attention weights
            #neighborE_innner_atts = v2e_weights_list[0][-1][main_ent_neighborE][:, 0, 1:]
            return h_out, node_weights, predictE_node, outE_ent_att
        else:
            return h_out
        
    def turn_matrix(self, ori_weights, id_dense, mask, max_col):
        # id_dense: each line denote the original index [m, n], max_idx as max_col
        # ori_weights: corresponding weights to id_dense, [m, n], the first is cls
        # mask: mask for id_dense
        # return : [m, max_col], each item denote the weights of original index
        
        # turn mask pos of ori_weights to 0
        #mask_expanded = mask.unsqueeze(-1) * (id_dense < max_col).unsqueeze(1)  # Ensure indices are within bounds
        mask_expanded = mask * (id_dense < max_col)
        result = torch.zeros((id_dense.size(0), max_col), dtype=torch.float, device=ori_weights.device)
        # simply understanding: add ori_weights[i] to result [id_dense[i]]
        result.scatter_add_(1, id_dense * mask_expanded, ori_weights)
        return result
        
        
    def cal_node_weights(self, v2e_attmatrix_list, e2v_attmatrix_list, predict_edge_index):
        '''
        Return the weights of each node in the subgraph for the predict_edge
        '''
        # v2e_weights_list: (K+1) * [subg_hedge, all_ent]
        # e2v_weights_list: K * [all_ent, subg_hedge]
        
        # turn each v2e_weights to [subg_hedge, subg_ent]
        last_v2e = v2e_attmatrix_list[-1][predict_edge_index] #[1, all_ent]
        tmp = last_v2e
        for i in range(self.p.K_hop-1, -1, -1):
            tmp = torch.mm(tmp, e2v_attmatrix_list[i]) #[1, all_ent] * [all_ent, subg_hedge] = [1, subg_hedge]
            tmp = torch.mm(tmp, v2e_attmatrix_list[i]) #[1, subg_hedge] * [subg_hedge, all_ent] = [1, all_ent]
        return tmp.squeeze(0)
            