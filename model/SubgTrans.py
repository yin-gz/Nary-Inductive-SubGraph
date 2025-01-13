import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from model.flashatt.FlashAtt import FalshAttEncoder
from argparse import Namespace
from torch_geometric.utils import to_dense_batch, coalesce

'''
Concat query sequence embedding [bsz, seq_len] with neiboring nodes (with hop biases) [bsz, neighbor_len]
'''

class SubgTrans(nn.Module):
    def __init__(self, param: Namespace = Namespace()):
        super(SubgTrans, self).__init__()
        self.p = param

        #Transformer
        self.position_embeddings = nn.Embedding(128, self.p.embed_dim)
        if self.p.use_flash:
            self.encoder  = FalshAttEncoder(param)
        else:
            encoder_layers = TransformerEncoderLayer(self.p.embed_dim, self.p.num_heads, self.p.hidden_dim, self.p.drop_decoder)
            self.encoder = TransformerEncoder(encoder_layers, self.p.trans_layer)

    def add_cls_emb(self, x_emb_dense, mask, mask_id):
        x_emb_dense = torch.cat([torch.zeros(x_emb_dense.size(0), 1,
                                 self.p.embed_dim).to(self.p.device), x_emb_dense], dim=1)
        if mask_id ==0:
            mask = torch.cat([torch.zeros(mask.size(0), 1).bool().to(self.p.device), mask], dim=1)
        else:
            mask = torch.cat([torch.ones(mask.size(0), 1).bool().to(self.p.device), mask], dim=1)
        return x_emb_dense, mask
    
    def add_cls_id(self, x_id_dense, mask, cls_id):
        x_id_dense = torch.cat([cls_id * torch.ones(x_id_dense.size(0), 1).long().to(self.p.device), x_id_dense], dim=1)
        mask = torch.cat([torch.ones(mask.size(0), 1).bool().to(self.p.device), mask], dim=1)
        return x_id_dense, mask


    def forward(self, batch_data, unity_embeddings, mode = 'train'):
        batch_input_ids = batch_data['batch_input_seqs']
        batch_query_embedding = unity_embeddings[batch_input_ids]

        # add positional embedding to query
        if self.p.position_mode is not None:
            qual_ind = 6
            positions_main = torch.arange(qual_ind - 2, dtype=torch.long, device = self.p.device)  # [0,1,2,3]
            positions_qual = torch.arange(qual_ind - 2, qual_ind, dtype=torch.long, device = self.p.device).repeat((self.p.max_seq_length - qual_ind + 2) // 2) # [4,5,4,5...]
            positions = torch.cat([positions_main, positions_qual]).repeat(batch_query_embedding.size(0), 1)
            pos_embeddings = self.position_embeddings(positions)
            batch_query_embedding = batch_query_embedding + pos_embeddings
        
        # generate neighbornodes ids embeddings
        G = batch_data['batch_graph']
        edge_index = G['node', 'to', 'Hedge'].edge_index
        subg_node_batch = G['node'].batch
        edge_source = edge_index[0]
        subg_node_id = G['node'].x
        #hop = G['node'].dis_source
        
        edge_index[1] = subg_node_batch[edge_source] # batch id
        #V2E_edge_hop = hop[edge_source] # node distance to hop
        edge_index[0] = subg_node_id[edge_source] # node id
        
        #delete duplicate edge
        #edge_index, V2E_edge_hop = coalesce(edge_index, edge_attr=V2E_edge_hop, sort_by_row=False, reduce = 'min')
        edge_index = coalesce(edge_index, sort_by_row=False)
        neighbor_id_dense, neighbor_id_mask = to_dense_batch(edge_index[0], edge_index[1], fill_value = 0, max_num_nodes = 2*self.p.max_VinE*self.p.max_arity)
        #dis_id_dense, dis_id_mask = to_dense_batch(V2E_edge_hop, edge_index[1], fill_value = 0, max_num_nodes = 2*self.p.max_VinE*self.p.max_arity)
        batch_neighbor_embedding = unity_embeddings[neighbor_id_dense] #[bsz, max_neighbor_len, dim]
        #batch_dis_embedding = self.position_embeddings(dis_id_dense) #[bsz, max_neighbor_len, dim]
        batch_embedding = torch.cat([batch_query_embedding, batch_neighbor_embedding], dim=1) #[bsz, max_query_len+max_neighbor_len, dim]
        

        # generate mask
        batch_pad_mask = (batch_input_ids == 0)
        batch_pad_mask = torch.cat([batch_pad_mask, ~neighbor_id_mask], dim=1)
        
        # add cls token on the begining of each sequence
        batch_embedding, batch_pad_mask= self.add_cls_emb(batch_embedding, batch_pad_mask, 0)
        batch_pad_inv = ~batch_pad_mask
        
        # put into Transformer
        if self.p.use_flash:
            x_seq = self.encoder(batch_embedding, batch_pad_inv)
            x_seq = x_seq.transpose(0, 1)
        else:
            stk_inp = batch_embedding.transpose(1, 0) #[padding_length, batch, dim]
            x_seq = self.encoder(stk_inp, src_key_padding_mask=batch_pad_mask).float() #[len, batch, dim]
        

        # Average query embedding as output
        x_query_seq = x_seq.transpose(0,1) #[batch, query_length, dim]
        x_query_seq = x_query_seq * (batch_pad_inv.unsqueeze(2).expand(-1,-1,x_query_seq.size(-1)))
        #count true num in batch_pad_inv
        batch_query_length = torch.sum(batch_pad_inv, dim=1) #[batch]
        batch_query_length = batch_query_length.unsqueeze(1).expand(-1,x_query_seq.size(-1)) #[batch, dim]
        x_query = torch.sum(x_query_seq, 1) / batch_query_length  #[batch, dim]/ [batch, dim]
        h_out_emb = x_query

        # get CLS emb as output
        # h_out_emb = x_seq[0]
        
        return h_out_emb