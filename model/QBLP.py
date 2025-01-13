import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from argparse import Namespace


class QBLP(nn.Module):
    def __init__(self, param: Namespace = Namespace()):
        super(QBLP, self).__init__()
        self.p = param

        #Transformer
        self.position_embeddings = nn.Embedding(8, self.p.embed_dim)
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
        batch_embedding = unity_embeddings[batch_input_ids]

        #add pos
        if self.p.position_mode is not None:
            qual_ind = 6
            positions_main = torch.arange(qual_ind - 2, dtype=torch.long, device = self.p.device)  # [0,1,2,3]
            positions_qual = torch.arange(qual_ind - 2, qual_ind, dtype=torch.long, device = self.p.device).repeat((self.p.max_seq_length - qual_ind + 2) // 2) # [4,5], 所有的qual embedding都是赋值的4、5，是同样的低维
            positions = torch.cat([positions_main, positions_qual]).repeat(batch_embedding.size(0), 1)
            pos_embeddings = self.position_embeddings(positions)
            batch_embedding = batch_embedding + pos_embeddings

        
        batch_pad_mask = (batch_input_ids == 0)
        #add cls token on the begining of each sequence
        batch_embedding, batch_pad_mask= self.add_cls_emb(batch_embedding, batch_pad_mask, 0)
        batch_pad_inv = ~batch_pad_mask
        stk_inp = batch_embedding.transpose(1, 0) #[padding_length, batch, dim]
        x_seq = self.encoder(stk_inp, src_key_padding_mask=batch_pad_mask).float() #[len, batch, dim]

        # Average query embedding
        x_query_seq = x_seq.transpose(0,1) #[batch, query_length, dim]
        x_query_seq = x_query_seq * (batch_pad_inv.unsqueeze(2).expand(-1,-1,x_query_seq.size(-1)))
        #count true num in batch_pad_inv
        batch_query_length = torch.sum(batch_pad_inv, dim=1) #[batch]
        batch_query_length = batch_query_length.unsqueeze(1).expand(-1,x_query_seq.size(-1)) #[batch, dim]
        x_query = torch.sum(x_query_seq, 1) / batch_query_length  #[batch, dim]/ [batch, dim]
        h_out_emb = x_query
        
        return h_out_emb