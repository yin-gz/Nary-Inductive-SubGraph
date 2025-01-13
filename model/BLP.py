import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Parameter
#from performer_pytorch import PerformerLM, Performer, SelfAttention
from torch_geometric.utils import to_dense_adj, to_dense_batch, unbatch, sort_edge_index, softmax
from argparse import Namespace
from torch_geometric.nn import Set2Set, SAGPooling, GraphMultisetTransformer
from torch_geometric.nn.dense.linear import Linear
from typing import Any
import math
from torch_scatter import scatter_add,scatter
from utils.utils_others import get_param
from torch_geometric.data import HeteroData
from model.transfer_model import TransLoss
from typing import Dict


class BLP(nn.Module):
    def __init__(self, ent_feature = None, param: Namespace = Namespace()):
        super(BLP, self).__init__()
        self.p = param

        if self.p.task == 'TR-EF':
            self.ent_pkl = ent_feature.to(self.p.device) # type: ignore
            self.other_emb = get_param((self.p.num_ent + self.p.num_rel + 3 - self.ent_pkl.size(0), self.p.embed_dim), norm=False)
            self.projection = nn.Linear(self.ent_pkl.size(-1), self.p.embed_dim, bias=False) # type: ignore
        #elif self.p.ent_feature == 'learn':
            #self.unities = get_param((self.p.num_rel + self.p.num_ent + 3, self.p.embed_dim), norm=False)

        self.normalize_embs = True
        #loss
        self.myloss = TransLoss(self.p)

    def load_unity_emd(self):
        '''
        load unity embeddings
        '''
        if self.p.task == 'TR-EF':
            #or '_' in self.p.dataset
            ent_emb_file = self.projection(self.ent_pkl)
            unity_embeddings = torch.cat([self.other_emb[0].unsqueeze(0), ent_emb_file, self.other_emb[1:]], 0)
        #elif self.p.ent_feature == 'learn':
            #unity_embeddings = self.unities

        return unity_embeddings




    def forward(self, predict_type, batch_data, target_ent_index, bigraph: Dict = {}, base_graph: Dict = {}, mode = 'train'):
        batch_input_ids = batch_data['batch_input_seqs']
        unity_embeddings = self.load_unity_emd()

        batch_mask_pos = batch_data['batch_mask_position'][0].item()

        query_input = torch.cat([batch_input_ids[:, :batch_mask_pos], batch_input_ids[:, batch_mask_pos+1:3]], dim = 1)
        query_emb = unity_embeddings[query_input] #[batch, query_length, dim]
        if self.normalize_embs:
            query_emb = F.normalize(query_emb, dim=-1)
        query_sum = torch.sum(query_emb, dim = 1) #[batch, dim]

        #* Get all target embeddings
        if predict_type == 'ent':
            E_embed = unity_embeddings[:self.p.num_ent + 1] #include pad
        else:
            E_embed = unity_embeddings[0] + unity_embeddings[self.p.num_ent + 1: -1]
        #* Get target emb
        # For BCE LOSS, target_emb is [num_ent, dim]
        # For Negative Sampling, target_emb is [batch, num_neg, dim]
        if self.p.use_neg and 'neg_target_index' in batch_data:
            batch_target = batch_data['neg_target_index']
            target_emb = E_embed[batch_target]
        elif target_ent_index is not None:
            if type(target_ent_index) != torch.Tensor:
                target_ent_index = torch.LongTensor(target_ent_index)
            target_emb = E_embed[target_ent_index]
        else:
            target_emb = E_embed


        if len(target_emb.size()) == 3:
            score = -torch.norm(query_sum.unsqueeze(1) - target_emb, dim=-1, p=1) # [batch, 1, dim] - [batch, num_ent_emb, dim] → [batch, num_ent]
        else:
            score = -torch.norm(query_sum.unsqueeze(1) - target_emb.unsqueeze(0), dim = -1, p = 1) # [batch, 1, dim] - [1, num_ent_emb, dim] → [batch, num_ent]
        return score
    
    def loss(self, pred, batch_data):
        batch_mask_label = batch_data['batch_mask_label']
        batch_label_all = batch_data['batch_target_all']
        return self.myloss(pred, batch_mask_label, batch_label_all)



        