import torch
from typing import Dict
from model.HART import *
from model.QBLP import *
from model.SubgTrans import *
from torch_scatter import scatter_add,scatter
from argparse import Namespace
from torch_geometric.data import HeteroData
from model.HyperAgg import HyperAggModel
from model.GRAN import GRAN

class TransLoss(torch.nn.Module):
    def __init__(self,  param: Namespace = Namespace()):
        super(TransLoss, self).__init__()
        self.bceloss = torch.nn.BCEWithLogitsLoss()
        self.celoss = torch.nn.CrossEntropyLoss(size_average = True)
        self.margin_loss = torch.nn.MarginRankingLoss()
        self.p = param

    def forward(self, pred, batch_mask_label, batch_label_all):
        if self.p.loss_name == 'CE':
            #pred [batch, n_label]
            #mask_label [batch, n_label]
            loss = self.celoss(pred, batch_mask_label)
        elif self.p.loss_name == 'BCE':
            #loss = self.bceloss(torch.sigmoid(pred), batch_label_all)  # [B]
            loss = self.bceloss(pred, batch_label_all)
        elif self.p.loss_name =='margin':
            #input1: pos score, [batch] → [batch, neg]
            #input2: neg score, [batch, neg]
            #target: [1,1,1...], [batch, neg]
            pos_score = pred[:,-1].unsqueeze(1).expand(-1, pred.size(1) -1) #[batch,neg]
            neg_score  = pred[:,:-1]
            target_label = torch.ones_like(pos_score)
            loss = self.margin_loss(pos_score, neg_score, target_label)
        else:
            raise ValueError("Invalid loss name")
        return loss


class NTransBase(nn.Module):
    '''
    Nary Semantic Models for TR-EF and TR-NEF
    '''
    def __init__(self, ent_feature = None, param: Namespace = Namespace()):
        super(NTransBase, self).__init__()
        self.p = param
        if self.p.unit_encode:
            if self.p.task == 'TR-EF':
                self.ent_pkl = ent_feature.to(self.p.device)
                self.other_emb = get_param((self.p.num_ent + self.p.num_rel + 3 - self.ent_pkl.size(0), self.p.embed_dim), norm=False)
                self.projection = nn.Linear(self.ent_pkl.size(-1), self.p.embed_dim, bias=False)
            elif "FI_" not in self.p.dataset:
                self.embs = get_param((self.p.num_ent + self.p.num_rel + 3, self.p.embed_dim), norm=False) #['PAD'], all ent, all rel, ['mask', 'cls']
            else:
                self.other_emb = get_param((self.p.num_rel + 3, self.p.embed_dim), norm=False) #['PAD'], all rel, ['mask', 'cls']
        else:
            self.entities = get_param((self.p.num_ent + 2, self.p.embed_dim), norm=False)
            self.relations = get_param((self.p.num_rel + 3, self.p.embed_dim), norm=False)
        self.LayerNorm = nn.LayerNorm(self.p.embed_dim)
        self.feature_drop = torch.nn.Dropout(self.p.feature_drop)

        #model
        if self.p.model_name == 'HART-Intra':
            self.out_model = HART(self.p)
        else:
            self.out_model = globals()[self.p.model_name](self.p)
        

        #loss
        self.myloss = TransLoss(self.p)

    def load_unity_emd(self, bigraph):
        '''
        load unity embeddings
        '''
        if self.p.task == 'TR-EF':
            ent_emb_file = self.projection(self.ent_pkl)
            unity_embeddings = torch.cat([self.other_emb[0].unsqueeze(0), ent_emb_file, self.other_emb[1:]], 0)
        elif "FI_" not in self.p.dataset:
            # load learned embeddings
            unity_embeddings = self.embs
        else:
            bi_edge, bi_edge_type = bigraph['edge_index'], bigraph['edge_type']
            type_to_ent = torch.stack([bi_edge_type, bi_edge[0]], dim=0)
            type_to_ent = coalesce(type_to_ent, sort_by_row = False)
            type_emb = self.other_emb[type_to_ent[0]]
            ent_emb = scatter(type_emb, type_to_ent[1], dim=0, reduce='mean', dim_size=self.p.num_ent + 1) #average rel emb as ent emd, [pad + ent_emb]
            unity_embeddings = torch.cat([ent_emb, self.other_emb[1:]], 0)

        return unity_embeddings

    def forward(self, predict_type, batch_data, target_ent_index, bigraph: Dict = {}, base_graph: Dict = {}, mode = 'train'):
        #* Load unity embeddings (entity and relation)
        unity_embeddings = self.load_unity_emd(bigraph)
        unity_embeddings = self.LayerNorm(unity_embeddings)
        unity_embeddings = self.feature_drop(unity_embeddings)

        #* Get all target embeddings
        if predict_type == 'ent':
            E_embed = unity_embeddings[:self.p.num_ent + 1] #include pad
        else:
            E_embed = unity_embeddings[0] + unity_embeddings[self.p.num_ent + 1: -1]

        #* Model Forward: get nary query representations [batch, dim]
        if self.p.explain_fact:
            h_out, node_weights, predictE_node, outE_ent_att  = self.out_model(batch_data, unity_embeddings, mode = mode)
            return h_out, node_weights, predictE_node, outE_ent_att
        else:
            h_out = self.out_model(batch_data, unity_embeddings, mode = mode)

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
        
        # For BCE LOSS, [batch, dim] * [dim, num_ent] → [batch, num_ent]
        # For Negative Sampling, [batch, 1, dim] * [batch, dim, num_neg] →[batch, 1, num_neg]→[batch, num_neg]
        if len(target_emb.size()) == 3:
            score = torch.bmm(h_out.unsqueeze(1), target_emb.transpose(1, 2)).squeeze(1)
        else:
            score = torch.mm(h_out, target_emb.transpose(1, 0))
        return score

    def loss(self, pred, batch_data):
        batch_mask_label = batch_data['batch_mask_label']
        batch_label_all = batch_data['batch_target_all']
        return self.myloss(pred, batch_mask_label, batch_label_all)
