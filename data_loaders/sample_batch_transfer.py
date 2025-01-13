from torch.utils.data import Dataset
import torch
import copy
import numpy as np
import random
from argparse import Namespace
from utils.utils_graphs import sample_khop_hedge, calculate_dis, calculate_dis_trans
from torch_geometric.data import HeteroData
from typing import Optional, List, Union, Tuple


class BatchData(Dataset):
    """
    Dataset class for batch data
    """
    def __init__(self, facts,  params:  Namespace = Namespace(), gt_dict=None, target_index= None, known_graph: HeteroData = HeteroData(), split = 'train'):
        self.facts = facts
        self.gt_dict = gt_dict
        self.target_index = target_index
        self.p = params
        self.tar_num = (self.p.num_ent + 1) if target_index is None else len(target_index)
        self.known_graph = known_graph
        self.split = split

    def __len__(self):
        return len(self.facts)

    # get info used in model
    def __getitem__(self, idx):
        fact = self.facts[idx]

        # hyperedge info
        input_seqs = fact.input_seqs
        hyper_edge_index = fact.hyper_edge_index #has added 'mask_node_id' to 'hyper_edge_index'
        hyper_edge_type =  fact.hyper_edge_type
        hyper_edge_attr = fact.hyper_edge_attr

        #mask info
        mask_node_id = fact.mask_node_id
        mask_position = fact.mask_position
        mask_label = fact.mask_label
        mask_type = fact.mask_type
        mask_etype = fact.mask_etype
        query_rel = fact.query_rel

        # target info
        use_neg = self.p.use_neg if (self.split == 'train') else False
        target_all, neg_target_index = self.get_target_all(input_seqs, mask_position, mask_node_id, mask_type, self.gt_dict, self.target_index, use_neg)
        
        return input_seqs, hyper_edge_index, hyper_edge_type, hyper_edge_attr, mask_node_id, mask_position, mask_label, mask_type, mask_etype, query_rel, target_all, neg_target_index


    # turn bs * [batch_seq, batch_adj,... ] in to bs * batch_seq, bs * batch_adj,...
    def collate_fn(self, data):
        max_seq_length = self.p.max_seq_length

        #load data
        batch_input_seqs = torch.LongTensor(np.array(
            [inst[0] for inst in data]).astype("int64")).reshape([-1, max_seq_length])
        # batch * Tensor (2, different n_edges)
        batch_edge_index = [inst[1] for inst in data] 
        batch_edge_type  = [inst[2] for inst in data]
        batch_edge_attr  = [inst[3] for inst in data]

        mask_node_id = torch.LongTensor(np.array(
            [inst[4] for inst in data]).astype("int64")).reshape([-1])
        batch_mask_position = torch.LongTensor(np.array(
            [inst[5] for inst in data]).astype("int64")).reshape([-1])
        batch_mask_label = torch.LongTensor(np.array(
            [inst[6] for inst in data]).astype("int64")).reshape([-1])
        batch_mask_type = torch.LongTensor(np.array(
            [inst[7] for inst in data]).astype("int64")).reshape([-1])
        batch_mask_etype = torch.LongTensor(np.array(
            [inst[8] for inst in data]).astype("int64")).reshape([-1])
        batch_query_rel = torch.LongTensor(np.array(
            [inst[9] for inst in data]).astype("int64")).reshape([-1])
        batch_target_all = torch.Tensor(np.array(
            [inst[10] for inst in data]).astype("float32")).reshape([batch_query_rel.size(0), -1])

        batch_data = {
            'batch_input_seqs': batch_input_seqs,
            'batch_mask_position': batch_mask_position,
            'batch_mask_label': batch_mask_label, 
            'batch_mask_type': batch_mask_type,
            'batch_mask_etype': batch_mask_etype,
            'batch_query_rel': batch_query_rel,
            'batch_target_all': batch_target_all}
        
        if self.p.use_subg:
            batch_data['batch_edge_index'] = batch_edge_index
            batch_data['batch_edge_type'] = batch_edge_type
            batch_data['batch_edge_attr'] = batch_edge_attr
            batch_data['batch_mask_node_id'] = mask_node_id
        if (self.split == 'train') and self.p.use_neg:
            neg_target_index = torch.LongTensor(np.array([inst[11] for inst in data])).reshape([-1, self.p.neg_num+1])
            batch_data['neg_target_index'] = neg_target_index
        
        return batch_data

    def get_target_all(self, input_seqs, mask_position, mask_node_id, mask_type, gt_dict, target_index, use_neg):
        """
        :param statements: array of shape (bs, seq_len) like (64, 43)
        :return: array of shape (bs, num_entities) like (64, 49113), the index for the correct targets are 1 and the rest are 0
        """
            
        tar_num = len(target_index)
        y = np.zeros((tar_num), dtype=np.float32)
        lbl_smooth = self.p.e_soft_label
        
        if self.split != 'train':
            lbl_smooth = 0

        key = tuple([input_seqs[i] for i in range(len(input_seqs)) if ((input_seqs[i] != 0  and i != mask_position))])
        lbls = gt_dict[mask_position][key]
        if len(lbls) == 0:
            print('can not find label for:', key)
        y[lbls] = 1.0

        if use_neg:
            #* strict sampling, too slow
            '''
            #inverse y
            prob = 1-y
            prob =prob/np.sum(prob)
            #sample neg_num according to y probability
            
            neg_samples = np.random.choice(target_index, size = self.p.neg_num, replace = False, p = prob)
            #neg_samples = random.choices(target_index, k = self.p.neg_num, weights = prob)
            #neg_target_index = [mask_label] + list(neg_samples) # index here is based on target ent
            neg_target_index = np.append(neg_samples, mask_label)
            '''
            #* fast sampling
            if target_index is not None:
                neg_samples = random.choices(target_index, k = self.p.neg_num)
            else:
                #create continuous index
                target_index = np.arange(1, tar_num)
                neg_samples = random.choices(target_index, k = self.p.neg_num)
            neg_target_index = np.append(neg_samples, mask_node_id) #all original id
            y = np.array([0]*self.p.neg_num + [1])
        else:
            neg_target_index = None
            if lbl_smooth != 0:
                y = (1.0 - lbl_smooth) * y + (1.0 / tar_num)

        return y, neg_target_index
    

    
@torch.no_grad()
#@profile
def construct_batch_graph(split, known_graph, batch_data, params):
    '''
    Construct batch graph for each batch and update batch_data by adding "batch_graph" and "batch_predict_hedge"
    "batch_graph": The sampled batch graph, pyG HeteroData format, including node, edge_index, edge_type, edge_attr
    "batch_predict_hedge": Mark the hyper edge index for predicting, tensor format, the hyperedge id in the batch graph
    
    Param:
    known_graph: the known base graph, pyG HeteroData format
    '''
    
    batch_edge_index = batch_data['batch_edge_index']
    batch_edge_attr = batch_data['batch_edge_attr']
    batch_edge_type = batch_data['batch_edge_type']

    #* combine batch hyperedges
    batch_predict_hedge = [] #predict hnode_id in graph
    combine_edge_index = []
    combine_edge_attr = []
    combine_edge_type = []
    if params.binary:
        combine_bedge_index = []
        combine_bedge_attr = []
        combine_bedge_type = []
    q_hedge_start = 0
    for index, edge_list in enumerate(batch_edge_index):
        #index denote hedge for predicting in the batch
        batch_predict_hedge.append(edge_list[0][1])
        if params.binary is False:
            combine_edge_index.extend([edge[0], edge[1]] for edge in edge_list)
            combine_edge_attr.extend(batch_edge_attr[index])
            combine_edge_type.extend(batch_edge_type[index])
        else:
            #! for binary test, split the edge into (head, tail) and i*(head, qual_entity)
            combine_edge_index.extend([edge[0], edge[1]] for edge in edge_list[:2])
            combine_edge_attr.extend(batch_edge_attr[index][:2])
            combine_edge_type.extend(batch_edge_type[index][:2])
            combine_bedge_index.extend([edge_list[0][0], q_hedge_start + index] for index, edge_tuple in enumerate(edge_list[2:]))
            combine_bedge_index.extend([edge_tuple[0], q_hedge_start + index] for index, edge_tuple in enumerate(edge_list[2:]))
            q_len = len(edge_list[2:])
            q_hedge_start += q_len
            combine_bedge_attr.extend(q_len*[batch_edge_attr[index][0]]) #head's relation
            combine_bedge_attr.extend(batch_edge_attr[index][2:]) #qual's relation
            combine_edge_type.extend(q_len*[0])
            combine_edge_type.extend(q_len*[1])
    
    if params.binary is False or split == 'train':
        combine_edge_index = torch.LongTensor(np.array(combine_edge_index).astype("int64")).t()
    else:
        # add main edge and qual edges for binary test
        combine_edge_index = torch.LongTensor(np.array(combine_edge_index).astype("int64")).t()
        combine_bedge_index = torch.LongTensor(np.array(combine_bedge_index).astype("int64")).t()
        combine_bedge_index[1] += known_graph['node', 'to', 'Hedge'].edge_index[1].max() + 1
        combine_edge_index = torch.cat([combine_edge_index, combine_bedge_index], dim=1)
        combine_edge_attr.extend(combine_bedge_attr)
        combine_edge_type.extend(combine_bedge_type)
        
    combine_edge_attr = torch.LongTensor(combine_edge_attr)
    combine_edge_type = torch.LongTensor(combine_edge_type)
    batch_predict_hedge = torch.LongTensor(batch_predict_hedge)
                    
    
    #* add predict batch edges to base graph
    # when training, substitute the target edge in known_graph with batch_graph (with target entity maskerd)
    # when testing, add the batch_graph to the known inference graph
    new_graph = HeteroData()
    # Train: query node already in the known_graph
    if split == 'train' or split == 'inference':
        #sustitute target edge in known_graph with batch_graph
        new_edge_index = known_graph['node', 'to', 'Hedge'].edge_index.clone()
        new_edge_type = known_graph['node', 'to', 'Hedge'].edge_type.clone()
        new_edge_attr = known_graph['node', 'to', 'Hedge'].edge_attr.clone()
        #in train+valid, batch_predict_hedge may exist original graph max_num, so min_index and max_index may return -1
        maybe_min = torch.where(new_edge_index[1] == batch_predict_hedge[0].item())[0]
        if maybe_min.size(0) != 0:
            min_index = maybe_min[0].item() 
        else:
            min_index = new_edge_index.size(1)
        maybe_max = torch.where(new_edge_index[1] == batch_predict_hedge[-1].item())[0]
        if maybe_max.size(0) != 0:
            max_index = maybe_max[-1].item() + 1
        else:
            max_index = -1

        #update new graph
        new_graph['node', 'to', 'Hedge'].edge_index = torch.cat([new_edge_index[:,:min_index], combine_edge_index, new_edge_index[:,max_index:]], dim = 1)
        new_graph['node', 'to', 'Hedge'].edge_type = torch.cat([new_edge_type[:min_index],combine_edge_type, new_edge_type[max_index:]], dim = 0)
        new_graph['node', 'to', 'Hedge'].edge_attr = torch.cat([new_edge_attr[:min_index],combine_edge_attr, new_edge_attr[max_index:]], dim = 0)
    else:
        num_hedge_known = known_graph['Hedge'].x.size(0)
        new_edge_index = combine_edge_index
        new_edge_index[1] = new_edge_index[1].unique(sorted=True, return_inverse=True)[1] + num_hedge_known
        batch_predict_hedge = batch_predict_hedge.unique(sorted=True, return_inverse=True)[1] + num_hedge_known
        new_graph['node', 'to', 'Hedge'].edge_index = torch.cat([known_graph['node', 'to', 'Hedge'].edge_index, new_edge_index], dim=1)
        new_graph['node', 'to', 'Hedge'].edge_type = torch.cat([known_graph['node', 'to', 'Hedge'].edge_type, combine_edge_type], dim=0)
        new_graph['node', 'to', 'Hedge'].edge_attr = torch.cat([known_graph['node', 'to', 'Hedge'].edge_attr, combine_edge_attr], dim=0)
    #complete new_graph info
    new_graph['Hedge', 'to', 'node'].edge_index = torch.stack([new_graph['node', 'to', 'Hedge'].edge_index[1], new_graph['node', 'to', 'Hedge'].edge_index[0]], dim=0)
    new_graph['Hedge', 'to', 'node'].edge_type = new_graph['node', 'to', 'Hedge'].edge_type
    new_graph['Hedge', 'to', 'node'].edge_attr = new_graph['node', 'to', 'Hedge'].edge_attr
    new_graph['node'].x = known_graph['node'].x #the same as unity vocab
    new_graph['Hedge'].x = torch.arange(0, end= new_graph['Hedge', 'to', 'node'].edge_index[0].max()+1)

    #* sample k-hop subgraph
    if params.task == "full-trans":
        batch_graph = new_graph
    elif split != "train" and "FI" not in params.dataset:
        batch_graph = sample_khop_hedge(new_graph, batch_predict_hedge, params.K_hop, -1)
    else:
        batch_graph = sample_khop_hedge(new_graph, batch_predict_hedge, params.K_hop, params.max_each_hop)
    
    #_, batch_graph = calculate_dis_trans(batch_graph, batch_predict_hedge, 'Hedge', params.K_hop)
    
    batch_data['batch_graph'] = batch_graph
    del batch_graph

    batch_data['batch_predict_hedge'] = batch_predict_hedge
    del batch_data['batch_edge_index'], batch_data['batch_edge_attr'], batch_data['batch_edge_type'], batch_predict_hedge, new_graph, combine_edge_index, combine_edge_attr, combine_edge_type
    return batch_data


        

