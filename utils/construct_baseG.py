import torch
import numpy as np
from torch_geometric.utils import dense_to_sparse, to_undirected, coalesce, to_networkx 
from torch_geometric.data import Data, HeteroData
from typing import List, Union
from collections import defaultdict
import copy
#from line_profiler import LineProfiler

@torch.no_grad()
def construct_base_hgraph(nary_hypers, device = 0):
    '''
    Args:
        nary_hypers: List of NaryHyperE
        add_hyqualifier: add qualifier ent to hypergraph or not
        add_biqualifier: add qualifier ent to bi-graph or not
    Returns:
        hyperG: pyG hetro graph with edgetype, node_type:['ent (including virtual cls)': torch.LongTensor,  'hyper_edge': torch.LongTensor], 
        bigraph: pyG graph with edgetype
    '''  
    hyperG_edge_index = []
    hyperG_edge_attr = []
    hyperG_edge_type = []
    biedge_index = []
    biedge_type = []

    for hypere in nary_hypers:
        hyperG_edge_index.extend(hypere.hyper_edge_index)
        hyperG_edge_attr.extend(hypere.hyper_edge_attr)
        hyperG_edge_type.extend(hypere.hyper_edge_type)
        biedge_index.extend(hypere.biedge_index)
        biedge_type.extend(hypere.biedge_type)

    #to tensor
    hyperG_edge_index = torch.LongTensor(np.array(hyperG_edge_index).astype("int64")).t()
    hyperG_edge_attr = torch.LongTensor(hyperG_edge_attr)
    hyperG_edge_type = torch.LongTensor(hyperG_edge_type)
    biedge_index = torch.LongTensor(np.array(biedge_index).astype("int64")).t()
    biedge_type = torch.LongTensor(biedge_type)


    #count hyperG_node_type
    hyperG_node = hyperG_edge_index[0].unique(sorted=True)
    hyperG_hedge = hyperG_edge_index[1].unique(sorted=True)

    #turn to pyG graph
    hyperG = HeteroData()
    #pad x_id to its max
    hyperG['node'].x = torch.arange(0, end= hyperG_node.max() + 1)
    hyperG['Hedge'].x = hyperG_hedge
    hyperG['node', 'to', 'Hedge'].edge_index = hyperG_edge_index.contiguous()
    hyperG['node', 'to', 'Hedge'].edge_attr = hyperG_edge_attr
    hyperG['node', 'to', 'Hedge'].edge_type = hyperG_edge_type
    hyperG['Hedge', 'to', 'node'].edge_index = torch.stack([hyperG_edge_index[1], hyperG_edge_index[0]], dim=0).contiguous()
    hyperG['Hedge', 'to', 'node'].edge_attr = hyperG_edge_attr
    hyperG['Hedge', 'to', 'node'].edge_type = hyperG_edge_type

    bigraph = Data(edge_index = biedge_index, edge_type = biedge_type)
    bigraph = bigraph.to(device)

    return hyperG, bigraph

@torch.no_grad()
def construct_base_binaryG(nary_hypers, device = 0):
    '''
    Only two nodes in each hyperedge
    Args:
        nary_hypers: List of NaryHyperE
        add_hyqualifier: add qualifier ent to hypergraph or not
        add_biqualifier: add qualifier ent to bi-graph or not
    Returns:
        hyperG: pyG hetro graph with edgetype, node_type:['ent(including virtual cls)': torch.LongTensor,  'hyper_edge': torch.LongTensor], 
                edge_type:[0(virtual_edge),1,2,3,4]
                edge_attr: [rel1, rel2, ...]
        bigraph: pyG graph with edgetype
    ''' 
    #first, add original main edge_index_i
    #then, add qual edge_index_i and modify the hedge_index
    hyperG_edge_index = []
    hyperG_edge_type = []
    hyperG_edge_attr = []
    hyperG_node_type = {'nodes': [], 'hedge_id': []}
    biedge_index = []
    biedge_type = []

    hyperG_qe_index = []
    hyperG_qe_type = []
    hyperG_qe_attr = []
    q_hedge_start = 0
    for hypere in nary_hypers:
        hyedge_index_i = hypere.hyper_edge_index[:2]
        hyedge_type_i = hypere.hyper_edge_type[:2]
        hyedge_attr_i = hypere.hyper_edge_attr[:2]

        hyperG_edge_index.extend(hyedge_index_i)
        hyperG_edge_attr.extend(hyedge_attr_i)
        hyperG_edge_type.extend(hyedge_type_i)

        #add [sub, new_hedge_id] and [each_equal, new_hedge_id]
        hyperG_qe_index.extend([hyedge_index_i[0][0], index +  q_hedge_start] for index, edge_tuple in enumerate(hypere.hyper_edge_index[2:]))
        hyperG_qe_index.extend([edge_tuple[0], index +  q_hedge_start] for index, edge_tuple in enumerate(hypere.hyper_edge_index[2:]))
        q_len = len(hypere.hyper_edge_type[2:])
        q_hedge_start += q_len
        hyperG_qe_attr.extend(q_len*[hyedge_attr_i[0]]) #sub_rel
        hyperG_qe_attr.extend(hypere.hyper_edge_attr[2:]) #each qual_rel
        hyperG_qe_type.extend(q_len*[0])
        hyperG_qe_type.extend(q_len*[1])

        biedge_index.extend(hypere.biedge_index)
        biedge_type.extend(hypere.biedge_type)

    #concat
    hyperG_edge_index_main = torch.LongTensor(np.array(hyperG_edge_index).astype("int64")).t()
    hyperG_qe_index = torch.LongTensor(np.array(hyperG_qe_index).astype("int64")).t()
    hyperG_qe_index[1] += hyperG_edge_index_main[1].max() + 1
    hyperG_edge_index = torch.cat([hyperG_edge_index_main, hyperG_qe_index], dim=1)
    hyperG_edge_attr.extend(hyperG_qe_attr)
    hyperG_edge_type.extend(hyperG_qe_type)
    #to tensor
    hyperG_edge_attr = torch.LongTensor(hyperG_edge_attr)
    hyperG_edge_type = torch.LongTensor(hyperG_edge_type)
    biedge_index = torch.LongTensor(np.array(biedge_index).astype("int64")).t()
    biedge_type = torch.LongTensor(biedge_type)

    #count hyperG_node_type
    hyperG_node = hyperG_edge_index[0].unique(sorted=True)
    hyperG_hedge = hyperG_edge_index[1].unique(sorted=True)

    #turn to pyG graph
    hyperG = HeteroData()
    #pad x_id to its max
    hyperG['node'].x = torch.arange(0, end= hyperG_node.max() + 1)
    hyperG['Hedge'].x = hyperG_hedge
    hyperG['node', 'to', 'Hedge'].edge_index = hyperG_edge_index.contiguous()
    hyperG['node', 'to', 'Hedge'].edge_attr = hyperG_edge_attr
    hyperG['node', 'to', 'Hedge'].edge_type = hyperG_edge_type
    hyperG['Hedge', 'to', 'node'].edge_index = torch.stack([hyperG_edge_index[1], hyperG_edge_index[0]], dim=0).contiguous()
    hyperG['Hedge', 'to', 'node'].edge_attr = hyperG_edge_attr
    hyperG['Hedge', 'to', 'node'].edge_type = hyperG_edge_type

    bigraph = Data(edge_index = biedge_index, edge_type = biedge_type)
    bigraph = bigraph.to(device)

    return hyperG, bigraph



@torch.no_grad()
def construct_base_hgraph_starE(nary_hypers, device = 0):
        """
        Quals are represented differently here, i.e., more as a coo matrix
        s1 p1 o1 qr1 qe1 qr2 qe2    [edge index column 0]
        s2 p2 o2 qr3 qe3            [edge index column 1]

        edge index:
        [ [s1, s2],
            [o1, o2] ]

        edge type:
        [ p1, p2 ]

        quals will looks like
        [ [qr1, qr2, qr3],
            [qe1, qe2, qe3],
            [0  , 0  , 1  ]       <- obtained from the edge index columns

        :param raw: [[s, p, o, qr1, qe1, qr2, qe3...], ..., [...]]
            (already have a max qualifier length padded data)
        :param config: the config dict
        :return: output dict
        """
        edge_index = []
        edge_type = []
        edge_type_inverse = []
        full_quals = []
        biedge_index = []
        biedge_type = []
        for index, hypere in enumerate(nary_hypers):
            seq = hypere.hyper_seq
            edge_index.append([seq[0], seq[2]]) #[head, tail]
            biedge_index.extend([[seq[0], seq[2]], [seq[2], seq[0]]])
            edge_type.append(seq[1])
            edge_type_inverse.append(seq[3])
            biedge_type.extend([seq[1], seq[3]])
            seq_len = 2 * (hypere.arity - 2)  + 4

            for i in range(4, seq_len, 2):
                full_quals.append([seq[i+1], seq[i], index])
                biedge_index.extend([[seq[0],seq[i]], [seq[i],seq[0]]])
                biedge_type.extend([seq[i+1], seq[i+1]])
            

        #turn to Tensor
        edge_index = torch.LongTensor(np.array(edge_index).astype("int64")).t()
        edge_type = torch.LongTensor(edge_type)
        edge_type_inverse = torch.LongTensor(edge_type_inverse)
        biedge_index = torch.LongTensor(np.array(biedge_index).astype("int64")).t().to(device).contiguous()
        biedge_type = torch.LongTensor(biedge_type).to(device).contiguous()
        

        # Add inverses
        edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1).to(device).contiguous()
        edge_type = torch.cat([edge_type, edge_type_inverse]).to(device)

        full_quals = torch.LongTensor(np.array(full_quals).astype("int64")).t()
        full_quals = torch.cat([full_quals, full_quals], dim=1).to(device).contiguous()

        bigraph = Data(edge_index = biedge_index, edge_type = biedge_type)

        return {'edge_index': edge_index,
                'edge_type': edge_type,
                'quals': full_quals}, bigraph



