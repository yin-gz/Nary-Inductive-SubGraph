from torch_geometric.loader import NeighborLoader
import torch
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data, HeteroData
from typing import Union, List
from torch_geometric.utils import degree
from torch_geometric.utils import unbatch, unbatch_edge_index
import math
from torch import Tensor
import numpy as np
from torch_geometric.typing import OptTensor
from igraph import Graph


def sample_khop_hedge(hyperG, hedge_id: Union[int, List[int], torch.LongTensor], k_hop= 0, k_max_neighbors = 8):
    '''
    Sample k-hop neighbor from the source hedges by pytorch_geometric
    Args:
        hyperG: pyG Hetro Data
        hedge_id: int, target hedge (or list)
        k_hop: int
        k_max_neighbors: List, sample how many labels in each hop
    Returns:
        subgraph in pyG HeteroData
    ''' 

    # turn k_max_neighbors, the original denotes the neighbor hyper, need to add neighbor node
    max_neighbor_list = [-1] # sample all nodes in edge
    next_layer = k_max_neighbors
    for i in range(0, k_hop):
        max_neighbor_list.append(next_layer) # sample hedge
        max_neighbor_list.append(-1) # sample all nodes in each hedge
        if next_layer != -1:
            next_layer = int(math.log2(k_max_neighbors))
        if next_layer == 0:
            break

    if isinstance(hedge_id, int):
        hedge_id = torch.LongTensor([hedge_id])
    elif isinstance(hedge_id, List):
        hedge_id = torch.LongTensor(hedge_id)
        
    loader = NeighborLoader(
        hyperG,
        num_neighbors= max_neighbor_list,
        batch_size = hedge_id.size(0),
        input_nodes=('Hedge', hedge_id),
        is_sorted = False,
        disjoint = True)
    try:
        sampled_hetero_data = next(iter(loader))
    except:
        print('edge', hyperG['node', 'to', 'Hedge'].edge_index[1])
        print('hedge_id', hedge_id)
        raise Exception('error')
    return sampled_hetero_data

def sample_khop_node(hyperG, node_id: Union[int, List[int], torch.LongTensor], k_hop= 0, k_max_neighbors = 8):
    '''
    Sample k-hop neighbor from the target nodes by pytorch_geometric
    Args:
        hyperG: pyG Hetro Data
        node_id: int, target node (or list)
        k_hop: int
        k_max_neighbors: List, sample how many labels in each hop
    Returns:
        subgraph in pyG HeteroData
    ''' 
    max_neighbor_list = []
    next_layer = k_max_neighbors
    for i in range(0, k_hop):
        max_neighbor_list.append(next_layer) #sample hedge
        max_neighbor_list.append(-1) #sample all nodes in each hedge
        if next_layer != -1:
            next_layer = int(math.log2(k_max_neighbors))
        if next_layer == 0:
            break

    if isinstance(node_id, int):
        node_id = torch.LongTensor([node_id])
    elif isinstance(node_id, List):
        node_id = torch.LongTensor(node_id)
        
    loader = NeighborLoader(
        hyperG,
        num_neighbors= max_neighbor_list,
        batch_size = node_id.size(0),
        input_nodes=('node', node_id),
        is_sorted = False,
        disjoint = True)
    try:
        sampled_hetero_data = next(iter(loader))
    except:
        print('edge', hyperG['node', 'to', 'Hedge'].edge_index[1])
        print('node_id', node_id)
        raise Exception('error')
    return sampled_hetero_data

#@profile
def unbatch_edge(edge_index, node_dim = 0, edge_attr: Union[OptTensor, List[Tensor], None] = None, node_batch: OptTensor = None, edge_batch: OptTensor = None):
    if edge_batch is None:
        edge_batch = node_batch[edge_index[node_dim]]

    #* reorder the batch from small to large and then unbatch
    sorted_edge_indices = torch.argsort(edge_batch)
    edge_batch = edge_batch[sorted_edge_indices]
    edge_index = edge_index[:, sorted_edge_indices]
    sizes = degree(edge_batch, dtype=torch.long).tolist()
    unbatch_edge_index = edge_index.split(sizes, dim = 1)

    if edge_attr is None:
        return unbatch_edge_index, sizes, None
    if isinstance(edge_attr, Tensor):
        edge_attr = edge_attr[sorted_edge_indices]
        edge_attr = list(unbatch(edge_attr, edge_batch, dim = 0))
        return unbatch_edge_index, sizes, edge_attr
    if isinstance(edge_attr, (list, tuple)):
        edge_attr_result = []
        for e in edge_attr:
            e = e[sorted_edge_indices]
            edge_attr_result.append(list(unbatch(e, edge_batch, dim = 0)))
        return list(unbatch_edge_index), sizes, edge_attr_result
    
#@profile
def calculate_dis(graph: HeteroData, target_id: int, target_type: str, max_hop = 3):
    node_id = graph['node'].x #original id
    Hedge_id = graph['Hedge'].x #original id
    edge_index = graph['node', 'to', 'Hedge'].edge_index #after reindex

    #turn original target_id to id in new graph
    if target_type == 'Hedge':
        index = torch.where(Hedge_id == target_id)[0][0]
        index += graph['node'].num_nodes
    else:
        try:
            index = torch.where(node_id == target_id)[0][0]
        except:
            print('no target id:', target_id)
            #target not in the sampled graph
            node_num_subg = edge_index[0].max().item()+ edge_index[1].max().item() + 2
            dis = torch.LongTensor(node_num_subg*[-1])
            graph['node'].dis_target = dis[:node_id.size(0)]
            graph['Hedge'].dis_target = dis[node_id.size(0):]
            return -1, graph
            
    G_homo = Graph(directed=False)
    G_homo.add_vertices(edge_index[0].max().item()+ edge_index[1].max().item() + 2)
    edge_list = np.array(edge_index.t())
    edge_list[:,1] = edge_list[:,1] + edge_index[0].max().item() + 1 #turn hedge_id following node id in subg
    G_homo.add_edges(edge_list)
    ori_dis = G_homo.shortest_paths(source=index.item())[0]
    dis = [(max_hop+1) if i==float('inf') else i for i in ori_dis]
    dis = torch.LongTensor(dis)
    node_id = node_id.unique()
    if target_type == 'Hedge':
        graph['node'].dis_source = dis[:node_id.size(0)]
        graph['Hedge'].dis_source = dis[node_id.size(0):]
        #print('Hedge', graph['node'].dis_source, graph['Hedge'].dis_source)
    else:
        graph['node'].dis_target = dis[:node_id.size(0)]
        graph['Hedge'].dis_target = dis[node_id.size(0):]
        #print('node', graph['node'].dis_target, graph['Hedge'].dis_target)

    return index.item(), graph

def calculate_dis_trans(graph: HeteroData, target_id: int, target_type: str, max_hop = 3):
    node_id = graph['node'].x #original id
    Hedge_id = graph['Hedge'].x #original id
    edge_index = graph['node', 'to', 'Hedge'].edge_index #after reindex

    #turn original target_id to id in new graph
    if target_type == 'Hedge':
        index = torch.where(Hedge_id == target_id)[0][0]
        index += graph['node'].num_nodes
    else:
        try:
            index = torch.where(node_id == target_id)[0][0]
        except:
            print('no target id:', target_id)
            #target not in the sampled graph
            node_num_subg = edge_index[0].max().item()+ edge_index[1].max().item() + 2
            dis = torch.LongTensor(node_num_subg*[-1])
            graph['node'].dis_target = dis[:node_id.size(0)]
            graph['Hedge'].dis_target = dis[node_id.size(0):]
            return -1, graph
            
    G_homo = Graph(directed=False)
    G_homo.add_vertices(edge_index[0].max().item()+ edge_index[1].max().item() + 2)
    edge_list = np.array(edge_index.t())
    edge_list[:,1] = edge_list[:,1] + edge_index[0].max().item() + 1 #turn hedge_id following node id in subg
    G_homo.add_edges(edge_list)
    ori_dis = G_homo.shortest_paths(source=index.item())[0]
    dis = [(max_hop+1) if i==float('inf') else i for i in ori_dis]
    dis = torch.LongTensor(dis)
    node_id = node_id.unique()
    if target_type == 'Hedge':
        graph['node'].dis_source = dis[:node_id.size(0)]
        graph['Hedge'].dis_source = dis[node_id.size(0):]
        #print('Hedge', graph['node'].dis_source, graph['Hedge'].dis_source)
    else:
        graph['node'].dis_target = dis[:node_id.size(0)]
        graph['Hedge'].dis_target = dis[node_id.size(0):]
        #print('node', graph['node'].dis_target, graph['Hedge'].dis_target)

    return index.item(), graph