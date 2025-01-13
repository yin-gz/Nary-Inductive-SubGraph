import torch
import random
from utils.utils_graphs import sample_khop_hedge, sample_khop_node, calculate_dis
from torch_geometric.data import HeteroData
from utils.utils_graphs import unbatch_edge
from torch_geometric.utils import coalesce
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset

class PairSubGDataset(InMemoryDataset):
    '''
    An inmemory dataset class for handling PSR tasks.
    '''

    def __init__(self, root, known_graph, facts, transform=None, pre_transform=None, params=None, split='train'):
        self.known_graph = known_graph
        self.facts = facts
        self.p = params
        self.split = split
        super(PairSubGDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        files_name = []
        name = self.split+"_pair_"+self.p.version+".pt"
        files_name.append(name)
        return files_name

    def process(self):
        self.data_list = construct_pair_data(self.known_graph, self.split, self.facts, self.p)
        data, slices = self.collate(self.data_list)
        torch.save((data, slices), self.processed_paths[0])
        

@torch.no_grad()
def construct_pair_data(known_graph, split, facts, params):
    '''
    In each batch, turn each fact to pair data, then merge them to a PyG data list
    '''
    all_result = []
    with tqdm(range(0, len(facts), params.batch_size),desc = 'construct pair data for '+split) as t:
        for i in t:
            if i+params.batch_size > len(facts):
                batch_facts = facts[i:]
            else:
                batch_facts = facts[i:i+params.batch_size]
            
            #* 1. construct batch graph based on query facts
            batch_predict_hnode = []
            batch_pos_target = []
            batch_edge_index = []
            batch_query_rel = []
            batch_input_seqs = []
            batch_mask_position = []
            combine_edge_index = []
            combine_edge_attr = []
            combine_edge_type = []
            known_graph_node = known_graph['node'].x
            for fact in batch_facts:
                if fact.mask_node_id in known_graph_node:
                    combine_edge_attr.extend(fact.hyper_edge_attr)
                    combine_edge_type.extend(fact.hyper_edge_type)
                    batch_edge_index.append(fact.hyper_edge_index)
                    batch_pos_target.append(fact.mask_node_id)
                    batch_query_rel.append(fact.query_rel)
                    batch_input_seqs.append(fact.input_seqs)
                    batch_mask_position.append(fact.mask_position)
                else:
                    print(fact.mask_node_id + ' not in known graph!!')
            # find predict_hnode in the original graph
            for index, edge_list in enumerate(batch_edge_index):
                batch_predict_hnode.append(edge_list[0][1])
                combine_edge_index.extend([edge[0], edge[1]] for edge in edge_list)
            combine_edge_type = torch.LongTensor(combine_edge_type)
            combine_edge_attr = torch.LongTensor(combine_edge_attr)
            combine_edge_index = torch.LongTensor(combine_edge_index).t()
            batch_predict_hnode = torch.LongTensor(batch_predict_hnode)
            batch_pos_target = torch.LongTensor(batch_pos_target)
            batch_input_seqs = torch.LongTensor(batch_input_seqs)
            batch_mask_position = torch.LongTensor(batch_mask_position)
              
            #* 2. Integrate batch graph to the known graph to get the new graph
            new_graph = HeteroData()
            if split == 'train':
                # Train: query node already in the known_graph
                new_edge_index = known_graph['node', 'to', 'Hedge'].edge_index.clone()
                new_edge_type = known_graph['node', 'to', 'Hedge'].edge_type.clone()
                new_edge_attr = known_graph['node', 'to', 'Hedge'].edge_attr.clone()
                # the batch_predict_hnode is sorted, so we can find the min and max index of the batch_predict_hnode and substitute them to the incomplete ones
                maybe_min = torch.where(new_edge_index[1] == batch_predict_hnode[0].item())[0]
                if maybe_min.size(0) != 0:
                    min_index = maybe_min[0].item() 
                else:
                    min_index = new_edge_index.size(1)
                maybe_max = torch.where(new_edge_index[1] == batch_predict_hnode[-1].item())[0]
                if maybe_max.size(0) != 0:
                    max_index = maybe_max[-1].item() + 1
                else:
                    max_index = -1
                new_edge_index = torch.cat([new_edge_index[:,:min_index], combine_edge_index, new_edge_index[:,max_index:]], dim = 1)
                new_edge_type = torch.cat([new_edge_type[:min_index],combine_edge_type, new_edge_type[max_index:]], dim = 0)
                new_edge_attr = torch.cat([new_edge_attr[:min_index],combine_edge_attr, new_edge_attr[max_index:]], dim = 0)
                #update new graph
                new_graph['node', 'to', 'Hedge'].edge_index = new_edge_index
                new_graph['node', 'to', 'Hedge'].edge_type = new_edge_type
                new_graph['node', 'to', 'Hedge'].edge_attr = new_edge_attr
            else:
                # merge predict hedge in batch_graph to known_graph
                num_hedge_known = known_graph['Hedge'].x.size(0)
                new_edge_index = combine_edge_index
                new_edge_attr = combine_edge_attr
                new_edge_type = combine_edge_type
                #reindex new_edge_index[1] and batch_predict_hnode to make sure the index is continuous
                new_edge_index[1] = new_edge_index[1].unique(sorted=True, return_inverse=True)[1] + num_hedge_known
                batch_predict_hnode = batch_predict_hnode.unique(sorted=True, return_inverse=True)[1] + num_hedge_known
                #add batch query hnode and edge to the known_graph(6969)
                new_graph['node', 'to', 'Hedge'].edge_index = torch.cat([known_graph['node', 'to', 'Hedge'].edge_index, new_edge_index], dim=1)
                new_graph['node', 'to', 'Hedge'].edge_type = torch.cat([known_graph['node', 'to', 'Hedge'].edge_type , new_edge_type], dim=0)
                new_graph['node', 'to', 'Hedge'].edge_attr = torch.cat([known_graph['node', 'to', 'Hedge'].edge_attr , new_edge_attr], dim=0)
            # complete the new graph information
            new_graph['node', 'to', 'Hedge'].edge_attr = new_graph['node', 'to', 'Hedge'].edge_attr - params.num_ent - 1
            new_graph['Hedge', 'to', 'node'].edge_index = torch.stack([new_graph['node', 'to', 'Hedge'].edge_index[1], new_graph['node', 'to', 'Hedge'].edge_index[0]], dim=0)
            new_graph['Hedge', 'to', 'node'].edge_type = new_graph['node', 'to', 'Hedge'].edge_type
            new_graph['Hedge', 'to', 'node'].edge_attr = new_graph['node', 'to', 'Hedge'].edge_attr
            new_graph['node'].x = known_graph['node'].x #the same as unity vocab
            new_graph['Hedge'].x = torch.arange(0, end= new_graph['Hedge', 'to', 'node'].edge_index[0].max()+1)


            #* 3. For each fact, sample k-hop neighborhood from the source hedge and the positive target node.
            #! sample from hedge
            batch_source_subg = sample_khop_hedge(new_graph, batch_predict_hnode, params.K_hop, params.max_each_hop)
            src_batch_edge, src_edge_attr= unbatch_hedge_subg(batch_source_subg) #ubatch to a list of edge_index and edge_attr
            #! sample from positive target node
            batch_tpos_subg = sample_khop_node(new_graph, batch_pos_target, params.K_hop, params.max_each_hop)
            tpos_batch_edge, tpos_edge_attr= unbatch_node_subg(batch_tpos_subg)    
            # merge
            for k in range(len(src_batch_edge)):
                try:
                    src_edge_k, pos_tar_edge_k = src_batch_edge[k], tpos_batch_edge[k]
                except:
                    continue
                #if pos target not in the sampled graph, skip
                if pos_tar_edge_k.size(1) == 0:
                    continue
                src_edge_type_k, pos_tar_edge_type_k = src_edge_attr[0][k], tpos_edge_attr[0][k]
                src_edge_attr_k, pos_tar_edge_attr_k = src_edge_attr[1][k], tpos_edge_attr[1][k]
                src_edge_attrs_k, pos_tar_edge_attrs_k = [src_edge_type_k, src_edge_attr_k], [pos_tar_edge_type_k, pos_tar_edge_attr_k]
                query_hypere_id, pos_pair_G = combine_pair(edge_index_pair = [src_edge_k, pos_tar_edge_k], edge_attrs_pair = [src_edge_attrs_k, pos_tar_edge_attrs_k], 
                                                                    source_hedge_id = batch_predict_hnode[k].item(), target_node_id = batch_pos_target[k].item(), max_hop = params.K_hop)
                if query_hypere_id == -1:
                    continue
                pos_pair_G.query_rel = batch_query_rel[k] - params.num_ent - 1
                pos_pair_G.mask_position = batch_mask_position[k].item()
                all_result.append(pos_pair_G)
      
    return all_result

@torch.no_grad()
def combine_pair(edge_index_pair, edge_attrs_pair, source_hedge_id, target_node_id, max_hop):
    '''
    Calculate each node's distance to source hedge id and target node id, return the combined pair graph
    '''
    edge_index_concat = torch.cat([edge_index_pair[0], edge_index_pair[1]], dim=1)
    edge_attrs_concat = []
    for i in range(len(edge_attrs_pair[0])):
        edge_attrs_concat.append(torch.cat([edge_attrs_pair[0][i], edge_attrs_pair[1][i]], dim=0))
    G = HeteroData()
    #reindex edge
    edge_index_concat, edge_attrs = coalesce(edge_index_concat, edge_attrs_concat, sort_by_row=False, reduce = 'min')
    G['node', 'to', 'Hedge'].edge_type = edge_attrs[0]
    G['node', 'to', 'Hedge'].edge_attr = edge_attrs[1]
    G['node'].x, edge_index_concat[0] = edge_index_concat[0].unique(sorted=True, return_inverse=True)
    G['Hedge'].x, edge_index_concat[1] = edge_index_concat[1].unique(sorted=True, return_inverse=True)
    G['node', 'to', 'Hedge'].edge_index = edge_index_concat
    
    #calculate dis to source hedge id and target node id, update G info
    new_hedge_id, G = calculate_dis(G, source_hedge_id, 'Hedge', max_hop)
    # new_tnode_id, G = calculate_dis(G, target_node_id, 'node', max_hop)
    G['node'].dis = torch.LongTensor([i for i in G['node'].dis_source.tolist()])
    G['Hedge'].mark_source = torch.LongTensor([1 if i == 0 else 0 for i in G['Hedge'].dis_source])
    G['node'].mark_target = (G['node'].x == target_node_id)
    #G['node'].mark_target = torch.LongTensor([1 if i == 0 else 0 for i in G['node'].dis_target])
    #randomly select 1 negative target node, avoid select the positive target node
    try:
        neg_target_id = random.choice(G['node'].x[G['node'].mark_target == 0])
        G['node'].mark_neg = torch.LongTensor([0]*G['node'].x.size(0))
        G['node'].mark_neg[G['node'].x == neg_target_id] = 1
    except:
        #print('no neg target node in the graph')
        new_hedge_id = -1
        

    return new_hedge_id, G

@torch.no_grad()
def unbatch_hedge_subg(batch_G):
    '''
    Return the unbatched edge_index (original idx in graph) and edge_attr
    '''
    ori_sample_x = batch_G['node'].x
    ori_sample_hedge = batch_G['Hedge'].x
    V2E_edge_index = batch_G['node', 'to', 'Hedge'].edge_index
    edge_type = batch_G['node', 'to', 'Hedge'].edge_type
    edge_attr = batch_G['node', 'to', 'Hedge'].edge_attr
    edge_attrs = [edge_type, edge_attr]
    
    #get edge batch
    node_batch = batch_G['node'].batch
    edge_batch = node_batch[V2E_edge_index[0]]
    #turn to original edge_index
    V2E_edge_index[0] = ori_sample_x[V2E_edge_index[0]]
    V2E_edge_index[1] = ori_sample_hedge[V2E_edge_index[1]]
    unbatch_edge_index, _, edge_attr_result = unbatch_edge(V2E_edge_index, 0, edge_attrs, edge_batch = edge_batch)
    return unbatch_edge_index, edge_attr_result
    
    
@torch.no_grad()
def unbatch_node_subg(batch_G):
    '''
    Return the merged edge_index (original idx in graph) and edge_attr
    '''
    ori_sample_x = batch_G['node'].x
    ori_sample_hedge = batch_G['Hedge'].x
    
    E2V_edge_index = batch_G['Hedge', 'to', 'node'].edge_index
    E2V_edge_type = batch_G['Hedge', 'to', 'node'].edge_type
    E2V_edge_attr = batch_G['Hedge', 'to', 'node'].edge_attr
    
    V2E_edge_index = batch_G['node', 'to', 'Hedge'].edge_index
    V2E_edge_type = batch_G['node', 'to', 'Hedge'].edge_type
    V2E_edge_attr = batch_G['node', 'to', 'Hedge'].edge_attr
    
    #! concat E2V and V2E
    edge_index = torch.cat([E2V_edge_index.flip(0), V2E_edge_index], dim = 1)
    edge_type = torch.cat([E2V_edge_type, V2E_edge_type], dim = 0)
    edge_attr = torch.cat([E2V_edge_attr, V2E_edge_attr], dim = 0)
    edge_attrs = [edge_type, edge_attr]
    
    #for i in range(source.size(0)):
        #assert(torch.isin(source[i],ori_sample_x[edge_index[1]]).item())


    #get edge batch
    node_batch = batch_G['node'].batch
    edge_batch = node_batch[edge_index[0]]
    #turn to original edge_index
    edge_index[0] = ori_sample_x[edge_index[0]]
    edge_index[1] = ori_sample_hedge[edge_index[1]]
    unbatch_edge_index, _, edge_attr_result = unbatch_edge(edge_index, 0, edge_attrs, edge_batch = edge_batch)
    

    return unbatch_edge_index, edge_attr_result