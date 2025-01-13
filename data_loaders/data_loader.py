from tqdm import tqdm
import numpy as np
from data_loaders.NaryData import NaryHyperE, NaryInstance
from data_loaders.read_datasets import *
from utils.construct_baseG import construct_base_hgraph, construct_base_hgraph_starE, construct_base_binaryG
from argparse import Namespace
from data_loaders.vocab import VocabularyWithFeature

class DataFacts:
    """
    Input file name, get train/valid/test Nary example list
    """
    def __init__(self, data_dir ='./data/JF17K/', param: Namespace = Namespace()):
        self.p = param
        #* Read datasets, return dict: {'train': List of NaryExamples, 'train_reverse': List of NaryExamples, ...}
        raw_examples, raw_reverse_examples = self.read_datasets(data_dir)
        param.max_seq_length  = 2 * (param.max_arity - 2)  + 4 #if max_arity has been updated, update max_seq_length
  
        #* Format vocab
        self.vocab = VocabularyWithFeature(raw_examples, data_dir, self.p)
        self.mask_id = self.vocab.unit_vocab['[MASK]'] if self.p.unit_encode else self.vocab.ent_vocab['[MASK]']

        #* Turn each example in train/valid/test and format to NaryHyperE (including its padded tokens, insiding edges and biedges)
        # Return dict: {'train': List of NaryHyperE, 'valid': List of NaryHyperE, ...}
        self.nary_hypers = self.turn_example_to_hedge(raw_examples, self.vocab, add_hyperqualifier=True)

        #* Construct base graph
        # train_hyperG, inference_hyperG (known graph when testing and validating) for sampling subgraphs
        if self.p.model_name == 'StarE' and self.p.use_subg is False:
           self.train_hyperG, self.train_bigraph  = construct_base_hgraph_starE(self.nary_hypers['train'], device = self.p.device)
        elif self.p.binary:
            self.train_hyperG, self.train_bigraph  = construct_base_binaryG(self.nary_hypers['train'], device = self.p.device)
        else:
            self.train_hyperG, self.train_bigraph  = construct_base_hgraph(self.nary_hypers['train'], device = self.p.device)
        if 'inference' in self.p.load_split_list :
            if self.p.model_name == 'StarE' and self.p.use_subg is False:
                self.inference_hyperG, self.inference_bigraph  = construct_base_hgraph_starE(self.nary_hypers['inference'], device = self.p.device)
            elif self.p.binary:
                self.inference_hyperG, self.inference_bigraph  = construct_base_binaryG(self.nary_hypers['inference'], device = self.p.device)
            else:
                self.inference_hyperG, self.inference_bigraph  = construct_base_hgraph(self.nary_hypers['inference'], device = self.p.device)

        #* Generate training, valid and testing masked instances; update gt_dict (their labels) at the same time
        # gt_dict_all: consider all instances in train/inference and update golden targets(labels)
        # gt_dict_split: consider instances in train/inference separately and update golden targets(labels)
        self.data = {}
        self.gt_dict_all = collections.defaultdict(lambda: collections.defaultdict(list))
        self.gt_dict_split = {'train': collections.defaultdict(lambda: collections.defaultdict(list)),
                              'inference': collections.defaultdict(lambda: collections.defaultdict(list))}
        for split in self.p.load_split_list :
            mask_pos_list = self.p.test_mask_pos_list if split == 'test' else self.p.train_mask_pos_list
            if split == 'train':
                gt_dicts = [self.gt_dict_all, self.gt_dict_split['train']]
            else:
                gt_dicts = [self.gt_dict_all, self.gt_dict_split['inference']]
            # return: {mask_pos: List of NaryInstances}
            self.data[split+ '_ent'], self.data[split+ '_rel'] = self.generate_data(split, self.nary_hypers, mask_pos_list, gt_dicts)
        print('Data loading finished!')

        #* Print data information
        for mask_pos in self.p.train_mask_pos_list:
            print('MASK ' +str(mask_pos)+' Feature Statistics:')
            print('Number of training mask features(ent):', len(self.data['train_ent'][mask_pos]))
            print('Number of training mask features(rel):', len(self.data['train_rel'][mask_pos]))
        for mask_pos in self.p.test_mask_pos_list:
            print('MASK ' +str(mask_pos)+' Feature Statistics:')
            print('Number of valid mask features(ent):', len(self.data['valid_ent'][mask_pos]))
            print('Number of valid mask features(rel):', len(self.data['valid_rel'][mask_pos]))
            print('Number of test mask features(ent):', len(self.data['test_ent'][mask_pos]))
            print('Number of test mask features(rel):', len(self.data['test_rel'][mask_pos]))
        del raw_examples, raw_reverse_examples
        
    def read_datasets(self, data_dir):
        '''
        Load data from data_dir according to the specific dataset and split_list
        Args:
            data_dir: path to data
        Returns:
            raw_examples:
            {
            'train': List of nary examples
            ...
            }
            raw_reverse_examples (add reverse to main relation):
            {
            'train': List of nary examples
            ...
            }

        '''
        self.read_examples = read_examples_inductive


        raw_examples = {}
        raw_reverse_examples = {}
        for split in self.p.load_split_list :
            split_raw, split_raw_reverse = self.read_examples(data_dir, split_type = split, version = self.p.version, param = self.p)
            raw_examples[split] = split_raw
            raw_reverse_examples[split] = split_raw_reverse
            print('Number of '+ split +' examples:', len(raw_examples[split]))

        
        #generate distribution graph
        examples_all = []
        for split in self.p.load_split_list :
            examples_all.extend(raw_examples[split])
        disbution = collections.Counter([example.arity for example in examples_all])
        sorted_disbution = sorted(disbution.items(), key=lambda x: x[0])
        print('Arity distribution:', sorted_disbution)
        print('Arity distribution(%):', [(item[0], round((100*item[1])/len(examples_all), 2)) for item in sorted_disbution])
        print('Average arity:', sum([example.arity for example in examples_all])/len(examples_all))
        self.generate_distrubution_graph(disbution)

        return raw_examples, raw_reverse_examples
    
    def generate_distrubution_graph(self, disbution):
        '''
        Generate a histogram graph to show the arity distribution
        '''
        import matplotlib.pyplot as plt
        plt.bar(range(len(disbution)), list(disbution.values()), align='center')
        plt.xticks(range(len(disbution)), list(disbution.keys()))
        #save as png
        #plt.savefig(self.p.dataset + '_arity_distribution.png')

    def get_node_neighbor(self, nary_hypers_split):
        '''
        Generate node neighbor dict for each node, key is node_id, value is a list of h_edge_id in nary_hypers_split
        '''
        node_neighbor = collections.defaultdict(list)
        for e_index, hypere in enumerate(nary_hypers_split):
            tokens = hypere.hyper_seq
            for index in range(len(tokens)):
                if index % 2 == 0 and tokens[index] != 0:
                    node_neighbor[tokens[index]].append(e_index)
        return node_neighbor



    def turn_example_to_hedge(self, split_examples, vocab_class, add_hyperqualifier = True):
        '''
        Convert a set of NaryExample into NaryHyperE
        Args:
            examples: List of NaryExample
                        NaryExample(
                        arity,
                        relation,
                        head,
                        tail,
                        auxiliary_info)
            vocab_class: Vocab class, for converting tokens to ids(unified for train/valid/test...)
        Returns:
            all_hedges: {'train': List of NaryHyperE, 'valid': List of NaryHyperE, ...}
            NaryHyperE(
                 split,
                 example_id,
                 arity,
                 input_tokens,
                 hyper_seq,
                 hyper_edge_index,
                 hyper_edge_type,
                 hyper_edge_attr,
                 biedge_index,
                 biedge_type)
        '''
        all_hedges = {} #restore all results
        pad_id = 0
        for split in self.p.load_split_list :
            examples = split_examples[split]
            hyper_edeges_split = []
            for (example_id, example) in enumerate(examples):
                arity = example.arity
                seq_length = 2 * (arity - 2) + 4

                # generate new token sequence by adding reverse relation
                rel = example.relation
                rel_reverse = rel.split('_reverse')[0] if '_reverse' in rel else rel + '_reverse'
                new_token = [example.head, rel, example.tail, rel_reverse]
                if example.auxiliary_info is not None:
                    for attribute in example.auxiliary_info.keys():
                        for value in example.auxiliary_info[attribute]:
                            if len(new_token) < seq_length:
                                new_token.append(value)
                                new_token.append(attribute)


                #! generate bi-edge based on the original tokens
                '''
                Example:
                edge_index:
                [ h t h qe
                  t h qe h]
                edge_type:
                [ r  r_rever  qr  qr]
                '''
                biedge_index = []
                biedge_type = []
                token_ids_biedge = vocab_class.convert_facts_to_tokens_biedge(new_token) #seperately index rel and ent
                biedge_index.extend([[token_ids_biedge[0],token_ids_biedge[2]],[token_ids_biedge[2],token_ids_biedge[0]]])
                biedge_type.extend([token_ids_biedge[1],token_ids_biedge[3]])
                #add qual info to head entity
                for index in range(4, seq_length-1, 2):
                    biedge_index.extend([[token_ids_biedge[0],token_ids_biedge[index]],[token_ids_biedge[index],token_ids_biedge[0]]])
                    biedge_type.extend([token_ids_biedge[index+1],token_ids_biedge[index+1]])


                # convert tokens to ids
                token_ids = vocab_class.convert_facts_to_tokens_hyper(new_token)

                # hyper_edge construction
                hyperG_edge_index = [] #hyperedge index
                hyperG_edge_attr = [] #semantic relation
                for index in range(0, seq_length -1,2):
                    hyperG_edge_index.append([token_ids[index], example_id])
                    hyperG_edge_attr.append(token_ids[index+1])
                '''
                Positional Encoding:
                full: (head:0， tail:1, others: random(2,n-1), rel = ent+n)
                random: entity: random, rel = random+3
                simple: ent:0, rel:3
                '''
                hyperG_edge_type = []
                if self.p.position_mode == 'full':
                    hyperG_edge_type.extend([0,1])
                    if add_hyperqualifier is True:
                        random_list = list(range(2,arity))
                        hyperG_edge_type.extend(random_list)
                elif self.p.position_mode == 'random':
                    # generate a random list of 0,1,2, ... , arity-1
                    random_list = list(range(arity))
                    random.shuffle(random_list)
                    hyperG_edge_type.extend(random_list)
                elif self.p.position_mode in ['simple', 'same']:
                    hyperG_edge_type.extend(arity*[0])
                assert len(hyperG_edge_index) == len(hyperG_edge_type)
                assert len(hyperG_edge_index) == len(hyperG_edge_attr)

                # PADDING to max_seq_length
                arity = example.arity if example.arity < self.p.max_arity else self.p.max_arity
                if len(new_token) > self.p.max_seq_length:
                    new_token = new_token[:self.p.max_seq_length]
                    token_ids = token_ids[:self.p.max_seq_length]
                    seq_length = self.p.max_seq_length
                else:
                    while len(new_token) < self.p.max_seq_length:  # padding to max length
                        new_token.append('[PAD]')
                        token_ids.append(pad_id)
                assert len(new_token) == self.p.max_seq_length

                hyper_e = NaryHyperE(
                 split = split,
                 example_id = example_id,
                 arity = arity,
                 input_tokens  = new_token, #original token (include PAD)
                 hyper_seq  = token_ids, #original tokens' ids (include PAD ID)
                 hyper_edge_index = hyperG_edge_index,
                 hyper_edge_type = hyperG_edge_type,
                 hyper_edge_attr = hyperG_edge_attr,
                 biedge_index = biedge_index,
                 biedge_type = biedge_type)
                assert (len(hyper_e.hyper_edge_index) == len(hyper_e.hyper_edge_type))
                hyper_edeges_split.append(hyper_e)
            all_hedges[split] = hyper_edeges_split
        return all_hedges

    def update_gtdicts(self, input_seqs, mask_position,  label, gt_dicts):
        key = tuple([input_seqs[i] for i in range(len(input_seqs)) if (input_seqs[i] != 0) and (i != mask_position)])
        # not add same feature (when parse valid and train, the same query may be added many times)
        key_is_ingt = False #if key in gt_dict[mask_position] else False
        for gt_dict in gt_dicts:
            if key in gt_dict[mask_position]:
                key_is_ingt = True
            gt_dict[mask_position][key].append(label)
        return key_is_ingt
    
    def generate_data(self, split, nary_hypers, mask_pos_list, gt_dicts):

        '''
        Convert each hyperedge to train/test data by masking one token in the hyperedge
        :param split: train/valid/test
        :param nary_hypers: all hyperedges
        :param mask_pos_list: mask position list
        :param split_all_target: split_all_target
        :param gt_dicts: ground truth dict to update
        '''
        feature_id = 0
        ent_masks = collections.defaultdict(list) #key is position
        rel_masks = collections.defaultdict(list)
        nary_hypers_all = nary_hypers[split]

        for hypere in tqdm(nary_hypers_all, desc="convert data for "+split):
            # analyse tokens and format hypere
            input_tokens = hypere.input_tokens
            token_ids = hypere.hyper_seq

            for mask_position in mask_pos_list:
                if input_tokens[mask_position] == "[PAD]":
                    continue
                else:
                    #no mask relation in this stage
                    mask_type = 0  # mask entity
                    mask_label = token_ids[mask_position]
                    mask_node_id = mask_label
                    input_seqs = token_ids[:]

                    #mask ent in hyper_edge_index(keep edge type and edge attr)
                    hyper_edge_index = hypere.hyper_edge_index[:]
                    hyper_edge_attr = hypere.hyper_edge_attr[:]
                    hyper_edge_type = hypere.hyper_edge_type[:]

                    new_edge_index = []
                    new_edge_type = []
                    new_edge_attr = []
                    found_target = False #ensure that one target only delete once
                    for index in range(len(hyper_edge_index)):
                        source_node, edge_id = hyper_edge_index[index]
                        if found_target or source_node != mask_label:
                            new_edge_index.append([source_node, edge_id])
                            new_edge_type.append(hyper_edge_type[index])
                            new_edge_attr.append(hyper_edge_attr[index])
                        else:
                            # not add target edge, mark the query rel and mask etype
                            query_rel = hyper_edge_attr[index]
                            mask_etype = hyper_edge_type[index]#edge type of the mask position
                            found_target = True
                            
                    if "FI" in self.p.dataset:
                        if split in ['valid', 'test']:
                            mask_label = self.vocab.ent_split_index['inference'].index(mask_label)
                        else:
                            mask_label = self.vocab.ent_split_index[split].index(mask_label)
                    else:
                        mask_label = self.vocab.ent_unit_index.index(mask_label)
                
                    feature = NaryInstance(
                        split = hypere.split,
                        example_id = hypere.example_id,
                        arity = hypere.arity,
                        input_seqs = input_seqs, #with target ent masked

                        hyper_edge_index = new_edge_index, #with target ent masked
                        hyper_edge_type =  new_edge_type,
                        hyper_edge_attr = new_edge_attr,

                        #mask info
                        mask_node_id = mask_node_id,
                        mask_position = mask_position,
                        mask_etype = mask_etype,
                        mask_label = mask_label,
                        mask_type = mask_type,
                        query_rel = query_rel) 
                    
                    key_is_ingt = self.update_gtdicts(input_seqs, mask_position,  mask_label, gt_dicts)
                    if mask_type == 0:
                        feature.input_seqs[mask_position] = self.mask_id 
                        ent_masks[mask_position].append(feature)
                        feature_id += 1
                    else:
                        feature.input_seqs[mask_position] = self.mask_id
                        rel_masks[mask_position].append(feature)
                        feature_id += 1
        return ent_masks, rel_masks