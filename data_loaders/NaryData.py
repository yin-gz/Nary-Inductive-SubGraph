'''
This file defines the data structure for N-ary dataset.
NaryExample: a single training/test example(fact) of n-ary fact.
NaryHyperE: a single set of features of an edge in N-ary semantic hypergraph.
NaryInstance: a single NaryInstance masked for train/valid/test.
'''

class NaryExample(object):
    """
    A single training/test example(fact) of n-ary fact. (read file and save as NaryExample)
    """
    def __init__(self,
                 arity,
                 relation,
                 head,
                 tail,
                 auxiliary_info=None):
        """
        Construct NaryExample.

        Args:
            arity (mandatory): arity of a given fact
            relation (mandatory): primary relation
            head (mandatory): primary head entity (subject)
            tail (mandatory): primary tail entity (object)
            auxiliary_info (optional): auxiliary attribute-value pairs,
                with attributes and values sorted in alphabetical order
        """
        self.arity = arity
        self.relation = relation
        self.head = head
        self.tail = tail
        self.auxiliary_info = auxiliary_info

class NaryHyperE(object):
    """
    A single set of features of HyperE.
    """
    def __init__(self,
                 split,
                 example_id,
                 arity,
                 input_tokens,
                 hyper_seq,
                 hyper_edge_index,
                 hyper_edge_type,
                 hyper_edge_attr,
                 biedge_index,
                 biedge_type):
        """
        Construct NaryHyperE.

        Args:
            split: train/dev/test
            example_id: corresponding example id in each split
            arity: arity of the corresponding example
            input_tokens: input sequence of tokens (include padding)
            hyper_seq: sequences in hyperedge
            hyper_edge_index: [ent, hyper_edge_id]
            hyper_edge_type: the same length as hyper_edge_index
            hyper_edge_attr: the same length as hyper_edge_index
            biedge_index: [2, n_bi_edge]:　relations between nodes
            biedge_type: the same length as biedge_index, indicates the relation type
        """
        self.split = split
        self.example_id = example_id
        self.arity = arity
        self.input_tokens = input_tokens
        self.hyper_seq = hyper_seq
        self.hyper_edge_index = hyper_edge_index
        self.hyper_edge_type =  hyper_edge_type
        self.hyper_edge_attr =  hyper_edge_attr
        self.biedge_index = biedge_index
        self.biedge_type = biedge_type

class NaryInstance(object):
    """
    A single NaryInstance masked for train/valid/test
    """
    def __init__(self,
                split,
                example_id,
                arity,
                input_seqs,

                hyper_edge_index,
                hyper_edge_type,
                hyper_edge_attr,

                mask_node_id,
                mask_position,
                mask_etype,
                mask_label,
                mask_type,
                query_rel
                ):

        """
        Args:
            split: train/dev/test
            example_id: corresponding example id
            arity: arity of the corresponding example
            input_seqs: input sequence of ids(with mask)

            mask_node_id: node id of masked token (no reindex)
            mask_position: position of masked token
            mask_etype: edge type of masked token
            mask_label: label of masked token (reindex according to split)
            mask_type: type of masked token, 1 for entities (values) and -1 for relations (attributes)
            query_rel: query relation (only when mask type is ent)
        """
        self.split = split
        self.example_id = example_id
        self.arity = arity
        self.input_seqs = input_seqs
            
        self.hyper_edge_index = hyper_edge_index
        self.hyper_edge_type =  hyper_edge_type
        self.hyper_edge_attr = hyper_edge_attr

        #mask info
        self.mask_node_id = mask_node_id
        self.mask_position = mask_position
        self.mask_etype = mask_etype
        self.mask_label = mask_label
        self.mask_type = mask_type
        self.query_rel = query_rel
