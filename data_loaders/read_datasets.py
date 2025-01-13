from data_loaders.NaryData import NaryExample
import collections
import random


def read_examples_inductive(data_dir, split_type='train', version = None, param = None):
    """
    Read a n-ary json file into a list of NaryExample.
    """
    
    result = []
    result_reverse = []

    if 'WD' in data_dir: # for WD dataset
        if split_type == 'train':
            split_type = 'transductive_train'
        elif split_type == 'valid':
            split_type = 'inductive_val'
        elif split_type == 'test':
            split_type = 'inductive_ts'
        elif split_type == 'inference':
            split_type = 'inductive_train'
        else:
            raise ValueError('split type error')
        f_path = f'{data_dir}statements/{version}/{split_type}.txt'
    else:
        f_path = f'{data_dir}/{split_type}.txt'
        

    with open(f_path, "r") as fr:
        print('load ' + split_type + ' examples')
        lines = fr.readlines()
        if param.explain_fact == False:
            random.shuffle(lines)
        for k in range(0, len(lines)):
            line = lines[k]
            content = line.strip("\n").split(",")
            if len(param.ary_list) != 0:
                content = content[:param.max_seq_length-1]
            else:
                content = content[:-1] #read all content
            arity = 2+ (len(content) - 3)//2
            param.max_arity = max(param.max_arity, arity) #! update max arity
            relation = content[1]
            head = content[0]
            tail =  content[2]


            auxiliary_info = collections.defaultdict(list)
            for i in range(3, len(content)-1, 2):
                q_rel = content[i]
                q_ent = content[i+1]
                auxiliary_info[q_rel].append(q_ent)
            example = NaryExample(
                arity=arity,
                relation=relation,
                head=head,
                tail=tail,
                auxiliary_info=auxiliary_info)
            result.append(example)
            example = NaryExample(
                arity=arity,
                relation=relation + '_reverse',
                head=tail,
                tail=head,
                auxiliary_info=auxiliary_info)
            result_reverse.append(example)
    return result, result_reverse