from argparse import Namespace
import collections

class VocabularyWithFeature(object):
    '''
    Load ent indexes according to the file (if task == 'TR-EF'), and construct vocabularies for ent, rel, and unit.
    '''
    def __init__(self, examples, data_dir, param: Namespace = Namespace()):
        '''
        Construct these variables:
        filter_ent: when constructing ent_split_index and ent_main, filter ent only appears in qual
        rel_vocab: ['PAD'] + all relations + ['MASK']
        ent_vocab: ['PAD'] + all entities + ['MASK']
        unit_vocab: ['[PAD]'] + self.ent_vocab + self.rel_vocab + ['[MASK]', '[CLS]']
        ent_split_index: ent sorted index in each split, eg. {'train': [0,1,2,3, ...], 'inference': [0,1,2,3,4,5,6,7,8,9,10,11,...]}
        ent_main: main ent (not include ent in qual)
        load_ent_index: ent index in pkl file
        '''
        self.p = param
        self.rel_vocab = []
        self.ent_vocab = []
        self.unit_vocab = []

        self.ent_split_index = collections.defaultdict(list)
        self.ent_unit_index = []
        self.ent_main = []
        self.load_ent_index = []
        
        # load meaning
        if self.p.explain_fact:
            meaning_vocab = {}
            with open(data_dir + 'id2word.txt', "r") as fr:
                for line in fr.readlines():
                    word_id, meaning = line.strip().split('\t')
                    meaning_vocab[word_id] = meaning
        
        #read examples in each split
        ent_split = collections.defaultdict(list) #ent in each split
        for split in self.p.load_split_list :
            split_examples = examples[split]
            for example in split_examples:
                ent_split[split].extend([example.head, example.tail])
                self.ent_main.extend([example.head, example.tail])
                self.ent_vocab.extend([example.head, example.tail])
                self.rel_vocab.append(example.relation)
                self.unit_vocab.extend([example.head, example.tail, example.relation])
                #if self.p.use_hyperwalk or self.p.add_reciprocals is False:
                if '_reverse' not in example.relation:
                    self.rel_vocab.append(example.relation + '_reverse')
                    self.unit_vocab.append(example.relation + '_reverse')
                if example.auxiliary_info is not None:
                    for k,v in example.auxiliary_info.items():
                        for each_v in v:
                            self.ent_vocab.append(each_v)
                            self.unit_vocab.append(each_v)
                            if self.p.filter_ent is False:
                                self.ent_main.append(each_v)
                                ent_split[split].append(each_v)
                        self.rel_vocab.append(k)
                        self.unit_vocab.append(k)

        
        if self.p.task == 'TR-EF':
            #load ent_index from file, and put ent in ent_index_file first
            vocab_path = data_dir + 'index.txt'
            ent_vocab_load = []
            with open(vocab_path, "r") as fr:
                for index, line in enumerate(fr.readlines()):
                    ent = line.strip()
                    if ent in self.ent_vocab:
                        self.load_ent_index.append(index)
                        ent_vocab_load.append(ent)
                print('load ent embedding from file success!!')
            #put ent in ent_index_file first
            self.ent_vocab = ent_vocab_load + sorted(list(set(self.ent_vocab) - set(ent_vocab_load)))
        else:
            self.ent_vocab = sorted(list(set(self.ent_vocab)))
        self.rel_vocab = sorted(list(set(self.rel_vocab)))
        self.p.num_ent = len(self.ent_vocab)
        self.p.num_rel = len(self.rel_vocab)
        

        if self.p.unit_encode is True:
            #combine ent_vocab and rel_vocab to unit_vocab
            self.unit_vocab = ['[PAD]'] + self.ent_vocab + self.rel_vocab + ['[MASK]', '[CLS]']
            self.unit_vocab = {value: index for index, value in enumerate(self.unit_vocab)}
            if self.p.explain_fact:
                self.id_meaning = {}
                for key, word_id in self.unit_vocab.items():
                    if key not in ['[PAD]', '[MASK]', '[CLS]'] and key in meaning_vocab:
                        self.id_meaning[word_id] = meaning_vocab[key]
                    elif key.endswith('_reverse') and key[:-8] in meaning_vocab:
                        self.id_meaning[word_id] = meaning_vocab[key[:-8]] + '_reverse'
                    else:
                        self.id_meaning[word_id] = key

        print('num_ent: ', self.p.num_ent, 'num_rel: ', self.p.num_rel, 'num_unit: ', len(self.unit_vocab))


        #add ['PAD'] and ['MASK']
        self.ent_vocab = ['[PAD]'] + self.ent_vocab + ['[MASK]']
        self.rel_vocab = ['[PAD]'] + self.rel_vocab + ['[MASK]']
        self.ent_vocab = {value: index for index, value in enumerate(self.ent_vocab)}
        self.rel_vocab = {value: index for index, value in enumerate(self.rel_vocab)}
        self.ent_main = sorted(list(set([self.ent_vocab[i] for i in self.ent_main])))

        for split in self.p.load_split_list :
            for item in ent_split[split]:
                self.ent_unit_index.append(self.unit_vocab[item] if self.p.unit_encode else self.ent_vocab[item])
                self.ent_split_index[split].append(self.unit_vocab[item] if self.p.unit_encode else self.ent_vocab[item])
            self.ent_split_index[split] = sorted(list(set(self.ent_split_index[split])))
        self.ent_split_index['inference'] = sorted(list(set(self.ent_split_index['valid'] + self.ent_split_index['test'] + self.ent_split_index['inference'])))
        self.ent_unit_index = sorted(list(set(self.ent_unit_index))) # not include '[PAD]', sort according to ent index
        del ent_split


    def convert_by_vocab(self, vocab, items):
        """
        Convert a sequence of [tokens|ids] using the vocab.
        """
        output = []
        for item in items:
            output.append(vocab[item])
        return output

    def convert_tokens_to_ids(self, vocabs, tokens):
        return self.convert_by_vocab(vocabs, tokens)

    def convert_facts_to_tokens_hyper(self, tokens):
        """
        Convert tokens in a nary facts using the vocab.
        """
        output = []
        if self.p.unit_encode is True:
            for i in range(0,len(tokens)):
                output.append(self.unit_vocab[tokens[i]])
            return output
        else:
            for i in range(0,len(tokens)-1,2):
                output.append(self.ent_vocab[tokens[i]])
                output.append(self.rel_vocab[tokens[i+1]])
            if len(tokens) %2 == 1:
                output.append(self.ent_vocab[tokens[-1]])
            return output

    def convert_facts_to_tokens_biedge(self, tokens):
        """
        Convert a fact using the vocab. (for bi-edge)
        """
        output = []
        for i in range(0, len(tokens) - 1, 2):
            output.append(self.ent_vocab[tokens[i]])
            output.append(self.rel_vocab[tokens[i + 1]])
        return output
    