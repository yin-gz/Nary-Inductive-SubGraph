import collections
import itertools
#from data_loaders.data_loader import DataFacts
from data_loaders.data_loader import DataFacts
from data_loaders.sample_batch_transfer import BatchData, construct_batch_graph
from torch_geometric.data import DataLoader as PyGDataset
from torch.utils.data import DataLoader
from model.transfer_model import *
from tqdm import tqdm
from torch.cuda import amp
import os
import time
import wandb
from utils.eval_metrics import *
from torch_geometric.data import Batch
import pickle
from model.StarE import StarEModel
from model.transfer_model import NTransBase
from model.GRAN import GRAN
from model.BLP import BLP
import numpy as np


class ExperimentTransfer:
    '''
    Task: Train on a graph, evalute on another distinct inference graph
    '''
    def __init__(self, param):
        self.p = param
        self.enable_amp = True if ("cuda" in self.p.device.type and self.p.use_fp16) else False
        self.scaler = amp.GradScaler(enabled=self.enable_amp)

        # load data and vocab, store them in self.d
        self.d = self.load_data(self.p.dataset)
        self.target_all = self.d.vocab.ent_main if self.p.filter_ent else None
        if self.p.task == 'TR-EF':
            with open(self.p.data_path + "/%s/" % self.p.dataset + "embs.pkl", 'rb') as f:
                with torch.no_grad():
                    load_ent_index = torch.LongTensor(self.d.vocab.load_ent_index)
                    self.ent_pkl = torch.Tensor(pickle.load(f))[load_ent_index]
                    print(f"Loaded embeddings of shape {self.ent_pkl.shape}")
        else:
            self.ent_pkl = None
        
        # format to batch
        self.data_iter = {}
        if 'main_ent' or 'q_ent' in self.p.mask_type_list:
            self.data_iter['train_ent'] = self.get_data_loader(self.d.data, 'train', self.p.batch_size, mask_type = 'ent')
            self.data_iter['valid_ent'] = self.get_data_loader(self.d.data, 'valid', self.p.batch_size, mask_type = 'ent')
            self.data_iter['test_ent'] = self.get_data_loader(self.d.data, 'test', self.p.eval_batch_size, mask_type = 'ent') 
        if 'main_rel' or 'q_rel' in self.p.mask_type_list:
            self.data_iter['train_rel'] = self.get_data_loader(self.d.data, 'train', self.p.batch_size, mask_type = 'rel')
            self.data_iter['valid_rel'] = self.get_data_loader(self.d.data, 'valid', self.p.batch_size, mask_type = 'rel')
            self.data_iter['test_rel'] = self.get_data_loader(self.d.data, 'test', self.p.eval_batch_size, mask_type = 'rel')


        #select mnodel
        if self.p.model_name == 'StarE':
            self.model = StarEModel(self.ent_pkl, param=self.p)
        elif self.p.model_name == 'BLP':
            self.model = BLP(self.ent_pkl, param=self.p)
        else:
            #N-ary semantic hypergraph-based model and QBLP, GRAN
            self.model = NTransBase(self.ent_pkl, param=self.p)
        self.model.to(self.p.device)

        #optimizer
        self.parameters = self.model.parameters()
        self.opt = torch.optim.AdamW(self.parameters, lr=self.p.lr, weight_decay=0.01)
        if self.p.dr_type == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode="min", patience=10, factor=0.9, min_lr=0.0001, verbose=False)
        elif self.p.dr_type == 'stepLR':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=50, gamma=self.p.dr_rate)
        elif self.p.dr_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.p.max_epochs * self.p.batch_size, verbose=False)
        elif self.p.dr_type == 'onecycle':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.opt, max_lr = 0.01, pct_start=0.1, anneal_strategy='linear', total_steps=self.p.max_epochs * self.p.batch_size + 1)
        else:
            self.scheduler = None

    def load_data(self, dataset):
        data_dir = self.p.data_path + "/%s/" % dataset
        d = DataFacts(data_dir=data_dir, param =self.p)
        return d

    def get_data_loader(self, all_facts, split, batch_size, shuffle=False, mask_type = 'ent'):
        '''
        Get data loader for each split, each load will return a merged batch of subgraphs or a batch of facts
        '''
        if "FI_" in self.p.dataset:
            if split == 'train': #for train
                gt_dict = self.d.gt_dict_split['train']
                target_index = self.d.vocab.ent_split_index['train']
                known_graph = self.d.train_hyperG
            else: #for fully inductive valid/test
                gt_dict = self.d.gt_dict_split['inference']
                target_index= self.d.vocab.ent_split_index['inference']
                known_graph = self.d.inference_hyperG
        else:
            gt_dict = self.d.gt_dict_all
            target_index = self.d.vocab.ent_unit_index
            known_graph = self.d.train_hyperG


        split_facts = all_facts[split+'_'+mask_type]
        result = {}
        for pos, pos_facts in split_facts.items():
            dataset_class = BatchData(pos_facts, self.p, gt_dict, target_index, known_graph, split)
            result[pos] = DataLoader(
                dataset_class,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=max(0, self.p.num_workers),
                collate_fn=dataset_class.collate_fn
            )
        return result

    def batch_to_device(self, batch, device):
        #if batch is pyG data
        if isinstance(batch, Batch):
            batch = batch.to(device) #type: ignore
            return batch
        else:
            for key, data in batch.items():
                if isinstance(data, list):
                    batch[key] = [d.to(device) for d in data]
                elif isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(v, torch.Tensor):
                            batch[key][k] = v.to(device)
                        elif isinstance(v, list):
                            batch[key][k] = [d.to(device) for d in v]
                else:
                    batch[key] = data.to(device)
            return batch

    #@profile
    def train_and_eval(self):
        self.initial_epoch = 0
        save_path = os.path.join('./checkpoints', self.p.store_name.replace(':', ''))

        # load
        if self.p.restore is not None:
            self.load_model(save_path)
            print('Successfully Loaded previous model')
            #self.evaluate_groupby()

        print('Training the model...')
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')
        print('Training Starts...')

        train_losses = [] #average loss of each epoch
        train_time = []
        best_valid_iter = 0
        best_metric = {'valid_mrr': 0, 'test_mrr': 0, 'test_hit10': 0, 'test_hit5': 0, 'test_hit1': 0, 'test_qamrr': 0, 'test_qmrr': 0, 'test_qhit10': 0, 'test_qhit5': 0, 'test_qhit1': 0, 'test_pmrr': 0, 'test_phit10': 0, 'test_phit5': 0, 'test_phit1': 0, 'test_amrr': 0, 'test_mrr': 0, 'test_hit10': 0, 'test_hit5': 0, 'test_hit1': 0}
        self.best_val = 0 # main metric for val (eg. mrr)
        self.best_epoch = 0


        for epoch in range(self.initial_epoch+1, self.p.num_iterations + 1):
            #first train based on entity masked data, then train based on relation masked data (if exists)
            train_type = []
            if 'main_ent' in self.p.mask_type_list or 'q_ent' in self.p.mask_type_list:
                train_type.append('ent')
            if 'main_rel' in self.p.mask_type_list or 'q_rel' in self.p.mask_type_list:
                train_type.append('rel')
            for predict_type in train_type:
                self.model.train()
                print('\nEpoch %d starts training...' % epoch)
                per_epoch_loss = [] #loss of each step
                start_train = time.time()
                train_iter = itertools.chain()
                loader_length = 0
                
                ##* Construct data_loader: each epoch, reinitialize the train_iter
                for mask_pos, data_loader in self.data_iter['train_' + predict_type].items():
                    loader_length += len(data_loader) #number of batches in one epoch
                    train_iter = itertools.chain.from_iterable([train_iter, iter(data_loader)])

                #* Each Step
                with tqdm(range(0, loader_length), desc= 'train_' + predict_type+':') as t:
                    for i in t:
                        batch_data = next(train_iter)
                        if self.p.use_subg:
                            try:
                                batch_data = construct_batch_graph('train', self.d.train_hyperG, batch_data, self.p)
                            except Exception as e:
                                print(':', e)
                                continue
                        batch_data = self.batch_to_device(batch_data, self.p.device)

                        #* Model Forward
                        with amp.autocast(enabled=self.enable_amp):
                            if 'FI_' in self.p.dataset:
                                target_index = self.d.vocab.ent_split_index['train']
                            else:
                                target_index = self.d.vocab.ent_unit_index
                            base_graph = self.d.train_hyperG
                            bigraph = self.d.train_bigraph
                            pred = self.model.forward(predict_type, batch_data, target_index, bigraph =bigraph, base_graph = base_graph, mode = 'train')
                        
                        #* Calcualte loss and optimize parameters
                        loss = self.model.loss(pred, batch_data)
                        per_epoch_loss.append(loss.detach().item())
                        self.opt.zero_grad()
                        if self.enable_amp:
                            self.scaler.scale(loss).backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            self.scaler.step(self.opt)
                            self.scaler.update()
                        else:
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            self.opt.step()
                        if np.isnan(loss.item()):
                            print('NaN Loss! Break.....')
                            break

                        train_time.append(time.time() - start_train)
                        t.set_postfix({  # average loss in per epoch
                            'trainloss': '{:.6f}'.format(np.mean(per_epoch_loss)),
                            'step': '{:02d}'.format(i)
                        })
                        del batch_data, pred, loss

                print('train time cost=%fs' % train_time[-1])
                print('Epoch %d train, LOSS=%f' % (epoch, np.mean(per_epoch_loss)))
                train_losses.append(per_epoch_loss)

                #* Adjust lr after each epoch
                if self.p.dr_type == 'plateau':
                    self.scheduler.step(np.mean(per_epoch_loss)) 
                elif self.p.dr_type is not None:
                    self.scheduler.step()
                if self.p.use_wandb:
                    if hasattr(self.scheduler, '_last_lr'):
                        wandb.log({"epoch": epoch, "loss": float(np.mean(per_epoch_loss)), "time": train_time[-1],  'lr': float(self.scheduler._last_lr[0])})
                    else:
                        wandb.log({"epoch": epoch, "loss": float(np.mean(per_epoch_loss)), "time": train_time[-1], 'lr': float(self.opt.param_groups[0]['lr'])})

                #* Evaluate on valid after each epoch, and test after each param.eval_step epochs
                if epoch <= self.p.num_iterations:
                    self.model.eval()
                    eval_result_all = {} #keep all metrics in this epoch, {'split':{'ent_type':{'metric_type':...}}}
                    with torch.no_grad():
                        for eval_split in self.p.eval_split_list:
                            if len(self.d.data[eval_split+ '_' + predict_type]) != 0:
                                print('\n ~~~~~~~~~~~~~ Eval on '+ eval_split +'~~~~~~~~~~~~~~~~')
                                start_eval = time.time()
                                #eval_result, pykeen_result = self.evaluate(self.model, eval_split, epoch)
                                eval_result = self.evaluate(self.model, eval_split, epoch)
                                eval_result_all[eval_split] = eval_result
                                eval_time = time.time() - start_eval
                                print('***** Eval time cost:' + str(eval_time))

                                if self.p.use_wandb:
                                    wandb.log({f"{eval_split}_eval/time": eval_time, "epoch": epoch})
                                    for test_type in eval_result.keys():
                                        for metric_type in eval_result[test_type].keys():
                                            wandb.log({f"{eval_split}/{test_type}/{metric_type}": eval_result[test_type][metric_type] ,"epoch": epoch})
                                for test_type in eval_result.keys():
                                    for metric_type in eval_result[test_type].keys():
                                        print({f"{eval_split}/{test_type}/{metric_type}": eval_result[test_type][metric_type] ,"epoch": epoch})


                        # update the best metric
                        v_mrr = eval_result_all['valid']['all_ent']['mrr']
                        if v_mrr >= best_metric['valid_mrr']:
                            best_metric['valid_mrr'] = v_mrr
                            print('Epoch=%d,  Valid MRR increases!' % epoch)
                            self.best_epoch = epoch
                            self.best_val = best_metric['valid_mrr']
                            self.save_model(save_path)
                        else:
                            print('Valid MRR didnt increase, Best valid MRR=%f, Best test MRR=%f' % (best_metric['valid_mrr'], best_metric['test_mrr']))
                    
                        # test when epochs = param.eval_step, 2*param.eval_step, 3*param.eval_step
                        if epoch % self.p.eval_step == 0:
                            test_model = self.load_test_model(save_path)
                            test_model.eval()
                            print('Start to test...')
                            with torch.no_grad():
                                #eval_result, pykeen_result = self.evaluate(test_model,'test', epoch)
                                eval_result = self.evaluate(test_model,'test', epoch)
                                if 'q_ent' in eval_result.keys():
                                    #best_metric['test_qamrr'] = pykeen_result['q_ent']['test.both/adjusted_arithmetic_mean_rank']
                                    best_metric['test_qmrr'] = eval_result['q_ent']['mrr']
                                    best_metric['test_qhit10'] = eval_result['q_ent']['hits@10']
                                    best_metric['test_qhit5'] = eval_result['q_ent']['hits@5']
                                    best_metric['test_qhit1'] = eval_result['q_ent']['hits@1']
                                if 'p_ent' in eval_result.keys():
                                    #best_metric['test_qamrr'] = pykeen_result['p_ent']['test.both/adjusted_arithmetic_mean_rank']
                                    best_metric['test_pmrr'] = eval_result['p_ent']['mrr']
                                    best_metric['test_phit10'] = eval_result['p_ent']['hits@10']
                                    best_metric['test_phit5'] = eval_result['p_ent']['hits@5']
                                    best_metric['test_phit1'] = eval_result['p_ent']['hits@1']
                                if 'all_ent' in eval_result.keys():
                                    #best_metric['test_amrr'] = pykeen_result['all_ent']['test.both/adjusted_arithmetic_mean_rank']
                                    best_metric['test_mrr'] =  eval_result['all_ent']['mrr']
                                    best_metric['test_hit10'] = eval_result['all_ent']['hits@10']
                                    best_metric['test_hit5'] = eval_result['all_ent']['hits@5']
                                    best_metric['test_hit1'] = eval_result['all_ent']['hits@1']
                                    
                            for k,v in best_metric.items():
                                print({f"best/{k}":v ,"epoch": epoch})
                            print({f"best/it":self.best_epoch ,"epoch": epoch})
                        
                        # mark the best value for the current epoch       
                        if self.p.use_wandb:
                            for k,v in best_metric.items():
                                wandb.log({f"best/{k}":v ,"epoch": epoch}) 
                            wandb.log({f"best/it":self.best_epoch ,"epoch": epoch})
                        

            #if epoch %100 == 0:
                #self.evaluate_groupby()


    @torch.no_grad()
    def evaluate(self, model, split, epoch):
        result = {}
        metr_all = {} #for all entity
        metr_p = {} #for only main entity
        metr_q = {} #for only qual entity
        #pykeen_results = {}
        #pykeen_evaluater_all = init_pykeen_metrics()
        #pykeen_evaluater_p = init_pykeen_metrics()
        #pykeen_evaluater_q = init_pykeen_metrics()
        test_type_list = ['ent']

        for test_type in test_type_list:
            test_iter = itertools.chain()
            loader_length = 0
            fact_length = 0
            main_length = 0
            qual_length = 0
            for mask_pos, data_loader in self.data_iter[split+'_' + test_type].items():
                test_iter = itertools.chain.from_iterable([test_iter, iter(data_loader)])
                loader_length += len(data_loader) #number of batches in one epoch
                fact_length += len(data_loader.dataset) #number of facts in one epoch


            with tqdm(range(0, loader_length),desc = 'eval_'+split+' '+test_type) as t:
                for i in t:
                    batch_data = next(test_iter)
                    #if self.p.pair_subg:
                        #pass
                    if split == 'train' and self.p.use_neg:
                        #evaluate on all ent, not the main ent
                        del batch_data["neg_target_index"]
                    if self.p.use_subg:
                        try:
                            if 'FI_' in self.p.dataset and split !='train':
                                batch_data = construct_batch_graph('inference', self.d.inference_hyperG, batch_data, self.p)
                            else:
                                batch_data = construct_batch_graph('train', self.d.train_hyperG, batch_data, self.p)
                        except Exception as e:
                            print('Error in constructing graph:', e)
                            continue
                    batch_data = self.batch_to_device(batch_data, self.p.device)

                    with amp.autocast(enabled=self.enable_amp):
                        if 'FI_' in self.p.dataset:
                            target_index = self.d.vocab.ent_split_index['inference'] if split !='train' else self.d.vocab.ent_split_index['train']
                            base_graph = self.d.inference_hyperG if split !='train' else self.d.train_hyperG
                            bigraph = self.d.inference_bigraph if split !='train' else self.d.train_bigraph
                        else:
                            target_index = self.d.vocab.ent_unit_index
                            base_graph = self.d.train_hyperG
                            bigraph = self.d.train_bigraph
                        pred = model.forward(test_type, batch_data, target_index, bigraph= bigraph, base_graph = base_graph, mode = 'test')
                        pred = pred.float()

                    batch_mask_position = batch_data['batch_mask_position']
                    pos_label = self.p.neg_num * torch.ones_like(batch_mask_position).long()
                    batch_label = batch_data['batch_mask_label'] if "neg_target_index" not in batch_data else pos_label
                    batch_label_all = batch_data['batch_target_all'].view(batch_label.size(0),-1)


                    #* Get the main entity index
                    first_pos_bool = (batch_mask_position == 0)
                    second_pos_bool = (batch_mask_position == 2)#[batch]
                    qual_pos_bool = ~(first_pos_bool | second_pos_bool)
                    first_pos_bool = first_pos_bool.nonzero() #nonzero will add a dimension, [first_num(0), 1]
                    second_pos_bool = second_pos_bool.nonzero() #[second_num, 1]

                    primary = True
                    if first_pos_bool.size(0) == 0 and second_pos_bool.size(0) == 0:
                        primary = False
                    elif first_pos_bool.size(0) == 0:
                        primary_index = second_pos_bool.squeeze(1)
                    elif second_pos_bool.size(0) == 0:
                        primary_index = first_pos_bool.squeeze(1)
                    else:
                        primary_index = torch.cat((first_pos_bool,second_pos_bool), 0).squeeze(1) #[first_num + second_num in batch]
                    if primary:
                        if primary_index.size() == torch.Size([]):
                            primary_index = primary_index.unsqueeze(0)
                        pred_p = pred[primary_index]
                        batch_label_p = batch_label[primary_index]
                        batch_label_all_p = batch_label_all[primary_index]
                        metr_p = compute(pred_p, batch_label_p, batch_label_all_p, metr_p, split)
                        main_length += pred_p.size(0)

                    qual_index = qual_pos_bool.nonzero().squeeze(1)
                    if qual_index.size() == torch.Size([]): #scalar tensor with only one element
                        qual_index = qual_index.unsqueeze(0)
                    if qual_index.size(0) != 0:
                        pred_q = pred[qual_index]
                        batch_label_q = batch_label[qual_index]
                        batch_label_all_q = batch_label_all[qual_index]
                        metr_q = compute(pred_q, batch_label_q, batch_label_all_q, metr_q, split)
                        qual_length += pred_q.size(0)

                    #pred: [batch, all_target]
                    metr_all = compute(pred, batch_label, batch_label_all, metr_all, split)
                    #evaluate_pykeen(pykeen_evaluater_all, pred, batch_label, batch_label_all, epoch, split, target = 'tail')
                    del batch_data, pred
            
            if qual_length != 0:
                metrics_q = summarize_metrics(metr_q, qual_length)
                result['q_' + test_type] = metrics_q # qual ent/rel
                #evaluate_pykeen(pykeen_evaluater_q, pred_q, batch_label_q, batch_label_all_q, epoch, split, target = 'tail')
                #pykeen_results['q_' + test_type] = average_pykeen_metrics(pykeen_evaluater_q, epoch, split, self.p.use_wandb)
            if main_length != 0:
                metrics_p = summarize_metrics(metr_p, main_length)
                result['p_' + test_type] = metrics_p # main ent/rel
                #evaluate_pykeen(pykeen_evaluater_p, pred_p, batch_label_p, batch_label_all_p, epoch, split, target = 'tail')
                #pykeen_results['p_' + test_type] = average_pykeen_metrics(pykeen_evaluater_p, epoch, split, self.p.use_wandb)

            metrics_all = summarize_metrics(metr_all, fact_length)
            result['all_' + test_type] = metrics_all # all refers to all the ent/rel           
            #pykeen_results['all_' + test_type] = average_pykeen_metrics(pykeen_evaluater_all, epoch, split, self.p.use_wandb)

        #return result, pykeen_results
        return result
    
    def save_model(self, save_path):
        state = {
            'state_dict': self.model.state_dict(),
            'best_val': self.best_val,
            'best_epoch': self.best_epoch,
            'optimizer': self.opt.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
            'args': vars(self.p)
        }
        torch.save(state, save_path)

    def load_model(self, load_path):
        state = torch.load(load_path)
        state_dict = state['state_dict']
        self.best_val = state['best_val']
        self.initial_epoch = state['best_epoch']

        self.model.load_state_dict(state_dict)
        self.opt.load_state_dict(state['optimizer'])
        if self.p.dr_type is not None:
            self.scheduler.load_state_dict(state["scheduler"])
            
    def load_test_model(self, load_path):
        state = torch.load(load_path)
        state_dict = state['state_dict']
        #construct a new model with the same parameters as self.model
        if self.p.model_name == 'StarE':
            model = StarEModel(self.ent_pkl, param=self.p)
        elif self.p.model_name == 'BLP':
            model = BLP(self.ent_pkl, param=self.p)
        else:
            #N-ary semantic hypergraph-based model
            model = NTransBase(self.ent_pkl, param=self.p)
        model.to(self.p.device)       
        model.load_state_dict(state_dict)
        return model
    
    @torch.no_grad()
    def fact_explain(self, fact_id = 0):
        save_path = os.path.join('./checkpoints', self.p.store_name.replace(':', ''))
        # load model
        if self.p.restore is not None:
            self.load_model(save_path)
            print('Successfully Loaded previous model')

        #load facts
        #facts = [self.d.data['test_ent'][2][fact_id]] #2 denotes pos
        facts = []
        for each_fact in fact_id:
            facts = [self.d.data['inference_ent'][2][each_fact]]

            #generate data loader
            gt_dict = self.d.gt_dict_split['inference']
            target_index= self.d.vocab.ent_split_index['inference']
            known_graph = self.d.inference_hyperG
            dataset_class = BatchData(facts, self.p, gt_dict, target_index, known_graph, 'test')
            test_loader = DataLoader(
                dataset_class,
                batch_size=len(facts),
                shuffle=False,
                num_workers=max(0, self.p.num_workers),
                collate_fn=dataset_class.collate_fn
            )

            #get batch data
            batch_data = next(iter(test_loader))
            batch_data = construct_batch_graph('inference', self.d.inference_hyperG, batch_data, self.p)
            batch_data = self.batch_to_device(batch_data, self.p.device)

            with amp.autocast(enabled=self.enable_amp):
                target_index = self.d.vocab.ent_split_index['inference'] 
                base_graph = self.d.inference_hyperG
                bigraph = self.d.inference_bigraph

                h_out, node_weights, predictE_node, outE_ent_att = self.model.forward('ent', batch_data, target_index, bigraph= bigraph, base_graph = base_graph, mode = 'test')
                #turn predictE_node to id
                predictE_node= predictE_node[0].tolist()
                outE_ent_att = outE_ent_att.tolist()
                print("***********************************************************************")
                print("---------The last layer attention of the fact:---------")
                for index,node_id in enumerate(predictE_node):
                    if self.d.vocab.id_meaning[node_id] == '[PAD]':
                        continue
                    print(self.d.vocab.id_meaning[node_id], ':', outE_ent_att[index])
                # the nested attention scores of entities for the source hyperedge
                # sort and return the sorted score and index of node_weights
                sorted_weights, original_indices = torch.sort(node_weights, descending=True)
                print("---------Nested attention scores of Top20 Entities and Relations:---------")
                for k in range(20):
                    print(self.d.vocab.id_meaning[original_indices[k].item()], ':', sorted_weights[k].item())
                print("\n")