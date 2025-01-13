import collections
import itertools
from data_loaders.data_loader import DataFacts
from data_loaders.sample_batch_pair import PairSubGDataset
from data_loaders.sample_batch_transfer import BatchData, construct_batch_graph
from torch_geometric.loader import DataLoader
from model.transfer_model import *
from tqdm import tqdm
import os
import time
import wandb
from utils.eval_metrics import *
from torch_geometric.data import Batch
import pickle
from model.pair_model import NPairBase
from model.StarE import StarEModel
import numpy as np
from sklearn import metrics
from torch.cuda import amp


class ExperimentPair:
    def __init__(self,param):
        self.p = param
        self.enable_amp = True if ("cuda" in self.p.device.type and self.p.use_fp16) else False
        self.scaler = amp.GradScaler(enabled=self.enable_amp)    

        # load data and vocab, store them in self.d
        self.d = self.load_data(self.p.dataset)
        self.target_all = self.d.vocab.ent_main if self.p.filter_ent else None
        self.ent_pkl = None
        
        # format to batch
        self.data_loader = {}
        if 'main_ent' or 'q_ent' in self.p.mask_type_list:
            self.data_loader['train_ent'] = self.get_data_loader(self.d.data, 'train', self.p.batch_size, mask_type = 'ent')
            self.data_loader['valid_ent'] = self.get_data_loader(self.d.data, 'valid', self.p.batch_size, mask_type = 'ent')
            self.data_loader['test_ent'] = self.get_data_loader(self.d.data, 'test', self.p.eval_batch_size, mask_type = 'ent') 
        if 'main_rel' or 'q_rel' in self.p.mask_type_list:
            self.data_loader['train_rel'] = self.get_data_loader(self.d.data, 'train', self.p.batch_size, mask_type = 'rel')
            self.data_loader['valid_rel'] = self.get_data_loader(self.d.data, 'valid', self.p.batch_size, mask_type = 'rel')
            self.data_loader['test_rel'] = self.get_data_loader(self.d.data, 'test', self.p.eval_batch_size, mask_type = 'rel')

        # select model          
        if self.p.model_name in ['HART', 'HyperAggModel']:
            self.model = NPairBase(param=self.p)
        else:
            print('Invalid model for local subgraph reasoning')
            return
        self.model.to(self.p.device)
            
        # set optimizer
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

    def get_data_loader(self, all_facts, split, batch_size, shuffle=True, mask_type = 'ent'):
        if split == 'train':
            known_graph = self.d.train_hyperG
        else:
            known_graph = self.d.inference_hyperG
    
        split_facts = all_facts[split+'_'+mask_type]
        merge_pos_facts = []
        #merge all positive facts in different positions
        for pos, pos_facts in split_facts.items():
            merge_pos_facts.extend(pos_facts)
        dataset = PairSubGDataset(self.p.data_path + "/" + self.p.dataset, known_graph, merge_pos_facts, params=self.p, split = split)
        result = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=max(0, self.p.num_workers)
        )
        return result

    def batch_to_device(self, batch, device):
        #if batch is pyG data
        if isinstance(batch, Batch):
            batch = batch.to(device)
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
        best_metric = {'valid_auc_pr': -1, 'test_qauc_pr': -1, 'test_qauc_roc': -1, 'test_pauc_pr': -1, 'test_pauc_roc': -1}
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
                train_iter = iter(self.data_loader['train_'+predict_type])
                loader_length = len(train_iter) #number of batches in one epoch

                #* Each Step
                with tqdm(range(0, loader_length), desc= 'train_'+predict_type+':') as t:
                    for i in t:
                        batch_data = next(train_iter)
                        batch_data =self.batch_to_device(batch_data, self.p.device)
                        with amp.autocast(enabled=self.enable_amp):
                            pos_score, neg_score = self.model.forward(batch_data = batch_data)
                        
                        #* Calcualte loss and optimize parameters
                        loss = self.model.loss(pos_score, neg_score)
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
                        del batch_data, loss

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

                #* Evaluate on valid and test splits after each epoch
                if epoch <= self.p.num_iterations:
                    self.model.eval()
                    eval_result_all = {}
                    with torch.no_grad():
                        for eval_split in self.p.eval_split_list:
                            if len(self.d.data[eval_split+ '_' + predict_type]) != 0:
                                print('\n ~~~~~~~~~~~~~ Eval on '+ eval_split +'~~~~~~~~~~~~~~~~')
                                start_eval = time.time()
                                eval_result = self.evaluate(eval_split)
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

                        
                        #update the best metric
                        v_auc = eval_result_all['valid']['all_ent']['auc_pr']
                        if v_auc >= best_metric['valid_auc_pr']:
                            best_metric['valid_auc_pr'] = v_auc
                            if 'q_ent' in eval_result_all['test'].keys():
                                best_metric['test_qauc_pr'] =  eval_result_all['test']['q_ent']['auc_pr']
                                best_metric['test_qauc_roc'] = eval_result_all['test']['q_ent']['auc_roc']
                            if 'p_ent' in eval_result_all['test'].keys():
                                best_metric['test_pauc_pr'] =  eval_result_all['test']['p_ent']['auc_pr']
                                best_metric['test_pauc_roc'] = eval_result_all['test']['p_ent']['auc_roc']
                            if 'all_ent' in eval_result_all['test'].keys():
                                best_metric['test_auc_pr'] =  eval_result_all['test']['all_ent']['auc_pr']
                                best_metric['test_auc_roc'] = eval_result_all['test']['all_ent']['auc_roc']
                            print('Epoch=%d,  Valid auc_pr increases!' % epoch)
                            self.best_epoch = epoch
                            self.best_val = best_metric['valid_auc_pr']
                            self.save_model(save_path)
                        else:
                            print('Valid auc_pr didnt increase, Best_auc_pr=%f' % (best_metric['test_auc_pr']))

                            
                        # mark the best value for the current epoch
                        if self.p.use_wandb:
                            for k,v in best_metric.items():
                                wandb.log({f"best/{k}":v ,"epoch": epoch}) 
                            wandb.log({f"best/it":self.best_epoch ,"epoch": epoch})                            
    @torch.no_grad()
    def evaluate(self, split):
        result_all = {'auc_roc':0.0, 'auc_pr':0.0} # for all entity
        result_p = {'auc_roc':0.0, 'auc_pr':0.0} # for primary entity
        result_q = {'auc_roc':0.0, 'auc_pr':0.0} # for qual entity
        test_iter = iter(self.data_loader[split+'_ent'])
        loader_length = len(test_iter)
        p_length = 0
        q_length = 0
        with tqdm(range(0, loader_length),desc = 'eval_'+split+' '+ 'ent') as t:
            for i in t:
                batch_data = next(test_iter)
                batch_mask_position = batch_data['mask_position']
                batch_data = self.batch_to_device(batch_data, self.p.device)
                with amp.autocast(enabled=self.enable_amp):
                    pos_score, neg_score = self.model.forward(batch_data) #[batch]
                
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
                    pos_score_p = pos_score[primary_index]
                    neg_score_p = neg_score[primary_index]
                    p_length += 1
                    result_p = self.calculate_auc(pos_score_p, neg_score_p, result_p)
                    result_all = self.calculate_auc(pos_score_p, neg_score_p, result_all)
                    
                qual_index = qual_pos_bool.nonzero().squeeze(1)
                if qual_index.size() == torch.Size([]): #scalar tensor with only one element
                    qual_index = qual_index.unsqueeze(0)
                if qual_index.size(0) != 0:
                    pos_score_q = pos_score[qual_index]
                    neg_score_q = neg_score[qual_index]
                    q_length += 1
                    result_q = self.calculate_auc(pos_score_q, neg_score_q, result_q)
                    result_all = self.calculate_auc(pos_score_q, neg_score_q, result_all)
                
        if loader_length != 0:
            result_all['auc_roc'] /= (2*loader_length)
            result_all['auc_pr'] /= (2*loader_length)
        if p_length != 0:
            result_p['auc_roc'] /= p_length
            result_p['auc_pr'] /= p_length
        if q_length != 0:
            result_q['auc_roc'] /= q_length
            result_q['auc_pr'] /= q_length

        result = {'all_ent':result_all, 'p_ent':result_p, 'q_ent':result_q}
        return result
    
    def calculate_auc(self, pos_score, neg_score, result):
        #calcule auc_roc, auc-pr
        pos_score = pos_score.cpu().numpy()
        neg_score = neg_score.cpu().numpy()
        y_true = np.concatenate((np.ones(pos_score.shape[0]), np.zeros(neg_score.shape[0])))
        y_score = np.concatenate((pos_score, neg_score))

        auc_roc = metrics.roc_auc_score(y_true, y_score)
        auc_pr = metrics.average_precision_score(y_true, y_score)
        result['auc_roc'] += auc_roc
        result['auc_pr'] += auc_pr
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
                print("The last layer attention of the fact:")
                for index,node_id in enumerate(predictE_node):
                    if self.d.vocab.id_meaning[node_id] == '[PAD]':
                        continue
                    print(self.d.vocab.id_meaning[node_id], ':', outE_ent_att[index])
                # the nested attention scores of entities for the source hyperedge
                # sort and return the sorted score and index of node_weights
                sorted_weights, original_indices = torch.sort(node_weights, descending=True)
                print("---------------------------------------------------")
                print("Nested attention scores of Top20 entities:")
                for k in range(20):
                    print(self.d.vocab.id_meaning[original_indices[k].item()], ':', sorted_weights[k].item())