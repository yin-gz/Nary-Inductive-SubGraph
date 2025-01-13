import numpy as np
import torch
import argparse
import time
import random
import wandb
import yaml
from loops_transfer import ExperimentTransfer
from loops_pair import ExperimentPair
from torch.cuda import amp
import os



if __name__ == "__main__":
    parser = argparse.ArgumentParser()   

    #! Task General Setting
    parser.add_argument("-use_wandb", type=bool, default=False, help="Use wandb or not")
    parser.add_argument("-config_file", type=bool, default=True, help="Use parameters in config file or not")
    parser.add_argument("-data_path", type=str, default="../data/nary",  help="Data path")
    parser.add_argument("-dataset", type=str, default="FI_WD20K100", choices= ["FI_WD20K100", "FI_WD20K66", "FI_MFB15K100","FI_MFB15K33"], 
                        help="Dataset name")
    parser.add_argument("-version", type=str, default="v1", choices= ["v1", "v2"], help="Dataset version (only use in wd20k)")
    parser.add_argument("-task", type=str, default="TR-EF", choices = ["TR-EF", "TR-NEF", "PSR", "full-trans"],help="select tasks")
    parser.add_argument("-ary_list", type=list, default=[2,3,4,5,6,7], action="append", help="List of Ary for Trai   n and and Test, if set to [], use all")
    parser.add_argument("-mask_type_list", type=list, default=["main_ent", "q_ent"], action=  "append", help="Predict for main_ent or q_ent or both")
    parser.add_argument("-load_split_list", type=list, default=["train", "valid", "test", "inference"], action="append", help="Load data from these splits")
    parser.add_argument("-eval_split_list", type=list, default=["train", "valid", "test"], action="append", help=" Evaluate performance in each splits")
    parser.add_argument("-restore", type=str, default=None, help="Restore from the previously saved model")

    #! Model General setting
    parser.add_argument("-model_name", type=str, default="HART", choices=["HART", "HART-Intra", "BLP", "QBLP", "StarE", "GRAN", "HyperAggModel", "SubgTrans"], help="Choose the model")
    parser.add_argument("-hagg_method", type=str, default="HART", choices = ["sage","att","settrans", "HART"], help="Select aggregators in HyperAggModel")
    parser.add_argument("-feature_drop", type=float, default=0.2, choices=[0.1, 0.2, 0.3], help= "Dropout for entities")
    parser.add_argument("-embed_dim", type=int, default=200, help="Embedding dimension to give as input to score function")
    parser.add_argument("-hidden_dim", type=int, default=512, help="Hidden dims in the model")
    parser.add_argument("-trans_layer", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("-num_heads", type=int, default=4, help=" Number of attention heads in the model")
    
    #! Model Training Parameters
    parser.add_argument("-batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("-eval_batch_size", type=int, default=128, help="Batch size when evaluating")
    parser.add_argument("-num_iterations", type=int, default=300, help="Number of iterations")
    parser.add_argument("-lr", type=float, default=5e-4, choices=[1e-5, 5e-5, 1e-4, 5e-4], help="Learning rate")
    parser.add_argument("-dr_type", type=str, default="plateau", help="stepLR/plateau/cosine/onecycle")
    parser.add_argument("-dr_rate", type=float, default=0.95, help="dr_rate for StepLR")
    parser.add_argument("-unit_encode", type=bool, default=True, help="Entity and Relation are encoded uniformity or not, only set True for StarE")
    parser.add_argument("-filter_ent", type=bool, default=False, help="If set true, only consider main ent")
    parser.add_argument("-neg_num", type=int, default= 50, help="If neg_num > 0, use neg_num")
    parser.add_argument("-e_soft_label", type=float, default=0.15, help="Soft label for BCE")
    parser.add_argument("-loss_name", type=str, default="BCE", help="CE/BCE/margin")

    #! Subgraph Sampling Setting
    parser.add_argument("-use_subg", type=bool, default=False, help="Sample subgraph or not, set True for HGNNs")
    parser.add_argument("-max_each_hop", default = 16, type=int, help="Max neighborhood of each hop")
    parser.add_argument("-K_hop", type=int, default= 2, help="K-Hop when aggretgate information from edge to vertex, 0 means its own edge")
    parser.add_argument("-expand_qual", type=bool, default=True, help="Expand subgraph based on qual or only on head/tail")

    #!Semantic-Matching Positional Encoding
    parser.add_argument("-position_mode", type=str, default="random", help="full(main,qual, and their associated rel)/random(entity and associated rel)/simple(ent 0, rel 1)/same(all 0)/None")
    parser.add_argument("-V2E_with_rel", type=bool, default=True, help="Add semantic roles when V2E")
    parser.add_argument("-mark_query", type=str, default="unique", help="Unique(mark query rel using a unique pos)/same/none(no query)")
    parser.add_argument("-max_VinE", type=int, default=32, help=  "Max sequence length for transformer")

    #! For GCN-based models
    parser.add_argument("-n_gcn_layer", type=int, default=2, help="Number of GCN Layers to use")
    parser.add_argument("-drop_gcn_in", type=float, default=0.1, help="Dropout used in GCN Layer Aggegration")
    parser.add_argument("-drop_decoder", type=float, default=0.1, help="Dropout after GCN")
    parser.add_argument("-intermediate_size", type=int, default=128, help="For GRAN")
    
    #! GPU and CPU settingv     
    parser.add_argument("-num_workers", type=int, default=0, help="Number of processes to construct batches")
    parser.add_argument("-gpu", type=str, default="0", help="Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0")
    parser.add_argument("-use_fp16", type=bool, default=True, help="Whether to use fp16 mixed precision.")
    parser.add_argument("-use_flash", type=bool, default= True,  help="Use flash att transformer or not")
    
    #! Others
    parser.add_argument("-binary", type=bool, default=False, help="Construct only binary hyper_edges or not")
    parser.add_argument("-only_test", default=False, action="store_true", help="only test(load previous model) or not")
    parser.add_argument("-explain_fact", default=False, action="store_true", help="explain the fact using nested attention scores")

    param = parser.parse_args()

    # load yaml (yaml priority)
    if param.config_file:
        config_path = "./config/"+ param.task +"/"+ param.dataset + "_" + param.version + ".yaml"
        # judge if config_path exist?
        if os.path.exists(config_path):
            opt = vars(param)
            with open (config_path) as c_file:
                args = yaml.safe_load(c_file)
            opt.update(args)
            param = argparse.Namespace(**opt)
    param.lr = float(param.lr)

    # check fp16
    if param.gpu == "-1":
        param.use_fp16 = False
        param.use_flash = False
    elif param.use_flash:
        param.use_fp16 = True
        # CuDNN must be enabled for FP16 training.
        torch.backends.cudnn.enabled = True
        
    # check parameter in models
    if param.model_name == "StarE":
        param.unit_encode = False
        #param.lr = 0.001
        param.position_mode = "full"
    elif param.model_name == "GRAN":
        param.use_fp16 = False
        param.use_flash = False
        param.use_neg = False
        #param.lr = 0.0001
    elif param.model_name == "HART":
        param.hagg_method = "HART"
        
    # check Negative sample
    if param.neg_num > 0:
        param.use_neg = True
    else:
        param.use_neg = False

    # check subG
    if param.model_name in ["HART", "HyperAggModel", "HART-Intra", "SubgTrans"]:
        param.use_subg = True
        if param.model_name == "HART-Intra":
            param.K_hop = 0
    else:
        param.use_subg = False
            
    #check evaluate
    if param.task == "PSR":
        # evaluate on test for each epoch
        param.eval_split_list = ["valid", "test"]
        param.num_iterations = 50
    else:
        # evaluate on test for every eval_step epochs
        param.eval_split_list = ["train", "valid"]
        param.eval_step = 25
        if param.model_name in ["HART", "HyperAggModel"] and "FI_" in param.dataset:
            param.eval_batch_size = 1
        

    # calculate max arity
    if len(param.ary_list) != 0 :
        param.max_arity = max(param.ary_list)
        param.max_seq_length = 2 * (param.max_arity - 2)  + 4
    else:
        param.max_arity = 0
        param.max_seq_length = -1

    # set the predict position
    param.train_mask_pos_list = []
    param.test_mask_pos_list = []
    if "main_ent" in param.mask_type_list:
        param.train_mask_pos_list.extend([0,2])
        param.test_mask_pos_list.extend([0,2])
    if "main_rel" in param.mask_type_list:
        param.train_mask_pos_list.append(1)
        param.test_mask_pos_list.append(1)
    if "q_ent" in param.mask_type_list:
        for i in range(param.max_arity-2):
            param.train_mask_pos_list.append(2*i+4)
            param.test_mask_pos_list.append(2*i+4)
    if "q_rel" in param.mask_type_list:
        for i in range(param.max_arity-2):
            param.train_mask_pos_list.append(2*i+5)
            param.test_mask_pos_list.append(2*i+5)


    # restore model
    name = param.model_name
    if param.restore is not None:
        param.store_name = param.restore
    else:
        if param.binary and param.model_name == "starE":
            name = "CompGCN"
        param.store_name = param.task + "_" + param.dataset + "-" + param.version +  "_" + name + "_" + time.strftime("%d_%m_%Y") + "_" + time.strftime("%H:%M:%S")
    
    # set device
    if param.gpu != '-1' and torch.cuda.is_available():
        param.device = torch.device('cuda')
        torch.cuda.set_rng_state(torch.cuda.get_rng_state())
        torch.backends.cudnn.deterministic = True
    else:
        param.device = torch.device('cpu')
    param.enable_amp = True if ("cuda" in param.device.type and param.use_fp16) else False
    param.scaler = amp.GradScaler(enabled=param.enable_amp)

    if param.use_wandb:
        wandb.login(key="") #! enter your own key

        wandb.init(project=param.task + "_" + param.dataset + "-" + param.version)
        
        
        wandb.run.name = f"{name}, hagg:{param.hagg_method}, \
                            fd: {param.feature_drop}, lr:{param.lr}, \
                            K_hop:{param.K_hop}, max_each_hop:{param.max_each_hop}, \
                            position_mode:{param.position_mode}, TL:{param.trans_layer}\
                            binary:{param.binary}, Flash:{param.use_flash}"
        wandb.config.update(param)
        

    # fix seed
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed = 3333
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) # for current GPU
    torch.cuda.manual_seed_all(seed) # for all available GPUs
    """

    print("parameters setting:")
    print(param)
    
    if param.task == "PSR":
        Exp = ExperimentPair(param=param)
    else:
        Exp = ExperimentTransfer(param=param)
    if param.explain_fact:
        param.use_flash = False
        Exp.fact_explain([0,1,2,3,4,5,6,7,8,9,10])
    else:
        Exp.train_and_eval()