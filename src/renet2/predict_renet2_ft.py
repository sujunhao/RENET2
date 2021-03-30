import argparse
import sys
import os
import random
import time
import re
import copy
import pickle
import pandas as pd
import numpy as np

import gc
import torch
from torch import nn, optim, cuda
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, \
                             TensorDataset, WeightedRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import KFold

# from tqdm import tqdm|

from renet2.raw import load_documents, load_documents_batch
from renet2.raw_handler import *
from renet2.model import *

_G_eval_time = 0

def set_t(df):
    df['pmid'] = df['pmid'].astype(str)
    df['geneId'] = df['geneId'].astype(str)
    df['diseaseId'] = df['diseaseId'].astype(str)
    return df

def free_cuda():
    gc.collect()
    with torch.cuda.device('cuda'):
        torch.cuda.empty_cache()
        
def renet2_evaluate(args, _read_batch_idx=0, loaded_feature=None):

    # read data
    print('reading input at {}'.format(args.raw_data_dir))
    features_ft_sub = None
    if _read_batch_idx == 0:
        features_ft_sub = load_and_cache_data(args)
    else:
        features_ft_sub = loaded_feature
    dataset_ft_sub, _, _ = convert_features_to_dataset_single(features_ft_sub)
    dataloader_ft_sub = DataLoader(dataset_ft_sub, batch_size=args.batch_size)
    #start_time = time.time()
    #print("--- %.3f seconds ---" % (time.time() - start_time))

    if args.models_number <= 0:
        args.models_number = 1

    print('begin evaluation ... at {} models'.format(args.models_number))


    _m_start_time = time.time()
    for _i in range(1, args.models_number+1):
        #continue
        model_name_prefix = 'bst_ft_build_%02d' % (_i)
        checkpoint_f = os.path.join(args.model_dir, model_name_prefix + ".ckp")
        config_save_f = os.path.join(args.model_dir,  model_name_prefix + ".cf")

        config = torch.load(config_save_f)
        model, _, _ = load_checkpoint(config, checkpoint_f)
        

        config.device =  args.device
        
        args.is_iterare_info = False
        args.threshold = config.threshold 
        args.l2_weight_decay = config.l2_weight_decay
        model.update_model_config(config)

        if torch.cuda.device_count() > 1:
    	    #print("use", torch.cuda.device_count(), "GPUs!")
    	    model = nn.DataParallel(model)
	
        model.to(args.device)
	        
        pred_l, tru_l, S, pred_o = eval(model, dataloader_ft_sub, args, 'test')
        if not args.no_cuda:
            free_cuda()
        y_info = features_ft_sub[1].copy()
        y_info['pred'] = pred_l
        y_info['prob'] = pred_o
        y_info.drop(columns=['label'], inplace=True)
        
        cls_rst_file = os.path.join(args.gda_fn_d, "gda_rst_%02d.tsv" % (_i))
        y_info.to_csv(cls_rst_file, sep='\t', index=False)
        print('rst at {}'.format(cls_rst_file))
    global _G_eval_time
    _G_eval_time += (time.time() - _m_start_time)

    m_df = 0
    for _i in range(args.models_number):
        cls_file = os.path.join(args.gda_fn_d, "gda_rst_%02d.tsv" % (_i+1))
        
        y_info = pd.read_csv(cls_file, sep='\t')
        y_info['pmid'] = y_info['pmid'].astype(str)
        y_info['geneId'] = y_info['geneId'].astype(str)
        y_info['diseaseId'] = y_info['diseaseId'].astype(str)
        #y_info['label'] = y_info['label'].astype(str)
        
        
        y_info.rename(columns={'pred':'pred_%02d'%(_i+1)}, inplace=True)
        y_info.rename(columns={'prob':'prob_%02d'%(_i+1)}, inplace=True)
        #m_df = y_info if _i == 0 else m_df.merge(y_info, on=['pmid', 'geneId','diseaseId', 'label'], how='outer')
        m_df = y_info if _i == 0 else m_df.merge(y_info, on=['pmid', 'geneId','diseaseId'], how='outer')
    
    m_df['hit_cnt'] = m_df.apply(lambda x: sum([x['pred_%02d'%(_i+1)] for _i in range(args.models_number)]), axis=1)
    
    _cutoff = int(args.models_number/2)
    if args.is_sensitive_mode:
        _cutoff = 1

    print('assemble models using cutoff {}'.format(_cutoff))
    ReNet_df = m_df[['pmid', 'geneId', 'diseaseId']].copy()
    ReNet_df['pred'] = m_df.apply(lambda x: 1 if x['hit_cnt'] >= _cutoff else 0, axis=1)
    cols_n = ['prob_%02d' % (_i+1) for _i in range(args.models_number)]
    #print(m_df.head())
    ReNet_df['prob_avg'] = m_df[cols_n].mean(axis=1)
    #ReNet_df['prob_avg'] = ReNet_df['prob_avg'].round(decimals = 5)
    ReNet_df['prob_avg'] = ReNet_df['prob_avg'].map('{:,.5f}'.format)
    for _i in range(args.models_number):
        m_df[cols_n[_i]] = m_df[cols_n[_i]] .map('{:,.5f}'.format)
    ReNet_df['prob_X'] = m_df[cols_n].apply(lambda row: ';'.join(row.values.astype(str)), axis=1)
    #print(ReNet_df.head())
    
    gda_rst = ReNet_df[ReNet_df['pred']==1].copy()
    gda_rst = set_t(gda_rst)


    def read_ann(file_p):
    	anns = []
    	with open(file_p, 'r') as F:
    	    while True:
    	        line = F.readline()
    	        if line == '':
    	            break
    	        if line == '\n':
    	            continue
    	        ann = line.strip().split('\t')
#   	          print(ann)
    	        anns.append(ann)
#   	          break
    	header_list = ["pmid", "off_a", "off_b", "name", "type", "id", 's_i']
    	anns = pd.DataFrame(anns, columns = header_list) 
    	return anns

    start_time = time.time()
    print('generate GDA table, getting NER names, and merge tatble')
    
    merg_df = gda_rst
    anns = read_ann(os.path.join(args.raw_data_dir, "anns.txt"))
    
    anns = anns.dropna()
    anns['id'] = anns['id'].astype(str)
    anns['pmid'] = anns['pmid'].astype(str)
    anns['id'] = anns['id'].apply(lambda x: x.replace("\t", " "))
    
    anns_gene = anns[anns['type'] == "Gene"]
    anns_dis = anns[anns['type'] == "Disease"]
    
    tar_set_df = merg_df[['pmid', 'geneId', 'diseaseId']]
    
    df_t = pd.merge(tar_set_df, anns_gene, how="left", left_on=['pmid', 'geneId'], right_on=['pmid', 'id'])
    df_t = df_t.drop_duplicates(['pmid', 'geneId', 'diseaseId', 'name'])
    #print(df_t.head(5))
    _df = df_t.groupby(['pmid', 'geneId', 'diseaseId', 'id'], as_index=False)['name'].apply(lambda x: "%s" % '|'.join(x)).reset_index()
    _df.rename(columns={0: "g_name"}, inplace=True)
    _df.sort_values(by=['pmid', 'geneId']).head()
    _df.drop(columns=['id'], inplace=True)
    df_g = _df[:]
    
    df_t = pd.merge(tar_set_df, anns_dis, how="left", left_on=['pmid', 'diseaseId'], right_on=['pmid', 'id'])
    df_t = df_t.drop_duplicates(['pmid', 'geneId', 'diseaseId', 'name'])
    _df = df_t.groupby(['pmid', 'geneId', 'diseaseId', 'id'], as_index=False)['name'].apply(lambda x: "%s" % '|'.join(x)).reset_index()
    _df.rename(columns={0: "d_name"}, inplace=True)
    _df.drop(columns=['id'], inplace=True)
    _df.sort_values(by=['pmid', 'geneId']).head()
    df_d = _df[:]
    # _df
    
    
    df_t = pd.merge(df_g, df_d, how='inner', on=['pmid', 'geneId', 'diseaseId'])
    df_t = df_t.merge(tar_set_df)
    
    tar_set = merg_df.merge(df_t, how='left', on=['pmid', 'geneId', 'diseaseId'])
    
    #tar_set = tar_set[['pmid','geneId','diseaseId','g_name','d_name']]
    tar_set = tar_set[['pmid','geneId','diseaseId','g_name','d_name', 'prob_avg', 'prob_X']]
    print(tar_set.shape[0] == gda_rst.shape[0], gda_rst.shape[0])
    
    print(tar_set.head())
    print("--- %s seconds ---" % (time.time() - start_time))

    cls_rst_file = os.path.join(args.gda_fn_d, "gda_rst.tsv")
    if _read_batch_idx != 0:
        cls_rst_file = os.path.join(args.gda_fn_d, "gda_rst_%03d.tsv" % (_read_batch_idx))
    print('Final Found Positive GDA records at [# %d]:\n***  %s  ***' % (tar_set.shape[0], cls_rst_file))
    tar_set.to_csv(cls_rst_file, sep='\t', index=False)



def init_self_parser():
    # set up
    parser = argparse.ArgumentParser(description='RENET2 testing models [assemble, full-text]')
    parser.add_argument(
            "--raw_data_dir",
            default = "../data/ft_data/",
            type=str,
            help="input data folder",
    )

    parser.add_argument(
            "--model_dir",
            default = "../models/ft_models/",
            type=str,
            help="modle data dir",
    )

    parser.add_argument(
            "--label_f_name",
            default = "labels.txt",
            type=str,
            help="modle label name",
    )
    parser.add_argument(
            "--file_name_doc",
            default = "docs.txt",
            type=str,
            help="document name",
    )
    parser.add_argument(
            "--file_name_snt",
            default = "sentences.txt",
            type=str,
            help="sentences file name",
    )
    parser.add_argument(
            "--file_name_ann",
            default = "anns_n.txt",
            type=str,
            help="anns file name",
    )
    parser.add_argument(
            "--word_index_fn",
            default = "./utils/word_index",
            type=str,
            help="word index data dir",
    )
    parser.add_argument(
            "--gda_fn_d",
            default = "../data/ft_gda/",
            type=str,
            help="found gda file folder",
    )

    parser.add_argument(
            "--pretrained_model_p",
            default = "../models/Bst_abs_10",
            type=str,
            help="pretrained based models",
    )
    parser.add_argument('--use_fix_pretrained_models', action='store_true', default=False,
                                help='use fix pretrained models trained on RENET2 full-text dataset default: %(default)s')
    parser.add_argument('--use_cuda', action='store_true', default=False,
                                help='enables CUDA training default: %(default)s')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                                help='random seed default: %(default)s')
    parser.add_argument('--fix_snt_n', type=int, default=400, metavar='N',
                                help='number of sentence in input, recommend [for abs 32, for ft 400] default: %(default)s')
    parser.add_argument('--fix_token_n', type=int, default=54, metavar='N',
                                help='number of tokens  default: %(default)s')
    parser.add_argument('--max_doc_num', type=int, default=0, metavar='N',
                                help='number of document, 0 for read all doc default:  %(default)s')
    parser.add_argument('--no_cache_file', action='store_true', default=False,
                                help='disables using cache file for data input default: %(default)s')
    parser.add_argument('--add_cache_suf', action='store_true', default=False,
                                help='cache file suffix default: %(default)s')
    parser.add_argument('--overwrite_cache', action='store_true', default=False,
                                help='overwrite_cache default: %(default)s')
    parser.add_argument('--batch_size', type=int, default=60, metavar='N',
                                help='input batch size for training default: %(default)s')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                                help='number of epochs to train default: %(default)s')
    parser.add_argument('--lr', type=float, default=8e-4, metavar='LR',
                                help='learning rate  default: %(default)s')
    parser.add_argument('--l2_weight_decay', type=float, default=5e-5, metavar='L2',
                                help='L2 weight decay default: %(default)s')
    parser.add_argument('--is_read_doc', action='store_true', default=True,
                                help='reading doc file, not renet1\'s abstracts file  default: %(default)s')
    parser.add_argument('--read_abs', action='store_true', default=False,
                                help='reading_abs_only default: %(default)s')

    parser.add_argument('--models_number', type=int, default=10, metavar='N',
                                help='number of models to train default: %(default)s')
    parser.add_argument('--raw_input_read_batch', type=int, default=-1, metavar='N',
                                help='raw data read max doc batch number default: %(default)s')
    parser.add_argument('--raw_input_read_batch_resume', type=int, default=-1, metavar='N',
                                help='raw data read max doc batch number resume default: %(default)s')

    # explore args, disable
    parser.add_argument('--read_ori_token', action='store_true', default=False,
                                help='get raw text')
    parser.add_argument('--not_x_feature', action='store_true', default=False,
                                help='do not use x_feature')
    parser.add_argument('--read_old_file', action='store_true', default=False,
                                help='reading dga files')
    parser.add_argument('--using_new_tokenizer', action='store_true', default=False,
                                help='using new tokenizer')
    parser.add_argument('--is_filter_sub', action='store_true', default=False,
                                help='filter pmid in abs data')
    parser.add_argument('--is_sensitive_mode', action='store_true', default=False,
                                help='using sensitive mode')
    return parser

def main():
    #print_renet2_log()
    # set up
    parser = init_self_parser()
    args = parser.parse_args()

    get_index_path(args)

    base_path = Path(__file__).parent
    if args.use_fix_pretrained_models:
        args.model_dir = (base_path / args.model_dir).resolve()

    base_dir = os.path.dirname(__file__)  
    sys.path.insert(0, base_dir)

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)
    #print(args)

    _start_time = time.time()

    args.no_cuda = not args.use_cuda
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device('cuda' if use_cuda else 'cpu')
    if use_cuda:
        if torch.cuda.device_count() > 1:
    	    print("will use", torch.cuda.device_count(), "GPUs!")
    #device = torch.device('cuda:1')

    torch.manual_seed(args.seed)

    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    set_seed(args)
    args.device = device
    print('using device', device)

    args.ori_tokenizer = loading_tokenizer(args)
    args.token_voc_l = len(args.ori_tokenizer)
    print('tokenizer size %d' % (args.token_voc_l))
    
    print('fix input sentences# %d, tokens# %d, batch size %d' % (args.fix_snt_n, args.fix_token_n, args.batch_size))

    global _G_eval_time
    _G_eval_time = 0
    if args.raw_input_read_batch == -1:
        renet2_evaluate(args)
    else:
        _read_batch_idx = 1
        args.no_cache_file = True
        fix_snt_n, fix_token_n = args.fix_snt_n, args.fix_token_n
        text_path = os.path.join(args.raw_data_dir, args.file_name_doc) 
        sentence_path = os.path.join(args.raw_data_dir, args.file_name_snt)
        ner_path = os.path.join(args.raw_data_dir, args.file_name_ann)
        label_path = os.path.join(args.raw_data_dir, args.label_f_name)

        if args.read_old_file:
            text_path = os.path.join(args.raw_data_dir, "abstracts.txt")
            label_path = os.path.join(args.raw_data_dir, "labels.csv")


        print("Creating features from dataset file at ", args.raw_data_dir)
        print('\n/*/\nreading input with batch {}, at {}'.format(args.raw_input_read_batch, _read_batch_idx))
        for X in load_documents_batch(text_path, sentence_path, ner_path, args.ori_tokenizer, args):
            if args.raw_input_read_batch_resume != -1:
                if _read_batch_idx < args.raw_input_read_batch_resume:
                    _read_batch_idx += 1
                    print('\n/*/\nreading input with batch {}, at {}'.format(args.raw_input_read_batch, _read_batch_idx))
                    continue
            gdas, word_seq, x_feature, all_ori_seq, x_len = X 
            m_y, y = read_labels(label_path, gdas)
            print("reading label at {}".format(label_path))
        
            #gdas, word_seq, x_feature = load_documents_ori(text_path, sentence_path, ner_path, tokenizer)
            print("loaded document # ", len(gdas), len(word_seq), len(x_feature))
        
            all_feature_f = None
            if args.using_new_tokenizer:
                print("using new tokenizer")
                all_feature_f = tokenizer_doc(args.tokenizer, all_ori_seq, len(word_seq), fix_snt_n, fix_token_n)
            else: 
                print("padding document with fix length", fix_snt_n, fix_token_n)
                all_feature_f = padding_raw_input(word_seq, x_feature, len(word_seq), fix_snt_n, fix_token_n)
        
        
            features = (all_feature_f, m_y)
            check_rec_full(all_feature_f)
            print("loading ended")
            renet2_evaluate(args, _read_batch_idx, features)

            _read_batch_idx += 1
            print('\n/*/\nreading input with batch {}, at {}'.format(args.raw_input_read_batch, _read_batch_idx))


        print("ended all evaluation\n--------\n")
        m_df = None
        for _i in range(1, _read_batch_idx):
            cls_rst_file = os.path.join(args.gda_fn_d, "gda_rst_%03d.tsv" % (_i))
            tmp_df = pd.read_csv(cls_rst_file, sep='\t')
            tmp_df['prob_avg'] = tmp_df['prob_avg'].map('{:,.5f}'.format)
            m_df = tmp_df if _i == 0 else pd.concat([m_df, tmp_df], ignore_index=True)

        print(m_df.head())
        cls_rst_file = os.path.join(args.gda_fn_d, "gda_rst.tsv")
        print('Final Found Positive GDA records at [# %d]:\n ***  %s  ***' % (m_df.shape[0], cls_rst_file))
        m_df.to_csv(cls_rst_file, sep='\t', index=False)



    _G_tot_time = time.time() - _start_time
    print("time | total: %.2fs | evaluation %.2fs | parsing data %.2fs" % (_G_tot_time, _G_eval_time, _G_tot_time - _G_eval_time))

if __name__ == "__main__":
    main()
