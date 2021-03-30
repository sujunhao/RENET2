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

from renet2.raw import load_documents
from renet2.raw_handler import *
from renet2.model import *



def free_cuda():
    gc.collect()
    with torch.cuda.device('cuda'):
        torch.cuda.empty_cache()

def set_t(df):
    df['pmid'] = df['pmid'].astype(str)
    df['geneId'] = df['geneId'].astype(str)
    df['diseaseId'] = df['diseaseId'].astype(str)
    return df


def get_ann_dataset(features_ft_sub, abs_s_df):
    # filtering al non annotated GDA
    tmp_ft_df = features_ft_sub[1].copy()
    tmp_ft_df['index1'] = tmp_ft_df.index
    # print(len(tmp_ft_df))

    tmp_ft_df = tmp_ft_df.merge(abs_s_df, on=['pmid', 'geneId', 'diseaseId'], how='right')
    
    tmp_ft_df = tmp_ft_df.dropna(subset=['index1'])
    _index = pd.Int64Index(tmp_ft_df['index1'].values, dtype=np.int64)
    # print(len(tmp_ft_df))

    x_train, y_train = features_ft_sub

    x_train = x_train[_index]
    y_train = y_train.loc[_index]

    y_train.index = pd.RangeIndex(len(y_train.index))

    n_features_ft_sub = x_train, y_train
    return n_features_ft_sub

# get addon data
def get_addon_data_ft(_ft_sub, mdl_df, _add_n, n_label, is_shuffle = False, s_idx = None):
#     s_idx = None
#     nan_add_n = 500
#     n_label = 0

    tmp_ft_df = _ft_sub[1].copy()
    tmp_ft_df['index1'] = tmp_ft_df.index

    tar_addon_df = tmp_ft_df.merge(mdl_df, on=['pmid', 'geneId', 'diseaseId'], how='left')
    # tar_addon_df = tar_addon_df[['pmid', 'geneId', 'diseaseId', 'is_tar']]
    tar_addon_df = tar_addon_df[~tar_addon_df['is_tar'].isnull()]
    #print(tar_addon_df.shape)
#     tar_addon_df.head()
    _idx = []
    _add_n = len(tar_addon_df) if _add_n==-1 else _add_n
    if s_idx != None:
        _idx = s_idx[:_add_n]
    else:
        _idx = list(tmp_ft_df['index1'].values)
        if is_shuffle:
            random.shuffle(_idx)
        _idx = _idx[:_add_n]

    #print(len(_idx), _idx[:5])

    _index = pd.Int64Index(_idx, dtype=np.int64)

    x_train, y_train = _ft_sub

    x_train = x_train[_index]
    y_train = y_train.loc[_index].copy()
    y_train['label'] = n_label

    y_train.index = pd.RangeIndex(len(y_train.index))
    #print(y_train.head(3))
    n_ft_sub = x_train, y_train
    return n_ft_sub

def get_G_F1(_p, _r, _beta = 2):
    if _beta*_beta*_p+_r != 0:
        return (1+_beta*_beta) * ((_p*_r*1.) / (_beta*_beta*_p+_r))
    else:
        return 0

# df1 have pmid, geneId, diseaseId, pred, prob
def get_PR_info(df0, df_b, _beta=1):
    df0['pmid'] = df0['pmid'].astype(str)
    df0['geneId'] = df0['geneId'].astype(str)
    df0['diseaseId'] = df0['diseaseId'].astype(str)
#     df1.drop(columns=['label'], inplace=True)
    
    df1 = df0.merge(df_b, how='left')
    N_pred, N_unknown = int(sum(df1['pred'])), int(sum(df1[df1['pred']==1]['new_label'].isna()))

    
    df2 = df_b.merge(df0, how='left')
    tru_l2 = df2.new_label.fillna(0).to_numpy(copy=True)
    tru_l2[tru_l2==.5] = 0

    pred_l = df2.pred.fillna(0).to_numpy(copy=True)
    pred_l = pred_l.astype(int)
    
    precision, recall, f1, _ = \
                        precision_recall_fscore_support(tru_l2, pred_l, average='binary',zero_division=1)
    S1 = precision, recall, f1
    
    if _beta != 1:
        S = S1[0], S1[1], get_G_F1(S1[0], S1[1], 1), N_pred-N_unknown, N_unknown, get_G_F1(S1[0], S1[1], _beta)
    else:
        S = S1[0], S1[1], get_G_F1(S1[0], S1[1], 1), N_pred-N_unknown, N_unknown
    return S

def get_s(_df, true_c, pred_c):
    tru_l = _df[true_c].to_numpy(copy=True)
    tru_l[tru_l==.5] = 0
    
    pred_l = _df[pred_c].to_numpy(copy=True)
    pred_l[pred_l==.5] = 0
    
    pred_l = pred_l.astype(int)
    precision, recall, f1, _ = \
                        precision_recall_fscore_support(tru_l, pred_l, average='binary',zero_division=1)
    return precision, recall, f1

def set_df_to_str(df, col=['pmid', 'geneId', 'diseaseId']):
    for c in col:
        df[c] = df[c].astype(str)
    return df

def get_scores_a(_df, _c1, ann_df, _c2):
    _df['pmid'] = _df['pmid'].astype(str)
    _df['geneId'] = _df['geneId'].astype(str)
    _df['diseaseId'] = _df['diseaseId'].astype(str)
    
    N_ann = int(len(ann_df[ann_df[_c2]==1]))
    df2 = ann_df.merge(_df, how='left')
    tru_l2 = df2[_c2].fillna(0).to_numpy(copy=True)
    tru_l2[tru_l2==.5] = 0

    pred_l = df2[_c1].fillna(0).to_numpy(copy=True)
    pred_l = pred_l.astype(int)

    precision, recall, f1, _ = \
                        precision_recall_fscore_support(tru_l2, pred_l, average='binary',zero_division=1)
    S1 = precision, recall, N_ann
    return S1

BeFree_df, DTMiner_df, BioBERT_df, ReNet_df = None, None, None, None
def evaluate_rst_all_info(ori_df, ori_c, ann_df, is_freeze_c = False, b_ann_df = False):
    global BeFree_df, DTMiner_df, BioBERT_df
    
    global ReNet_df
    #if not is_freeze_c:
    #    global ReNet_df

    _rst = []
    _df = ori_df.copy()
    _c = ori_c
    if not is_freeze_c:
        _c = 'pred_tmp'
        _df.rename(columns={ori_c:'pred_tmp'}, inplace=True)

    # predicted GDA
    _pred_N = sum(_df[_c])
    _rst.append(_pred_N)

    # precision on ann & based #
    S1 = get_scores_a(_df, _c, ann_df, 'new_label')
    _precision1, _recall, N_ann = S1
    _rst.extend([_precision1])

    # merging renet|befree|dtminer result first
    if not is_freeze_c:
        merge_t_df = _df.merge(BeFree_df, on=['pmid', 'geneId','diseaseId'], how='outer')
    else:
        merge_t_df = ReNet_df.merge(BeFree_df, on=['pmid', 'geneId','diseaseId'], how='outer')
        
    merge_t_df = merge_t_df.merge(DTMiner_df, on=['pmid', 'geneId','diseaseId'], how='outer')
    merge_t_df = merge_t_df.merge(BioBERT_df, on=['pmid', 'geneId','diseaseId'], how='outer')
    merge_t_df = merge_t_df.fillna(0)
    ann_positive = ann_df.copy()
    
    df=merge_t_df
    F=[1,1,1,1]
    c=[_c,'is_dtminer','is_befree','is_biobert']
#     print(c)
    if is_freeze_c:
        c=['is_renet','is_dtminer','is_befree','is_biobert']
    
    # c[0], c[2] = c[2], c[0]
    mdl_positive = df[(df[c[0]]==F[0]) & ((df[c[1]]==F[1]) & (df[c[2]]==F[2]) & (df[c[3]]==F[3]))]
    
    if isinstance(b_ann_df, pd.DataFrame):
        mdl_positive = b_ann_df.merge(mdl_positive, on=['pmid', 'geneId','diseaseId'], how='outer')
        mdl_positive = mdl_positive[mdl_positive['is_ner_error'].isna()]
        del mdl_positive['is_ner_error']
        
    olp_positive = ann_positive.merge(mdl_positive, on=['pmid', 'geneId','diseaseId'], how='outer')
    
    ann_n, mdl_n, olp_n = len(ann_positive), len(mdl_positive), len(olp_positive)
#     print('---', ann_n, mdl_n, olp_n)
    
#     print('model confidence GDA {}, adding {} to TP, total TP {}'.format(mdl_n, olp_n-ann_n, olp_n))
#     print(olp_positive.head(3))
    
    olp_positive['ann_label'] = olp_positive['new_label']
    olp_positive['ann_label'] = olp_positive['ann_label'].fillna(0)
    olp_positive = olp_positive[['pmid', 'geneId','diseaseId', 'new_label', 'ann_label']]
    olp_positive['new_label'] = olp_positive['new_label'].fillna(1)
    olp_positive.shape
    targe_df = merge_t_df.merge(olp_positive, on=['pmid', 'geneId','diseaseId'], how='left')
    targe_df['new_label'] = targe_df['new_label'].fillna(0)

    tar_df = _df.copy()
    if isinstance(b_ann_df, pd.DataFrame):
        tar_df = tar_df.merge(b_ann_df, on=['pmid', 'geneId','diseaseId'], how='left')
        tar_df['pred'] = tar_df.apply(lambda x: 1 if x[_c] ==1 and pd.isnull(x['is_ner_error']) else 0, axis=1)
    else:
        tar_df['pred'] = _df.apply(lambda x: 1 if x[_c] ==1 else 0, axis=1)
    S2 = get_scores_a(tar_df, 'pred', targe_df, 'new_label')
    _precision2, _, N_p2 = S2
#     print(S2)
    _rst.extend([_precision2])



    _rst.append(_recall)
#     print(','.join(list(map(str, _rst))))
    return _rst

def evaluate_rst_all_info_err(ori_df, ori_c, ann_df, _ann_df, abs_s_df_ner, is_freeze_c = False):
    _rst1 = evaluate_rst_all_info(ori_df, ori_c, ann_df, is_freeze_c)
    _rst2 = evaluate_rst_all_info(ori_df, ori_c, _ann_df, is_freeze_c, abs_s_df_ner)

    _rst = _rst1[0], _rst1[2], _rst2[2], _rst1[3], _rst2[3]
    return _rst


def renet2_train(args):

    # initalize setting
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device('cuda' if use_cuda else 'cpu')
    torch.manual_seed(args.seed)

    if use_cuda:
        if torch.cuda.device_count() > 1:
    	    print("will use", torch.cuda.device_count(), "GPUs!")
        torch.cuda.manual_seed(args.seed)
    set_seed(args)
    args.device = device
    print('using device', device)

    # loading tokens vocabulary
    args.ori_tokenizer = loading_tokenizer(args)
    args.token_voc_l = len(args.ori_tokenizer)
    print('tokenizer size %d' % (args.token_voc_l))
    
    print('fix input sentences# %d, tokens# %d, batch size %d' % (args.fix_snt_n, args.fix_token_n, args.batch_size))

    # parse manual annotated GDA 
    old_data_dir = args.annotation_info_dir
    # write annotated table to model input file
    tar_path = os.path.join(old_data_dir, "ft_500_n.tsv")
    s_df = pd.read_csv(tar_path, sep = '\t')
    
    print('annotated table shape', s_df.shape)
    s_df = s_df[s_df['2nd_label'] != 0][['pmid', 'geneId', 'diseaseId', '2nd_label']]
    s_df.rename(columns={'2nd_label':'label'}, inplace=True)
    
    
    new_label_f_n = 'labels_n.txt'
    s_df.to_csv(args.raw_data_dir + new_label_f_n, sep=',', index=False)

    args.label_f_name = 'labels_n.txt'

    # read data [doc, set, ann, label]
    print('reading input at {}'.format(args.raw_data_dir))
    
    
    start_time = time.time()
    features_ft_sub = load_and_cache_data(args)
    dataset_ft_sub, _, _ = convert_features_to_dataset_single(features_ft_sub)
    dataloader_ft_sub = DataLoader(dataset_ft_sub, batch_size=args.batch_size)
    print("--- %.3f seconds ---" % (time.time() - start_time))


    # get manual annotated infoformation, for evaluate NER erros
    tar_path = os.path.join(old_data_dir, "ft_500_n.tsv")
    s_df = pd.read_csv(tar_path, sep = '\t')
    
    s_df['pmid'] = s_df['pmid'].astype(str)
    s_df['geneId'] = s_df['geneId'].astype(str)
    s_df['diseaseId'] = s_df['diseaseId'].astype(str)
    s_df.rename(columns={'2nd_label':'new_label'}, inplace=True)
    
    # annotated GDA
    abs_s_df = s_df[['pmid', 'geneId', 'diseaseId', 'new_label']].copy()
    abs_s_df_ner = s_df[['pmid', 'geneId', 'diseaseId', 'is_ner_error']].copy()
    
    # get the annotated GDA for model training
    n_features_ft_sub = get_ann_dataset(features_ft_sub, abs_s_df)
    N_ann_positive = len(n_features_ft_sub[1][n_features_ft_sub[1]['label']==1])
    N_ann_all = len(n_features_ft_sub[1])
    print(N_ann_positive, N_ann_all)



    # annotated non-NER errors GDA
    _s_df = s_df[ (s_df['is_ner_error'].isna())]
    _abs_s_df = _s_df[['pmid', 'geneId', 'diseaseId', 'new_label']].copy()
    
    ner_error_df = s_df[ ~(s_df['is_ner_error'].isna())]
    ner_error_df = ner_error_df[['pmid', 'geneId', 'diseaseId']].copy()
    
    # reading BeFree and DTMiner, BioBERT result
    global BeFree_df, DTMiner_df, BioBERT_df, ReNet_df   

    
    # reading BeFree GDA
    # _f = os.path.join(old_data_dir, "classification_result_befree.txt")
    # gdas = pd.read_csv(_f, sep=',')
    befree_f = os.path.join(old_data_dir, "classification_result_befree.txt")
    gdas = pd.read_csv(befree_f, sep=',')
    predicted_positive = gdas.drop_duplicates(['pmid', 'geneId', 'diseaseId'])
    BeFree_df = predicted_positive[['pmid', 'geneId', 'diseaseId']].copy()
    BeFree_df = set_df_to_str(BeFree_df)
    BeFree_df['is_befree'] = 1
    #print(BeFree_df.shape, BeFree_df.head(3))

    # reading DTMiner GDA
    # _f = os.path.join(old_data_dir, "classification_result_dtminer.txt")
    # gdas = pd.read_csv(_f, sep=',')
    dtminer_f = os.path.join(old_data_dir, "classification_result_dtminer.txt")
    gdas = pd.read_csv(dtminer_f, sep=',')
    predicted_positive = gdas.drop_duplicates(['pmid', 'geneId', 'diseaseId'])
    DTMiner_df = predicted_positive[['pmid', 'geneId', 'diseaseId']].copy()
    DTMiner_df = set_df_to_str(DTMiner_df)
    DTMiner_df['is_dtminer'] = 1
    #print(DTMiner_df.shape, DTMiner_df.head(3))
#     print('get BeFree and DT-Miner data, shape', BeFree_df.shape, DTMiner_df.shape)

    
    biobert_f = os.path.join(old_data_dir, "classification_result_biobert.txt")
    gdas = pd.read_csv(biobert_f, sep=',')
    predicted_positive = gdas.drop_duplicates(['pmid', 'geneId', 'diseaseId'])
    BioBERT_df = predicted_positive[['pmid', 'geneId', 'diseaseId']].copy()
    BioBERT_df = set_df_to_str(BioBERT_df)
    BioBERT_df['is_biobert'] = 1
    
    print('get BeFree, DTMiner, and BioBERT data, shape', BeFree_df.shape, DTMiner_df.shape, BioBERT_df.shape)
    

    addon_nan_feature = None
    if args.have_SiDa:
        print('reading [silver data]')
        m_df = 0
        s_arr = []
        for _i in range(10):
            cls_file = args.have_SiDa + "_%02d.tsv" % (_i+1)

            y_info = pd.read_csv(cls_file, sep='\t')
            y_info['pmid'] = y_info['pmid'].astype(str)
            y_info['geneId'] = y_info['geneId'].astype(str)
            y_info['diseaseId'] = y_info['diseaseId'].astype(str)
            y_info['label'] = y_info['label'].astype(str)

            S = evaluate_rst_all_info_err(y_info, 'pred', abs_s_df, _abs_s_df, abs_s_df_ner)
        #     print(S)
            s_arr.append(list(S))

            y_info.rename(columns={'pred':'pred_%02d'%(_i+1)}, inplace=True)
        #     y_info = y_info[y_info['pred'] == 1]
            m_df = y_info if _i == 0 else m_df.merge(y_info, on=['pmid', 'geneId','diseaseId', 'label'], how='outer')


        m_df['hit_cnt'] = m_df.apply(lambda x: sum([x['pred_%02d'%(_i+1)] for _i in range(10)]), axis=1)
        ori_m_df = m_df.copy()
        s_arr = np.array(s_arr)

        print('10X modes\' positive #, precision, precision[-], recall, recall[-]')
        for r in s_arr:
            print(','.join([str(i) for i in r]))

        avg_l = list(np.mean(s_arr, axis=0))
        avg_l = ['avg'] + avg_l

        print(','.join(list(map(str, avg_l))))

        print('assemble, positive #, precision, precision[-], recall, recall[-]')
        for _i in range(1, 11):
            ReNet_df = m_df[['pmid', 'geneId', 'diseaseId']].copy()
            ReNet_df['is_renet'] = m_df.apply(lambda x: 1 if x['hit_cnt'] >= _i else 0, axis=1)

            _rst = evaluate_rst_all_info_err(ReNet_df, 'is_renet', abs_s_df, _abs_s_df, abs_s_df_ner)
            _rst = [_i] + list(_rst)
            print(','.join(list(map(str, _rst))))

        the_p_cnt = 1
        ReNet_df = m_df[['pmid', 'geneId', 'diseaseId']].copy()
        ReNet_df['is_renet'] = m_df.apply(lambda x: 1 if x['hit_cnt'] >= the_p_cnt else 0, axis=1)

        print('selected assemble models at {}+, shape {}, result:'.format(the_p_cnt, ReNet_df.shape))
        print(evaluate_rst_all_info_err(ReNet_df, 'is_renet', abs_s_df, _abs_s_df, abs_s_df_ner))

        print('BEFREE result:')
        _rst = evaluate_rst_all_info(BeFree_df, 'is_befree', abs_s_df, is_freeze_c=True)
        _rst_pred_n, _rst_precision, _rst_recall = _rst[0], _rst[2], _rst[3]
        _rst_f1 = get_G_F1(_rst_precision, _rst_recall, 1)
        print('result: Precision, Recall, F1 : {}, {}, {}'.format(_rst_precision, _rst_recall, _rst_f1))

        print('DTMiner result:')
        _rst = evaluate_rst_all_info(DTMiner_df, 'is_dtminer', abs_s_df, is_freeze_c=True)
        _rst_pred_n, _rst_precision, _rst_recall = _rst[0], _rst[2], _rst[3]
        _rst_f1 = get_G_F1(_rst_precision, _rst_recall, 1)
        print('result: Precision, Recall, F1 : {}, {}, {}'.format(_rst_precision, _rst_recall, _rst_f1))

        print('BioBERT result:')
        _rst = evaluate_rst_all_info(BioBERT_df, 'is_biobert', abs_s_df, is_freeze_c=True)
        _rst_pred_n, _rst_precision, _rst_recall = _rst[0], _rst[2], _rst[3]
        _rst_f1 = get_G_F1(_rst_precision, _rst_recall, 1)
        print('result: Precision, Recall, F1 : {}, {}, {}'.format(_rst_precision, _rst_recall, _rst_f1))



        print('get silver data')
        # addon_olp3p_feature = get_addon_data_ft(features_ft_sub, mdl_df, -1, 1, False)
        # merging renet|befree|dtminer result first
        merge_t_df = ReNet_df.merge(BeFree_df, on=['pmid', 'geneId','diseaseId'], how='outer')
        merge_t_df = merge_t_df.merge(DTMiner_df, on=['pmid', 'geneId','diseaseId'], how='outer')
        merge_t_df = merge_t_df.merge(BioBERT_df, on=['pmid', 'geneId','diseaseId'], how='outer')
        merge_t_df = merge_t_df[~merge_t_df.is_renet.isnull()]
        merge_t_df = merge_t_df.fillna(0)

        F=[0, 0, 0, 0] #all negative
        # F=[1, 1, 1] #all positive
        c=['is_renet','is_dtminer','is_befree', 'is_biobert']
        df = merge_t_df
        mdl_df = df[(df[c[0]]==F[0]) & ((df[c[1]]==F[1]) & (df[c[2]]==F[2]) & (df[c[3]]==F[3]))].copy()
        mdl_df['is_tar'] = 1

        # opt 1 filter the GDA annotated
        mdl_df = mdl_df.merge(abs_s_df, how='left')
        mdl_df = mdl_df[mdl_df.new_label.isnull()]
        mdl_df_n = mdl_df[['pmid', 'geneId','diseaseId', 'is_tar']].copy()

        N_ann_positive = len(n_features_ft_sub[1][n_features_ft_sub[1]['label']==1])
        N_ann_all = len(n_features_ft_sub[1])
        #print(N_ann_positive, N_ann_all)
        # N_ann_positive += len(addon_olp3p_feature[1])
        add_to_rate = 2
        N_add_nan = int(N_ann_positive * add_to_rate - N_ann_all)
        # N_add_nan = int(500 * (5/4))
        print(N_ann_positive, N_ann_all, N_add_nan, 100.*N_add_nan/N_ann_positive, N_add_nan+N_ann_all)
        addon_nan_feature = get_addon_data_ft(features_ft_sub, mdl_df_n, N_add_nan, 0, True)
    
    
    ## reading pmid list order, for CV
    #_f = os.path.join(old_data_dir, 'PMID_lst')
    #with open(_f, 'rb' ) as fp:
    #    PMID_lst = pickle.load(fp)
    #print(len(PMID_lst))
    ##len(PMID_lst), PMID_lst[:3]

    # should read all pmid from id list other than annotated file, as some article may not in the annotated file
    tar_lst = pd.read_csv(os.path.join(old_data_dir, "ft_id_lst.csv"), dtype=str)
    PMID_lst = list(tar_lst.PMID.values)
    np.random.shuffle(PMID_lst)
    print(PMID_lst[:5])

    
    # build best model
    def build_dev(_idx):
        IS_DEBUG = False
        IS_RUN_1 = False
        IS_ADD_FT_N = False

        if args.have_SiDa:
            IS_ADD_FT_N = True

    
        _start_time = time.time()
    
        # testing ann
        tar_feature = n_features_ft_sub
        all_feature = features_ft_sub
    
        add_feature_n = addon_nan_feature if IS_ADD_FT_N else None
    
        train_index = tar_feature[1].index
        _train_dev_ds = convert_features_to_dataset_cv(tar_feature, train_index)
    
        if IS_ADD_FT_N:
            #add data in training dataset
            add_feature = add_feature_n
            #add_index = add_feature[1][(add_feature[1].pmid.isin(train_pmid_lst))].index
            add_index = add_feature[1].index
            #print('using add feature: %s' % len(add_index))
            add_dataset = convert_features_to_dataset_cv(add_feature, add_index)
            _train_dev_ds = torch.utils.data.ConcatDataset([_train_dev_ds, add_dataset]) 
    
    
        train_dataloader_ds = DataLoader(_train_dev_ds, batch_size=args.batch_size, shuffle=True)
        print('training dataset size {}'.format(len(_train_dev_ds)))
    
        train_dt, dev_dt, test_dt = train_dataloader_ds, None, None
   
    
        # loading pretrained abstract model
        checkpoint_f = os.path.join(args.pretrained_model_p + ".ckp")
        config_save_f = os.path.join(args.pretrained_model_p + ".cf")
    
    
        config = torch.load(config_save_f)
        model, _, _ = load_checkpoint(config, checkpoint_f)
    
    
        # model config
        args.EB_dp = 0.3
        args.FC_dp = 0.1
    
        # training config
        args.use_new_loss = False
        args.use_cls_loss = False
    
    
        #args.epochs = 10
        #args.epochs = 1
        args.warmup_epoch = 0
        args.patience_epoch = 3
    
        args.learning_rate = args.lr
        #args.learning_rate = 8e-4
        args.weight_decay = 8e-6
        args.l2_weight_decay = 5e-5
        args.max_grad_norm = 2.0
        args.lr_reduce_factor = 0.5
        args.lr_cooldown = 0
        args.threshold = .5
        args.adam_epsilon = 1e-8
        args.use_loss_sh = False
    #     args.is_iterare_info = not IS_DEBUG
        args.is_iterare_info = False

        args.device = device
    
        config = update_model_config(args, config, False)
    
        model.update_model_config(config)

        if torch.cuda.device_count() > 1:
            print("use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
    
        model.to(device)
        optimizer, scheduler = init_model_optimizer(model, args)
        scheduler.step(0)
    
        is_save_new_train = True
        model_name_prefix = 'bst_ft_build_%02d' % (_idx)
    
        args.checkpoint_f = os.path.join(args.model_dir, model_name_prefix + ".ckp")
        args.config_save_f = os.path.join(args.model_dir,  model_name_prefix + ".cf")
        torch.save(config, args.config_save_f)
        if is_save_new_train:
            torch.save(config, args.config_save_f)
    
    
        print("training------")
        train(model, optimizer, scheduler, train_dt, dev_dt, args, test_dt, is_save_new_train)
    
        model = None
        if not args.no_cuda:
            free_cuda()
        print("*********")
    
    
        print("--- %s seconds ---" % (time.time() - _start_time))   
        return 
    
        
    start_time = time.time()
    if args.models_number <= 0:
        args.models_number = 1
    print('will train {} models at {}'.format(args.models_number, args.model_dir))
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    for _i in range(1, args.models_number+1):
        #continue
        print('****\nbegin training 10X RENET2 models {}\n'.format(_i))
        build_dev(_i)

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
            print("use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

        model.to(device)
    
        pred_l, tru_l, S, pred_o = eval(model, dataloader_ft_sub, args, 'test')
        y_info = features_ft_sub[1].copy()
        y_info['pred'] = pred_l
        y_info['prob'] = pred_o
        
        cls_rst_file = os.path.join(args.gda_fn_d, "gda_rst_%02d.tsv" % (_i))
        y_info.to_csv(cls_rst_file, sep='\t', index=False)
        print('rst at {}'.format(cls_rst_file))


        y_info['pmid'] = y_info['pmid'].astype(str)
        y_info['geneId'] = y_info['geneId'].astype(str)
        y_info['diseaseId'] = y_info['diseaseId'].astype(str)
        y_info['label'] = y_info['label'].astype(str)
        S = evaluate_rst_all_info_err(y_info, 'pred', abs_s_df, _abs_s_df, abs_s_df_ner)
        print(S)

    print("training models ended (%.2f)" % (time.time() - start_time))
   
    s_arr = []

    m_df = 0
    for _i in range(args.models_number):
        cls_file = os.path.join(args.gda_fn_d, "gda_rst_%02d.tsv" % (_i+1))
        
        y_info = pd.read_csv(cls_file, sep='\t')
        y_info['pmid'] = y_info['pmid'].astype(str)
        y_info['geneId'] = y_info['geneId'].astype(str)
        y_info['diseaseId'] = y_info['diseaseId'].astype(str)
        y_info['label'] = y_info['label'].astype(str)
        
        S = evaluate_rst_all_info_err(y_info, 'pred', abs_s_df, _abs_s_df, abs_s_df_ner)
    #     print(S)
        s_arr.append(list(S))
        
        y_info.rename(columns={'pred':'pred_%02d'%(_i+1)}, inplace=True)
        y_info.rename(columns={'prob':'prob_%02d'%(_i+1)}, inplace=True)
    #     y_info = y_info[y_info['pred'] == 1]
        m_df = y_info if _i == 0 else m_df.merge(y_info, on=['pmid', 'geneId','diseaseId', 'label'], how='outer')
    
    m_df['hit_cnt'] = m_df.apply(lambda x: sum([x['pred_%02d'%(_i+1)] for _i in range(args.models_number)]), axis=1)
    
    s_arr = np.array(s_arr)
    
    print('{}X #, positive #, precision, precision[-], recall, recall[-]'.format(args.models_number))
    for r in s_arr:
        print(','.join([str(i) for i in r]))
    

    avg_l = list(np.mean(s_arr, axis=0))
    avg_l = ['avg'] + avg_l
    
    print(','.join(list(map(str, avg_l))))
    
    the_5th_rst = ''
    print('{}X assemble, positive #, precision, precision[-], recall, recall[-]'.format(args.models_number))
    for _i in range(1, args.models_number+1):
        ReNet_df = m_df[['pmid', 'geneId', 'diseaseId']].copy()
        ReNet_df['is_renet'] = m_df.apply(lambda x: 1 if x['hit_cnt'] >= _i else 0, axis=1)

        cols_n = ['prob_%02d' % (_i+1) for _i in range(args.models_number)]
        #print(m_df.head())
        ReNet_df['prob_avg'] = m_df[cols_n].mean(axis=1)
    
        _rst = evaluate_rst_all_info_err(ReNet_df, 'is_renet', abs_s_df, _abs_s_df, abs_s_df_ner)
        _rst = [_i] + list(_rst)
        print(','.join(list(map(str, _rst))))

        if _i == int(args.models_number/2):
            the_5th_rst = ','.join(list(map(str, _rst)))

    print('----\nthe assemble result:\n%s' % (the_5th_rst))
        

    ReNet_df = m_df[['pmid', 'geneId', 'diseaseId']].copy()
    ReNet_df['pred'] = m_df.apply(lambda x: 1 if x['hit_cnt'] >= int(args.models_number/2) else 0, axis=1)
    cols_n = ['prob_%02d' % (_i+1) for _i in range(args.models_number)]
    #print(m_df.head())
    ReNet_df['prob_avg'] = m_df[cols_n].mean(axis=1)
    #ReNet_df['prob_avg'] = ReNet_df['prob_avg'].round(decimals = 5)
    ReNet_df['prob_avg'] = ReNet_df['prob_avg'].map('{:,.5f}'.format)
    for _i in range(args.models_number):
        m_df[cols_n[_i]] = m_df[cols_n[_i]] .map('{:,.5f}'.format)
    ReNet_df['prob_X'] = m_df[cols_n].apply(lambda row: ';'.join(row.values.astype(str)), axis=1)
    
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
    print('getting NER names')
    
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
    _df = df_t.groupby(['pmid', 'geneId', 'diseaseId', 'id'], as_index=False)['name'].apply(lambda x: "%s" % ' | '.join(x)).reset_index()
    _df.rename(columns={0: "g_name"}, inplace=True)
    _df.sort_values(by=['pmid', 'geneId']).head()
    _df.drop(columns=['id'], inplace=True)
    df_g = _df[:]
    
    df_t = pd.merge(tar_set_df, anns_dis, how="left", left_on=['pmid', 'diseaseId'], right_on=['pmid', 'id'])
    df_t = df_t.drop_duplicates(['pmid', 'geneId', 'diseaseId', 'name'])
    _df = df_t.groupby(['pmid', 'geneId', 'diseaseId', 'id'], as_index=False)['name'].apply(lambda x: "%s" % ' | '.join(x)).reset_index()
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
    
    print("--- %s seconds ---" % (time.time() - start_time))

    cls_rst_file = os.path.join(args.gda_fn_d, "gda_rst.tsv")
    print('Found GDA records in %s' % (cls_rst_file))
    tar_set.to_csv(cls_rst_file, sep='\t', index=False)




def init_self_parser():
    # set up
    parser = argparse.ArgumentParser(description='RENET2 training models [assemble, full-text] and testing')
    parser.add_argument(
            "--raw_data_dir",
            default = "../data/ft_data/",
            type=str,
            help="input data folder",
    )
    parser.add_argument(
            "--annotation_info_dir",
            default = "../data/ft_info",
            type=str,
            help="annotation data folder",
    )

    parser.add_argument(
            "--model_dir",
            default = "../model",
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
    parser.add_argument(
            "--have_SiDa",
            default = None,
            type=str,
            help="SiDa file",
    )

    parser.add_argument(
            "--rst_file_prefix",
            default = 'ft_base',
            type=str,
            help="predicted GDP file",
    )

    parser.add_argument('--use_cuda', action='store_true', default=False,
                                help='enables CUDA training default: %(default)s')
    #parser.add_argument('--no_cuda', action='store_true', default=False,
    #                            help='disables CUDA training default: %(default)s')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
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
    parser.add_argument('--is_read_doc', action='store_true', default=True,
                                help='reading doc file, not renet2\'s abstracts file  default: %(default)s')
    parser.add_argument('--read_abs', action='store_true', default=False,
                                help='reading_abs_only default: %(default)s')
    parser.add_argument('--models_number', type=int, default=10, metavar='N',
                                help='number of models to train default: %(default)s')


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
    return parser

def main():
    # set up
    parser = init_self_parser()
    args = parser.parse_args()
    base_dir = os.path.dirname(__file__)  
    sys.path.insert(0, base_dir)

    get_index_path(args)

    args.no_cuda = not args.use_cuda

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    #print(args)

    
    renet2_train(args)





if __name__ == "__main__":
    main()
