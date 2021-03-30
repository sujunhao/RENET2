import argparse
import os
import re
import pickle
import random
import pandas as pd
import numpy as np
import torch
from pathlib import Path

from renet2.raw import load_documents, load_documents_ori
from sklearn.model_selection import train_test_split
#from tokenizers import BertWordPieceTokenizer
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, \
                             TensorDataset, WeightedRandomSampler

#from tqdm import tqdm
from tqdm.autonotebook import tqdm

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

# word_index
def loading_tokenizer(args):
    word_index_path = args.word_index_fn
    print("loading word index from", word_index_path)
    with open(word_index_path, 'rb') as fp:
        word_index = pickle.load(fp)
    print("loaded word index, voc size %s" % (len(word_index)))
    if "[X]" not in word_index:
        word_index["[X]"] = word_index['UUUNKKK']
    return word_index

def read_labels(label_path, gdas):
    pos_labels = pd.read_csv(label_path)
    pos_labels.pmid = pos_labels.pmid.astype(str)
    pos_labels.geneId = pos_labels.geneId.astype(str)
#     pos_labels['ann_label'] = pos_labels['label']
#     pos_labels['label'] = pos_labels['ori_label']\
#     print(len(pos_labels[pos_labels['label'] == 0.5]))
# #     pos_labels[pos_labels['label'] == 0.5]['label'] = 0
#     pos_labels.loc[pos_labels['label'] == 0.5, 'label'] = 0
    pos_labels = pos_labels.drop_duplicates()
    n_gdas = pd.merge(gdas, pos_labels, on=['pmid', "diseaseId", "geneId"], how="left")
    pos_labels.drop_duplicates()
    n_gdas = n_gdas.fillna(0)
    y = n_gdas.label.values
    return n_gdas, y

def padding_raw_input(word_seq, x_feature, doc_n, fix_snt_n, fix_token_n):
#     doc_n, fix_snt_n, fix_token_n = len(word_seq), 150, 150
    all_feature_f = np.zeros((doc_n, fix_snt_n, 2, fix_token_n), dtype='int32')
    for i in range(doc_n):
        i_n = min(fix_snt_n, len(word_seq[i]))
        for j in range(i_n):
            j_n = min(fix_token_n, len(word_seq[i][j]))
            all_feature_f[i, j, 0, :j_n] = word_seq[i][j][:fix_token_n]
#             all_feature_f[i, j, 1, :j_n] = [t[0] for t in x_feature[i][j]][:fix_token_n]
            all_feature_f[i, j, 1, :j_n] = [t for t in x_feature[i][j]][:fix_token_n]
    return all_feature_f

def tokenizer_doc(tokenizer, all_ori_seq, doc_n, fix_snt_n, _fix_token_n):
    all_feature_f = np.zeros((doc_n, fix_snt_n, _fix_token_n), dtype='int32')
#     fix_token_n = _fix_token_n - 2
    fix_token_n = _fix_token_n
    for i in tqdm(range(doc_n)):
        tmp_snt_i = 0
#         lst_l = tokenizer.encode(' '.join(all_ori_seq[i][0]), max_length=fix_token_n)
#         lst_l = (tokenizer.encode(' '.join(all_ori_seq[i][0])).ids)[:fix_token_n]
        lst_l = (tokenizer.cst_encode(' '.join(all_ori_seq[i][0]))[0])[:fix_token_n]
#         print('a', lst_l)
        for j in range(1, len(all_ori_seq[i])):
            if len(all_ori_seq[i][j]) <= 0:
                break
#             s_l = tokenizer.encode(' '.join(all_ori_seq[i][j]), max_length=fix_token_n)
#             s_l = (tokenizer.encode(' '.join(all_ori_seq[i][j])).ids)[:fix_token_n]
            s_l = (tokenizer.cst_encode(' '.join(all_ori_seq[i][j]))[0])[:fix_token_n]
#             print('b', s_l)
#             if len(s_l) <= 0:
#                 break
            if tmp_snt_i < fix_snt_n and len(lst_l) + len(s_l) > fix_token_n:
#                 tar_l = [tokenizer.cls_token_id] + lst_l[:][:_fix_token_n-2] + [tokenizer.sep_token_id]
                tar_l = lst_l[:]
#                 print('c', tar_l)
                all_feature_f[i, tmp_snt_i, :len(tar_l)] = tar_l[:]
                lst_l = s_l[:]
                tmp_snt_i += 1
            elif tmp_snt_i < fix_snt_n and len(lst_l) + len(s_l) <= fix_token_n:
                lst_l = lst_l + s_l
        if tmp_snt_i < fix_snt_n:
#             tar_l = [tokenizer.cls_token_id] + lst_l[:][:_fix_token_n-2] + [tokenizer.sep_token_id]
            tar_l = lst_l[:]
            all_feature_f[i, tmp_snt_i, :len(tar_l)] = tar_l[:]
            tmp_snt_i += 1
    return all_feature_f

def check_rec_full(df):
    # how many tokens reach last pos
    # how many snt reach last pos
    # entity empty rate
    snt_cnt = df.shape[0] * df.shape[1]
    snt_r_cnt = np.sum(df[:, :, 0, -1] > 0)
    
    doc_cnt = df.shape[0]
    doc_r_cnt = np.sum(df[:, -1, 0, 0] > 0)
    
    token_cnt = df.shape[0] * df.shape[1] * df.shape[-1]
    token_valid_cnt = np.sum(df[:, :, 0, :] > 0)

    print("total doc # %d, reach end # %d (%.2f%%)" % (doc_cnt, doc_r_cnt, doc_r_cnt*100./doc_cnt))
    print("total snt # %d, reach end # %d (%.2f%%)" % (snt_cnt, snt_r_cnt, snt_r_cnt*100./snt_cnt))
    print("total token # %d, not empty # %d valid rate: %.2f%%" % (token_cnt, token_valid_cnt, token_valid_cnt*100./token_cnt))
    return

    
def load_and_cache_data(args, get_id_l=False):
    fix_snt_n, fix_token_n = args.fix_snt_n, args.fix_token_n
    cached_features_file = os.path.join(
        args.raw_data_dir,
        "cached_{}_{}_{}_{}_{}".format(
            "all", 'abs' if args.read_abs else 'doc',
            args.max_doc_num,
            fix_snt_n, fix_token_n,
        ),
    )
    if args.using_new_tokenizer:
        cached_features_file += "_nt"

    if args.add_cache_suf:
        cached_features_file += args.add_cache_suf
    if os.path.exists(cached_features_file) and not args.overwrite_cache and not get_id_l:
        print("loading features from cached file %s", cached_features_file)
        features = pickle.load(open(cached_features_file, "rb" ))    
    else:
        print("no cached file %s" % (cached_features_file))
        print("creating features from dataset file at ", args.raw_data_dir)
        # loading raw data -> datasets
        text_path = os.path.join(args.raw_data_dir, args.file_name_doc) 
        sentence_path = os.path.join(args.raw_data_dir, args.file_name_snt)
        ner_path = os.path.join(args.raw_data_dir, args.file_name_ann)

        #text_path = os.path.join(args.raw_data_dir, "docs.txt")           
        ##sentence_path = os.path.join(args.raw_data_dir, "sentences.txt")
        #sentence_path = os.path.join(args.raw_data_dir, "sentences_p.txt")
        #ner_path = os.path.join(args.raw_data_dir, "anns.txt")

#         label_path = os.path.join(args.raw_data_dir, "labels.txt")
#         label_path = os.path.join(args.raw_data_dir, "labels_5.txt")
        label_path = os.path.join(args.raw_data_dir, args.label_f_name)

        print('reading files %s,%s,%s,%s' % (args.file_name_doc, args.file_name_snt, args.file_name_ann, args.label_f_name))
        
        if args.read_old_file:
            text_path = os.path.join(args.raw_data_dir, "abstracts.txt")
            label_path = os.path.join(args.raw_data_dir, "labels.csv")
        

#         gdas, word_seq, x_feature, bert_feature, x_len = load_documents(text_path, sentence_path, ner_path, tokenizer, args)
        gdas, word_seq, x_feature, all_ori_seq, x_len = load_documents(text_path, sentence_path, ner_path, args.ori_tokenizer, args)
        m_y, y = read_labels(label_path, gdas)
        print("reading label at {}".format(label_path))
        
        #gdas, word_seq, x_feature = load_documents_ori(text_path, sentence_path, ner_path, tokenizer)
        print("loaded document # ", len(gdas), len(word_seq), len(x_feature))
        
        all_feature_f = None
        if args.using_new_tokenizer:
            print("using new tokenizer")
#             all_feature_f = tokenizer_text(args.tokenizer, all_ori_seq, len(word_seq), fix_snt_n, fix_token_n)
            all_feature_f = tokenizer_doc(args.tokenizer, all_ori_seq, len(word_seq), fix_snt_n, fix_token_n)
        else: 
            print("padding document with fix length", fix_snt_n, fix_token_n)
            all_feature_f = padding_raw_input(word_seq, x_feature, len(word_seq), fix_snt_n, fix_token_n)
        
#         x_train, x_val, y_train, y_val = train_test_split(all_feature_f, m_y, test_size=0.2, random_state=42)
#         features = (x_train, x_val, y_train, y_val)
        
        features = (all_feature_f, m_y)
        check_rec_full(all_feature_f)
        if not args.no_cache_file:
            print("saving features into cached file ", cached_features_file)
            pickle.dump(features, open(cached_features_file, 'wb'), protocol=4)
#     x_train, x_val, y_train, y_val = features
    print("loading ended")
            
    return features

class doc_re_dataset(Dataset):
    def __init__(self, _x, _y):
        self.data_x = _x
        self.data_y = _y

    def __len__(self):
        return len(self.data_x)
    
    
    def __getitem__(self, index):
        tx = torch.tensor(self.data_x[index,:,:], dtype=torch.long)
#         ty = torch.tensor(self.data_y[index], dtype=torch.long)
        ty = torch.tensor(self.data_y[index], dtype=torch.float)
        return tx,ty

            
            
            
def convert_features_to_dataset(features, sp_arr = (0.6, 0.2, 0.2), is_shuffle = False):
    all_feature_f, m_y = features
        
    all_c, lst_c = len(m_y), 5000.
    if len(sp_arr) == 2 and ((lst_c / all_c) < sp_arr[1]):
        sp_arr[1] = lst_c / all_c
        sp_arr[0] = 1 - sp_arr[1]

    x_train, x_sub, y_train, y_sub = train_test_split(all_feature_f, m_y, test_size=1-sp_arr[0]/sum(sp_arr), \
                                shuffle=is_shuffle)
    if len(sp_arr) == 3:
        x_val, x_test, y_val, y_test = train_test_split(x_sub, y_sub, test_size=1-sp_arr[1]/sum(sp_arr[1:]), \
                                shuffle=is_shuffle)
    else:
        x_val, y_val = x_sub, y_sub
#     x_train, x_val, y_train, y_val = features
    y_train = y_train.label.values
    _k, _a = np.sum(y_train), len(y_train)
    print('dev', _k, _a, _k/_a)
#     train_x = torch.tensor(x_train, dtype=torch.long)
#     train_y = torch.tensor(y_train, dtype=torch.long)
#     train_dataset = TensorDataset(train_x, train_y)
    
    y_val = y_val.label.values
    _k, _a = np.sum(y_val), len(y_val)
    print('dev', _k, _a, _k/_a)
#     dev_x = torch.tensor(x_val, dtype=torch.long)
#     dev_y = torch.tensor(y_val, dtype=torch.long)
#     dev_dataset = TensorDataset(dev_x, dev_y)
    
    train_dataset = doc_re_dataset(x_train, y_train)
    dev_dataset = doc_re_dataset(x_val, y_val)
#     train_d_s = int(len(train_dataset) * 0.8)
#     dev_d_s = len(train_dataset) - train_d_s
#     train_dataset, dev_dataset = torch.utils.data.random_split(train_dataset, [train_d_s, dev_d_s])
    
    if len(sp_arr) == 3:
        y_test = y_test.label.values
        test_x = torch.tensor(x_test, dtype=torch.long)
        test_y = torch.tensor(y_test, dtype=torch.long)
        test_dataset = TensorDataset(test_x, test_y)
        return train_dataset, dev_dataset, test_dataset
    else:
        return train_dataset, dev_dataset

#def convert_features_to_dataset_single(features):
#    x_train, y_train = features
#    
#    y_train = y_train.label.values
#    _k, _a = np.sum(y_train), len(y_train)
#    print('creading dataset, positive GDA %d, all GDA %d, positive rate %.2f%%' % ( _k, _a, 100.*_k/_a))
##     train_x = torch.tensor(x_train, dtype=torch.long)
##     train_y = torch.tensor(y_train, dtype=torch.long)
#    
##     train_dataset = TensorDataset(train_x, train_y)
#    
#    train_dataset = doc_re_dataset(x_train, y_train)
#    
#    
#    return train_dataset
##     return train_dataset, x_train, y_train


def convert_features_to_dataset_cv_aug(features, index, features_aug):
    x_train, y_train = features
    y_train = y_train.label.values
    
    x_train = x_train[index]
    y_train = y_train[index]
    
    x_train_aug, y_train_aug = features_aug
    y_train_aug = y_train_aug.label.values   
    
    #merge
    x_train_m = np.concatenate((x_train, x_train_aug))
    y_train_m = np.concatenate((y_train, y_train_aug))
    
    _dataset = doc_re_dataset(x_train_m, y_train_m)
    return _dataset

def get_dataloader(_dataset, batch_size):
    _sampler = RandomSampler(_dataset)
#     weights = [10 if label == 1 else 1 for data, label in _dataset]
#     _sampler = WeightedRandomSampler(weights,num_samples=int(len(_dataset)/10),replacement=False)
    _dataloader = DataLoader(_dataset, sampler=_sampler, batch_size=batch_size, num_workers=8)
#     _dataloader = DataLoader(_dataset, sampler=_sampler, batch_size=batch_size, drop_last=True)
#     _dataloader = DataLoader(_dataset, sampler=_sampler, batch_size=args.batch_size,
#                             num_workers=4, pin_memory=True)
    return _dataloader

def convert_features_to_dataset_cv(features, index):
    x_train, y_train = features
    y_train = y_train.label.values
    
    x_train = x_train[index]
    y_train = y_train[index]
    
    _k, _a = np.sum(y_train), len(y_train)
    #print('postive cnt: ', _k, _a, _k/_a)

    _dataset = doc_re_dataset(x_train, y_train)
    return _dataset


def convert_features_to_dataset_single(features):
    x_train, y_train = features
    
    y_train = y_train.label.values
    _k, _a = np.sum(y_train), len(y_train)
    #print('creading dataset, positive GDA %d, all GDA %d, positive rate %.2f%%' % ( _k, _a, 100.*_k/_a))
    
    train_dataset = doc_re_dataset(x_train, y_train)
    return train_dataset, x_train, y_train





# dataset, psitive_rate and result sampler sampled number
def get_weitghted_sampler(_dataset, positive_rate = 5, require_n = None):
    label_lst = [int(i[1]) for i in _dataset]

    num_samples = len(label_lst)
    class_counts = [num_samples - sum(label_lst), sum(label_lst)]
    print(class_counts)
    class_weights = [num_samples/class_counts[i] for i in range(len(class_counts))]
    class_weights[0] *= positive_rate
    if require_n == None:
        require_n = class_counts[1] + class_counts[1] * positive_rate
    print(class_counts, require_n)
    weights = [class_weights[label_lst[i]] for i in range(int(num_samples))]
    weighted_sampler = WeightedRandomSampler(torch.DoubleTensor(weights), require_n)
    return weighted_sampler


# weight sample, the 0:0.5:1 class rate is in class rate
def get_weitghted_sampler_b(_dataset, positive_rate = 5, require_n = None, add_s_wgt = None):
    def get_c_index(x):
        if x == 1:
            return 2
        if x == 0.5:
            return 1
        else:
            return 0
    label_lst = [get_c_index(i[1]) for i in _dataset]

    num_samples = len(label_lst)
    c2_cnt = sum([1 for i in _dataset if i[1] == 0.5])
    c3_cnt = sum([1 for i in _dataset if i[1] == 1])
    c1_cnt = num_samples - c2_cnt - c3_cnt

    class_counts = [c1_cnt, c2_cnt, c3_cnt]

    #print('class count ', class_counts)
    class_weights = [num_samples/class_counts[i] for i in range(len(class_counts))]

    class_weights[0] *= positive_rate
    class_weights[1] *= (c2_cnt * 1.0 / c3_cnt)

    if require_n == None:
        require_n = int(class_counts[2] * sum([1, (c2_cnt * 1.0 / c3_cnt), positive_rate]))

    print('class cnt and sample number [0, 1/2, 1]: ', class_counts, require_n)

    weights = [class_weights[label_lst[i]] for i in range(int(num_samples))]
    if add_s_wgt:
        weights = [weights[i] * add_s_wgt[i] for i in range(len(weights))]

    weighted_sampler = WeightedRandomSampler(torch.DoubleTensor(weights), require_n, replacement=False)
    #weighted_sampler = WeightedRandomSampler(torch.DoubleTensor(weights), require_n)
    return weighted_sampler

#class BertWordPieceTokenizer_add_s(BertWordPieceTokenizer):
#    # force
#    def add_cst_special_tokens(self, lst):
#        self.cst_special_tokens = lst[:]
#        _sz = self.get_vocab_size()
#        self.cst_map = {lst[i]: i+_sz for i in range(len(lst))}
#        self.cst_sz = _sz + len(self.cst_map)
#        print('ori sz: %d, fx: sz %d, end sz: %d' % (self.ori_sz, _sz, self.cst_sz))
#        
#    def add_fx_map(self, fx_lst):
#        self.ori_sz = self.get_vocab_size()
#        self.fx_map = {fx_lst[i]: i+1 for i in range(len(fx_lst))}
#        
#    # get feature base on id
#    def get_fx_map(self, tokens):
#        tar_f = [0] * len(tokens)
#        tar_f[-1] = self.fx_map[tokens[-1]]
#        return tar_f
#        
#    def cst_encode_get_f(self, s):
#        rst_id = []
#        rst_tokens = []
#        rst_fx = []
#        all_s_l = s.split("$@")
#        for i in range(len(all_s_l)):
#            tar_s = all_s_l[i].strip()
#            tar_id, tar_tks = [], []
#            if i % 2 == 0:
#                encoded = self.encode(tar_s)
#                tar_id =  encoded.ids
#                tar_tks = encoded.tokens
#                
#                if len(tar_id) > 0:
#                    if i != len(all_s_l) - 1:
#                        tar_id = tar_id[:-1]
#                        tar_tks = tar_tks[:-1]
#                        
#                        rst_fx.extend(self.get_fx_map(encoded.tokens))
#                    else:
#                        tar_id = tar_id
#                        tar_tks = tar_tks
#                        rst_fx.extend([0] * len(tar_id))
#            else:
#                tar_tks = [tar_s]
#                if tar_s in self.cst_map:
#                    tar_id = [self.cst_map[tar_s]]
#                else:
#                    tar_id = [self.cst_map['<None>']]
#            rst_id.extend(tar_id)
#            rst_tokens.extend(tar_tks)
#        return rst_id, rst_tokens, rst_fx
#                
#        
#    def cst_encode(self, s):
#        rst_id = []
#        rst_tokens = []
#        all_s_l = s.split("$@")
#        for i in range(len(all_s_l)):
#            tar_s = all_s_l[i].strip()
#            if i % 2 == 0:
#                encoded = self.encode(tar_s)
#                tar_id =  encoded.ids
#                tar_tks = encoded.tokens
#            else:
#                tar_tks = [tar_s]
#                if tar_s in self.cst_map:
#                    tar_id = [self.cst_map[tar_s]]
#                else:
#                    tar_id = [self.cst_map['<None>']]
#            rst_id.extend(tar_id)
#            rst_tokens.extend(tar_tks)
#        return rst_id, rst_tokens



#def get_self_tokenizer(voc_f, id_tk_f):
#    vocab = voc_f
#    wp_tokenizer = BertWordPieceTokenizer_add_s(vocab, add_special_tokens=False)
#
#    # add special tokens
#    special_tokens_l = ['[D1]', '[eD1]','[G1]','[eG1]','[D2]','[eD2]','[G2]','[eG2]']
#    special_tokens_l.extend(['[M1]', '[eM1]','[M2]','[eM2]'])
#    for i in range(22):
#        special_tokens_l.append('[K' + str(i+1) + ']')
#
#    wp_tokenizer.add_fx_map(['[D1]', '[G1]', '[M1]', '[D2]', '[G2]','[M2]'])
#    wp_tokenizer.add_special_tokens(special_tokens_l)
#    
#    # add id tokens
#    s_f = id_tk_f
#    all_addtional_l = []
#    with open(s_f, 'r') as file:
#        for line in file:
#            line = line.strip()
#            all_addtional_l.append(line)
#
#    wp_tokenizer.add_cst_special_tokens(all_addtional_l)
#    return wp_tokenizer



def init_parser():
    # set up
    parser = argparse.ArgumentParser(description='RENET2 training models [assemble, full-text] and testing')
    parser.add_argument(
            "--raw_data_dir",
            default = "/mnt/bal31/jhsu/old/data/RENET_PMC_data",
            type=str,
            help="raw data dir",
    )
    parser.add_argument(
            "--modle_dir",
            default = "/mnt/bal31/jhsu/home/git/renet2/model",
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
            default = "anns.txt",
            type=str,
            help="anns file name",
    )
    parser.add_argument(
            "--word_index_fn",
            default = "./utils/word_index",
            type=str,
            help="word index data dir",
    )

    parser.add_argument('--no_cache_file', action='store_true', default=False,
                                help='disables using cache file for data input')

    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                                help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                                help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                                help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                                help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                                help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                                help='random seed (default: 42)')
    parser.add_argument('--save-interval', type=int, default=10, metavar='N',
                                help='how many batches to wait before checkpointing')
    parser.add_argument('--resume', action='store_true', default=False,
                                help='resume training from checkpoint')
    parser.add_argument('--read_abs', action='store_true', default=False,
                                help='reading_abs_only')
    parser.add_argument('--overwrite_cache', action='store_true', default=False,
                                help='overwrite_cache')
    parser.add_argument('--fix_snt_n', type=int, default=150, metavar='N',
                                help='number of snt')
    parser.add_argument('--fix_token_n', type=int, default=150, metavar='N',
                                help='number of tokens')
    parser.add_argument('--max_doc_num', type=int, default=0, metavar='N',
                                help='number of document')
    parser.add_argument('--is_read_doc', action='store_true', default=True,
                                help='reading doc file')
    parser.add_argument('--is_filter_sub', action='store_true', default=False,
                                help='filter pmid in abs data')
    parser.add_argument('--add_cache_suf', action='store_true', default=False,
                                help='cache file suffix')
    parser.add_argument('--read_old_file', action='store_true', default=False,
                                help='reading dga files')
    parser.add_argument('--read_ori_token', action='store_true', default=False,
                                help='get raw text')
    parser.add_argument('--using_new_tokenizer', action='store_true', default=False,
                                help='using new tokenizer')
    parser.add_argument('--not_x_feature', action='store_true', default=False,
                                help='do not use x_feature')
    return parser

def get_index_path(args):
    base_path = Path(__file__).parent
    args.word_index_fn = (base_path / args.word_index_fn).resolve()

def print_renet2_log():
    _log = """
  ____  _____ _   _ _____ _____ ____  
 |  _ \| ____| \ | | ____|_   _|___ \ 
 | |_) |  _| |  \| |  _|   | |   __) |
 |  _ <| |___| |\  | |___  | |  / __/ 
 |_| \_\_____|_| \_|_____| |_| |_____|

           """
    print(_log)



#def init_parser(parser):
#    parser.add_argument(
#            "--raw_data_dir",
#            default = "/mnt/bal31/jhsu/old/data/RENET_PMC_data",
#            type=str,
#            help="raw data dir",
#    )
#    parser.add_argument(
#            "--modle_dir",
#            default = "/mnt/bal31/jhsu/home/git/renet2/model",
#            type=str,
#            help="modle data dir",
#    )
#    parser.add_argument(
#            "--label_f_name",
#            default = "labels.txt",
#            type=str,
#            help="modle label name",
#    )
#    parser.add_argument(
#            "--word_index_fn",
#            default = "/mnt/bal31/jhsu/home/git/renet2/src/utils/word_index",
#            type=str,
#            help="word index data dir",
#    )
#
#
#    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
#                                help='input batch size for training (default: 64)')
#    parser.add_argument('--epochs', type=int, default=30, metavar='N',
#                                help='number of epochs to train (default: 30)')
#    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
#                                help='learning rate (default: 0.001)')
#    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
#                                help='SGD momentum (default: 0.5)')
#    parser.add_argument('--threshold', type=float, default=0.5, metavar='M',
#                                help='regression threshold (default: 0.5)')
#    parser.add_argument('--no-cuda', action='store_true', default=False,
#                                help='disables CUDA training')
#    parser.add_argument('--seed', type=int, default=42, metavar='S',
#                                help='random seed (default: 42)')
#    parser.add_argument('--save-interval', type=int, default=10, metavar='N',
#                                help='how many batches to wait before checkpointing')
#    parser.add_argument('--resume', action='store_true', default=False,
#                                help='resume training from checkpoint')
#    parser.add_argument('--read_abs', action='store_true', default=False,
#                                help='reading_abs_only')
#    parser.add_argument('--overwrite_cache', action='store_true', default=False,
#                                help='overwrite_cache')
#    parser.add_argument('--fix_snt_n', type=int, default=32, metavar='N',
#                                help='number of snt')
#    parser.add_argument('--fix_token_n', type=int, default=54, metavar='N',
#                                help='number of tokens')
#    parser.add_argument('--max_doc_num', type=int, default=0, metavar='N',
#                                help='number of document')
#    parser.add_argument('--is_read_doc', action='store_true', default=True,
#                                help='reading doc file')
#    parser.add_argument('--is_filter_sub', action='store_true', default=False,
#                                help='filter pmid in abs data')
#    parser.add_argument('--read_old_file', action='store_true', default=False,
#                                help='reading dga files')
#    parser.add_argument('--read_ori_token', action='store_true', default=False,
#                                help='get raw text')
#    parser.add_argument('--using_new_tokenizer', action='store_true', default=False,
#                                help='using new tokenizer')
#    parser.add_argument('--not_x_feature', action='store_true', default=False,
#                                help='do not use x_feature')
#    return
