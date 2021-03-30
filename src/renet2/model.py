import pandas as pd
import numpy as np
import random
import math
import torch
import re
#from tqdm import tqdm
from tqdm.autonotebook import tqdm


from torch import nn, optim, cuda
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import KFold



from pprint import pprint
from timeit import default_timer as timer


#args.num_embedding = 64
#args.cnn_out_c = 100
#args.rnn_out_f_n = 50
#args.rnn_num_directions = 2
#args.rnn_layers = 1
#args.window_sizes = [2, 3, 4, 5]

class Config: pass

def set_model_config(args, is_print_info=True):
    # model configure
    config = Config()
    config.max_token_n  = args.fix_token_n
    config.device = args.device
    config.not_x_feature = args.not_x_feature
    config.num_words = args.token_voc_l
    
    config.window_sizes = args.window_sizes
    config.num_embedding = args.num_embedding
    config.cnn_out_c = args.cnn_out_c
    config.rnn_out_f_n = args.rnn_out_f_n
    config.rnn_num_directions = args.rnn_num_directions
    config.rnn_layers = args.rnn_layers
    config.FC_dp = args.FC_dp
    config.EB_dp = args.EB_dp

    # training config
    config.use_new_loss = args.use_new_loss
    config.epochs = args.epochs
    config.warmup_epoch = args.warmup_epoch
    # in dev, if loss not improve wait x more epochs
    config.patience_epoch = args.patience_epoch
    config.learning_rate = args.learning_rate
    config.adam_epsilon = args.adam_epsilon 
    config.weight_decay = args.weight_decay 
    config.max_grad_norm = args.max_grad_norm 
    config.lr_reduce_factor = args.lr_reduce_factor 
    config.threshold = args.threshold 
    config.lr_cooldown = args.lr_cooldown 
    config.l2_weight_decay = args.l2_weight_decay 
    config.batch_size = args.batch_size

    if is_print_info:
        print('config ----------------')
        pprint(vars(config))
        print('       ----------------')

    return config


def update_model_config(args, config, is_print_info=True):
    config.FC_dp = args.FC_dp
    config.EB_dp = args.EB_dp

    config.device = args.device

    # training config
    config.use_new_loss = args.use_new_loss
    config.epochs = args.epochs
    config.warmup_epoch = args.warmup_epoch
    config.patience_epoch = args.patience_epoch
    config.learning_rate = args.learning_rate
    config.adam_epsilon = args.adam_epsilon 
    config.weight_decay = args.weight_decay 
    config.max_grad_norm = args.max_grad_norm 
    config.lr_reduce_factor = args.lr_reduce_factor 
    config.threshold = args.threshold 
    config.lr_cooldown = args.lr_cooldown 
    config.l2_weight_decay = args.l2_weight_decay 
    config.batch_size = args.batch_size

    if is_print_info:
        print('config ----------------')
        pprint(vars(config))
        print('       ----------------')

    return config

class Base_Net_Ori(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_token_n = config.max_token_n
        self.num_words = config.num_words
        self.num_embedding = config.num_embedding
        self.cnn_out_c = config.cnn_out_c
        self.window_sizes = config.window_sizes
        self.cnn_out_f_n = self.cnn_out_c * len(self.window_sizes)
        self.rnn_in_f_n = self.cnn_out_f_n
        self.rnn_out_f_n = config.rnn_out_f_n
        self.device = config.device
        self.not_x_feature = config.not_x_feature
        
        self.rnn_num_layers = config.rnn_layers
        self.rnn_num_directions = config.rnn_num_directions
        
        
        self.embeds = nn.Embedding(self.num_words, self.num_embedding, padding_idx=0)
        
        x_weight = torch.FloatTensor([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0], \
                                    [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, 1]])
        
        self.embeds_f = nn.Embedding.from_pretrained(x_weight, freeze=True)
        
        self.num_feature = self.num_embedding
        if not self.not_x_feature:
            self.num_feature += 4
        
        self.convs = nn.ModuleList([
                            nn.Sequential(
                            nn.Conv1d(in_channels=self.num_feature, 
                                      out_channels=self.cnn_out_c, stride=1, kernel_size=h),
                            nn.BatchNorm1d(self.cnn_out_c),
                            nn.ReLU(),
                            nn.MaxPool1d(kernel_size= math.floor((self.max_token_n-h)/1.+1)))
                            for h in self.window_sizes
                            ])

        self.rnn  = nn.GRU(self.rnn_in_f_n, self.rnn_out_f_n, batch_first=True,\
                             num_layers = self.rnn_num_layers, \
                             bidirectional=False if self.rnn_num_directions == 1 else True)
    
        
        self.fc_h = nn.Linear(self.rnn_out_f_n * self.rnn_num_directions,
                              int(self.rnn_out_f_n * self.rnn_num_directions))
        self.fc = nn.Linear(int(self.rnn_out_f_n * self.rnn_num_directions), 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def update_model_config(self, config):
        self.dropout_FC = nn.Dropout(p=config.FC_dp)
        self.dropout_EB = nn.Dropout(p=config.EB_dp)
        self.device = config.device

    def init_hidden(self, _batch_n):
        return torch.zeros(self.rnn_num_layers * self.rnn_num_directions, _batch_n, self.rnn_out_f_n).to(self.device)
    
    def forward(self, feature_x=None):
        _batch = len(feature_x)               
        self.rnn_hidden = self.init_hidden(_batch)       

        # word embedding
        snt_lengths = torch.sum(torch.gt(feature_x[:,:,0,0],0), 1)

        x1 = self.embeds(feature_x[:,:,0,:])
        x2 = self.embeds_f(feature_x[:,:,1,:])

        x = torch.cat((x1, x2), -1)
        x = torch.transpose(x.view(-1, self.max_token_n, self.num_feature), 1, 2)
        
        # cnn
        out = [conv(x) for conv in self.convs]
        x = torch.cat(out, dim=1)
        x = x.view(_batch, -1, self.cnn_out_f_n)
        
        # rnn
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(x, snt_lengths, batch_first=True, enforce_sorted=False)
        x, self.rnn_hidden = self.rnn(x_packed, self.rnn_hidden)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        last_seq = [x[e, i-1, :].unsqueeze(0) for e, i in enumerate(snt_lengths)]
        x = torch.cat(last_seq, dim=0)

        # FN
        x = self.fc_h(x)
        x = self.relu(x)
        
        x = self.fc(x)  
        x = self.sigmoid(x)
        x = torch.squeeze(x)
        return x

class Base_Net_Ori_V(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_token_n = config.max_token_n
        self.num_words = config.num_words
        self.num_embedding = config.num_embedding
        self.cnn_out_c = config.cnn_out_c
        self.window_sizes = config.window_sizes
        self.cnn_out_f_n = self.cnn_out_c * len(self.window_sizes)
        self.rnn_in_f_n = self.cnn_out_f_n
        self.rnn_out_f_n = config.rnn_out_f_n
        self.device = config.device
        self.not_x_feature = config.not_x_feature
        
        self.rnn_num_layers = config.rnn_layers
        self.rnn_num_directions = config.rnn_num_directions
        
        self.dropout_FC = nn.Dropout(p=config.FC_dp)
        self.dropout_EB = nn.Dropout(p=config.EB_dp)
        
        self.embeds = nn.Embedding(self.num_words, self.num_embedding, padding_idx=0)
        
        x_weight = torch.FloatTensor([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0], \
                                    [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, 1]])
        
        self.embeds_f = nn.Embedding.from_pretrained(x_weight, freeze=True)
        
        self.num_feature = self.num_embedding
        if not self.not_x_feature:
            self.num_feature += 4
        
        self.convs = nn.ModuleList([
                            nn.Sequential(
                            nn.Conv1d(in_channels=self.num_feature, 
                                      out_channels=self.cnn_out_c, stride=1, kernel_size=h),
                            nn.BatchNorm1d(self.cnn_out_c),
                            nn.ReLU(),
                            nn.MaxPool1d(kernel_size= math.floor((self.max_token_n-h)/1.+1)))
                            for h in self.window_sizes
                            ])

        self.rnn  = nn.GRU(self.rnn_in_f_n, self.rnn_out_f_n, batch_first=True,\
                             num_layers = self.rnn_num_layers, \
                             bidirectional=False if self.rnn_num_directions == 1 else True)
    
        
        self.fc_h = nn.Linear(self.rnn_out_f_n * self.rnn_num_directions,
                              int(self.rnn_out_f_n * self.rnn_num_directions))
        self.fc = nn.Linear(int(self.rnn_out_f_n * self.rnn_num_directions), 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
    def init_hidden(self, _batch_n):
        return torch.zeros(self.rnn_num_layers * self.rnn_num_directions, _batch_n, self.rnn_out_f_n).to(self.device)
    
    def forward(self, feature_x=None):
        _batch = len(feature_x)               
        self.rnn_hidden = self.init_hidden(_batch)       

        # word embedding
        snt_lengths = torch.sum(torch.gt(feature_x[:,:,0,0],0), 1)

        x1 = self.embeds(feature_x[:,:,0,:])
        x1 = self.dropout_EB(x1)
        x2 = self.embeds_f(feature_x[:,:,1,:])

        x = torch.cat((x1, x2), -1)
        x = torch.transpose(x.view(-1, self.max_token_n, self.num_feature), 1, 2)
        
        # cnn
        out = [conv(x) for conv in self.convs]
        x = torch.cat(out, dim=1)
        x = x.view(_batch, -1, self.cnn_out_f_n)
        
        # rnn
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(x, snt_lengths, batch_first=True, enforce_sorted=False)
        x, self.rnn_hidden = self.rnn(x_packed, self.rnn_hidden)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        last_seq = [x[e, i-1, :].unsqueeze(0) for e, i in enumerate(snt_lengths)]
        x = torch.cat(last_seq, dim=0)

        # FN
        x = self.fc_h(x)
        #x = self.relu(x)
        x = self.tanh(x)
        x = self.dropout_FC(x)
        
        x = self.fc(x)  
        x = self.sigmoid(x)
        x = torch.squeeze(x)
        return x

class Base_Net(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_token_n = config.max_token_n
        self.num_words = config.num_words
        self.num_embedding = config.num_embedding
        self.cnn_out_c = config.cnn_out_c
        self.window_sizes = config.window_sizes
        self.cnn_out_f_n = self.cnn_out_c * len(self.window_sizes)
        self.rnn_in_f_n = self.cnn_out_f_n
        self.rnn_out_f_n = config.rnn_out_f_n
        self.device = config.device
        self.not_x_feature = config.not_x_feature
        
        self.rnn_num_layers = config.rnn_layers
        self.rnn_num_directions = config.rnn_num_directions
        
        
        self.embeds = nn.Embedding(self.num_words, self.num_embedding, padding_idx=0)
        
        x_weight = torch.FloatTensor([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0], \
                                    [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, 1]])
        
        self.embeds_f = nn.Embedding.from_pretrained(x_weight, freeze=True)
        
        self.num_feature = self.num_embedding
        if not self.not_x_feature:
            self.num_feature += 4
            #self.rnn_in_f_n += 4
        
        self.convs = nn.ModuleList([
                            nn.Sequential(
                            nn.Conv1d(in_channels=self.num_feature, 
                                      out_channels=self.cnn_out_c, stride=1, kernel_size=h),
                            #nn.BatchNorm1d(num_features=self.cnn_out_c, affine=False), 
                            nn.ReLU(),
#                           nn.PReLU(),
#                           nn.Softplus(beta=1, threshold=20),
                            nn.MaxPool1d(kernel_size= math.floor((self.max_token_n-h)/1.+1)))
                            for h in self.window_sizes
                            ])
        self.dropout = nn.Dropout(p=0.5)
        self.dropout_s = nn.Dropout(p=0.3)
        self.dropout_1 = nn.Dropout(p=0.1)

        self.dropout_FC = nn.Dropout(p=config.FC_dp)
        self.dropout_EB = nn.Dropout(p=config.EB_dp)
    
        self.rnn  = nn.LSTM(self.rnn_in_f_n, self.rnn_out_f_n, batch_first=True,\
                            num_layers = self.rnn_num_layers, \
                            #dropout = 0.3, 
                            bidirectional=False if self.rnn_num_directions == 1 else True)
        #self.rnn  = nn.GRU(self.rnn_in_f_n, self.rnn_out_f_n, batch_first=True,\
        #                     num_layers = self.rnn_num_layers, \
        #                     bidirectional=False if self.rnn_num_directions == 1 else True)
    

        
        #self.fc = nn.Linear(int(self.rnn_out_f_n * self.rnn_num_directions/2), 1)
        #self.fc_h = nn.Linear(self.rnn_out_f_n * self.rnn_num_directions,
        #                      int(self.rnn_out_f_n * self.rnn_num_directions/2))
        self.fc_h = nn.Linear(self.rnn_out_f_n * self.rnn_num_directions,
                              int(self.rnn_out_f_n * self.rnn_num_directions))
        self.fc_h1 = nn.Linear(self.rnn_out_f_n * self.rnn_num_directions, 
                              self.rnn_out_f_n * self.rnn_num_directions)
        self.fc = nn.Linear(int(self.rnn_out_f_n * self.rnn_num_directions), 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def update_model_config(self, config):
        self.dropout_FC = nn.Dropout(p=config.FC_dp)
        self.dropout_EB = nn.Dropout(p=config.EB_dp)
        self.device = config.device

        
    def init_hidden(self, _batch_n):
        return (torch.zeros(self.rnn_num_layers * self.rnn_num_directions, _batch_n, self.rnn_out_f_n).to(self.device), 
                torch.zeros(self.rnn_num_layers * self.rnn_num_directions, _batch_n, self.rnn_out_f_n).to(self.device))
        #return torch.zeros(self.rnn_num_layers * self.rnn_num_directions, _batch_n, self.rnn_out_f_n).to(self.device)
    
    def forward(self, feature_x=None):
        _batch = len(feature_x)               
        self.rnn_hidden = self.init_hidden(_batch)       

        # word embedding
        x = None
        snt_lengths = None
        if self.not_x_feature:
            x = self.embeds(feature_x[:,:,:])
            ori_token_l = 32000
#             ori_token_l = 30522
            subset_x = x<ori_token_l
            x = self.dropout(x * subset_x.float()) + x * (~subset_x).float()
#             x[subset_x] = self.dropout(x[subset_x])
            
            snt_lengths = torch.sum(torch.gt(feature_x[:,:,0],0), 1)
        else:
            snt_lengths = torch.sum(torch.gt(feature_x[:,:,0,0],0), 1)
            x1 = self.embeds(feature_x[:,:,0,:])
            x1 = self.dropout_EB(x1)
            # feature embedding
            x2 = self.embeds_f(feature_x[:,:,1,:])
#             print(x2.shape)
            snt_x2 = torch.sum(x2, 2)
#             print(snt_x2.shape, snt_x2)
            # total embedding
            x = torch.cat((x1, x2), -1)

        x = torch.transpose(x.view(-1, self.max_token_n, self.num_feature), 1, 2)
        
        # cnn
        out = [conv(x) for conv in self.convs]
        x = torch.cat(out, dim=1)
        x = x.view(_batch, -1, self.cnn_out_f_n)
        
#         if not self.not_x_feature:
        #x = torch.cat((snt_x2, x), -1)
        
        # rnn
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(x, snt_lengths, batch_first=True, enforce_sorted=False)
        # fpr DataParallel
        self.rnn.flatten_parameters()
        x, self.rnn_hidden = self.rnn(x_packed, self.rnn_hidden)

        # (1) get rnn out from hidden
#         x = torch.squeeze(self.rnn_hidden[0][-1, :, :])
        
# #         get rnn out from all out
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        last_seq = [x[e, i-1, :].unsqueeze(0) for e, i in enumerate(snt_lengths)]
        x = torch.cat(last_seq, dim=0)

        #all_seq = [torch.cat([x[e, 0, :].unsqueeze(0), x[e, i-1, :].unsqueeze(0)], dim=1) for e, i in enumerate(snt_lengths)]
        ##print(last_seq[0].shape, all_seq[0].shape)
        #x = torch.cat(all_seq, dim=0)

        # FN

        x = self.fc_h(x)
        x = self.tanh(x)
        x = self.dropout_FC(x)
        
        x = self.fc_h1(x)
        x = self.tanh(x)
        x = self.dropout_FC(x)
        
        
        x = self.fc(x)  
        x = self.sigmoid(x)
        #x = torch.squeeze(x)


        #print(x.shape)
        x = torch.squeeze(x, 1)
        #print(x.shape)
#         x = torch.cat((x, 1-x), dim=1)
#         print(x)
        
        return x

#         # word embedding
#         x1 = self.embeds(feature_x[:,:,0,:])
#         x1 = self.dropout(x1)
        
#         # feature embedding
#         x2 = self.embeds_f(feature_x[:,:,1,:])
# #         x2 = self.dropout_1(x2)
        
#         # total embedding
#         x = torch.cat((x1, x2), -1)
# #         x = self.dropout_s(x)

#         snt_lengths = torch.sum(torch.gt(feature_x[:,:,0,0],0), 1)




#         # (2) integreate in pack seq
#         rnn_x = x.data
#         rnn_x = self.relu(self.fc_h(rnn_x))
#         rnn_x = self.dropout(rnn_x)
#         rnn_x = self.fc(rnn_x)
#         new_x = nn.utils.rnn.PackedSequence(rnn_x, x.batch_sizes)
#         new_x, _ = torch.nn.utils.rnn.pad_packed_sequence(new_x, batch_first=True)
        
        
#         # find max of rnn output
#         base = self.softmax(new_x)
#         tt, ti = torch.max(base[:,:,1], 1)
#         tx = new_x.gather(1, ti.repeat(2, 1).transpose(0, 1).view(-1, 1, 2)).squeeze()
#         x1 = tx
        
#         # get last part
#         last_pred = [new_x[e, i-1, :].unsqueeze(0) for e, i in enumerate(snt_lengths)]
#         x2 = torch.cat(last_pred, dim=0)
    
#         x = (x1 + x2) / 2
#         return x
    
def load_checkpoint(config, ck_f_name, Model_class=Base_Net):
    checkpoint = torch.load(ck_f_name, map_location='cpu')
    model = Model_class(config)
    model.load_state_dict(checkpoint['model'])
    #print("loaded model")
    #model.to(config.device)

    optimizer, scheduler = init_model_optimizer(model, config)
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler = checkpoint['scheduler']
    return model, optimizer, scheduler

class Regularization(torch.nn.Module):
    def __init__(self,weight_decay=1e-2,p=2):
        '''
        :param model 
        :param weight_decay: lambda
        :param p: the order of norm  2 for l2
        '''
        super(Regularization, self).__init__()
        self.weight_decay=weight_decay
        self.p=p

    def to(self,device):
        self.device=device
        super().to(device)
        return self

    def forward(self, model):
        self.weight_list=self.get_weight(model)#update
        reg_loss = self.regularization_loss(self.weight_list)
        #return reg_loss
        return reg_loss.unsqueeze(0)

    def get_weight(self,model):
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name and param.requires_grad:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self,weight_list):
        # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
        # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
        # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
        # reg_loss=torch.FloatTensor([0.]).to(self.device)
        reg_loss=0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=self.p)
            reg_loss = reg_loss + l2_reg

        reg_loss=self.weight_decay*reg_loss
        return reg_loss

    def weight_info(self,model):
        print("---------------regularization weight---------------")
        weight_list = self.get_weight(model)
        for name ,w in weight_list:
            print(name)
        print("---------------------------------------------------")


def mse_loss(input, target):
#     return torch.mean((input - target) ** 2)
    SE = (input - target)**2
    FP = (input >= 0.33) & (target < 0.5)
    FN = (input <= 0.66) & (target > 0.5)
    #CLASS_1 = (target >= 0.9)
    #CLASS_0 = (target <= 0.1)
    #CLASS_u = (target > 0.1) & (target < 0.9)
    #TP = (input >= 0.6) & (target > 0.5)
    #TN = (input <= 0.4) & (target < 0.5)
    #
    #Total_E = SE
    #Total_E += (0.5 - torch.abs(input - 0.5)) * (TP.float() * 8. +  TN.float() * 1.)
#   #  Total_E = SE + ((0.5 - torch.abs(input - 0.5)) **2) * (0.3)
# # #    Total_E = SE  + SE * CLASS_1.float() * 2.
    #Total_E += (    SE * CLASS_1.float() * 1.  \
    #              + SE * CLASS_0.float() * 1.  \
    #              + SE * CLASS_u.float() * 0.)
    Total_E = SE
    #Total_E += (    SE * CLASS_1.float() * 5.  \
    #              + SE * CLASS_0.float() * 0.  \
    #              + SE * CLASS_u.float() * 0.)
    #Total_E = SE + SE * FP.float() * .5 + SE * FN.float()* 2.
    Total_E = SE + SE * FP.float() * 2. + SE * FN.float()* .5
    #Total_E = SE + SE * FP.float() * .5 + SE * FN.float()* 1.
# # #     ABS = torch.abs(input - target)
# # #     ABSE = ABS * (ABS > 0.3).float()
# # #     Total_E = SE + ABSE    
# # # #     Total_E = SE
# # #     SE_W = (target > 0.5).float()
# # #     Total_E = Total_E + Total_E * SE_W * 2.
    
# # #     Total_E = SE + input * target
    return torch.mean(Total_E)


DEBEG_ONE_STEP = False 
# if not save_model only run a epoch
# def train(model, optimizer, train_dataloader, dev_dataloader, args, save_model = True):
def train(model, optimizer, scheduler, train_dataloader, dev_dataloader, args, test_dataloader=None, save_model = True):   
    start = timer()
    reg_loss = None
    reg_loss = Regularization(weight_decay = args.l2_weight_decay, p=2).to(args.device)
    
    train_losses = []

    lst_f1, lst_loss = 0, 1
    bst_dev_f1, bst_t = 0, 0
    bst_dev_auc, bst_dev_loss = 0, 1
    gloabl_step, best_step_i = 0, 0
    end_train_f = 0
    test_score = None
    _rst = None
    print("training begin")
    args.save_step = 10
    threshold = args.threshold
    is_disable = args.is_iterare_info
    try:
        for e_i in range(args.epochs):
    #         if e_i >= 2:
    #             break
            model.train()

            pred_o = [0, 1]
            pred_l, tru_l = [0, 1], [0, 1]
            #epoch_iterate = tqdm(train_dataloader, desc = "Iteration", disable = is_disable, ncols=800)
            epoch_iterate = tqdm(train_dataloader, desc = "Iteration", disable = is_disable)
            for i, (feature_x, labels) in enumerate(epoch_iterate):
                # sort the data first from rnn
#                 snt_lengths = torch.sum(torch.gt(feature_x[:,:,0,0],0), 1)
#                 lengths, perm_index = snt_lengths.sort(0, descending=True)
                
                try:
                    if args.no_ambiguous_label:
                        labels[labels==.5] = 0
                except:
                    pass


#                 labels[labels==.5] = 0
#                 print(labels)
#                 labels = labels[perm_index]
#                 feature_x = feature_x[perm_index]

        #         print(feature_x, labels)
    #             feature_x = feature_x[:, :54, :, :]
                feature_x = feature_x.to(device=args.device, non_blocking=True)
                labels = labels.to(device=args.device, non_blocking=True)

                optimizer.zero_grad()
                
                output = model(feature_x)
                output = output.view(-1)   #batch_n = 1 case
#                 pred = output.argmax(1, keepdim=True)
                pred = (output > threshold).to(dtype=int)
                pred_o.extend([i for i in output.tolist()])
        #         print(pred)
                pred_l.extend([i for i in pred.tolist()])


                loss_F = F.mse_loss
                try:
                    if args.use_new_loss:
                        loss_F = mse_loss
                    if args.use_cls_loss:
                        loss_F = F.binary_cross_entropy
                        labels[labels==.5] = 0
                        #print('use new loss')
                except:
                    pass
                loss = loss_F(output, labels)

#                 loss = F.cross_entropy(output, labels, weight=weights)
    #             loss = focal_loss(output, labels)
    #             loss = F.cross_entropy(output, labels)

                if reg_loss:
                    loss = loss + reg_loss(model)

                loss.backward()
                train_losses.append(loss.item())
                
#                 nn.utils.clip_grad_norm_(model.parameters(), 1.)
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    
                optimizer.step()
#                 scheduler.step()
#                 print(pred, labels)
                labels = (labels>threshold).to(dtype=int)
                tru_l.extend([i for i in labels.tolist()])
#                 print(pred, labels)

        

                gloabl_step += 1
                if DEBEG_ONE_STEP:
                    return
                if i % args.save_step == 0:
                    precision, recall, f1, _ = \
                    precision_recall_fscore_support(tru_l, pred_l, average='binary',zero_division=1)
                    auc_s = roc_auc_score(tru_l, pred_o)

                    # at least 1000 records to compute a valid F1 score
                    if f1 < 1.0 and f1 > lst_f1 and loss.item() <= lst_loss and \
                            i >= (len(epoch_iterate) / 3):
    #                         (i * args.batch_size) >= 5000:
                        #torch.save(model.state_dict(), args.model_save_dir+"_t")
    #                     torch.save(optimizer.state_dict(), '../temp/optimizer.pth')
    #                     torch.save(train_losses, '../temp/train_losses.pth')
                        lst_f1 = f1
                        if lst_loss - loss.item() <= (lst_loss / 5):
                            lst_loss = loss.item()
                        best_step = gloabl_step % args.save_step
    #                     epoch_iterate.set_description("Iteration  update pth ")

                    epoch_iterate.set_description("It { L %.5f, %.3f, %.3f, %.3f,  %.3f/ BF %.3f} " \
                                    % (loss.item(), \
                                    precision, recall, f1, auc_s, lst_f1))
#                     epoch_iterate.set_description("It { L %.5f, %.3f, %.3f, %.3f / BF %.3f} " % (loss.item(),\
#                                     precision, recall, f1, lst_f1))
                    epoch_iterate.refresh() # to show immediately the update
    #                 scheduler.step()
    #                 epoch_iterate.write('Epoch {}, lr {}'.format(
    #                             e_i, optimizer.param_groups[0]['lr']))
                if f1 < lst_f1 - 0.2 and ((gloabl_step % args.save_step) > (best_step % args.save_step) + 20):
                    end_train_f = 1
                    print("cnt_1")
                    break
                del loss, output, pred
                
            print("e_%d" % (e_i+1), end=' ')
            dev_loss, dev_precision, dev_recall, dev_f1, dev_auc = 0., 0, 0, 0, 0
            if dev_dataloader:
                _, _, S, _ = eval(model, dev_dataloader, args)
                dev_loss, dev_precision, dev_recall, dev_f1, dev_auc = S
    #             scheduler.step(dev_auc)
                print("dev: %.5f, %.3f, %.3f, %.3f, %.3f" % (dev_loss, dev_precision, dev_recall, dev_f1, dev_auc), end=' ')
                if args.warmup_epoch > e_i:
                    print("w", end=' ')
    #                 print("w")
    #                 continue

    #             _, _, _, _, _ = eval(model, test_dataloader, args, 'test')
    #             if bst_dev_f1 < dev_f1:
    #                 bst_dev_f1 = dev_f1
    #             if e_i >= 3 and bst_dev_auc - dev_auc < 5e-2 and (dev_loss - bst_dev_loss < 2e-3) \
    #                 and (bst_dev_auc - dev_auc < 0 or dev_loss - bst_dev_loss < 0):
    #                 bst_dev_auc = dev_auc if bst_dev_auc < dev_auc else bst_dev_auc
    #                 bst_dev_loss = dev_loss if dev_loss < bst_dev_loss else bst_dev_loss
    #                 bst_dev_auc = (dev_auc + bst_dev_auc)/2.
    #                 bst_dev_loss = (dev_loss + bst_dev_loss)/2.

            #if (not args.use_loss_sh) or (e_i >= args.warmup_epoch and bst_dev_auc - dev_auc < 1e-2 and bst_dev_f1 - dev_f1 < 1e-2 \
            #                and (dev_loss - bst_dev_loss < 5e-3) \
            #      and (bst_dev_auc <= dev_auc or bst_dev_f1 <= dev_f1) and (dev_loss - bst_dev_loss < 3e-3)):
#                 and (bst_dev_auc <= dev_auc or bst_dev_f1 <= dev_f1 or dev_loss <= bst_dev_loss):
            IF_save_model = False
            if (not args.use_loss_sh) or (e_i >= args.warmup_epoch and bst_dev_loss > dev_loss):
                bst_dev_f1 = dev_f1 if bst_dev_f1 <= dev_f1 else bst_dev_f1
                bst_dev_auc = dev_auc if bst_dev_auc <= dev_auc else bst_dev_auc
                bst_dev_loss = dev_loss if dev_loss < bst_dev_loss else bst_dev_loss

                if save_model:
                    IF_save_model = True

                best_step_i = e_i
                print('*', end='')
            else:
                print('', end='')
                   
            if test_dataloader:
                _rst = eval(model, test_dataloader, args, 'test')
                test_score = _rst[2]
                _loss, _precision, _recall, _f1, _auc_s = test_score
                print(" test rst: [%.5f, %.4f, %.4f, %.4f, %.4f]" % (_loss, _precision, _recall, _f1, _auc_s))
            else:
                print("")

            if scheduler:
#                 scheduler.step(dev_loss)
                scheduler.step(dev_loss if args.use_loss_sh else 1)

            IF_SAVE_TMP_MODEL = False
            if IF_save_model and IF_SAVE_TMP_MODEL:
                checkpoint = {'model': model.state_dict(),
                              'optimizer': optimizer.state_dict(),
                              'scheduler': scheduler}
                torch.save(checkpoint, args.checkpoint_f + '_%03d' % (e_i+1))
                print('saved checkpoint in %s' % (args.checkpoint_f + '_%03d' % (e_i+1)))
            
            if args.warmup_epoch > e_i:
                continue
            #if e_i >= args.zero_patience_epoch and test_score and test_score[3] < 1e-3:
            #    break
            if e_i > best_step_i + args.patience_epoch:
                break
            if end_train_f == 1:
                break
            if DEBEG_ONE_STEP:
                break
    except (KeyboardInterrupt) as e:
        print("KeyboardInterrupt")
    except:
        raise
    
    if save_model:
        checkpoint = {'model': model.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'scheduler': scheduler}
        torch.save(checkpoint, args.checkpoint_f)
        print('saved checkpoint in %s' % (args.checkpoint_f))
    end = timer()
    print("training end, used %.2f s" % (end - start))

    return _rst

    
def get_pred_threshold(model, dataloader, args):
    pred_l, tru_l, f1, auc, pre_o = eval(model, dataloader, args)

    def to_labels(pos_probs, threshold):
        return (pos_probs >= threshold).astype('int')
    
    thresholds = np.arange(0, 1, 0.01)
    # evaluate each threshold
    scores = [f1_score(tru_l, to_labels(pre_o, t)) for t in thresholds]
    # get best threshold
    ix = np.argmax(scores)
    print('Threshold, f1\n%.3f, %.5f' % (thresholds[ix], scores[ix]))
    return thresholds[ix]
    
def eval(model, the_dataloader, args, pre_f = 'dev'):

    reg_loss = None
    reg_loss = Regularization(weight_decay = args.l2_weight_decay, p=2).to(args.device)

    model.eval()
    test_loss, correct = 0, 0
    #pred_o = [0, 1]
    #pred_l, tru_l = [0, 1], [0, 1]
    pred_l = []
    pred_o = []
    tru_l = []
    threshold = args.threshold
    is_disable = args.is_iterare_info
    #is_disable = True
    with torch.no_grad():
        #epoch_iterate = tqdm(the_dataloader, desc = "evaluation", disable = is_disable, ncols=800)
        epoch_iterate = tqdm(the_dataloader, desc = "evaluation", disable = is_disable)
        for feature_x, labels in epoch_iterate:
#             snt_lengths = torch.sum(torch.gt(feature_x[:,:,0,0],0), 1)
#             lengths, perm_index = snt_lengths.sort(0, descending=True)
#             labels = labels[perm_index]
#             feature_x = feature_x[perm_index]
        
            #feature_x = feature_x.to(device=args.device, non_blocking=True)
            #labels_1 = labels.to(device=args.device, non_blocking=True)
            feature_x = feature_x.to(device=args.device)
            labels_1 = labels.to(device=args.device)
#             lengths = lengths.to(device=args.device, non_blocking=True)
            
#             _batch_n = len(feature_x)
#             model.rnn_hidden = model.init_hidden(_batch_n)
#             output = model(feature_x, lengths, _batch_n)
            output = model(feature_x)
            output = output.view(-1)   #batch_n = 1 case
#             pred = output.argmax(1, keepdim=True)
            pred = (output > threshold).to(dtype=int)
#             labels = labels_1.view(_batch_n, 1).to(device=args.device, non_blocking=True, dtype=torch.float)
#             labels = labels_1.view(-1, 1).to(device=args.device, non_blocking=True, dtype=torch.float)
            labels = labels_1
#             labels = torch.cat((labels, 1-labels), dim=1).to(device=args.device, non_blocking=True, dtype=torch.float)
#             output = torch.log(output).to(device=args.device, non_blocking=True, dtype=torch.float)
#             loss = F.kl_div(output, labels, reduction='batchmean')
            loss_F = F.mse_loss
            try:
                if args.use_new_loss:
                    loss_F = mse_loss
                if args.use_cls_loss:
                    loss_F = F.binary_cross_entropy
                    labels[labels==.5] = 0
                    #print('use new loss')
            except:
                pass
            loss = loss_F(output, labels)
            if reg_loss:
                loss = loss + reg_loss(model)
#             loss = mse_loss(output, labels)
    
    
#             test_loss += F.cross_entropy(output, labels)
            test_loss += loss.item()
            
#             pred = output.argmax(1, keepdim=False)
            ori_output = torch.zeros_like(output)
#             ori_output[perm_index] = output
            ori_output = output
    
#             try:
#                 pred_o.extend([i for i in ori_output.tolist()])
#             except:
#                 print(labels, ori_output)
#                 pred_o.extend([ori_output])
            pred_o.extend([i for i in ori_output.tolist()])
            
            ori_pred = torch.zeros_like(pred)
#             ori_pred[perm_index] = pred
            ori_pred = pred
            
            ori_labels = torch.zeros_like(labels_1)
#             ori_labels[perm_index] = labels_1
            ori_labels = labels_1
            ori_labels = (ori_labels>threshold).to(dtype=int)
            
            
            pred_l.extend([i for i in ori_pred.tolist()])
            tru_l.extend([i for i in ori_labels.tolist()])
            del loss, output, pred, ori_output
            
#             correct += pred.eq(labels.view_as(pred)).sum().item()
#     print(pred_o)
    test_loss /= len(pred_l)
    acc = correct / len(pred_l)
    precision, recall, f1, auc_s = 0, 0, 0, 0 
    try:
        precision, recall, f1, _ = precision_recall_fscore_support(tru_l, pred_l, average='binary', zero_division=1)
        auc_s = roc_auc_score(tru_l, pred_o)
    except ValueError:
        pass
#     print("precision %.3f, recall %.3f, F1 %.3f " % (precision, recall, f1))
#     print("%.3f, %.3f, %.3f, %.3f" % (precision, recall, f1, auc_s), end=' ')
    Score = (test_loss, precision, recall, f1, auc_s)
    return (pred_l, tru_l, Score, pred_o)

#def model_init_w(model):
#    print("init model")
#    def weight_init(m):
#        if isinstance(m, nn.Embedding):
#            #print(m)
#            #nn.init.xavier_normal_(m.weight,  gain=nn.init.calculate_gain('relu'))
#            nn.init.xavier_uniform_(m.weight,  gain=nn.init.calculate_gain('relu'))
#            #nn.init.xavier_uniform_(m.weight)
#
#
#        if isinstance(m, nn.Conv1d):
#            #print(m)
#            nn.init.xavier_normal_(m.weight,  gain=nn.init.calculate_gain('relu'))
#            #nn.init.xavier_uniform_(m.weight)
#            #nn.init.xavier_uniform_(m.weight,  gain=nn.init.calculate_gain('relu'))
#            if m.bias is not None:
#                m.bias.data.fill_(0.)
#            
#        if isinstance(m, nn.LSTM):
#            #print('x', m)
#            #for name, param in m.named_parameters():
#            #    if 'weight_ih' in name:
#            #        #nn.init.xavier_normal_(param.data, gain=nn.init.calculate_gain('relu'))
#            #        nn.init.xavier_normal_(param.data)
#            #        #for idx in range(4):
#            #        #    mul = param.shape[0]//4
#            #        #    nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
#            #    elif 'weight_hh' in name:
#            #        nn.init.orthogonal_(param.data)
#            #        #for idx in range(4):
#            #        #    mul = param.shape[0]//4
#            #        #    nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
#            #    elif 'bias' in name:
#            #        nn.init.zeros_(param.data)
#            #nn.init.xavier_uniform_(m.weight.data,  gain=nn.init.calculate_gain('relu'))
#            #nn.init.xavier_normal_(m.weight.data,  gain=nn.init.calculate_gain('relu'))
#            # high forget gate
#            for name, param in m.named_parameters():
#                if 'bias' in name:
##                     print(name, param.shape)
#                    for idx in range(4):
#                        mul = param.shape[0]//4
#                        param.data[mul:mul*2].fill_(2.)
##                         print(mul)              
#        if isinstance(m, nn.Linear):
#            nn.init.xavier_normal_(m.weight)
#            nn.init.zeros_(m.bias)
#            #y = m.in_features
#            #m.weight.data.normal_(0.0,1/np.sqrt(y))
#            #m.bias.data.fill_(0)
#            
#
#    for name, m in model.named_children():
##         print(name, m)
##         print(name, m)
#        if 'embeds_f' not in name:
##             weight_init(m)
#            m.apply(weight_init)
## model_init_w(model)
## print("init weight")

def model_init_w_ori(model):
    print("init model")
    def weight_init(m):
        if isinstance(m, nn.Embedding):
            print(m)
            nn.init.uniform_(m.weight, -0.05, 0.05)
            #nn.init.xavier_normal_(m.weight,  gain=nn.init.calculate_gain('relu'))

        #if isinstance(m, nn.Conv1d):
        #    print(m)
        #    nn.init.xavier_uniform_(m.weight)
        #    m.bias.data.fill_(0.)
        
        if isinstance(m, nn.Linear):
            print(m)
            y = m.in_features
            #m.weight.data.normal_(0.0,.1/np.sqrt(y))
            m.weight.data.normal_(0.0,.1/np.sqrt(y))
            m.bias.data.fill_(0.1)
            

    for name, m in model.named_children():
        if 'embeds_f' not in name:
            m.apply(weight_init)

def model_init_w(model):
    print("init model")
    def weight_init(m):
        if isinstance(m, nn.Embedding):
            #print(m)
            nn.init.xavier_normal_(m.weight,  gain=nn.init.calculate_gain('relu'))

        if isinstance(m, nn.Conv1d):
            #print(m)
            nn.init.xavier_normal_(m.weight,  gain=nn.init.calculate_gain('relu'))
            if m.bias is not None:
                m.bias.data.fill_(0.)
            
        if isinstance(m, nn.LSTM):
            #print(m)
            for name, param in m.named_parameters():
                if 'bias' in name:
#                     print(name, param.shape)
                    for idx in range(4):
                        mul = param.shape[0]//4
                        param.data[mul:mul*2].fill_(2.)
#                         print(mul)              
        
        if isinstance(m, nn.Linear):
            #print(m)
            y = m.in_features
            #m.weight.data.normal_(0.0,.1/np.sqrt(y))
            m.weight.data.normal_(0.0,.1/np.sqrt(y))
            m.bias.data.fill_(0)
            
#             nn.init.xavier_normal_(m.weight,  gain=nn.init.calculate_gain('relu'))
#             nn.init.xavier_normal_(m.weight)
#             nn.init.zeros_(m.bias)

    for name, m in model.named_children():
        if 'embeds_f' not in name:
            m.apply(weight_init)


def init_model_optimizer(model, config, patience_epochs = 1):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=config.adam_epsilon)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.lr_reduce_factor, \
                                                     min_lr = 1e-5, cooldown = config.lr_cooldown, \
                                                     patience = patience_epochs, verbose=True)
#     scheduler = None
    
    return optimizer, scheduler


