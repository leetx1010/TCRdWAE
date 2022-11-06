import numpy as np
import pickle 
import sys
import argparse
import os

import torch
import math
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertSelfAttention, BertLayer, BertEmbeddings, BertEncoder, BertForSequenceClassification
from transformers.models.bert_generation.modeling_bert_generation import BertGenerationPreTrainedModel, BertGenerationDecoder
from transformers.models.bert_generation.configuration_bert_generation import BertGenerationConfig
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions

from transformers.file_utils import ModelOutput

from typing import Optional, Tuple

from utils import *
from dataset import *
from torch.utils.data import DataLoader

from encoders import *
from generators import *
from classifiers import *
from model import *

from config import *

import time

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=10, type=int, help="epochs")
parser.add_argument("--shuffle_epochs", default=10, type=int, help="epochs")
parser.add_argument("--shuffle_prob", default=0.0, type=float, help="shuffle_prob")
parser.add_argument("--per_gpu_batch_size", default=128, type=int, help="batch_size")
parser.add_argument("--lr", default=5e-5, type=float, help="lr")
parser.add_argument("--i_cls_weight", default=0.5, type=float, help="i_cls_weight")
parser.add_argument("--recon_weight", default=0.25, type=float, help="recon_weight")
parser.add_argument("--s_recon_weight", default=0.25, type=float, help="s_recon_weight")
parser.add_argument("--wass_weight", default=1.0, type=float, help="wass_weight")
parser.add_argument("--epsilon", default=0.0, type=float, help="epsilon")
parser.add_argument("--sigma_rbf", default=1.0, type=float, help="sigma_rbf")
parser.add_argument("--dataset", default=None, type=str, help="dataset")
parser.add_argument("--data_file", default=None, type=str, help="data_file")
parser.add_argument("--load_model_prefix", default=None, type=str, help="load_model_prefix")
parser.add_argument("--load_model_path", default=None, type=str, help="load_model_path")
parser.add_argument("--output_result_prefix", default=None, type=str, help="output_result_prefix")
parser.add_argument("--output_path", default=None, type=str, help="output_path")
parser.add_argument("--peptide", default=None, type=str, help="peptide")

args = parser.parse_args()
args.n_gpu = torch.cuda.device_count()

torch.manual_seed(42)
random.seed(42)

epochs = args.epochs
shuffle_epochs = args.shuffle_epochs
shuffle_prob = args.shuffle_prob
batch_size = args.per_gpu_batch_size * max(1, args.n_gpu)
lr = args.lr
epsilon = args.epsilon

dataset = args.dataset
data_file = args.data_file
load_model_path = args.load_model_path

if args.peptide == 'all':
    peptide = ''
else:
    peptide = '_' + args.peptide

opt['s_decoder_config']['epsilon'] = epsilon
opt['decoder_config']['epsilon'] = epsilon

   
opt['sigma_rbf'] = args.sigma_rbf

opt['batch_size'] = args.per_gpu_batch_size * max(1, args.n_gpu)

output_path=args.output_path
model_prefix=args.output_result_prefix

with open(data_file, 'rb') as f:
    dat_test = pickle.load(f)

pep_list = list(set(i.split('-')[-1] for i in dat_test))

batch_size = 64
peptide = args.peptide

dat_pos = {}
dat_neg = {}

for pair in dat_test:
    if peptide in pair:
        pair_new = pair.split('_')[0]
        tcr_seq, pep_seq, label = dat_test[pair]
        if label == 1:
            dat_pos[pair_new] = dat_test[pair]
        else:
            dat_neg[pair_new] = dat_test[pair]
        
dset_pos = PairedTCRPeptideDataset(dat_pos)
dset_neg = PairedTCRPeptideDataset(dat_neg)

dset_pos_loader = DataLoader(dset_pos, batch_size=batch_size, shuffle=False, drop_last=False)
dset_neg_loader = DataLoader(dset_neg, batch_size=batch_size, shuffle=False, drop_last=False)

model = DisentangledBERTForBindingPrediction(opt)
model.load_state_dict(torch.load(load_model_path+'/pytorch_model.bin'))
model = model.cuda()
model = (model.module if hasattr(model, "module") else model)
model.eval()

### Random
"""n = 10000
model = (model.module if hasattr(model, "module") else model)

d_i_emb = model.opt['i_encoder_config'].mlp_output_size
i_emb_random = torch.tensor(np.random.normal(0, 1, size=(n, 1, d_i_emb))).float()

seq, seq_attn_mask, pep, pep_attn_mask, i_label, label = [i.cuda() for i in dset_neg[0:1]]

p_emb, s_emb, i_emb = model.encode(seq, seq_attn_mask, pep, pep_attn_mask)

i_cls_logits_list = []
for ii in range(n//batch_size+1):
    i_emb_sub = i_emb_random[(ii*batch_size):min((ii+1)*batch_size, n)].cuda()
    p_emb_sub = torch.cat([p_emb]*i_emb_sub.shape[0],0)
    i_label_sub = torch.cat([i_label]*i_emb_sub.shape[0],0)
    i_cls_loss, i_cls_logits, _ = model.i_classifier(torch.cat([p_emb_sub, i_emb_sub], dim=-1), labels=i_label_sub)
    i_cls_logits_list.append(i_cls_logits)
i_cls_logits_list = torch.cat(i_cls_logits_list)

i_emb_new_random = i_emb_random[i_cls_logits_list.argmax()].reshape(1,1,i_emb_random.shape[-1]).cuda()"""

### Best and avg
i_emb_list = []
cls_logits_list = []

for i, batch in enumerate(dset_pos_loader):
    seq, seq_attn_mask, pep, pep_attn_mask, i_label = [i.cuda() for i in batch]

    p_emb, s_emb, i_emb = model.encode(seq, seq_attn_mask, pep, pep_attn_mask)
    i_cls_loss, i_cls_logits, _ = model.i_classifier(torch.cat([p_emb, i_emb], dim=-1), labels=i_label)
    
    i_emb_list.append(i_emb)
    cls_logits_list.append(i_cls_logits)

i_emb_list = torch.cat(i_emb_list)
cls_logits_list = torch.cat(cls_logits_list)

i_emb_new_best = i_emb_list[cls_logits_list.argmax()].reshape(1,1,i_emb_list.shape[-1])
i_emb_new_avg = i_emb_list.mean(dim=0).reshape(1,1,i_emb_list.shape[-1])

    
### Generate
    
for opt_type in ['random', 'best', 'avg', 'null']:
    start_time = time.time()
    if not os.path.isdir(output_path + '/'+ model_prefix + '-' + opt_type + '/'):
        os.mkdir(output_path + '/'+ model_prefix + '-' + opt_type + '/')
    generate_seq_list = []
    orig_seq_list = []

    for ii in range(len(dset_neg)):
        seq, seq_attn_mask, pep, pep_attn_mask, i_label = [i.cuda() for i in dset_neg[ii:(ii+1)]]
        orig_seq_list.append(seq)
        
        p_emb, s_emb, i_emb = model.encode(seq, seq_attn_mask, pep, pep_attn_mask)
        seq_emb_in = model.tcr_embedding(seq)

        # Random
        if opt_type == 'random':
            i_emb_new_random = i_emb_list[torch.randperm(i_emb_list.shape[0])[0],:,:].reshape(1,1,i_emb_list.shape[-1])
            full_emb = torch.cat((torch.cat([s_emb]*model.max_seqlen,1), torch.cat([p_emb]*model.max_seqlen,1), torch.cat([i_emb_new_random]*model.max_seqlen,1)), dim=-1)
            generate_seq = model.generate(full_emb, seq_emb_in)[0]
            generate_seq_list.append(generate_seq)

        # Best
        elif opt_type == 'best':
            full_emb = torch.cat((torch.cat([s_emb]*model.max_seqlen,1), torch.cat([p_emb]*model.max_seqlen,1), torch.cat([i_emb_new_best]*model.max_seqlen,1)), dim=-1)
            generate_seq = model.generate(full_emb, seq_emb_in)[0]
            generate_seq_list.append(generate_seq)

        # Avg
        elif opt_type == 'avg':
            full_emb = torch.cat((torch.cat([s_emb]*model.max_seqlen,1), torch.cat([p_emb]*model.max_seqlen,1), torch.cat([i_emb_new_avg]*model.max_seqlen,1)), dim=-1)
            generate_seq = model.generate(full_emb, seq_emb_in)[0]
            generate_seq_list.append(generate_seq)

        # Null
        elif opt_type == 'null':
            d_i_emb = model.opt['i_encoder_config'].mlp_output_size
            i_emb_null = torch.tensor(np.random.normal(0, 1, size=(1, 1, d_i_emb))).float().cuda()
            full_emb = torch.cat((torch.cat([s_emb]*model.max_seqlen,1), torch.cat([p_emb]*model.max_seqlen,1), torch.cat([i_emb_null]*model.max_seqlen,1)), dim=-1)
            generate_seq = model.generate(full_emb, seq_emb_in)[0]
            generate_seq_list.append(generate_seq)
    end_time = time.time()
    print('dAE running time' + opt_type + ': ', end_time-start_time)
    generate_seq_list = torch.cat(generate_seq_list, dim=0).detach().cpu()
    orig_seq_list = torch.cat(orig_seq_list, dim=0).detach().cpu()

    with open(output_path + '/'+ model_prefix + '-' + opt_type + '/' + dataset + '-' + peptide + '.pkl', 'wb') as f:
        pickle.dump((orig_seq_list, generate_seq_list), f)


