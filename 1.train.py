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
import logging

from utils import *
from dataset import *
from torch.utils.data import DataLoader

from encoders import *
from generators import *
from classifiers import *
from model import *

from config import *

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=10, type=int, help="epochs")
parser.add_argument("--eval_step", default=100, type=int, help="eval_step")
parser.add_argument("--per_gpu_batch_size", default=128, type=int, help="batch_size")
parser.add_argument("--lr", default=5e-5, type=float, help="lr")
parser.add_argument("--ergo_weight", default=1.0, type=float, help="ergo_weight")
parser.add_argument("--i_cls_weight", default=0.5, type=float, help="i_cls_weight")
parser.add_argument("--recon_weight", default=0.25, type=float, help="recon_weight")
parser.add_argument("--s_recon_weight", default=0.25, type=float, help="s_recon_weight")
parser.add_argument("--wass_weight", default=1.0, type=float, help="wass_weight")
parser.add_argument("--epsilon", default=0.0, type=float, help="epsilon")
parser.add_argument("--sigma_rbf", default=1.0, type=float, help="sigma_rbf")
parser.add_argument("--data_path", default=None, type=str, help="data_path")
parser.add_argument("--data_prefix", default=None, type=str, help="data_prefix")
parser.add_argument("--dataset", default=None, type=str, help="dataset")
parser.add_argument("--pretrained_path", default=None, type=str, help="pretrained_path")
parser.add_argument("--save_model_path", default=None, type=str, help="save_model_path")
parser.add_argument("--seed", default=None, type=int, help="seed")
parser.add_argument("--peptide", default=None, type=str, help="peptide")

args = parser.parse_args()
args.n_gpu = torch.cuda.device_count()

if args.seed is not None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)


epochs = args.epochs
eval_step = args.eval_step
batch_size = args.per_gpu_batch_size * max(1, args.n_gpu)
lr = args.lr
epsilon = args.epsilon

data_path = args.data_path
dataset = args.dataset
data_prefix = args.data_prefix
pretrained_path = args.pretrained_path
save_model_path = args.save_model_path

if args.peptide == 'all':
    peptide = ''
else:
    peptide = '_' + args.peptide

device='cuda'

opt['s_decoder_config']['epsilon'] = epsilon
opt['decoder_config']['epsilon'] = epsilon

    
opt['sigma_rbf'] = args.sigma_rbf

opt['batch_size'] = args.per_gpu_batch_size * max(1, args.n_gpu)

# Train model
### Load data
print("Loading data")
with open(data_path+data_prefix+'_train.pkl', 'rb') as f:
    dat_train = pickle.load(f)
    
dset_train = PairedTCRPeptideDataset(dat_train)
opt['dataset_size'] = dset_train.__len__()

with open(data_path+data_prefix+'_val.pkl', 'rb') as f:
    dat_val = pickle.load(f)
    
dset_val = PairedTCRPeptideDataset(dat_val)

### Train model
#### PHASE 1: standard autoencoder training
print("*******Training *******")

print("Initializing model")
model = DisentangledBERTForBindingPrediction(opt)
model = model.cuda()

print("CUDA on: ", model.is_cuda)
print("Number of gpus: ", args.n_gpu)

if args.n_gpu > 1:
    model = torch.nn.DataParallel(model)
    
model_params = []
for i in model.named_parameters():
    if 'final_classifier' in i[0]: 
        continue
    model_params.append(i)

        
opt_model = torch.optim.Adam([p for n, p in model_params], lr=lr)


start_epoch = 0
step = 1
for epoch in range(start_epoch, epochs):
    train_dataloader = DataLoader(dset_train, batch_size=batch_size, shuffle=True, num_workers=0)

    print('### Epoch {} begin:'.format(epoch))
    loss_records = []
    s_acc_records = []
    #s2i_records = []
    
    for ii, batch in enumerate(train_dataloader):
        model.train()
        batch = [i.cuda() for i in batch]
        seq, seq_attn_mask, pep, pep_attn_mask, i_label = batch

        full_recon_seq, full_recon_loss, s_recon_seq, s_recon_loss, i_cls_loss, i_cls_logits, i_wass_loss = model(batch)
        loss = full_recon_loss * args.recon_weight + s_recon_loss * args.s_recon_weight + i_cls_loss * args.i_cls_weight + i_wass_loss * args.wass_weight

        opt_model.zero_grad()
        if args.n_gpu > 1:
            loss = loss.mean()
        loss.backward(retain_graph=True)
        print('Step ', ii, ' loss = ', float(loss))
        
        opt_model.step()
        
        step += 1
        

        ### Evaluate
        if step % eval_step==0:
            val_dataloader = DataLoader(dset_val, batch_size=batch_size, shuffle=True, num_workers=0)
            
            model.eval()
            full_recon_loss_list = []
            s_recon_loss_list = []
            i_cls_loss_list = []
            full_cls_loss_list = []
            s_cls_loss_list = []
            i_wass_loss_list = []

            i_cls_preds = []
            i_cls_probs = []
            i_cls_labels = []

            full_cls_preds = []
            full_cls_probs = []
            full_cls_labels = []

            for jj, batch in enumerate(val_dataloader):
                batch = [i.cuda() for i in batch]
                seq, seq_attn_mask, pep, pep_attn_mask, i_label = batch

                full_recon_seq, full_recon_loss, s_recon_seq, s_recon_loss, i_cls_loss, i_cls_logits, i_wass_loss = model(batch)
                loss = full_recon_loss * args.recon_weight + s_recon_loss * args.s_recon_weight + i_cls_loss * args.i_cls_weight + i_wass_loss * args.wass_weight

                full_recon_loss_list.append(full_recon_loss.detach().cpu().numpy())
                s_recon_loss_list.append(s_recon_loss.detach().cpu().numpy())
                i_cls_loss_list.append(i_cls_loss.detach().cpu().numpy())
                i_wass_loss_list.append(i_wass_loss.detach().cpu().numpy())

                i_cls_prob = torch.sigmoid(i_cls_logits)
                i_cls_preds.append(((i_cls_prob>0.5)*1).cpu().numpy())
                i_cls_probs.append(i_cls_prob.cpu().detach().numpy())
                i_cls_labels.append(i_label.cpu().numpy())
                

            i_cls_preds = np.concatenate(i_cls_preds).reshape(-1)
            i_cls_probs = np.concatenate(i_cls_probs).reshape(-1)
            i_cls_labels = np.concatenate(i_cls_labels)
            i_cls_metrics = acc_f1_mcc_auc_aupr_pre_rec(i_cls_preds, i_cls_labels, i_cls_probs)

            full_cls_metrics = {}
            full_cls_metrics['full_recon_loss'] = np.mean(full_recon_loss_list)
            full_cls_metrics['s_recon_loss'] = np.mean(s_recon_loss_list)
            full_cls_metrics['i_cls_loss'] = np.mean(i_cls_loss_list)
            full_cls_metrics['i_wass_loss'] = np.mean(i_wass_loss_list)

            print('****** Eval ******')
            for metric in i_cls_metrics:
                print('*** Interation predictor Eval ' + metric + ' : ' + str(i_cls_metrics[metric]))
            for metric in full_cls_metrics:
                print('*** Final predictor Eval ' + metric + ' : ' + str(full_cls_metrics[metric]))
                
    if epoch % 50 == 49 or epoch in [9,19,29]:
        ckpt_path = save_model_path + '-ckpt' + str(epoch+1)
        if not os.path.isdir(ckpt_path):
            os.mkdir(ckpt_path)
        model_to_save = (model.module if hasattr(model, "module") else model)
        model_to_save.save_pretrained(ckpt_path)
        
model_to_save = (model.module if hasattr(model, "module") else model)
model_to_save.save_pretrained(save_model_path)

### Test performance
print('****** Testing ******')
with open(data_path+data_prefix+'_test.pkl', 'rb') as f:
    dat_test = pickle.load(f)
    
dset_test = PairedTCRPeptideDataset(dat_test)

test_dataloader = DataLoader(dset_test, batch_size=batch_size, shuffle=True, num_workers=0)

model.eval()
full_recon_loss_list = []
s_recon_loss_list = []
i_cls_loss_list = []
full_cls_loss_list = []
s_cls_loss_list = []
i_wass_loss_list = []

i_cls_preds = []
i_cls_probs = []
i_cls_labels = []

full_cls_preds = []
full_cls_probs = []
full_cls_labels = []

for jj, batch in enumerate(test_dataloader):
    batch = [i.cuda() for i in batch]
    seq, seq_attn_mask, pep, pep_attn_mask, i_label = batch

    full_recon_seq, full_recon_loss, s_recon_seq, s_recon_loss, i_cls_loss, i_cls_logits, i_wass_loss = model(batch)
    loss = full_recon_loss * args.recon_weight + s_recon_loss * args.s_recon_weight + i_cls_loss * args.i_cls_weight + i_wass_loss * args.wass_weight

    full_recon_loss_list.append(full_recon_loss.detach().cpu().numpy())
    s_recon_loss_list.append(s_recon_loss.detach().cpu().numpy())
    i_cls_loss_list.append(i_cls_loss.detach().cpu().numpy())
    i_wass_loss_list.append(i_wass_loss.detach().cpu().numpy())

    i_cls_prob = torch.sigmoid(i_cls_logits)
    i_cls_preds.append(((i_cls_prob>0.5)*1).cpu().numpy())
    i_cls_probs.append(i_cls_prob.cpu().detach().numpy())
    i_cls_labels.append(i_label.cpu().numpy())


i_cls_preds = np.concatenate(i_cls_preds).reshape(-1)
i_cls_probs = np.concatenate(i_cls_probs).reshape(-1)
i_cls_labels = np.concatenate(i_cls_labels)
i_cls_metrics = acc_f1_mcc_auc_aupr_pre_rec(i_cls_preds, i_cls_labels, i_cls_probs)

full_cls_metrics = {}
full_cls_metrics['full_recon_loss'] = np.mean(full_recon_loss_list)
full_cls_metrics['s_recon_loss'] = np.mean(s_recon_loss_list)
full_cls_metrics['i_cls_loss'] = np.mean(i_cls_loss_list)
full_cls_metrics['i_wass_loss'] = np.mean(i_wass_loss_list)

print('****** Test ******')
for metric in i_cls_metrics:
    print('*** Interation predictor Test ' + metric + ' : ' + str(i_cls_metrics[metric]))
for metric in full_cls_metrics:
    print('*** Final predictor Test ' + metric + ' : ' + str(full_cls_metrics[metric]))
    
ckpt_path = save_model_path
if not os.path.isdir(ckpt_path):
    os.mkdir(ckpt_path)
model_to_save = (model.module if hasattr(model, "module") else model)
model_to_save.save_pretrained(ckpt_path)    
