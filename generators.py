import numpy as np

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import random

from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertSelfAttention, BertLayer, BertEmbeddings, BertEncoder, BertForSequenceClassification
from transformers.models.bert_generation.modeling_bert_generation import BertGenerationPreTrainedModel, BertGenerationDecoder
from transformers.models.bert_generation.configuration_bert_generation import BertGenerationConfig
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions

from transformers.file_utils import ModelOutput

from typing import Optional, Tuple

from utils import *

class LSTMDecoder(nn.Module):
    def __init__(self, config):
        super(LSTMDecoder, self).__init__()
        self.d_hidden = config['d_hidden']
        self.n_layer = config['n_layer']
        self.epsilon = config['epsilon']
        
        self.lstm = nn.LSTM(input_size=config['d_wordvec']+config['d_input'], hidden_size=config['d_hidden'], num_layers=config['n_layer'], batch_first=True)
        self.fc = nn.Linear(config['d_hidden'], config['n_vocab'])
        self.config = config   
        
    def forward(self, emb, input_seq_emb, scheduled_sampling=False, embedding_weight = None, g_hidden = None):
        batch_size, n_seq, n_embed = input_seq_emb.size()
        if not scheduled_sampling:
            #emb = torch.cat([emb]*n_seq, 1).view(batch_size, n_seq, -1) #Replicate z inorder to append same z at each time step

            x= torch.cat([input_seq_emb, emb], dim=-1) #append z to generator word input at each time step
           # print(x.shape)
            if g_hidden is None: #if we are validating
                output, out_hidden = self.lstm(x)
            else: #if we are train
                output, out_hidden = self.lstm(x, g_hidden)

            #Get top layer of h_T at each time step and produce logit vector of vocabulary words
            output = self.fc(output)
            return output, out_hidden #Also return complete (h_T, c_T) incase if we are testing
    
    
        else:
            """Training with scheduled sampling"""
            batch_size, n_seq, n_embed = input_seq_emb.size()

            h_0 = torch.zeros(self.config['n_layer'],  batch_size, self.config['d_hidden']).detach()
            c_0 = torch.zeros(self.config['n_layer'],  batch_size, self.config['d_hidden']).detach() 
            g_hidden = (h_0.cuda(), c_0.cuda())

            #word_vec = torch.cat([embedding_weight[0,:].reshape(1, 1, embedding_weight.shape[1])]*batch_size,0)
            word_vec = input_seq_emb[:,0:1,:]
            sent_embed_list = [word_vec]
            output = []

            for j in range(n_seq):
                x = torch.cat([word_vec, emb[:,j:(j+1),:]], dim=-1)
                logits, g_hidden = self.lstm(x, g_hidden)  # logits [batch, 1, n_vocab]
                logits = self.fc(logits)
                word_idx = torch.argmax(logits, dim = -1)                   # [batch, 1]

                onehot_idx = torch.softmax(logits, dim = -1) #STE(logits)
                output.append(logits)
                
                if random.random() < self.epsilon:
                    word_vec = torch.einsum('ijk,kl->ijl', onehot_idx ,  embedding_weight)    # [batch, 1, d_wordvec]
                else:
                    word_vec = input_seq_emb[:,(j+1):(j+2),:]    # [batch, 1, d_wordvec]
                

            output = torch.cat(output, dim=1)
            return output, g_hidden #Also return complete (h_T, c_T) incase if we are testing
        
    
  