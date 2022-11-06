import esm

import numpy as np

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.autograd as autograd

from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertSelfAttention, BertLayer, BertEmbeddings, BertEncoder, BertModel, BertForSequenceClassification
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert_generation.modeling_bert_generation import BertGenerationPreTrainedModel, BertGenerationDecoder
from transformers.models.bert_generation.configuration_bert_generation import BertGenerationConfig
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions, SequenceClassifierOutput

from transformers.file_utils import ModelOutput

from typing import Optional, Tuple

from utils import *
import torch.nn.functional as F

class LinearClassifier(nn.Module):
    def __init__(self, opt):
        super(LinearClassifier, self).__init__()
        classifier_dropout = 0.1
        
        self.is_cuda = opt['cuda']
        self.dropout = nn.Dropout(classifier_dropout)
        self.relu = nn.LeakyReLU()
        
        if opt['d_hidden'] is not None:
            d_hidden = opt['d_hidden']
            hidden_list = []
            for i in range(len(d_hidden)):
                if i == 0:
                    hidden_list.append(nn.Linear(opt['dim_input'], d_hidden[i]))
                else:
                    hidden_list.append(nn.Linear(d_hidden[i-1], d_hidden[i]))     
            self.hidden = nn.ModuleList(hidden_list)
            self.classifier = nn.Linear(d_hidden[-1], opt['num_class'])
                
        else:
            self.hidden = None
            self.classifier = nn.Linear(opt['dim_input'], opt['num_class'])
        
    def forward(self, emb, labels = None):
        if self.hidden is not None:
            logits = self.dropout(emb)
            for hidden in self.hidden:
                logits = hidden(logits)
                logits = self.relu(logits)
                logits = self.dropout(logits)
            logits = self.classifier(logits)
            
        else:
            logits = self.classifier(self.dropout(emb))
            
        
        if labels is None:
            return logits, prediction
        else:
            prediction = (logits>0)*1
            loss_fct = BCEWithLogitsLoss()
            if self.is_cuda:
                loss_fct = BCEWithLogitsLoss().cuda()
        loss = loss_fct(logits.view(-1), labels.float())
        return loss, logits, prediction
    
    
class DoubleLSTMClassifier(nn.Module):
    """
    A rewrite of ERGO's DoubleLSTMClassifier, with matrix multiplication instead of embedding, so it can be backpropagated
    """
    def __init__(self, embedding_dim, lstm_dim, dropout, device):
        super(DoubleLSTMClassifier, self).__init__()
        # GPU
        self.device = device
        # Dimensions
        self.embedding_dim = embedding_dim
        self.lstm_dim = lstm_dim
        self.dropout = dropout
        # Embedding matrices - 20 amino acids + padding
        self.tcr_embedding = nn.Linear(20 + 1, embedding_dim, bias=False)
        self.pep_embedding = nn.Linear(20 + 1, embedding_dim, bias=False)
        # RNN - LSTM
        self.tcr_lstm = nn.LSTM(embedding_dim, lstm_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.pep_lstm = nn.LSTM(embedding_dim, lstm_dim, num_layers=2, batch_first=True, dropout=dropout)
        # MLP
        self.hidden_layer = nn.Linear(lstm_dim * 2, lstm_dim)
        self.relu = torch.nn.LeakyReLU()
        self.output_layer = nn.Linear(lstm_dim, 1)
        self.dropout = nn.Dropout(p=dropout)

    def init_hidden(self, batch_size):
        return (autograd.Variable(torch.zeros(2, batch_size, self.lstm_dim)).to(self.device),
                autograd.Variable(torch.zeros(2, batch_size, self.lstm_dim)).to(self.device))

    def lstm_pass(self, lstm, padded_embeds, lengths):
        # Before using PyTorch pack_padded_sequence we need to order the sequences batch by descending sequence length
        #lengths, perm_idx = lengths.sort(0, descending=True)
        #padded_embeds = padded_embeds[perm_idx]
        # Pack the batch and ignore the padding
        #padded_embeds = torch.nn.utils.rnn.pack_padded_sequence(padded_embeds, lengths, batch_first=True, enforce_sorted = False)
        # Initialize the hidden state
        batch_size = len(lengths)
        hidden = self.init_hidden(batch_size)
        # Feed into the RNN
        lstm_out, hidden = lstm(padded_embeds, hidden)
        # Unpack the batch after the RNN
        #lstm_out, lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        # Remember that our outputs are sorted. We want the original ordering
        #_, unperm_idx = perm_idx.sort(0)
        #lstm_out = lstm_out[unperm_idx]
        #lengths = lengths[unperm_idx]
        return lstm_out

    def forward(self, tcrs, tcr_lens, peps, pep_lens):
        # TCR Encoder:
        # Embedding
        tcr_embeds = self.tcr_embedding(tcrs)
        # LSTM Acceptor
        tcr_lstm_out = self.lstm_pass(self.tcr_lstm, tcr_embeds, tcr_lens)
        tcr_last_cell = torch.cat([tcr_lstm_out[i, j.data - 1] for i, j in enumerate(tcr_lens)]).view(len(tcr_lens), self.lstm_dim)

        # PEPTIDE Encoder:
        # Embedding
        pep_embeds = self.pep_embedding(peps)
        # LSTM Acceptor
        pep_lstm_out = self.lstm_pass(self.pep_lstm, pep_embeds, pep_lens)
        pep_last_cell = torch.cat([pep_lstm_out[i, j.data - 1] for i, j in enumerate(pep_lens)]).view(len(pep_lens), self.lstm_dim)

        # MLP Classifier
        tcr_pep_concat = torch.cat([tcr_last_cell, pep_last_cell], 1)
        hidden_output = self.dropout(self.relu(self.hidden_layer(tcr_pep_concat)))
        mlp_output = self.output_layer(hidden_output)
        output = F.sigmoid(mlp_output)
        return output
