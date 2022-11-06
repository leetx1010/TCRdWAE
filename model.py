import numpy as np

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, BCELoss
import torch.autograd as autograd
import random

from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertSelfAttention, BertLayer, BertEmbeddings, BertEncoder, BertForSequenceClassification
from transformers.models.bert_generation.modeling_bert_generation import BertGenerationPreTrainedModel, BertGenerationDecoder
from transformers.models.bert_generation.configuration_bert_generation import BertGenerationConfig
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions

from transformers.file_utils import ModelOutput

from typing import Optional, Tuple

from utils import *

from encoders import * 
from generators import * 
from classifiers import * 

import sys

class DisentangledBERTForBindingPrediction(BertPreTrainedModel):
    def __init__(self, opt):
        super(DisentangledBERTForBindingPrediction, self).__init__(opt['s_encoder_config'])
        self.tcr_embedding = nn.Embedding(num_embeddings=opt['vocab_size'], embedding_dim=opt['d_input_emb'])
        
        #if opt['tcr_embedding_weight'] is not None:
        #    self.tcr_embedding.weight = opt['tcr_embedding_weight']
            
        #if opt['tcr_embedding_weight'] is not None:
        #    self.pep_embedding.weight = opt['pep_embedding_weight']
        
        self.tcr_bert_embedding = PretrainedInputEmbeddings(config=opt['s_encoder_config'])
        
        self.is_cuda = opt['cuda']
        
        ### Encoders
        self.s_encoder = BertLowerDimEncoder(config=opt['s_encoder_config'])
        self.i_encoder = BertLowerDimEncoder(config=opt['i_encoder_config'])
        
        ### Decoders
        self.s_decoder = LSTMDecoder(config=opt['s_decoder_config'])
        self.decoder = LSTMDecoder(config=opt['decoder_config'])
        
        ### Classifiers
        opt_i_cls = opt['i_classifier_config']
        opt_i_cls['cuda'] = self.is_cuda
        self.i_classifier = LinearClassifier(opt_i_cls)
        
        ### Classifiers
        opt_i_cls = opt['i_classifier_config']
        opt_i_cls['cuda'] = self.is_cuda
        self.i_classifier = LinearClassifier(opt_i_cls)
        
        self.softmax = nn.Softmax(dim=-1)
        
        self.max_seqlen = opt['max_seq_len']
        self.max_peplen = opt['max_pep_len']
        self.vocab_size = opt['vocab_size']
        self.ergo_vocab_size = opt['ergo_vocab_size']
        self.sigma_rbf = opt['sigma_rbf']

        self.start_token_id = opt['start_token_id']
        self.end_token_id = opt['end_token_id'] 
                               
        self.opt = opt
        
    def encode(self, seq, seq_attn_mask, pep, pep_attn_mask):
        seq_emb_in = self.tcr_embedding(seq)
        
        seq_emb_in_bert = self.tcr_bert_embedding(
            input_ids=None,
            position_ids=None,
            token_type_ids=None,
            inputs_embeds=seq_emb_in,
        )
        
        
        seq_attn = self.extend_attention(seq_emb_in_bert, seq_attn_mask)
        
        #p_emb = self.p_encoder(pep)
        s_emb_out = self.s_encoder(hidden_states=seq_emb_in_bert, encoder_attention_mask=seq_attn)
        i_emb_out = self.i_encoder(hidden_states=seq_emb_in_bert, encoder_attention_mask=seq_attn)
        p_emb_out = pep.reshape(pep.shape[0],1,pep.shape[1])
        return p_emb_out, s_emb_out[0][:,0:1,:], i_emb_out[0][:,0:1,:]
    
    def decode(self, seq, pep, label, p_emb, s_emb, i_emb):
        seq_emb_in = self.tcr_embedding(seq)
        pep_emb_in = pep
        
        i_cls_loss, i_cls_logits, _ = self.i_classifier(torch.cat([p_emb, i_emb], dim=-1), labels=label)
            
        ### Context-based decoder output
        s_recon = self.s_decoder(torch.cat([s_emb]*self.max_seqlen,1), seq_emb_in, scheduled_sampling=True, embedding_weight = self.tcr_embedding.weight)
        s_recon_loss, s_recon_acc = self.recon_loss_and_acc(seq, s_recon[0])
        #s_recon_seq = self.softmax(s_recon[0][:,:,:self.ergo_vocab_size])
        s_recon_seq = self.softmax(s_recon[0])

        ### Full decoder output
        full_emb = torch.cat((torch.cat([s_emb]*self.max_seqlen,1), torch.cat([p_emb]*self.max_seqlen,1), torch.cat([i_emb]*self.max_seqlen,1)), dim=-1)
        full_recon = self.decoder(full_emb, seq_emb_in, scheduled_sampling=True, embedding_weight = self.tcr_embedding.weight)
        full_recon_loss, full_recon_acc = self.recon_loss_and_acc(seq, full_recon[0])
        #full_recon_seq = self.softmax(full_recon[0][:,:,:self.ergo_vocab_size])
        full_recon_seq = self.softmax(full_recon[0])

        ### Wasserstein & MI constraints
        i_wass_loss = self.kernel_mmd_loss(torch.cat([i_emb,s_emb],2), sigma_rbf=self.sigma_rbf)
                              
        return full_recon_seq, full_recon_loss, s_recon_seq, s_recon_loss, i_cls_loss, i_cls_logits, i_wass_loss
    
    
    def forward(self, batch, shuffle_prob = 0, opt_type = None):
        seq, seq_attn_mask, pep, pep_attn_mask, i_label = batch
        batch_size = len(seq)

        ### Encode
        p_emb, s_emb, i_emb = self.encode(seq, seq_attn_mask, pep, pep_attn_mask)
        
        output_list = self.decode(seq, pep, i_label, p_emb, s_emb, i_emb)
                       
        return output_list
    
    def extend_attention(self, emb, amask):
        input_shape = emb.size()[:-1]
        device = emb.device       
        
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = None
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
                amask, input_shape, device
        )
            
    def recon_loss_and_acc(self, target, logit):
        logit = logit
        logit = logit[:,:logit.shape[1]-1].contiguous().view(-1, self.vocab_size)
        
        recon_sent = logit.argmax(dim=-1)
        sent=target[:, 1:target.size(1)].contiguous().view(-1)
        sent = sent
        
        loss = F.cross_entropy(logit, sent).mean()
        acc = float(sum(recon_sent==sent)/len(recon_sent))
        
        return loss, acc
    
    def kernel_mmd_loss(self, emb, sigma_rbf, mu=0, sigma=1):
        X = emb
        Y = self.sample_gaussian(mu, sigma, emb.shape)
        if self.is_cuda:
            Y = Y.cuda()
        return self.rbf_mmd2(X, Y, sigma=sigma_rbf)
    
    def rbf_mmd2(self, X, Y, sigma=0):
        ### From https://github.com/djsutherland/opt-mmd/blob/master/two_sample/mmd.py
        # n = (T.smallest(X.shape[0], Y.shape[0]) // 2) * 2
        n = (X.shape[0] // 2) * 2
        gamma = 1 / (2 * sigma**2)
        rbf = lambda A, B: torch.exp(-gamma * ((A - B) ** 2).sum(axis=1))
        mmd2 = (rbf(X[:n:2], X[1:n:2]) + rbf(Y[:n:2], Y[1:n:2])
          - rbf(X[:n:2], Y[1:n:2]) - rbf(X[1:n:2], Y[:n:2])).mean()
        return mmd2 
    
    def sample_gaussian(self, mu, sigma, size):
        return torch.tensor(np.random.normal(loc=mu, scale=sigma, size=size))
    
    def generate(self, emb, input_seq, first_tok=5, max_seqlen=None):
        batch_size = emb.shape[0]

        max_seqlen = self.max_seqlen

        sent = torch.zeros([batch_size, max_seqlen], dtype = torch.int64).cuda()
        sentlen = torch.ones(batch_size, dtype = torch.int64).cuda() * max_seqlen
        sent_embed_seq = []

        h_0 = torch.zeros(self.opt['decoder_config']['n_layer'],  batch_size, self.opt['decoder_config']['d_hidden']).detach()
        c_0 = torch.zeros(self.opt['decoder_config']['n_layer'],  batch_size, self.opt['decoder_config']['d_hidden']).detach() 
        g_hidden = (h_0.cuda(), c_0.cuda())

        embedding_weight = self.tcr_embedding.weight
        input_embedding_weight = self.tcr_embedding.weight

        sent[:,0] = self.start_token_id
        word_vec = torch.cat([input_embedding_weight[self.start_token_id,:].reshape(1, 1, input_embedding_weight.shape[1])]*batch_size,0)
        sent_embed_list = [word_vec]

        for j in range(1, max_seqlen):
            logits, g_hidden = self.decoder(emb[:,(j-1):j,:], word_vec, scheduled_sampling=False, g_hidden=g_hidden)  # logits [batch, 1, n_vocab]
            word_idx = torch.argmax(logits, dim = -1)                   # [batch, 1]
            sent[:, j:j+1] = word_idx
            sentlen[word_idx[:,0] == self.end_token_id] = j+1

            onehot_idx = torch.softmax(logits, dim = -1) #STE(logits)
            word_vec = torch.einsum('ijk,kl->ijl', onehot_idx ,  embedding_weight)    # [batch, 1, d_wordvec]
            sent_embed_list.append(word_vec)

        sent_embed = torch.cat(sent_embed_list, 1)
        return sent, sent_embed, sentlen
        sent_embed = torch.cat(sent_embed_list, 1)
        return sent, sent_embed, sentlen
