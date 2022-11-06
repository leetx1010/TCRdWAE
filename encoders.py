import numpy as np

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertSelfAttention, BertLayer, BertEmbeddings, BertEncoder, BertForSequenceClassification
from transformers.models.bert_generation.modeling_bert_generation import BertGenerationPreTrainedModel, BertGenerationDecoder
from transformers.models.bert_generation.configuration_bert_generation import BertGenerationConfig
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions

from transformers.file_utils import ModelOutput

from typing import Optional, Tuple

from utils import *

class PretrainedInputEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    """Remove token type embeddings"""

    def __init__(self, config):
        super().__init__()
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        embeddings = inputs_embeds.clone()
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids) 
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertLowerDimEncoderConfig(BertConfig):
    def __init__(
        self,
        vocab_size=50358,
        hidden_size=1024,
        mlp_hidden_size=None,
        mlp_output_size=1024, # Added by TXL
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=1,
        position_embedding_type="absolute",
        use_cache=True,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.mlp_hidden_size = mlp_hidden_size
        self.mlp_output_size = mlp_output_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache

    
class BertLowerDimEncoder(nn.Module):
    '''
    Bert encoder, but the final embedding has lower dimension (specified in config)
    '''
    def __init__(self,config):
        super(BertLowerDimEncoder, self).__init__()
        classifier_dropout = 0.1
        self.encoder = BertEncoder(config=config)
        self.dropout = nn.Dropout(classifier_dropout)
        self.relu = nn.LeakyReLU()
        self.config = config
        
        if config.mlp_hidden_size is not None:
            d_hidden = config.mlp_hidden_size
            hidden_list = []
            for i in range(len(d_hidden)):
                if i == 0:
                    hidden_list.append(nn.Linear(config.hidden_size, d_hidden[i]))
                else:
                    hidden_list.append(nn.Linear(d_hidden[i-1], d_hidden[i]))     
            self.mlp_hidden = nn.ModuleList(hidden_list)
            self.linear = nn.Linear(d_hidden[-1], config.mlp_output_size)
                
        else:
            self.mlp_hidden = None
            self.linear = nn.Linear(config.hidden_size, config.mlp_output_size)
        
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
               ):
        
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        x = encoder_outputs.last_hidden_state
        if self.mlp_hidden is not None:
            for hidden in self.mlp_hidden:
                x = hidden(x)
                x = self.relu(x)
                x = self.dropout(x)
            
        outputs = self.linear(x)
        sequence_output = outputs[0]

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=outputs,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
