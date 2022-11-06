from transformers import BertConfig
import esm
import torch

from encoders import *
from generators import *
from classifiers import *
from model import *

amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
amino_to_ix = {amino: index for index, amino in enumerate(['PAD'] + amino_acids + ['<cls>','<pad>','<eos>','<unk>'])}


d_input_emb = 128
d_decoder_embed = d_input_emb

d_s_embed = 32
d_i_embed = 8
d_p_embed = 25

vocab_size = len(amino_to_ix)
ergo_vocab_size = 21
num_hidden_layers = 1
num_attention_heads = 8
intermediate_size = 128

max_seq_len = 25
max_pep_len = 25

s_encoder_config = BertLowerDimEncoderConfig(vocab_size = vocab_size,
                                             hidden_size=d_input_emb,
                                             mlp_hidden_size = [128],
                                             mlp_output_size = d_s_embed,
                                             intermediate_size=intermediate_size,
                                             num_hidden_layers=num_hidden_layers,
                                             num_attention_heads=num_attention_heads,)


i_encoder_config = BertLowerDimEncoderConfig(vocab_size = vocab_size,
                                             hidden_size = d_input_emb,
                                             mlp_hidden_size = [128],
                                             mlp_output_size = d_i_embed,
                                             intermediate_size=intermediate_size,
                                             num_hidden_layers=num_hidden_layers,
                                             num_attention_heads=num_attention_heads,)

p_encoder_config = BertConfig(vocab_size = vocab_size,
                                             hidden_size = d_p_embed,
                                             intermediate_size=32,
                                             num_hidden_layers=num_hidden_layers,
                                             num_attention_heads=1,)



s_decoder_config = {'d_hidden':256,
                    'n_layer':2,
                    'd_input':d_s_embed,
                    'd_wordemb':d_input_emb,
                    'd_wordvec':d_decoder_embed,
                    'n_vocab':vocab_size}

decoder_config = {'d_hidden':256,
                  'n_layer':2,
                  'd_input':d_s_embed+d_i_embed+d_p_embed,
                  'd_wordemb':d_input_emb,
                  'd_wordvec':d_decoder_embed,
                  'n_vocab':vocab_size}


i_classifier_config = {'dim_input':d_p_embed+d_i_embed, 
                       'd_hidden':[32],
                       'num_class':1}


classifier_config = {'ckpt_path':'/home/tl444/Projects/ProteinDisentangle/2.eval/tcr_optimization/ERGO/models/lstm_mcpas1.pt'}

                                           
opt = {}
opt['tcr_embedding_weight'] = None
opt['pep_embedding_weight'] = None


opt['s_encoder_config'] = s_encoder_config
opt['i_encoder_config'] = i_encoder_config
opt['s_decoder_config'] = s_decoder_config
opt['decoder_config'] = decoder_config
opt['p_encoder_config'] = p_encoder_config
opt['i_classifier_config'] = i_classifier_config
opt['classifier_config'] = classifier_config
opt['max_seq_len'] = max_seq_len
opt['max_pep_len'] = max_pep_len
opt['d_input_emb'] = d_input_emb
opt['d_s_embed'] = d_s_embed
opt['d_i_embed'] = d_i_embed
opt['d_p_embed'] = d_p_embed
opt['d_decoder_embed'] = d_decoder_embed

opt['vocab_size'] = vocab_size
opt['ergo_vocab_size'] = vocab_size
opt['sigma_rbf'] = 1
opt['dataset_size'] = 18906
opt['batch_size'] = 128
opt['stoi_epoch'] = 5

opt['start_token_id'] = amino_to_ix['<cls>']
opt['end_token_id'] = amino_to_ix['<eos>']

opt['cuda'] = torch.cuda.is_available()

epsilon = 0.1
opt['s_decoder_config']['epsilon'] = epsilon
opt['decoder_config']['epsilon'] = epsilon