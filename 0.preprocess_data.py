import pickle
import glob 
import sys
import os.path

import torch
#import esm
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

input_file = sys.argv[1]
output_file = sys.argv[2]


### Load blosum embeddings
blosum50 = {}

with open('blosum50.txt', 'r') as f:
    line = f.readline()
    line = f.readline()
    
    while line:
        tmp = line.split('\t')[0].split()
        blosum50[tmp[0]] = np.array([int(i) for i in tmp[1:]])
        line = f.readline()

### Token ids (same as ERGO)
amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
amino_to_ix = {amino: index for index, amino in enumerate(['PAD'] + amino_acids + ['<cls>','<pad>','<eos>','<unk>'])}


def seq2ergo(seq):
    seq_new = [amino_to_ix['<cls>']]
    for i in seq:
        seq_new.append(amino_to_ix[i])

    seq_new += [amino_to_ix['<eos>']]
    seq_new = torch.tensor(seq_new)
    return seq_new

def seq2blosum(seq):
    seq_new = []
    for i in seq:
        seq_new.append(blosum50[i])

    seq_new = torch.tensor(seq_new).float()
    return seq_new

### Load input csv
dat_raw = pd.read_csv(input_file, sep='\t')

### Process data
dat = {}

for idx,row in dat_raw.iterrows():
    idx = row[0]
    cdr = row[1]
    pep = row[2]
    label = row[3]
    dat[idx] = (seq2ergo(cdr), seq2blosum(pep).mean(0), label)
    

with open(output_file, 'wb') as f:
     pickle.dump(dat, f)