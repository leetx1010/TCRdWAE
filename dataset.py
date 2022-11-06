import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
amino_to_ix = {amino: index for index, amino in enumerate(['PAD'] + amino_acids + ['<cls>','<eos>','<unk>'])}


class PairedTCRPeptideDataset(Dataset):
    def __init__(self, dset, max_seq_len=25, max_pep_len=25, vocab_size=33, blosum_dim=25):
        self.max_seq_len = max_seq_len
        self.max_pep_len = max_pep_len
        self.blosum_dim = blosum_dim
        self.vocab_size = vocab_size
        
        seq_emb_mat = []
        seq_attn_mask_mat = []
        pep_emb_mat = []
        pep_attn_mask_mat = []
        i_label_mat = []
        
        for pair in list(dset.keys()):
            seq_emb, pep_emb, label = dset[pair]
            n = len(seq_emb)
            seq_amask = np.ones(n)  

            if n < self.max_seq_len:
                seq_emb = torch.tensor(np.concatenate((np.array(seq_emb), np.zeros(self.max_seq_len - n))), dtype=torch.int64)
                seq_amask = np.concatenate((np.ones(n), 
                                        np.zeros(self.max_seq_len - n)))
                
            
            m = len(pep_emb)
            pep_amask = np.ones(m)
             
            seq_emb_mat.append(torch.tensor(seq_emb))
            pep_emb_mat.append(torch.tensor(pep_emb))
            
            seq_attn_mask_mat.append(torch.tensor(seq_amask))
            pep_attn_mask_mat.append(torch.tensor(pep_amask))

            i_label_mat.append(torch.tensor(label))

        self.seq = torch.stack(seq_emb_mat)
        self.pep = torch.stack(pep_emb_mat)
        
        self.seq_attn_mask = torch.stack(seq_attn_mask_mat).float()
        self.pep_attn_mask = torch.stack(pep_attn_mask_mat).float()
             
        self.i_label = torch.stack(i_label_mat)
        
    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        seq = self.seq[idx]
        seq_attn_mask = self.seq_attn_mask[idx]
        
        pep = self.pep[idx]
        pep_attn_mask = self.pep_attn_mask[idx]
        
        i_label = self.i_label[idx]

        return seq, seq_attn_mask, pep, pep_attn_mask, i_label
    
 