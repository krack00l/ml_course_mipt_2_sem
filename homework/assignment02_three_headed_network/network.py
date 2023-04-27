import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F

import tqdm


class Reorder(nn.Module):
    def forward(self, input):
        return input.permute((0, 2, 1))

class ThreeInputsNet(nn.Module):
    def __init__(self, n_tokens, n_cat_features, concat_number_of_features, hid_size=64):
        super(ThreeInputsNet, self).__init__()
        alpha = 0.5
        dropout = 0.2
        #first head - emb-conv-pool
        self.title_emb = nn.Embedding(n_tokens, embedding_dim=hid_size)
        self.title_reorder = Reorder()
        self.title_conv = nn.Conv1d(in_channels=hid_size,
                                    out_channels=hid_size,
                                    kernel_size=2)
        self.title_relu = nn.ReLU()
        self.title_adapt_avg_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.title_dropout = nn.Dropout(dropout)
        self.title_flatten = nn.Flatten()
        self.title_linear = nn.Linear(in_features=hid_size,
                                      out_features=1) 

        
        #second head - emb-conv-pool (full_description)
        self.full_emb = nn.Embedding(num_embeddings=n_tokens, embedding_dim=hid_size)
        self.full_reorder = Reorder()
        self.full_conv = nn.Conv1d(in_channels=hid_size,
                                    out_channels=hid_size,
                                    kernel_size=2)
        self.full_relu = nn.ReLU()
        self.full_adapt_avg_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.full_dropout = nn.Dropout(dropout)
        self.full_flatten = nn.Flatten()
        self.full_linear = nn.Linear(in_features=hid_size,
                                      out_features=1) 
        
        #third head - fully-connected
        self.category_linear_1 = nn.Linear(n_cat_features, int(n_cat_features / 2)) #number_category_features
        self.category_ELU = nn.ELU(alpha)
        self.category_linear_2 = nn.Linear(in_features = int(n_cat_features / 2),
                                           out_features=1)

        # Example for the final layers (after the concatenation)
        self.inter_dense = nn.Linear(in_features=concat_number_of_features, out_features=hid_size*2)
        self.out_relu = nn.ReLU()
        self.final_dense = nn.Linear(in_features=hid_size*2, out_features=1)

        

    def forward(self, whole_input):
        input1, input2, input3 = whole_input

        #first head
        title_beg = self.title_emb(input1).permute((0, 2, 1))
        title = self.title_conv(title_beg)
        #title = self.title_reorder() смена осей уже есть (понть до конца зачем здесь нужна смена осей)
        title = self.title_relu(title)
        title = self.title_adapt_avg_pool(title)
        title = self.title_dropout(title)
        title = self.title_flatten(title)
        title = self.title_linear(title)

        #second head
        full_beg = self.full_emb(input2).permute((0, 2, 1))
        #full = self.full_reorder(full_beg)
        full = self.full_conv(full_beg)
        full = self.full_relu(full) 
        full = self.full_adapt_avg_pool(full)
        full = self.full_dropout(full)
        full = self.full_flatten(full)
        full = self.full_linear(full)    
        
        #third head
        category = self.category_linear_1(input3)
        category = self.category_ELU(category)
        category = self.category_linear_2(category)
        
        concatenated = torch.cat(
            [
            title.view(title.size(0), -1),
            full.view(full.size(0), -1),
            category.view(category.size(0), -1)
            ],
            dim=1)
        

        out = self.inter_dense(concatenated)
        out = self.out_relu(out)
        out = self.final_dense(out)
        
        return out