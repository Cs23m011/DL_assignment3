import numpy as np
import pandas as pd
import torch
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as L           #importing necessary modules

class BahdanauAttention(L.LightningModule):
    def __init__(self,hidden_size,max_seq_length,bidirectional,dec_layer,flag=1):
        """
        Initialize the Bahdanau Attention mechanism.

        Args:
            hidden_size (int): Size of the hidden state.
            max_seq_length (int): Maximum sequence length.
            bidirectional (bool): Indicates whether the model is bidirectional.
            dec_layer (int): Number of decoder layers.
            flag (int): Flag indicating attention mechanism variant.
        """
        
        super().__init__()
        self.bidirectional=bidirectional
        if bidirectional:
            hidden_size=2*hidden_size
        self.Wa = nn.Linear(hidden_size*dec_layer, max_seq_length)  # Linear layers for attention
        self.Ua = nn.Linear(hidden_size, max_seq_length)
        self.Va = nn.Linear(max_seq_length, 1)
    def forward(self, query, keys):
        """
        Perform forward pass through the attention mechanism.

        Args:
            query (torch.Tensor): Query tensor.
            keys (torch.Tensor): Keys tensor.

        Returns:
            tuple: Tuple containing context tensor and attention weights.
        """
        
        # Compute attention scores
        
        b=self.Wa(query)
        a=self.Ua(keys)
        c=torch.tanh(a+b)
        scores = self.Va(c)  # Compute attention weights
        scores = scores.squeeze(2).unsqueeze(1)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys) # Compute context vector
        return context, weights