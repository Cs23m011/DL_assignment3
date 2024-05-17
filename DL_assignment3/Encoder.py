import numpy as np
import pandas as pd
import torch
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as L
from torch.nn.utils.rnn import pad_sequence   #importing necessary modules

if torch.cuda.is_available():
    device = torch.device('cuda')  # Get the GPU device
else:
    device = torch.device('cpu') # Get the CPU device


class encoder(L.LightningModule):
    def __init__(self,architecture_type,encoder_vocabulary_size,embedding_size,hidden_size,num_layer,drop_out,birectional=False):
        """
        Initialize the Encoder.

        Args:
            architecture_type (str): Type of the architecture ('GRU', 'RNN', 'LSTM').
            encoder_vocabulary_size (int): Size of the encoder vocabulary.
            embedding_size (int): Size of the embedding vectors.
            hidden_size (int): Number of features in the hidden state.
            num_layers (int): Number of recurrent layers.
            dropout (float): Dropout probability.
            bidirectional (bool): If True, becomes a bidirectional encoder.
        """
        super().__init__()
        self.encoder_vocabulary_size=encoder_vocabulary_size
        self.embedding_size=embedding_size
        self.hidden_size=hidden_size
        self.num_layer=num_layer
        if num_layer==1:   # Ensure dropout is set to 0 if num_layers is 1
            drop_out=0
        self.architecture_type=architecture_type
        self.drop_out=nn.Dropout(drop_out)
        self.embedding=nn.Embedding(self.encoder_vocabulary_size+1,self.embedding_size)
        self.birectional=birectional
        # Initialize the appropriate model type
        if self.architecture_type=='GRU': 
            self.model_type=nn.GRU(self.embedding_size,self.hidden_size,self.num_layer,bias=True, dropout=drop_out, bidirectional=self.birectional)
        if self.architecture_type=='RNN':
            self.model_type=nn.RNN(self.embedding_size,self.hidden_size,self.num_layer,bias=True, dropout=drop_out, bidirectional=self.birectional)
        if self.architecture_type=='LSTM':
            self.model_type=nn.LSTM(self.embedding_size,self.hidden_size,self.num_layer,bias=True, dropout=drop_out, bidirectional=self.birectional)
    def forward(self,x):
        """
        Forward pass for the Encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size).

        Returns:
            tuple: Depending on the architecture type, returns:
                - (hidden, cell, output) for LSTM
                - (hidden, None, output) for GRU and RNN
        """
    
    
    
        x=self.drop_out(self.embedding(x))  # Apply embedding layer followed by dropout
        if self.architecture_type=='LSTM':    # Forward pass through the recurrent model
            output,(hidden,cell)=self.model_type(x)
            return hidden,cell,output
        else:
            output,hidden=self.model_type(x)
            return hidden,None,output