import numpy as np
import pandas as pd
import torch
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as L
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.loggers import WandbLogger
import csv
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import os
import wandb     #importing necesary modules

if torch.cuda.is_available():
    device = torch.device('cuda')  # Get the GPU device
else:
    device = torch.device('cpu')

class decoder(L.LightningModule):
    def __init__(self,architecture_type,decoder_vocabulary_size,embedding_size,hidden_size,num_layer,drop_out,epoch,teacher_forcing_ratio=0.5,birectional=False):
        """
        Initialize the Decoder.

        Args:
            architecture_type (str): Type of the architecture ('GRU', 'RNN', 'LSTM').
            decoder_vocabulary_size (int): Size of the decoder vocabulary.
            embedding_size (int): Size of the embedding vectors.
            hidden_size (int): Number of features in the hidden state.
            num_layers (int): Number of recurrent layers.
            dropout (float): Dropout probability.
            epoch (int): Current epoch.
            teacher_forcing_ratio (float): Ratio for teacher forcing.
            bidirectional (bool): If True, becomes a bidirectional decoder.
        """
        
        
        
        super().__init__()
        self.epoch=epoch
        self.decoder_vocabulary_size=decoder_vocabulary_size
        self.embedding_size=embedding_size
        self.hidden_size=hidden_size
        self.num_layer=num_layer
        if num_layer==1:    #make sure dropout is 0 if number of layer is 1
            drop_out=0
        self.drop_out=nn.Dropout(drop_out)
        self.birectional=birectional
        self.architecture_type=architecture_type
        self.teacher_forcing_ratio=teacher_forcing_ratio
        self.embedding=nn.Embedding(self.decoder_vocabulary_size+1,self.embedding_size)
        if self.birectional==True:          #adjusting dimention if birectional cell
            h2=2*self.hidden_size
        else:
            h2=self.hidden_size
        self.fc=nn.Linear(h2,decoder_vocabulary_size)
        if self.architecture_type=='GRU':   # Initialize the appropriate model type
            self.model_type=nn.GRU(self.embedding_size,self.hidden_size,self.num_layer,bias=True, dropout=drop_out, bidirectional=self.birectional)
        if self.architecture_type=='RNN':
            self.model_type=nn.RNN(self.embedding_size,self.hidden_size,self.num_layer,bias=True, dropout=drop_out, bidirectional=self.birectional)
        if self.architecture_type=='LSTM':
            self.model_type=nn.LSTM(self.embedding_size,self.hidden_size,self.num_layer,bias=True, dropout=drop_out, bidirectional=self.birectional)

    def forward(self,epoch,target,hidden,cell=None,tf_bit=1):
        """
        Forward pass for the Decoder.

        Args:
            epoch (int): Current epoch.
            target (torch.Tensor): Target sequence tensor of shape (seq_length, batch_size).
            hidden (torch.Tensor): Initial hidden state tensor of shape (num_layers * num_directions, batch_size, hidden_size).
            cell (torch.Tensor, optional): Initial cell state tensor for LSTM of shape (num_layers * num_directions, batch_size, hidden_size).
            tf_bit (int): Flag to indicate whether to use teacher forcing (1) or not (0).

        Returns:
            torch.Tensor: Outputs of shape (seq_length, batch_size, decoder_vocabulary_size).
        """
        
        
        
        
        
        
        batch_size=target.shape[1]
        seq_length=target.shape[0]
        outputs=torch.zeros(seq_length,batch_size,self.decoder_vocabulary_size,device=device) # Initialize the outputs tensor
        x=target[0]  # Get the first input token
        for i in range(1,seq_length):  # Perform a decoding step
            if self.architecture_type=='LSTM':
                output,hidden,cell=self.decoder_step(x,hidden,cell)
            else:
                output,hidden,cell=self.decoder_step(x,hidden,None)
            outputs[i]=output      # Store the output
            pred_char=output.argmax(1)     # Get the predicted character
            if tf_bit==1:                  #applying teacher forcing if trainnig in case of testing tf_bit=0
                x = target[i] if random.random() < self.teacher_forcing_ratio else pred_char
            else:
                x=pred_char         
        return outputs
    def decoder_step(self,x,hidden,cell=None):
        """
        Perform a single decoding step.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size).
            hidden (torch.Tensor): Hidden state tensor.
            cell (torch.Tensor, optional): Cell state tensor for LSTM.

        Returns:
            tuple: (pred, hidden, cell)
                - pred (torch.Tensor): Prediction tensor of shape (batch_size, decoder_vocabulary_size).
                - hidden (torch.Tensor): Updated hidden state tensor.
                - cell (torch.Tensor, optional): Updated cell state tensor for LSTM.
        """
    
    
        x=x.unsqueeze(0)   #adjusting dimentions
        x=self.drop_out(self.embedding(x))
        if self.architecture_type=='LSTM':  #depending on cell type running the decoder
            output,(hidden,cell)=self.model_type(x,(hidden,cell))
        else:
            output,hidden=self.model_type(x,hidden)

        pred=self.fc(output)
        pred=pred.squeeze(0)
        return pred,hidden,cell   #return the output
