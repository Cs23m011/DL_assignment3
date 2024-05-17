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
import wandb
from Bahdanau_attention import BahdanauAttention


# Check if GPU is available, otherwise use CPU
if torch.cuda.is_available():
    device = torch.device('cuda')  # Get the GPU device
else:
    device = torch.device('cpu')

class decoder_with_Attention(L.LightningModule):
    def __init__(self,max_seq_length,architecture_type,decoder_vocabulary_size,embedding_size,hidden_size,num_layer,drop_out,epoch,teacher_forcing_ratio=0.5,birectional=False):
        """
        Decoder model with Bahdanau Attention.

        Args:
            max_seq_length (int): Maximum sequence length.
            architecture_type (str): Type of RNN architecture (e.g., 'GRU', 'LSTM', 'RNN').
            decoder_vocabulary_size (int): Size of the decoder vocabulary.
            embedding_size (int): Size of the embedding layer.
            hidden_size (int): Size of the hidden state.
            num_layer (int): Number of RNN layers.
            drop_out (float): Dropout probability.
            epoch (int): Number of epochs.
            teacher_forcing_ratio (float): Probability of teacher forcing during training.
            bidirectional (bool): Indicates whether the model is bidirectional.
        """
        
        
        
        
        
        super().__init__()
        self.epoch=epoch
        self.decoder_vocabulary_size=decoder_vocabulary_size
        self.max_seq_length=max_seq_length
        self.embedding_size=embedding_size
        self.hidden_size=hidden_size
        self.num_layer=num_layer
        if num_layer==1:       # Adjust dropout if num_layer is 1
            drop_out=0
        self.drop_out=nn.Dropout(drop_out)
        self.birectional=birectional
        self.architecture_type=architecture_type
        self.teacher_forcing_ratio=teacher_forcing_ratio
        self.embedding=nn.Embedding(self.decoder_vocabulary_size+1,self.embedding_size)  # Embedding layer defining
        if num_layer==1:
            flag=0
        else:
            flag=1
        self.attention=BahdanauAttention(self.hidden_size,self.max_seq_length,self.birectional,num_layer,flag)  #creating object for attention class
        if self.birectional==True:             #adjusting shapes if bidirectional
            h2=2*self.hidden_size
        else:
            h2=self.hidden_size
        self.fc=nn.Linear(h2,decoder_vocabulary_size)  #defing feed forward layer
        if self.architecture_type=='GRU':          #based on the architecture type defining cells
            self.model_type=nn.GRU(self.embedding_size+h2,self.hidden_size,self.num_layer,bias=True, dropout=drop_out, bidirectional=self.birectional)
        if self.architecture_type=='RNN':
            self.model_type=nn.RNN(self.embedding_size+h2,self.hidden_size,self.num_layer,bias=True, dropout=drop_out, bidirectional=self.birectional)
        if self.architecture_type=='LSTM':
            self.model_type=nn.LSTM(self.embedding_size+h2,self.hidden_size,self.num_layer,bias=True, dropout=drop_out, bidirectional=self.birectional)

    def forward(self,epoch,en_output,target,hidden,cell=None,tf_bit=1):
        """
        Forward pass of the decoder model with attention.

        Args:
            epoch (int): Current epoch.
            en_output (torch.Tensor): Encoder output.
            target (torch.Tensor): Target sequence.
            hidden (torch.Tensor): Initial hidden state.
            cell (torch.Tensor, optional): Initial cell state for LSTM.
            tf_bit (int, optional): Teacher forcing bit (0 or 1).

        Returns:
            torch.Tensor: Output sequence.
            torch.Tensor: Attention weights.
        """
        
        
        
        
        
        batch_size=target.shape[1]
        seq_length=target.shape[0]
        outputs=torch.zeros(seq_length,batch_size,self.decoder_vocabulary_size,device=device) # Initialize the outputs tensor
        attention_weights=torch.zeros(seq_length,batch_size,en_output.shape[0],device=device)  # Initialize the attention tensor
        x=target[0]       # Get the first input token
        for i in range(1,seq_length):    # Perform a decoding step
            if self.architecture_type=='LSTM':
                attn,output,hidden,cell=self.decoder_step(en_output,x,hidden,cell)
            else:
                attn,output,hidden,cell=self.decoder_step(en_output,x,hidden,None)
            outputs[i]=output # Store the output
            attention_weights[i]=attn       # Store the attention
            pred_char=output.argmax(1)   # Get the predicted character
            if tf_bit==1:
                x = target[i] if random.random() < self.teacher_forcing_ratio else pred_char #applying teacher forcing if trainnig in case of testing tf_bit=0
            else:
                x=pred_char
        return outputs,attention_weights
    def decoder_step(self,en_output,x,hidden,cell=None):
        """
        Perform a single decoding step.

        Args:
            en_output (torch.Tensor): Encoder output.
            x (torch.Tensor): Input tensor.
            hidden (torch.Tensor): Hidden state.
            cell (torch.Tensor, optional): Cell state for LSTM.

        Returns:
            torch.Tensor: Attention weights.
            torch.Tensor: Output prediction.
            torch.Tensor: Updated hidden state.
            torch.Tensor: Updated cell state for LSTM.
        """
        
        
        
        
        x=x.unsqueeze(0)  #adjusting shapes
        x=self.drop_out(self.embedding(x))
        query = hidden.permute(1, 0, 2)
        if self.birectional==True or query.shape[1]>1:
            query1=torch.cat([query[:, i, :] for i in range(query.shape[1])], dim=1) # Concatenate query for bidirectional or multi-layer models
            query=query1.unsqueeze(1)
        key=en_output.permute(1,0,2)
        context, attn_weights = self.attention(query, key)
        context1=context.permute(1,0,2)
        input_model = torch.cat((x, context1), dim=2)
        if self.architecture_type=='LSTM':         # Perform RNN step
            output,(hidden,cell)=self.model_type(input_model,(hidden,cell))
        else:
            output,hidden=self.model_type(input_model,hidden)

        pred=self.fc(output)       # Linear layer for prediction
        pred=pred.squeeze(0)
        return attn_weights.permute(1,0,2).squeeze(0),pred,hidden,cell