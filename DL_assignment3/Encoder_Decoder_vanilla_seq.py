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
import wandb          #importing necessary modules

if torch.cuda.is_available():
    device = torch.device('cuda')  # Get the GPU device
else:
    device = torch.device('cpu')   # Get the CPU device

from Accesories_functions import trainDataset,Helper_functions   #importing necessary modules
class eng_ben(L.LightningModule):
    def __init__(self,encoder,decoder,target_vocab_size,decoder_index_to_char,encoder_index_to_char,teacher_forcing_ratio=-0.5):
        """
        Initialize the EngBen model.

        Args:
            encoder (nn.Module): Encoder model.
            decoder (nn.Module): Decoder model.
            target_vocab_size (int): Size of the target vocabulary.
            decoder_index_to_char (dict): Mapping from index to character for the decoder.
            encoder_index_to_char (dict): Mapping from index to character for the encoder.
            teacher_forcing_ratio (float): Ratio for teacher forcing.
        """
        
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.target_vocab_size=target_vocab_size
        self.teacher_forcing_ratio=teacher_forcing_ratio
        self.decoder_index_to_char=decoder_index_to_char
        self.encoder_index_to_char=encoder_index_to_char
        self.automatic_optimization = False
        self.val_true_word=[]              # Lists to store words for validation and testing
        self.val_pred_word=[]
        self.test_true_word=[]
        self.test_pred_word=[]
        self.test_eng_word=[]
        self.test12=[]
        self.test13=[]
        self.test14=[]
        self.architecture_type=encoder.architecture_type
        self.obj=Helper_functions()
    def forward(self, inputs, target,tf_bit):
        
        """
        Forward pass for the EngBen model.

        Args:
            inputs (torch.Tensor): Input tensor for the encoder.
            target (torch.Tensor): Target tensor for the decoder.
            tf_bit (int): Teacher forcing flag.

        Returns:
            torch.Tensor: Output tensor from the decoder.
        """
        
        
        
        
        
        hidden,cell,en_output=self.encoder(inputs)
        #print(en_output.shape)

        if self.encoder.birectional==True:           # Adjust hidden and cell states if the encoder is bidirectional
            hidden=hidden[[self.encoder.num_layer-1,-1]]
            hidden=hidden.mean(axis=0).unsqueeze(0)
            hidden=hidden.repeat(2*self.decoder.num_layer,1,1) #depending on decoder num of layer fill the adjust the hidden dimention
            if self.architecture_type=='LSTM':
                cell=cell[[self.encoder.num_layer-1,-1]]
                cell=cell.mean(axis=0).unsqueeze(0)
                cell=cell.repeat(2*self.decoder.num_layer,1,1)      #depending on decoder num of layer fill the adjust the cell dimention

        else:                                         #Adjust hidden and cell states if the encoder is not bidirectional
            hidden=hidden[-1,:,:]
            hidden=hidden.unsqueeze(0)
            hidden=hidden.repeat(self.decoder.num_layer,1,1)
            if self.architecture_type=='LSTM':
                cell=cell[-1,:,:]
                cell=cell.unsqueeze(0)
                cell=cell.repeat(self.decoder.num_layer,1,1)
        outputs=self.decoder(self.trainer.current_epoch,target,hidden,cell,tf_bit)   #call decoder forward method .
        return outputs                                                                 #return the output

    def training_step(self, batch, batch_idx):
        
        """
        Training step for the model.

        Args:
            batch (tuple): Batch of data containing inputs and targets.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Training loss.
        """
        
        
        
        
        
        inputs, target = batch
        inputs1=inputs.permute(1,0)
        target1=target.permute(1,0)
        output = self(inputs1, target1,1)  #calling forward method of this class
        _,pred_tensor=torch.max(output,2)  #predicting classes 
        output=output[1:].reshape(-1,output.shape[2])
        output1=output.to(device)
        target1=target1[1:].reshape(-1)
        target12=target1.to(device)
        optimizer=self.optimizers()            #manual backtraing calling optimizer
        optimizer.zero_grad()
        criteria=nn.CrossEntropyLoss(ignore_index=0)   #ignoring the padding index for loss calculation
        loss = criteria(output1,target12)
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)    #clipping the gradient
        optimizer.step()
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)  #logging the loss to wandb
        return loss

    def configure_optimizers(self):
        """
        Configure optimizers for the model.

        Returns:
            torch.optim.Optimizer: Optimizer.
        """
        
        return torch.optim.Adam(self.parameters(), lr=0.001)   
        #return torch.optim.Adam(self.decoder.parameters(), lr=0.1)

    def validation_step(self,batch,batch_idx):
        
        """
        Validation step for the model.

        Args:
            batch (tuple): Batch of validation data containing inputs and targets.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Validation loss.
        """
    
    
    
        val_inputs,val_outputs = batch
        true_word=val_outputs
        val_outputs=val_outputs.permute(1,0)
        val_inputs=val_inputs.permute(1,0)
        output = self.forward(val_inputs,val_outputs,0)  #calling forward function
        _,pred_tensor=torch.max(output,2)              # calculating prediction class
        pred_tensor1=pred_tensor.permute(1,0)           
        t,p=self.obj.tensor_To_word(pred_tensor1,true_word,self.decoder_index_to_char)  #getting true and predicted word
        self.val_true_word.extend(t)
        self.val_pred_word.extend(p)
        output=output[1:].reshape(-1,output.shape[2])      #reshaping to calculate loss
        output1=output.to(device)
        target1=val_outputs[1:].reshape(-1)    #reshaping to calculate loss
        target12=target1.to(device)
        criteria=nn.CrossEntropyLoss(ignore_index=0)   
        loss = criteria(output1, target12)        #calculating loss

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)  #logging the loss to wandb
        return loss

    def on_train_epoch_end(self):
        '''
        End-of-epoch processing for training, including logging validation accuracy.
        
        '''
        
        
        count=0
        for i in range(len(self.val_true_word)):
            if self.val_true_word[i]==self.val_pred_word[i]:   #calculating number of correctly predicted word
                count=count+1
            if self.trainer.current_epoch==25:
                print(self.val_true_word[i],self.val_pred_word[i])

        print("val accuracy is",count/(len(self.val_true_word)))  #printing validation accuracy
        val_acc=count/(len(self.val_true_word))           
        self.val_true_word=[]
        self.val_pred_word=[]
        self.log('val_acc', val_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)   #logging validation accuracy
    def test_step(self,batch,batch_idx):
        
        """
        Test step for the model.

        Args:
            batch (tuple): Batch of test data containing inputs and targets.
            batch_idx (int): Index of the batch.

        Returns:
            dict: Dictionary containing test accuracy and loss.
        """
        
        
        
        test_inputs,test_outputs = batch
        true_word=test_outputs
        eng_words=self.obj.tensor_To_english(test_inputs,self.encoder_index_to_char)  #calculating english word from the tensor
        self.test_eng_word.extend(eng_words)
        test_outputs=test_outputs.permute(1,0)
        test_inputs=test_inputs.permute(1,0)
        output = self.forward(test_inputs,test_outputs,0)  #calling forward function
        _,pred_tensor=torch.max(output,2)     #predicting labels
        pred_tensor1=pred_tensor.permute(1,0)
        t,p=self.obj.tensor_To_word(pred_tensor1,true_word,self.decoder_index_to_char) ##getting true and predicted word
        self.test_true_word.extend(t)
        self.test_pred_word.extend(p)
        output=output[1:].reshape(-1,output.shape[2])   #reshaping to calculate loss
        output1=output.to(device)
        target1=test_outputs[1:].reshape(-1)    #reshaping to calculate loss
        target12=target1.to(device)
        criteria=nn.CrossEntropyLoss(ignore_index=0) #calculating loss
        loss = criteria(output1, target12)
        accuracy=0
        if self.trainer.num_test_batches[0]==1+batch_idx:   #when we are in last batch we compute the accuracy
            count=0
            for i in range(len(self.test_true_word)):
                if self.test_true_word[i]==self.test_pred_word[i]: #counting number of correct words
                    count=count+1
                else:
                    self.test12.append(self.test_eng_word[i])
                    self.test13.append(self.test_true_word[i])
                    self.test14.append(self.test_pred_word[i])
                    #test144[self.test_eng_word[i]]=self.test_pred_word[i]
            #print(count,len(self.test_true_word))
            accuracy=count/(len(self.test_true_word))   #calculating test accuracy
            accuracy=accuracy*self.trainer.num_test_batches[0]
            #saveFile(self.test_eng_word,self.test_true_word,self.test_pred_word)
        self.log("test_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)  #logging the loss and accuracy
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)


    def on_test_end(self):
        """
        End-of-test processing, including writing predictions to a CSV file.
        """
    
    
        data = zip(self.test12,self.test13,self.test14)
        column_widths = (20,20,20)
        file_name = 'prediction_vanilla.csv'
        with open(file_name, 'w', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['English', 'Telegu', 'Predicted_Telegu'])
            for row in data:
                formatted_row = [f"{col:<{width}}" for col, width in zip(row, column_widths)]
                csvwriter.writerow(formatted_row)   #the test predictin written into the CSV
        print(f"Data written to {file_name}")