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
import argparse
from Accesories_functions import trainDataset,Helper_functions
from Encoder import encoder
from Vanila_decoder import decoder
from Encoder_Decoder_vanilla_seq import eng_ben
from Attention_Decoder import decoder_with_Attention
from seq_to_seq_attention import eng_ben_attention
wandb.login()


if torch.cuda.is_available():
    device = torch.device('cuda')  # Get the GPU device
else:
    device = torch.device('cpu')   # Use CPU if GPU is not available

def main(args):
    if 145==145:
        obj=Helper_functions()
        #config=wandb.config
        #wandb.run.name='bs-'+str(config.batch_size)+'-ct-'+ str(config.cell_type)+'-ep-'+str(config.epochs)+ '-es-'+str(config.embedding_size)+'-hs-'+str(config.hidden_size)+'-nel-'+str(config.encoder_num_layers)+'-ndl-'+str(config.decoder_num_layers)+'-do-'+str(config.drop_out)+'-bd-'+str(config.bidirectional)
        path=args.path
        # Extracting and preprocessing training, validation, and test data
        train_data=obj.Extract_data(path+'/tel_train.csv')
        train_test_data=obj.Extract_data(path+'/tel_test.csv')
        train_val_data=obj.Extract_data(path+'/tel_valid.csv')
        encoder_input_text,decoder_input_text,encoder_char_to_index,decoder_char_to_index,encoder_vocab,decoder_vocab=obj.helper(train_data)
        # Creating character-to-index and index-to-character mappings for encoder and decoder
        decoder_index_to_char={index: char for char, index in decoder_char_to_index.items()}
        encoder_index_to_char={index: char for char, index in encoder_char_to_index.items()}
        batch_size=args.batch_size
        # Creating training, validation, and test datasets and dataloaders
        p,p1=obj.words_to_tensor(encoder_input_text,encoder_char_to_index,decoder_input_text,decoder_char_to_index)
        dataset1=trainDataset(p,p1)
        dataloader=DataLoader(dataset=dataset1,batch_size=batch_size,shuffle=True,num_workers=1)
        encoder_input_text_test,decoder_input_text_test,_,_,_,_=obj.helper(train_test_data)
        encoder_input_text_val,decoder_input_text_val,_,_,_,_=obj.helper(train_val_data)
        p_test,p1_test=obj.words_to_tensor(encoder_input_text_test,encoder_char_to_index,decoder_input_text_test,decoder_char_to_index)
        testdataset=trainDataset(p_test,p1_test)  #create train dataset
        testDataloader=DataLoader(dataset=testdataset,batch_size=batch_size,shuffle=False,num_workers=1)
        p_val,p1_val=obj.words_to_tensor(encoder_input_text_val,encoder_char_to_index,decoder_input_text_val,decoder_char_to_index)
        val_dataset=trainDataset(p_val,p1_val)
        valDataloader=DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False,num_workers=1)
        heatMap_dataloader=DataLoader(dataset=testdataset,batch_size=9,shuffle=True,num_workers=1)
        first_batch = next(iter(heatMap_dataloader))
         # Retrieving model parameters from command-line arguments
        cell_type=args.cell_type
        embedding_size=args.embedding_size
        hidden_size=args.hidden_size
        encoder_num_layer=args.encoder_num_layer
        decoder_num_layer=args.decoder_num_layer
        drop_out=args.dropout_rate
        epoch=args.epochs
        bidirectional=args.bidirectional
        max_seq_length=p.shape[1]
        # Initializing Wandb
        wandb.init(entity=args.wandb_entity,project=args.project_name)
        wandb_logger = WandbLogger(project=args.project_name, entity=args.wandb_entity)
        # Initializing encoder and decoder models
        en=encoder(cell_type,encoder_vocab,embedding_size,hidden_size,encoder_num_layer,drop_out,bidirectional)
        de=decoder(cell_type,decoder_vocab,embedding_size,hidden_size,decoder_num_layer,drop_out,epoch,0.5,bidirectional)
        de_att=decoder_with_Attention(max_seq_length,cell_type,decoder_vocab,embedding_size,hidden_size,decoder_num_layer,drop_out,epoch,0.5,bidirectional)
         # Selecting the model based on attention flag
        if args.attention==False:
            model=eng_ben(en,de,decoder_vocab,decoder_index_to_char,encoder_index_to_char)
        else:
            model=eng_ben_attention(en,de_att,decoder_vocab,decoder_index_to_char,encoder_index_to_char)
            
         # Training the model
        trainer = L.Trainer(accelerator='auto',devices="auto",max_epochs=epoch,num_sanity_val_steps=0,logger=wandb_logger)
        trainer.fit(model,dataloader,valDataloader)
        trainer.test(dataloaders=testDataloader,ckpt_path='best')  # Testing the model
        if args.attention==False and args.plot_heatmap==True:  # Printing a message if heatmap plotting is requested but attention is not applied
            print("attention has not applied to the model")
        elif args.attention==True and args.plot_heatmap==True:
            trainer.predict(dataloaders=heatMap_dataloader,ckpt_path='best')
		
if  __name__ =="__main__":
    parser = argparse.ArgumentParser()  #taking arguments from command line arguments
    # Adding command-line arguments
    parser.add_argument('-wp','--project_name',type=str,default='CS6910-Assignment',help='Project name used to track experiments in Weights & Biases dashboard')
    parser.add_argument('-we','--wandb_entity',type=str,default='amar_cs23m011',help='Project name used to track experiments in Weights & Biases dashboard')
    parser.add_argument('-p','--path',type=str,help='provide the path where your data is stored in memory,Read the readme for more description')
    parser.add_argument('-e','--epochs',type=int,default=15,help='Number of epochs to train')
    parser.add_argument('-b','--batch_size',type=int,default=128,help='Batch size used to train RNN')
    parser.add_argument('-es','--embedding_size',type=int,default=256,help='default size of Enbedding Layer')
    parser.add_argument('-hs','--hidden_size',type=int,default=256,help='default size of context vector')
    parser.add_argument('-enl','--encoder_num_layer',type=int,default=2,help='number of encoder Layer')
    parser.add_argument('-dnl','--decoder_num_layer',type=int,default=1,help='number of decoder Layer')
    parser.add_argument('-ct','--cell_type',type=str,default='LSTM',choices=['GRU','RNN','LSTM'],help='which cell gonna use either GRU or LSTM or RNN')
    parser.add_argument('-do','--dropout_rate',type=float,default=0.3,choices=[0.2,0.3,0.4],help='drop out rate to regularilze the model')
    parser.add_argument('-bd','--bidirectional',type=bool,default=True,choices=[True,False],help='biderctional model or not')
    parser.add_argument('-a','--attention',type=int,default=1,choices=[1,0],help='We will use attention or not')
    parser.add_argument('-hp','--plot_heatmap',type=int,default=0,choices=[1,0],help='We will plot the heatmap or not')
    args = parser.parse_args()
    main(args)