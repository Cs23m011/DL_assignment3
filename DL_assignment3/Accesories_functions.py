import numpy as np
import pandas as pd
import torch
import random
import torch.nn as nn                   #importing necessary libraries
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as L
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import wandb
from matplotlib.font_manager import FontProperties as fp

if torch.cuda.is_available():
    device = torch.device('cuda')  # Get the GPU device
else:
    device = torch.device('cpu')    #if GPU not available then device is CPU

class trainDataset(Dataset):     #creating Dataset class which will be neaded to create dataloader class
    def __init__(self,p,p1):               #constructor to trainDataset class
        self.encoder_embeddings,self.decoder_embeddings=p,p1  #p,p1 are padded tensor for input and output data
        self.encoder_embeddings.detach_()       # Ensuring tensors are detached from the computation graph
        self.decoder_embeddings.detach_()

    def __getitem__(self,idx):
        return self.encoder_embeddings[idx],self.decoder_embeddings[idx]   #returning idx corresponding data

    def __len__(self):
        return self.encoder_embeddings.shape[0]             #returning the length of the data 
class Helper_functions:                          #this class contain necessary functions
    def Extract_data(self,path):
        """
        Extract data from a CSV file, processing each row to split by commas.

        Args:
            path (str): The file path to the CSV (Tab-Separated Values) file.

        Returns:
            list: A list of lists, where each inner list contains the comma-separated values from a row in the file.
        """
        panda_data = pd.read_csv(path, sep='\t',on_bad_lines='skip',header=None)
        Data = [row[0].split(',') for row in panda_data.values.tolist()]
        return Data
        
    def helper(self,data):
        ''' this takes input as the data which is a list contain tuple of input and output words

            returns:
            
                first list containing input data ,second list contain corresponding output data,third and fourth are
                dictionary containing mapping to input and output alphabet to corresponding index respectively and fifth and
                sixth are the length of input and output vocabulary size.
        '''
    
    
    
    
        encoder_alphabets=set()    #set containing input alphabet
        decoder_alphabets=set()     #set containing output alphabet
        encoder_input_text=[]        # contain input words
        decoder_input_text=[]          # contain output words
        max_encoder_word_length=-1
        max_decoder_word_length=-1
        for x in data:
            e,tl=x[0],x[1]
            e=e+'$'                        #adding end symbol to input word
            encoder_input_text.append(e)
            tl='*'+tl+'$'                  #adding start and end symbol to output word
            decoder_input_text.append(tl)
            max_encoder_word_length=max(max_encoder_word_length,len(e))
            max_decoder_word_length=max(max_decoder_word_length,len(tl))
            for c in e:
                if c!='$':
                    encoder_alphabets.add(c)
            for c in tl:
                if c!='$' and c!='*':
                    decoder_alphabets.add(c)
        encoder_alphabet=sorted(encoder_alphabets)
        decoder_alphabet=sorted(decoder_alphabets)
        encoder_char_to_index={char: index+1 for index, char in enumerate(encoder_alphabet)}  #creating char to index mapping dictionary 
        decoder_char_to_index={char: index+1 for index, char in enumerate(decoder_alphabet)}
        encoder_char_to_index['$']=len(encoder_alphabets)+1
        decoder_char_to_index['$']=len(decoder_alphabets)+1
        decoder_alphabets.add('$')     #adding end symbol to dictionary
        encoder_alphabets.add('$')
        decoder_char_to_index['*']=len(decoder_alphabets)+1
        decoder_alphabets.add('*')
        encoder_alphabets.add('*')

        encoder_index_to_char={index: char for char, index in encoder_char_to_index.items()}  #creating index to char mapping dictionary 
        decoder_index_to_char={index: char for char, index in decoder_char_to_index.items()}

        return encoder_input_text,decoder_input_text,encoder_char_to_index,decoder_char_to_index,len(encoder_alphabets),len(decoder_alphabets)
    
    def words_to_tensor(self,encoder_input_text,encoder_char_to_index,decoder_input_text,decoder_char_to_index):
        '''
           input encoder input text and a dictionary corresponding to input vocabulary to index .Similar thing for output also.
           return :
               two tensor which contain index corresponding to character in the dictionary padded with 0.
    
         '''
        word_indices_encoder = [[encoder_char_to_index[char] for char in word] for word in encoder_input_text]     #vectorize the words
        padded_sequences = pad_sequence([torch.tensor(indices) for indices in word_indices_encoder],batch_first=True,padding_value=0)  #added padding in the tensor
        word_indices_decoder = [[decoder_char_to_index[char] for char in word] for word in decoder_input_text]       #vectorize the words
        padded_sequences1 = pad_sequence([torch.tensor(indices) for indices in word_indices_decoder],batch_first=True, padding_value=0) #added padding in the tensor
        return padded_sequences,padded_sequences1

    def tensor_To_english(self,data,encoder_index_to_char):
        
        '''
        input: tensor , encoder_index_to_char: dictionary index from english vocabulary to index.
        return:
              corresponding english word
        
        '''
    
    
    
        words=[]
        for i in range(0,data.shape[0]):
            str1=''
            for j in range(0,data.shape[1]):
                if data[i][j]==0 or data[i][j]==27:
                    continue
                else:
                    str1=str1+encoder_index_to_char[data[i][j].item()]    # adding english character correspond to index in the string
            words.append(str1)
        return words                 

    def saveFile(self,data1,data2,data3):
        '''
            input: three list data1,data2,data3
            output : return a CSV file adding this data column wise
        
        
        
        '''
    
    
    
        data = zip(data1,data2,data3)
        column_widths = (20,20,20)
        file_name = 'prediction_vanilla.csv'

        with open(file_name, 'w', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['English', 'Telegu', 'Predicted_Telegu'])   
            for row in data:
                formatted_row = [f"{col:<{width}}" for col, width in zip(row, column_widths)]
                csvwriter.writerow(formatted_row)     #writting the file in same folder
        print(f"Data written to {file_name}")

    def tensor_To_word(self,pred,true,decoder_index_to_char):
        '''
                input: tensor , encoder_index_to_char: dictionary index from telegu vocabulary to index.
        return:
              corresponding telegu words
        
        '''
        
        
        
        true_word=[]        # Initialize lists to store the decoded true and predicted words
        pred_word=[]
        decoder_char_to_index={char: index for index, char in decoder_index_to_char.items()} # Create a reverse mapping from characters to indices for easy lookup
        c=0
        for i in range(0,true.shape[0]):        # Iterate over each sequence in the batch
            str1=''
            str2=''                           # Iterate over each character index in the sequence, starting from the second position
            for j in range(1,true.shape[1]):
                if true[i][j]==decoder_char_to_index['$']:
                        break                    # Stop if the end-of-sequence character ('$') is encountered in the true sequence
                else:
                    str1=str1+decoder_index_to_char[true[i][j].item()] # Append the character corresponding to the true index to str1
                    str2=str2+decoder_index_to_char[pred[i][j].item()]
            true_word.append(str1)                  # Add the decoded true and predicted words to their respective lists
            pred_word.append(str2)
        return true_word,pred_word
    
    def plot_heatmaps_in_grid(self,heatmap_images):

        fig, axes = plt.subplots(3, 3, figsize=(24, 24)) # Create a figure with a 3x3 grid of subplots
        for i, ax in enumerate(axes.flatten()):    # Iterate over each subplot axis and the heatmap images
            ax.imshow(heatmap_images[i], cmap="YlGnBu", interpolation='nearest')  # Display the heatmap image on the axis with specified colormap and interpolation
            ax.axis('off')   # Remove axis ticks and labels
        plt.tight_layout()        #adjust the layout to prevent overlapping
        wandb.log({"attention_heatmaps": wandb.Image(plt)})  #logging into wandb
        wandb.finish()
        plt.show()   

    def print_attention_map(self,english_word, true_word, pred_word, attention_value):
        """
        Generate attention heatmaps and log them using Weights and Biases (wandb).

        Args:
            attention_value (torch.Tensor): The attention values tensor with shape (batch_size, seq_length, seq_length).
            english_word (list of str): List of English words corresponding to the attention values.
            pred_word (list of str): List of predicted words corresponding to the attention values.
        """
        
        
        
        heatmap_images = []
        att = attention_value.permute(1, 0, 2)  # Extract the relevant attention values and convert to NumPy array
        for i in range(att.shape[0]):
            att1 = att[i]
            eng1 = english_word[i]
            bngp1 = pred_word[i]
            attention = att1[1:len(bngp1)+1, 1:len(eng1)+1].cpu().numpy() # Extract the relevant attention values and convert to NumPy array
            predicted = list(bngp1)
            fig, ax = plt.subplots(1, 1, figsize=(8,8))
            img = sns.heatmap(attention, cmap="YlGnBu", cbar=True,ax=ax) # Plot the heatmap
            labels = list(eng1)
            ax.set_xticklabels(labels)
            ax.set_yticklabels(predicted, fontdict={'fontsize':16}, rotation=0, fontproperties=fp(fname='telegu.ttf'))
            # Convert the Matplotlib figure to a NumPy array
            fig.canvas.draw()
            image_array = np.array(fig.canvas.renderer.buffer_rgba())
            plt.close(fig)  # Close the figure to prevent displaying individual plots
            heatmap_images.append(image_array)
        self.plot_heatmaps_in_grid(heatmap_images) # Call the method to plot heatmaps in a gri