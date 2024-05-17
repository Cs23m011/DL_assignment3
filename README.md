# DL_assignment3
please install the dependencies before running the program other wise it may give error .I have added the requirement.txt file in the same folder. Please download the code from github and extract it .Then install the dependencies and run trainRNN.py .Note that all the other necessary module should be there in the same folder as with trainRNN.py as it import all the other classes from different file . If you want to run it in kaggle/colab it will also run normally as command promt.
```
pip install requirement.txt

python trainRNN.py -p path
```
My code is very much flexible to add in command line arguments . I am adding the list of possible argument below for your reference.Please try to run this on local PC or from command promt by ensuring all the libraries in requirements.txt are already installed in your system. Because in google colab this might give problem .
python trainRNN.py -p path to the folder where data reside. For example C:\Users\USER\Downloads\aksharantar_sampled\aksharantar_sampled\tel this where my train data,validation data,test data reside then you just given till 
C:\Users\USER\Downloads\aksharantar_sampled\aksharantar_sampled\tel , program will add train and val portion itself. My model is prepare to work on telegu language so giving location to other dataset for example asami or urdu give error .

| Name        | Default Value   | Description |
| ------------- |:-------------:| -----:|
| `-wp ,--project_name`     | CS6910-Assignment | it will make login into wandb in the project_name project |
| `-e,--epochs` | 15      |    number of epochs your algorithm iterate |
|`-b,--batch_size`|64      |batch size your model used to train |
|`-es,--cell_type`|LSTM|Choices=['LSTM', 'RNN', 'GRU']|
|`-we,--wandb_entity`|amar_cs23m011|Project name used to track experiments in Weights & Biases dashboard|
| `-es,--embedding_size` | 256     |    Embedding size of the decoder and encoder|
|`-p,--path`|mandatory field      |location where your data stored. |
| `-hs,--hidden_size` | 256     |    hidden size of the decoder and encoder|
| `-enl,--encoder_num_layer` | 2     |   number of layer in the encoder|
| `-dnl,--decoder_num_layer` | 1     |   number of layer in the decoder|
| `-do,--dropout_rate` | 0.3     |   dropout rate to control overfitting|
| `-bd,--bidirectional` | True     |Choices=[True,False], bidirectional cell will be used or|
| `-a,--attention` | 1     |Choices=[1,0], attention will be applied or not|
| `-hp,--plot_heatmap` | 0     |Choices=[1,0], heatmap will be created and plot and logged to wandb|



Few example are shown below to how to give inputs:-
```
```
This will run my best model which i get by validation accuracy. after that it will create a log in a project named CS6910-assignment12 by default until user dont specify project name.
```
"parameters":
    {
        'hidden_size': {"values":[256]},
        'batch_size': {"values":[64]},
        'encoder_num_layers': {"values":[2]},
        'decoder_num_layers': {"values":[1]},
        'embedding_size': {"values":[128]},
        'drop_out': {"values":[0.3]},
        'epochs':{"values":[15]},
        'cell_type':{"values":["LSTM"]},
        'bidirectional':{"values":[True]}
}
      
    }```
Now if you want to change the number of layer I just have to execute the following the command.
```
python trainRNN.py -e 6
```
this will change the number of epoch to 6. Similarly we can use other commands as well.
My code plots to  wandb so please provide wandb key and project name and wandb entity appropriately so it can plot logs to wandb. 
