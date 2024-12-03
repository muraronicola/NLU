import torch.nn as nn
import torch

class LM_LSTM(nn.Module): #This is the model we will use for this second part of the assignment
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, variational_dropout=0, n_layers=1, device="cpu"):
        super(LM_LSTM, self).__init__()
        self.device = device
        
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.locked_dropout = torch.full((emb_size, emb_size), 1.0 - variational_dropout, device=self.device).bernoulli() #We create a dropout mask for the embeddings and for the output of the LSTM
        
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)    

        self.output = nn.Linear(hidden_size, output_size)
        
        
        self.embedding.weight = self.output.weight
        self.variational_dropout = variational_dropout
        self.emb_size = emb_size
        self.hidden_size = hidden_size 
    
    
    def changeDropoutMask(self): #Update the dropout mask, used every epoch
        if self.variational_dropout != 0:
            self.locked_dropout = torch.full((self.emb_size, self.emb_size), 1.0 - self.variational_dropout, device=self.device).bernoulli()
    
    
    def forward(self, input_sequence, train=True):
        emb = self.embedding(input_sequence)
        if self.variational_dropout != 0 and train:
            emb = self.locked_dropout[:emb.shape[1]] * emb
        
        lstm_out, _  = self.lstm(emb)
        
        if self.variational_dropout != 0 and train:
            lstm_out = self.locked_dropout[:lstm_out.shape[1]] * lstm_out
        
        output = self.output(lstm_out).permute(0,2,1) #We permute the dimensions to have the output compatible with the CrossEntropyLoss
        return output 
