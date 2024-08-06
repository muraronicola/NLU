import torch.nn as nn
import torch

class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, variational_dropout=0, n_layers=1, device="cpu"):
        super(LM_LSTM, self).__init__()
        # Token ids to vectors, we will better see this in the next lab 
        self.device = device
        
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        
        self.locked_dropout = torch.full((emb_size, emb_size), 1.0 - variational_dropout, device=self.device).bernoulli()
        
        # Pytorch's RNN layer: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)    
        self.pad_token = pad_index

        # Linear layer to project the hidden layer to our output space 
        self.output = nn.Linear(hidden_size, output_size)
        
        self.embedding.weight = self.output.weight
        
        self.variational_dropout = variational_dropout
        self.emb_size = emb_size
        self.hidden_size = hidden_size 
    
    
    def changeDropoutMask(self):
        if self.variational_dropout != 0:
            self.locked_dropout = torch.full((self.emb_size, self.emb_size), 1.0 - self.variational_dropout, device=self.device).bernoulli()
    
    
    def forward(self, input_sequence, train=True):
        emb = self.embedding(input_sequence)
        if self.variational_dropout != 0 and train:
            emb = self.locked_dropout[:emb.shape[1]] * emb
        
        lstm_out, _  = self.lstm(emb)
        
        if self.variational_dropout != 0 and train:
            lstm_out = self.locked_dropout[:lstm_out.shape[1]] * lstm_out
        
        output = self.output(lstm_out).permute(0,2,1)
        return output 





""" class LM_LSTM_part1(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1, emb_dropout=0.1, n_layers=1):
        super(LM_LSTM_part1, self).__init__()
        # Token ids to vectors, we will better see this in the next lab 
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # Pytorch's RNN layer: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)    
        self.pad_token = pad_index
        # Linear layer to project the hidden layer to our output space 
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        lstm_out, _  = self.lstm(emb)
        output = self.output(lstm_out).permute(0,2,1)
        return output 


class LM_LSTM_part2and3(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1, emb_dropout=0.1, n_layers=1):
        super(LM_LSTM_part2and3, self).__init__()
        # Token ids to vectors, we will better see this in the next lab 
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.drop1 = nn.Dropout(emb_dropout)
        # Pytorch's RNN layer: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)    
        self.pad_token = pad_index
        self.drop2 = nn.Dropout(out_dropout)
        # Linear layer to project the hidden layer to our output space 
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        drop_emb = self.drop1(emb)
        lstm_out, _  = self.lstm(drop_emb)
        drop_lstm = self.drop2(lstm_out)
        output = self.output(drop_lstm).permute(0,2,1)
        return output  """