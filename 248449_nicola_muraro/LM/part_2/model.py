import torch.nn as nn
import torch


class VariationalDropout(nn.Module):
    def __init__(self, size, dropout_value=0.5, device="cpu"):
        super(VariationalDropout, self).__init__()
        
        self.dropout_value = dropout_value
        self.size = size
        self.device = device
    
    def forward(self, input):
        one_mask_initialization = torch.full((input.shape[0], self.size), 1.0 - self.dropout_value, device=self.device)
        one_instance_mask = torch.bernoulli(one_mask_initialization)
        one_instance_mask = one_instance_mask / (1.0 - self.dropout_value) #We rescale the mask to have the same expected value as the original input. 
        mask = one_instance_mask.unsqueeze(1).expand(-1, input.shape[1], -1)
        
        result = input * mask
        return result


class LM_LSTM(nn.Module): #This is the model we will use for this second part of the assignment
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, emb_dropout=0, out_dropout=0, n_layers=1, device="cpu"):
        super(LM_LSTM, self).__init__()
        self.device = device
        
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.variational_dropout_1 = VariationalDropout(emb_size, emb_dropout, device=self.device)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)    
        self.variational_dropout_2 = VariationalDropout(hidden_size, out_dropout, device=self.device)
        self.output = nn.Linear(hidden_size, output_size)
        
        self.embedding.weight = self.output.weight
        self.emb_dropout = emb_dropout
        self.out_dropout = out_dropout
        self.emb_size = emb_size
        self.hidden_size = hidden_size 
    
    
    def forward(self, input_sequence, train=True):
        emb = self.embedding(input_sequence)
        
        if self.emb_dropout != 0 and train:
            emb = self.variational_dropout_1(emb)
        
        lstm_out, _  = self.lstm(emb)
        
        if self.out_dropout != 0 and train:
            lstm_out = self.variational_dropout_2(lstm_out)
        
        output = self.output(lstm_out).permute(0,2,1) #We permute the dimensions to have the output compatible with the CrossEntropyLoss
        return output 
