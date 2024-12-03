import torch.nn as nn


class LM_LSTM(nn.Module): #This is the model we will use for this first part of the assignment
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0, emb_dropout=0, n_layers=1):
        super(LM_LSTM, self).__init__()

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.drop1 = nn.Dropout(emb_dropout) #Dropout on the embeddings

        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)  #LSTM layer
        #self.pad_token = pad_index
        self.drop2 = nn.Dropout(out_dropout) #Dropout on the output of the LSTM

        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        drop_emb = self.drop1(emb)
        
        lstm_out, _  = self.lstm(drop_emb)
        
        drop_lstm = self.drop2(lstm_out)
        output = self.output(drop_lstm).permute(0,2,1) #We permute the dimensions to have the output compatible with the CrossEntropyLoss
        return output 

