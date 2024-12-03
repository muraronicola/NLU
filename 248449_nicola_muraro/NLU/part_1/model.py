import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ModelIAS(nn.Module):

    def __init__(self, out_slot, out_int, vocab_len, emb_size=300, hid_size=200, dropout_value=0, n_layer=1, pad_index=0):
        super(ModelIAS, self).__init__()
        
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        
        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=True, batch_first=True)  
        
        self.slot_out = nn.Linear(hid_size * 2, out_slot)
        self.intent_out = nn.Linear(hid_size, out_int)
        
        self.dropout = nn.Dropout(dropout_value)
        
    
    def forward(self, utterance, seq_lengths):
        utt_emb = self.embedding(utterance) 
        
        #Dropout
        utt_emb = self.dropout(utt_emb)
        
        # pack_padded_sequence avoid computation over pad tokens reducing the computational cost
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        # Process the batch
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input) 

        # Unpack the sequence
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        # Get the last hidden state
        last_hidden = last_hidden[-1,:,:]
        
        #Dropout
        utt_encoded = self.dropout(utt_encoded)
        last_hidden = self.dropout(last_hidden)
        
        slots = self.slot_out(utt_encoded)
        intent = self.intent_out(last_hidden)
        
        slots = slots.permute(0,2,1) # We need this for computing the loss
        return slots, intent

