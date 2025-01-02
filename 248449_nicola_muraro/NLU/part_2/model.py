import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ModelIAS(nn.Module):

    def __init__(self, model_bert, hiddenSize, out_slot, out_int, drop_value=0.1, device="cpu"):
        super(ModelIAS, self).__init__()
        self.bert = model_bert #The petrained model
        
        self.dropout = nn.Dropout(drop_value)
        
        self.slotFillingLayer = nn.Linear(hiddenSize, out_slot) #The projection head for the slot filling task
        self.intentLayer = nn.Linear(hiddenSize, out_int) #The projection head for the intent classification task
        
        self.__device = device
    
    
    def forward(self, utterance, tokenizedUtterance):
        predictionBert = self.bert(**utterance) #Get the output of the pretrained model
        
        last_hidden_states = predictionBert.last_hidden_state
        last_hidden_states = self.dropout(last_hidden_states)
        
        results_slotFilling = self.slotFillingLayer(last_hidden_states[:, 1:]) #The prediction for the slot filling task
        results_intent = self.intentLayer(last_hidden_states[:, 0]) #The prediction for the intent classification task
        
        return results_slotFilling, results_intent

