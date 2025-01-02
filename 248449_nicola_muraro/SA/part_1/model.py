import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ModelIAS(nn.Module):

    def __init__(self, model_bert, hiddenSize, out_slot, drop_value=0.1, device="cpu"):
        super(ModelIAS, self).__init__()
        self.bert = model_bert #The petrained model
        
        self.dropout = nn.Dropout(drop_value)
        
        self.slotFillingLayer = nn.Linear(hiddenSize, out_slot)  #The projection head for the slot filling task
        
        self.__device = device
    
    
    def forward(self, inputs_model):
        
        predictionBert = self.bert(**inputs_model)  #Get the output of the pretrained model
        
        last_hidden_states = predictionBert.last_hidden_state
        last_hidden_states = self.dropout(last_hidden_states)
        
        results_slotFilling = self.slotFillingLayer(last_hidden_states[:, 1:]) #The prediction for the slot filling task
        
        return results_slotFilling

