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
        
        
        
        #We need to pad the sequences to have the same length
        shapeDim_1 = results_slotFilling[0].shape[1]
        
        for i in range(results_slotFilling.shape[0]):
            frase = []
            contatore = 0
            for j in range(results_slotFilling[i].shape[0]):
                
                token = tokenizedUtterance[i][j]
                if "##" not in token and "'" not in token and "." not in token: #We need to check if the token is a special token
                    frase.append(results_slotFilling[i,j]) 
                else:
                    contatore += 1 #We need to count how many tokens we need to add at the end of the sequence
            
            for l in range(contatore):
                frase.append(torch.zeros(shapeDim_1).to(self.__device))
                
            newEntry = torch.stack(frase, dim=0)
            results_slotFilling[i] = newEntry
        
        
        results_slotFilling = results_slotFilling.permute(0,2,1)
        
        return results_slotFilling, results_intent

