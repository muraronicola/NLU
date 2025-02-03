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
        
        #Initializing only the weights of the projection heads
        torch.nn.init.uniform_(self.slotFillingLayer.weight, -0.01, 0.01)

        
    def align_hidden_representation(self, hidden_representation, text_original, text_suddivided, length_token_bert): #Align the slots with the original text
        #We need to pad the sequences to have the same length
        shapeDim_1 = hidden_representation[0].shape[1]
        
        for i in range(len(text_original)):
            phrase = []
            counter = 0
            
            parole = text_suddivided[i]
            
            index_slot = 0
            for j in range(0, len(parole)):
                word = parole[j]
                length_token = length_token_bert[0][word]

                if(length_token > 1):
                    media = torch.mean(torch.stack([hidden_representation[i][index_slot+k] for k in range(length_token) if index_slot + k < len(hidden_representation[i])]), dim=0)
                    phrase.append(media)
                    counter += length_token - 1 
                    index_slot += length_token
                else:
                    phrase.append(hidden_representation[i, index_slot]) #We need to count how many tokens we need to add at the end of the sequence
                    index_slot += 1
            
            num_padding = hidden_representation.shape[1] - len(phrase)
            for l in range(num_padding):
                phrase.append(torch.zeros(shapeDim_1).to(self.__device))
                
            newEntry = torch.stack(phrase, dim=0)
            hidden_representation[i] = newEntry

        return hidden_representation
        
        
    def forward(self, inputs_model,  text_original, text_suddivided, length_token_bert):
        
        predictionBert = self.bert(**inputs_model)  #Get the output of the pretrained model
        last_hidden_states = predictionBert.last_hidden_state

        aligned_hidden_states = self.align_hidden_representation(last_hidden_states, text_original, text_suddivided, length_token_bert)#Align the slots with the original text
        aligned_hidden_states = self.dropout(aligned_hidden_states)
        
        results_slotFilling = self.slotFillingLayer(aligned_hidden_states[:, 1:]) #The prediction for the slot filling task
        
        return results_slotFilling.permute(0,2,1)
