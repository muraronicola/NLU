import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ModelIAS(nn.Module):

    def __init__(self, model_bert, hiddenSize, out_slot, device="cpu"):
        super(ModelIAS, self).__init__()
        self.bert = model_bert
        
        self.dropout = nn.Dropout(0.1)
        
        self.slotFillingLayer = nn.Linear(hiddenSize, out_slot)
        
        self.__device = device
    
    
    def forward(self, utterance, frase_testo, length_token_bert, text_suddiviso):
        
        predictionBert = self.bert(**utterance)
        
        last_hidden_states = predictionBert.last_hidden_state
        last_hidden_states = self.dropout(last_hidden_states)
        
        results_slotFilling = self.slotFillingLayer(last_hidden_states[:, 1:])
        
        
        shapeDim_1 = results_slotFilling[0].shape[1]
        
        for i in range(len(frase_testo)): #Prima crashava
            frase = []
            daFondere = []
            contatore = 0
            
            parole = text_suddiviso[i]
            
            indice_slot = 0
            for j in range(0, len(parole)): #L'uno la skippo
                parola = parole[j]
                length_token = length_token_bert[0][parola] #Come fa a non crashare??

                if(length_token > 1):
                    media = torch.mean(torch.stack([results_slotFilling[i][indice_slot+k] for k in range(length_token) if indice_slot + k < len(results_slotFilling[i])]), dim=0)
                    frase.append(media)
                    contatore += length_token - 1 
                    indice_slot += length_token
                else:
                    frase.append(results_slotFilling[i, indice_slot])
                    indice_slot += 1
            
            num_padding = results_slotFilling.shape[1] - len(frase)
            for l in range(num_padding):
                frase.append(torch.zeros(shapeDim_1).to(self.__device))
                
            newEntry = torch.stack(frase, dim=0)
            results_slotFilling[i] = newEntry
        
        
        results_slotFilling = results_slotFilling.permute(0,2,1)
        
        return results_slotFilling

