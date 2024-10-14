# Global variables
import os
import json
from pprint import pprint
import random
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from conll import evaluate
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import pandas as pd
from transformers import BertTokenizer, BertModel
from pprint import pprint
from torcheval.metrics.functional import multiclass_f1_score #L'ho aggiunta io
import time
from evals import evaluate_ote, evaluate_ts, ot2bieos_ote
import re
import nltk
from utils import label2tag

nltk.download('punkt')

device = 'mps' # cuda:0 means we are using the GPU with id 0, if you have multiple GPU
train_batch_size = 64 #Con 128 crashava...
test_batch_size = 32

os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # Used to report errors on CUDA side
PAD_TOKEN = -1

def current_milli_time():
    return round(time.time() * 1000)

def getPadToken(lang):
    return lang.slot2id["O"]

def getConvertedInput(input_line, train=False):
    data = []
    contatoreErrori = 0
    
    for line in input_line:
        data_split = line.split("####")
        x = data_split[0]
        
        """ x = x if x[0] != "," else x[1:]
        
        x = x.replace('(', '')
        x = x.replace(')', '')
        
        x = re.sub(r'(\d+)\.(\w+)', r'\1\2', x)
        x = re.sub(r'(\d+)\,(\w+)', r'\1\2', x)
        x = re.sub(r'\.{2,}', '.', x)

        x = x.replace("I've", "I£ve") #beforeTricl
        x = x.replace("I'v", "I#v") #beforeTricl
        x = x.replace(".com", "com")
        
        x = x.replace(".", " . ")
        x = x.replace(':', ', ')
        x = x.replace(",", " , ")
        x = x.replace("!", " ! ")
        x = x.replace("?", " ? ")
        x = x.replace("' ", " ")
        x = x.replace("'", " '")
        x = x.replace("n 't", " n't")
        x = x.replace(" -- ", " - ")
        x = x.replace("-- ", " ")
        x = x.replace("--", " -- ")
        x = x.replace("$", " $ ")
        x = x.replace("%", " % ")
        x = x.replace(";", " ; ")
        x = x.replace('"', "")
        
        x = x.replace("I#v", "I'v") #afterTrick
        x = x.replace("I£ve", "I' ve") #beforeTricl
        x = x.replace("dont", "do n't")
        x = x.replace("cant", "ca n't")
        x = x.replace("cannot", "can not") """
        
        x = nltk.word_tokenize(x)
        plain_text = " ".join(x)
        
        y_one_line = []
        y_label = data_split[1].split()

        for i in range(len(y_label)):
            if "=O" in y_label[i]:
                y_one_line.append("O")
            else:
                index_equal = y_label[i].index("=")
                y_one_line.append("T")
                
        slots = " ".join(y_one_line)
        
        data.append({'plain_text': plain_text, 'text_suddiviso': x, 'slots': slots})
        
        """if (len(x.split()) != len(y_one_line)):
            print("\nErrore Splitting")
            print(data_split[0])
            print(x.split())
            print(y_label
            contatoreErrori += 1
        
        if (len(x.split()) == len(slots.split()) or not train):
            data.append({'plain_text': x, 'text_suddiviso': x.split(), 'slots': slots}) """
    
    if train:
        print("\n\nDati ignorati nel test set: ", contatoreErrori)
    """ print("\n\nContatore errori: ", contatoreErrori)
    print("\nData: ", data[0])
    print("\nData: ", data[1])
    print("\nData: ", data[2])
    print("\nData: ", data[3])
    exit(0) """
    return data

class Lang():
    def __init__(self, tokens, tokenizer, slots, cutoff=0):
        self.word2id = self.w2id(tokens["input_ids"], tokenizer, cutoff=cutoff)
        self.slot2id = self.lab2id(slots, pad=False) #Casomai provo a mettere False / True
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2slot = {v:k for k, v in self.slot2id.items()}
    
    def w2id(self, index, tokenizer, cutoff=None, unk=True):
        vocab = {'pad': PAD_TOKEN}
        if unk:
            vocab['unk'] = len(vocab)
        count = Counter(index)
        for k, v in count.items():
            for i in k:
                """ print(i.item())
                print(tokenizer.convert_ids_to_tokens(i.item())) """
                if v > cutoff:
                    vocab[str(tokenizer.convert_ids_to_tokens(i.item()))] = i.item()
        return vocab
    
    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab
    
class IntentsAndSlots (data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, tokenizer, dataset, lang, unk='unk'):
        self.utterances = []
        self.slots = []
        self.unk = unk
        self.tokenizer = tokenizer
        self.text_suddiviso = []
        
        for x in dataset:
            self.utterances.append(x['plain_text'])
            self.slots.append(x['slots'])
            self.text_suddiviso.append(x['text_suddiviso'])
            
        self.utt_tokenized = tokenizer(self.utterances, padding=True)
        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        #I believe with Apple - you get what you pay for.####I=O believe=O with=O Apple=O ,=O you=O get=O what=O you=O pay=O for=O .=O
        self.O_slot = lang.slot2id["O"]
        self.length_token_bert = self.getLengthBert(self.text_suddiviso, self.tokenizer)
        #self.utt_tokenized = self.mapping_seq_bert(self.utterances, lang.word2id, tokenizer) #Quale devo fare?
        
    def getLengthBert(self, text_suddiviso, tokenizer):
        lengths = {}
        for txt_line in text_suddiviso:
            for parola in txt_line:
                tokenized = tokenizer(parola)
                lengths[parola] = len(tokenized['input_ids']) - 2  #C'è il cls e il sep
        return lengths
        
    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        attention_mask = self.utt_tokenized['attention_mask'][idx]
        input_ids = self.utt_tokenized['input_ids'][idx]
        token_type_ids = self.utt_tokenized['token_type_ids'][idx]
        
        frase_testo = self.utterances[idx]

        """ for i in range(len(attention_mask)):            #Rimozione ## da input_ids
            token = self.tokenizer.convert_ids_to_tokens(input_ids[i])
            if "##" in token:
                #indexsRemove.append(i)
                attention_mask.pop(i)
                input_ids.pop(i)
                token_type_ids.pop(i)
                
                attention_mask.append(0)
                input_ids.append(0)
                token_type_ids.append(0)
        """
        
        """ attention_mask = [self.utt_tokenized['attention_mask'][idx][i] for i in range(len(self.utt_tokenized['attention_mask'][idx])) if i not in indexsRemove]
        input_ids = [self.utt_tokenized['input_ids'][idx][i] for i in range(len(self.utt_tokenized['input_ids'][idx]))  if i not in indexsRemove  ]
        token_type_ids = [self.utt_tokenized['token_type_ids'][idx][i] for i in range(len(self.utt_tokenized['token_type_ids'][idx]))  if i not in indexsRemove  ]
        """
        
        """ print(self.utt_tokenized['attention_mask'][idx])
        print(attention_mask)
        
        print(self.utt_tokenized['input_ids'][idx])
        print(input_ids)
        
        print(self.utt_tokenized['token_type_ids'][idx])
        print(token_type_ids)
        exit(0) """


        """ attention_mask = self.utt_tokenized['attention_mask'][idx]
        input_ids = self.utt_tokenized['input_ids'][idx]
        token_type_ids = self.utt_tokenized['token_type_ids'][idx] """
            
        slots = torch.Tensor(self.slot_ids[idx])
        for i in range(len(slots), len(attention_mask) - 1):
            slots = torch.cat((slots, torch.tensor([PAD_TOKEN])))
            #print("padded")
        
        """ print(slots)
        print(attention_mask)
        exit(0) """
        
        #intent = self.intent_ids[idx]
        sample = {'text_suddiviso': self.text_suddiviso[idx],  'length_token_bert': self.length_token_bert, 'frase_testo':frase_testo, 'plain_text': utt, 'slots': slots, 'attention_mask': attention_mask, 'input_ids': input_ids, 'token_type_ids': token_type_ids, 'tokenizedUtterance': self.tokenizer.convert_ids_to_tokens(input_ids)}
        
        """ print("input_ids_ text:", [self.tokenizer.convert_ids_to_tokens(x) for x in input_ids])
        print("input_ids:", input_ids)
        print("Slots:", slots)
        print("attention_mask:", attention_mask)
        exit(0) """
        return sample
    
    # Auxiliary methods
    
    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]
    
    def mapping_seq(self, data, mapper): # Map sequences to number
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            res.append(tmp_seq)
        return res

class ModelIAS(nn.Module):

    def __init__(self, model_bert, seqLen, hiddenSize, out_slot, tokenizer, lang, n_layer=1, pad_index=0):
        super(ModelIAS, self).__init__()
        self.bert = model_bert
        self.dropout = nn.Dropout(0.1)
        self.slotFillingLayer = nn.Linear(hiddenSize, out_slot)
        self.tokenizer = tokenizer
        self.seq_length = seqLen
        
    def forward(self, utterance, tokenizedUtterance, frase_testo, length_token_bert, text_suddiviso):
        
        #with torch.no_grad():
        predictionBert = self.bert(**utterance)
        last_hidden_states = predictionBert.last_hidden_state
        
        last_hidden_states = self.dropout(last_hidden_states)
        
        results_slotFilling = self.slotFillingLayer(last_hidden_states[:, 1:])
        
        
        shapeDim_0 = results_slotFilling[0].shape[0] #51
        shapeDim_1 = results_slotFilling[0].shape[1] #130
        
        prima = current_milli_time()
        #print(shapeDim_0)
        
        for i in range(len(frase_testo)): #Prima crashava
            frase = []
            daFondere = []
            contatore = 0
            
            #parole_old = frase_testo[i].split()
            parole = text_suddiviso[i]
            
            """ print("\n\n")
            print(parole, parole_new) """
            
            """ print("\n\n")
            print(results_slotFilling[i], "\n")
            print(parole, "\n") """
            indice_slot = 0
            for j in range(0, len(parole)): #L'uno la skippo
                #print(length_token_bert)
                parola = parole[j]
                length_token = length_token_bert[0][parola] #Come fa a non crashare??
                #print(parola, length_token)
                if(length_token > 1):
                    media = torch.mean(torch.stack([results_slotFilling[i][indice_slot+k] for k in range(length_token) if indice_slot + k < len(results_slotFilling[i])]), dim=0)
                    frase.append(media)
                    
                    """ primo_valore = results_slotFilling[i][indice_slot] 
                    frase.append(primo_valore) """
                    #print("media", media)
                    #j += length_token - 1 #Il meno 1 deriva dal fatto che c'è già un j++
                    contatore += length_token - 1 
                    indice_slot += length_token
                else:
                    frase.append(results_slotFilling[i, indice_slot])
                    """ if "##" not in parola and "'" not in parola and "." not in parola:
                        frase.append(results_slotFilling[i, indice_slot]) """
                        #print("aggiunta")
                    indice_slot += 1
                
                """ if "'" in parole[j]:
                    contatore += 1
                    if j >= (2+contatore) and  j-(2+ contatore) < len(parole) and parole[j-(2+ contatore)].find("'") not in [-1, 0] and j+1 < results_slotFilling[i].shape[0]:
                        media = torch.mean(torch.stack([results_slotFilling[i][j-1], results_slotFilling[i][j], results_slotFilling[i][j+1]]), dim=0)
                        frase.append(media)
                    else:
                        contatore += 1
                else:
                    frase.append(results_slotFilling[i,j]) """

            
            #print("Frase prima del padding", len(frase))
            #print("Contatore", contatore)
            num_padding = results_slotFilling.shape[1] - len(frase)
            for l in range(num_padding):
                frase.append(torch.zeros(shapeDim_1).to(device))
            #print("Frase dopo il padding", len(frase))
                
            newEntry = torch.stack(frase, dim=0)
            results_slotFilling[i] = newEntry
            
            """ print("\n\nresults_slotFilling")
            print(results_slotFilling[i][0:10]) """
            """ print("frase", newEntry)
            exit(0) """
        #exit(0)
        dopo = current_milli_time()
        #print("Tempo impiegato", dopo - prima)
        
        """ for i in range(results_slotFilling.shape[0]):
            frase = []
            contatore = 0
            for j in range(results_slotFilling[i].shape[0]):
                
                token = tokenizedUtterance[i][j] #Come fa a non crashare??
                if "##" not in token and "'" not in token and "." not in token:
                    frase.append(results_slotFilling[i,j])
                else:
                    contatore += 1
            
            for l in range(contatore):
                frase.append(torch.zeros(shapeDim_1).to(device))
                
            newEntry = torch.stack(frase, dim=0)
            
            results_slotFilling[i] = newEntry
        """
        
        results_slotFilling = results_slotFilling.permute(0,2,1)
        
        return results_slotFilling


def load_data(path):
    '''
        input: path/to/data
        output: json 
    '''
    """  dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset """
    
    dataset = []
    file = open(path, 'r')
    lines = file.readlines()        
    
    for line in lines:
        dataset.append(line)
        
    file.close()
    return dataset


def collate_fn(data): #Why utterance??
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape 
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        # print(padded_seqs)
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths
    # Sort data by seq lengths
    data.sort(key=lambda x: len(x['plain_text']), reverse=True) 
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
        
    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, _ = merge(new_item['plain_text'])
    y_slots, y_lengths = merge(new_item["slots"])
    
    src_utt = src_utt.to(device) # We load the Tensor on our selected device
    y_slots = y_slots.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)
    
    new_item["utterance"] = src_utt
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    return new_item


def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)
                    
def train_loop(data, optimizer, criterion_slots, criterion_intents, model, lang, clip=5 ):
    model.train()
    loss_array = []
    #f = open("./debug.txt", "w")
    for indice, sample in enumerate(data):
        
        optimizer.zero_grad() # Zeroing the gradient
        
        #print("indice loop", indice)
        franco = torch.Tensor(sample['input_ids']).to(device).to(torch.int64)
        alberto = torch.Tensor(sample['attention_mask']).to(device).to(torch.int64)
        giovanni = torch.Tensor(sample['token_type_ids']).to(device).to(torch.int64)
        
        tokenizedUtterance = sample['tokenizedUtterance']
        length_token_bert = sample['length_token_bert']
        
        
        inputs_bert = {'input_ids': franco, 'attention_mask': alberto, 'token_type_ids': giovanni}
        
        slots = model(inputs_bert, tokenizedUtterance, sample['frase_testo'], length_token_bert, sample['text_suddiviso'])
        
        #loss_intent = criterion_intents(intent, sample['intents'])
        loss_slot = criterion_slots(slots, sample['y_slots'])
        
        
        #f1score = multiclass_f1_score(torch.argmax(slots[0], dim=0), sample['y_slots'][0])
        
        vec_pred = []
        vec_value = []
        argmaxAMano =  torch.argmax(slots[0], dim=0)
        for i in range(len(argmaxAMano)):
            vec_value.append(argmaxAMano[i].detach().cpu().item())
            vec_pred.append(lang.id2slot[argmaxAMano[i].detach().cpu().item()])
        
        """ print("\n\nPred, true")
        print(vec_value)
        print(sample['y_slots'][0]) """
        #Print tokenization
        """ print("\n\nINDICE")
        for frase in tokenizedUtterance:
            for w in frase:
                f.write("'" + w + "' ")
            f.write("\n") """
        
        """ print("\n\n\nINDICE:", indice)
        print("\nPrimaFrase", sample['frase_testo'][0])
        print("\ntokenizedUtterance", tokenizedUtterance[0])
        print("\ninputs_bert", inputs_bert['input_ids'][0])
        print("\nslotsPrimaFrase", slots[0])
        print("\nTrue slots", sample['y_slots'][0])
        print("\nPred value: ", vec_value)
        print("\nPred slots: ", vec_pred)
        print("\nF1 score: ", f1score, "loss_slot: ", loss_slot, "loss_intent: ", loss_intent) """
        
        loss = loss_slot # In joint training we sum the losses. 
        
        #loss = loss_intent*0.1 + loss_slot*0.9 # In joint training we sum the losses. 
                                       # Is there another way to do that?
                                       # (alpha) * loss_intent + (1-alpha) * loss_slot
        
        loss_array.append(loss.item())
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() # Update the weight
    """ f.close()
    exit(0) """
    return loss_array


def eval_loop(data, criterion_slots, criterion_intents, model, lang, test=False):
    model.eval()
    loss_array = []
    
    ref_intents = []
    hyp_intents = []
    
    ref_slots = []
    hyp_slots = []
    
    f1_score_arr = []
    #softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            
            franco = torch.Tensor(sample['input_ids']).to(device).to(torch.int64)
            alberto = torch.Tensor(sample['attention_mask']).to(device).to(torch.int64)
            giovanni = torch.Tensor(sample['token_type_ids']).to(device).to(torch.int64)
            
            inputs_bert = {'input_ids': franco, 'attention_mask': alberto, 'token_type_ids': giovanni}
            
            tokenizedUtterance = sample['tokenizedUtterance']
            
            #print("\n\n\n\n---------------- EVAL ---------------- ")
            slots = model(inputs_bert, tokenizedUtterance, sample['frase_testo'], sample['length_token_bert'], sample['text_suddiviso'])
            loss_slot = criterion_slots(slots, sample['y_slots'])
            loss = loss_slot 
            loss_array.append(loss.item())
            
            # Intent inference
            # Get the highest probable class
            
            # Slot inference 
            output_slots_hyp = torch.argmax(slots, dim=1).detach().cpu().numpy()
            true_y = sample['y_slots'].detach().cpu().numpy()
            """ print("HYP SLOTS", output_slots_hyp[0])
            print("REF SLOTS", true_y[0])
            print(lang.id2slot[0])
            print(lang.id2slot[1]) """
            
            
            """ output_slots_hyp = np.argmax(slots.npvalue())
            true_y = np.argmax(sample['y_slots'].npvalue()) """
            #len_frase = len(sample['frase_testo'][0].split())
            #print(sample['frase_testo'][0], output_slots_hyp[0][:len_frase], true_y[0][:len_frase])
            
            for sequenza in range(len(output_slots_hyp)):
                #len_frase = len(sample['text_suddiviso'][sequenza])
                try:
                    first_index_of_pad = true_y[sequenza].tolist().index(PAD_TOKEN)#Tricky, devo salvarmi quanta lunga è la predizione
                    
                    val_true_y = [lang.id2slot[element] for element in true_y[sequenza][:first_index_of_pad]]
                    val_hat_y = [lang.id2slot[element] for element in output_slots_hyp[sequenza][:first_index_of_pad]]
                    
                    converted_true_y = ot2bieos_ote(val_true_y)  #ot2bio_ote does not work
                    converted_hat_y = ot2bieos_ote(val_hat_y)   #t2 era: ot2bieos_ote
                    
                    """ print("og true_y:", true_y[sequenza][:first_index_of_pad])
                    print("og hat_y:", output_slots_hyp[sequenza][:first_index_of_pad])
                    
                    print("val_hat_y: ", val_true_y)
                    print("val_hat_y: ", val_hat_y)
                    
                    print("converted_true_y: ", converted_true_y)
                    print("converted_hat_y: ", converted_hat_y) """
                    
                    ref_slots.append(converted_true_y)
                    hyp_slots.append(converted_hat_y)
                    
                except:
                    val_true_y = [lang.id2slot[element] for element in true_y[sequenza]]
                    val_hat_y = [lang.id2slot[element] for element in output_slots_hyp[sequenza]]
                    
                    converted_true_y = ot2bieos_ote(val_true_y)
                    converted_hat_y = ot2bieos_ote(val_hat_y)
                    
                    
                    """ print("og true_y:", true_y[sequenza][:first_index_of_pad])
                    print("og hat_y:", output_slots_hyp[sequenza][:first_index_of_pad])
                    
                    print("val_hat_y: ", val_true_y)
                    print("val_hat_y: ", val_hat_y)
                    
                    print("converted_true_y: ", converted_true_y)
                    print("converted_hat_y: ", converted_hat_y) """
                    
                    ref_slots.append(converted_true_y)
                    hyp_slots.append(converted_hat_y)
                        


    
    
    scores = evaluate_ote(ref_slots, hyp_slots)
    #scores = evaluate_ote([[0,1,0,1,1]], [[0,1,0,1,1]])
    #scores = evaluate_ote([["O", "S", "O", "E", "E"]], [["O", "S", "O", "E", "E"]])
    precision = scores[0]
    recall = scores[1]
    f1_score = scores[2]

    return f1_score, loss_array


def main():
    
    filename = "ponte_new_v1"
    
    tokenizer_bert_base = BertTokenizer.from_pretrained("bert-base-uncased") # Download the tokenizer
    model_bert_base = BertModel.from_pretrained("bert-base-uncased") # Download the model

    tmp_train_raw = load_data(os.path.join('./data','laptop14_train.txt'))
    test_raw = load_data(os.path.join('./data','laptop14_test.txt'))

    tmp_train_raw = getConvertedInput(tmp_train_raw, train=True) #x is the sentence (plain text), y is the slot (vector)
    test_raw = getConvertedInput(test_raw) #x is the sentence (plain text), y is the slot (vector)
    
    """  print(train_x[0])
    print(train_y[0])
    
    print("\n\n")
    print(test_x[0])
    print(test_y[0])
    
    exit(0) """
    #test_raw = getConvertedInput(test_raw)

    """intents = [x['intent'] for x in tmp_train_raw] # We stratify on intents
    count_y = Counter(intents)
    
    labels = []
    inputs = []
    mini_train = []

    for id_y, y in enumerate(intents):
        if count_y[y] > 1: # If some intents occurs only once, we put them in training
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y]) """
            
    inputs = [x for x in tmp_train_raw]
    labels = [0 for x in tmp_train_raw] #fake labels
    
    portion = 0.10
    
    X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=portion, random_state=42, shuffle=True, stratify=labels) #Metto shuffle a true
    train_raw = X_train
    dev_raw = X_dev

    #words = sum([x['utterance'].split() for x in train_raw], []) # No set() since we want to compute # the cutoff
    
    phrase_train = [x['plain_text'] for x in train_raw]
    """ phrase_dev = [x['utterance'] for x in dev_raw]
    phrase_test = [x['utterance'] for x in test_raw] """
    
    tokens = tokenizer_bert_base(phrase_train, return_tensors='pt', padding=True)

    corpus = train_raw + dev_raw + test_raw # We do not wat unk labels, 
                                            # however this depends on the research purpose
    
    slots = set(sum([line['slots'].split() for line in corpus],[]))
    lang = Lang(tokens, tokenizer_bert_base, slots, cutoff=0)
    
    
    
    """ print('Vocabulary size:', len(lang.word2id))
    print('Slot size:', len(lang.slot2id))
    print('Intent size:', len(lang.intent2id)) """
    
    
    train_dataset = IntentsAndSlots(tokenizer_bert_base, train_raw, lang)
    dev_dataset = IntentsAndSlots(tokenizer_bert_base, dev_raw, lang)
    test_dataset = IntentsAndSlots(tokenizer_bert_base, test_raw, lang)
    
    
    """ train_dataset = tokenizer_bert_base(phrase_train, return_tensors='pt', padding=True)
    dev_dataset = tokenizer_bert_base(phrase_dev, return_tensors='pt', padding=True)
    test_dataset = tokenizer_bert_base(phrase_test, return_tensors='pt', padding=True) """
    
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, collate_fn=collate_fn, shuffle=True) #Metto shuffle a True
    dev_loader = DataLoader(dev_dataset, batch_size=test_batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, collate_fn=collate_fn)

    seq_length = 91
    hiddenSize_new = 768

    hid_size = 200
    emb_size = 300

    lr = 0.00005  #For sgd # learning rate
    clip = 5 # Clip the gradientz
    
    out_slot = len(lang.slot2id)
    vocab_len = len(lang.word2id)
    
    n_epochs = 200
    runs = 1 #Era 5... ma ci mette troppo tempo
    
    best_model = None
    best_f1_ever = 0

    results = pd.DataFrame(columns=['run', 'epoch', 'train_loss', 'dev_loss', 'test_loss', 'f1_train', 'f1_dev', 'f1_test'])

    slot_f1s, intent_acc = [], []
    
    contatore_0 = 0
    contatore_T = 0
    for riga in train_raw:
        for token in riga['slots']:
            if "T" == token:
                contatore_T += 1
            else:
                contatore_0 += 1
                
    if contatore_0 > contatore_T:
        coefficente_cross_entropy = contatore_0 / contatore_T
        pesi_cross_entropy = torch.tensor([1.0, coefficente_cross_entropy]).to(device)
    else:
        coefficente_cross_entropy = contatore_T / contatore_0
        pesi_cross_entropy = torch.tensor([coefficente_cross_entropy, 1.0]).to(device)
        
    #Override pesi
    pesi_cross_entropy = torch.tensor([1.0, 1.0]).to(device)
    
    
    if out_slot > pesi_cross_entropy.shape[0]: #Se ho 3 calssi, necessito di 3 pesi
        pesi_cross_entropy = torch.cat((torch.tensor([1.0]).to(device), pesi_cross_entropy))
    
    
    print("Pesi cross entropy: ", pesi_cross_entropy)
    #t = tqdm(range(1, n_epochs), leave=True)
    for i in range(0, runs):
        model = ModelIAS(model_bert_base, seq_length, hiddenSize_new, out_slot, tokenizer_bert_base, lang, pad_index=PAD_TOKEN).to(device)
        model.apply(init_weights)

        optimizer = optim.Adam(model.parameters(), lr=lr) #Era Adam
        criterion_slots = nn.CrossEntropyLoss(weight=pesi_cross_entropy, ignore_index=PAD_TOKEN) #devo mettere ignore_index = [pad]  
        
        #Devo mettere la ignore_index al pad di bert BERT
        
        #Devo togliere il pad dalla evaluation ( amano , (dopo il sep ))
        #Non devo torgliere il #dall'input. ma devo rimodulare l'output ottenuto
        criterion_intents = nn.CrossEntropyLoss()
        
        patience = 3
        losses_train = []
        losses_dev = []
        sampled_epochs = []
        best_f1 = 0
        for x in tqdm(range(1, n_epochs)):
            #loss = 1
            loss = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model, lang, clip=clip)
            if x % 1 == 0:
                sampled_epochs.append(x)
                losses_train.append(np.asarray(loss).mean())
                
                results_dev, loss_dev = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang)
                results_train, loss_train = eval_loop(train_loader, criterion_slots, criterion_intents, model, lang)
                results_test, loss_test = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang, test=False)
                
                losses_dev.append(np.asarray(loss_dev).mean())
                
                f1_dev = results_dev
                f1_train = results_train
                f1_test = results_test
                
                """ f1_dev = results_dev['total']['f']
                f1_train = results_train['total']['f']
                f1_test = results_test['total']['f'] """
                
                ls_train = np.asarray(loss).mean()
                ls_dev = np.asarray(loss_dev).mean()
                ls_test = np.asarray(loss_test).mean()
                
                #t.set_description("Epoch: %i, Train Loss: %.4f, Dev Loss: %.4f, Test Loss: %.4f, Train f1: %.4f, Dev F1: %.4f, Test F1: %.4f" % (x, ls_train, ls_dev, ls_test, f1_train, f1_dev, f1_test))
                print("Epoch: %i, Train Loss: %.4f, Dev Loss: %.4f, Test Loss: %.4f, Train f1: %.4f, Dev F1: %.4f, Test F1: %.4f" % (x, ls_train, ls_dev, ls_test, f1_train, f1_dev, f1_test))
                new_row = pd.DataFrame({'run': [i], 'epoch': [x], 'train_loss': [round(ls_train, 4)], 'dev_loss': [round(ls_dev,4)], 'test_loss': [round(ls_test,4)], 'f1_train': [round(f1_train,4)], 'f1_dev': [round(f1_dev,4)], 'f1_test': [round(f1_test,4)]})
                results = pd.concat([results, new_row])
                results.to_csv("./results/" + filename + ".csv")
                
                if f1_dev >= best_f1:  #>= altrimente crascha... visto che è 0
                    patience = 10
                    best_f1 = f1_dev
                    #The patient is reset every time we find a new best f1
                    if f1_dev >= best_f1_ever:
                        best_f1_ever = f1_dev
                        best_model = copy.deepcopy(model).to('cpu')
                else:
                    patience -= 1
                if patience <= 0: # Early stopping with patient
                    break # Not nice but it keeps the code clean

        results_test, _ = eval_loop(test_loader, criterion_slots, 
                                                criterion_intents, model, lang)
        #slot_f1s.append(results_test['total']['f'])
        slot_f1s.append(results_test)
    slot_f1s = np.asarray(slot_f1s)
    
    
    slot_f1s_mean = round(slot_f1s.mean(),3)
    slot_f1s_std = round(slot_f1s.std(),3)
    print('Slot F1', slot_f1s_mean, '+-', slot_f1s_std)
    
    saving_object = {"epoch": x, 
                      "model": best_model, 
                      "model_state_dict": best_model.state_dict(), 
                      "optimizer_state_dict": optimizer.state_dict(), 
                      "lang": lang}
    torch.save(saving_object, "./models/{}.pt".format(filename))
    best_model.to(device)
    
    
    print("\n\nBEST RESULTS (on dev set)\n")
    results_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, best_model, lang)    
    print('Test f1: ', results_test)
    
    results_eval, _ = eval_loop(dev_loader, criterion_slots, criterion_intents, best_model, lang)      
    print('Eval f1: ', results_eval)
    
    results_train, _ = eval_loop(train_loader, criterion_slots, criterion_intents, best_model, lang)    
    print('Train f1: ', results_train)
    
    
    
    results.to_csv("./results/" + filename + ".csv")
    file = open("./results/" + filename + ".txt", "w") 
    
    file.write("TEST MEAN VALUES:\n")
    
    file.write("Slot F1: " + str(slot_f1s_mean) + " +- " + str(slot_f1s_std) + "\n")
    
    file.write("\n\nBEST RESULTS (on dev set)\n")
    file.write("Test f1: " + str(results_test) + "\n\n")
    
    file.write("Eval f1: " + str(results_eval) + "\n\n")
    
    file.write("Train f1: " + str(results_train) + "\n\n")
    
    file.close()


if __name__ == '__main__':
    main()