# Add functions or classes used for data loading and preprocessing
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
from functools import partial
import torch
import json
from sklearn.model_selection import train_test_split
from collections import Counter
import nltk

class LoadData():
    
    def __init__(self, train_path, test_path, tokenizer, device="cpu", pad_token=0):
        self.__device = device
        self.__pad_token = pad_token
        
        __original_train_raw = self.__read_file(train_path)
        __test_raw = self.__read_file(test_path)
        
        __tmp_tokenized_raw_train = self.getTokenizedInput(__original_train_raw)
        __tokenized_test = self.getTokenizedInput(__test_raw)
        
        __tokenized_train, __tokenized_dev = self.__createDevSet(__tmp_tokenized_raw_train)
        
        __tokens, __corpus, __slots = self.__createDatasetInformation(__tokenized_train, __tokenized_dev, __tokenized_test, tokenizer)
        self.__lang = Lang(__tokens, __slots, tokenizer, pad_token, cutoff=0)
        
        self.__train_dataset = Slots(__tokenized_train, self.__lang, tokenizer, self.__pad_token)
        self.__dev_dataset = Slots(__tokenized_dev, self.__lang, tokenizer, self.__pad_token)
        self.__test_dataset = Slots(__tokenized_test, self.__lang, tokenizer, self.__pad_token)


    def __read_file(self, path):
        dataset = []
        file = open(path, 'r')
        lines = file.readlines()        
        
        for line in lines:
            dataset.append(line)
            
        file.close()
        return dataset
    
    def __createDatasetInformation(self, train_raw, dev_raw, dev_test, tokenizer):
        corpus = train_raw + dev_raw + dev_test 
        
        slots = set(sum([line['slots'].split() for line in corpus],[]))
        
        phrases_train = [x['plain_text'] for x in train_raw]
        tokens = tokenizer(phrases_train, return_tensors='pt', padding=True)
        
        return tokens, corpus, slots
    
    def __createDevSet(self, original_train_raw, portion_dev=0.10):
        inputs = [x for x in original_train_raw]
        labels = [0 for x in original_train_raw] #fake labels, used only for the split
        
        X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=portion_dev, random_state=42, shuffle=True, stratify=labels) #Metto shuffle a true
        return X_train, X_dev
    
    def get_raw_train_data(self): #Boh vediamo. Credo che la toglierò
        return self.__train_raw
    
    def get_lang(self):
        return self.__lang
    
    def get_dataset_loaders(self, batch_size_train=64, batch_size_val=128, batch_size_test=128):
        train_loader = DataLoader(self.__train_dataset, batch_size=batch_size_train, collate_fn=partial(collate_fn, pad_token=self.__pad_token, device=self.__device), shuffle=True)
        dev_loader = DataLoader(self.__dev_dataset, batch_size=batch_size_val, collate_fn=partial(collate_fn, pad_token=self.__pad_token, device=self.__device))
        test_loader = DataLoader(self.__test_dataset, batch_size=batch_size_test, collate_fn=partial(collate_fn, pad_token=self.__pad_token, device=self.__device))
    
        return train_loader, dev_loader, test_loader
    
    def getTokenizedInput(self, input_line):
        data = []
        
        for line in input_line:
            data_split = line.split("####")
            x = data_split[0]
            x = nltk.word_tokenize(x)
            plain_text = " ".join(x)
            
            y_one_line = []
            y_label = data_split[1].split()

            for i in range(len(y_label)):
                if "=O" in y_label[i]:
                    y_one_line.append("O")
                else:
                    y_one_line.append("T")
                    
            slots = " ".join(y_one_line)
            data.append({'plain_text': plain_text, 'text_suddiviso': x, 'slots': slots})
        
        return data


class Lang():
    def __init__(self, tokens, slots, tokenizer, pad_token, cutoff=0):
        self.pad_token = pad_token
        
        self.word2id = self.w2id(tokens["input_ids"], tokenizer, cutoff=cutoff)
        self.slot2id = self.lab2id(slots, pad=False) #Casomai provo a mettere False / True
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2slot = {v:k for k, v in self.slot2id.items()}
    
    def w2id(self, index, tokenizer, cutoff=None, unk=True):
        vocab = {'pad': self.pad_token}
        if unk:
            vocab['unk'] = len(vocab)
        count = Counter(index)
        for k, v in count.items():
            for i in k:
                if v > cutoff:
                    vocab[str(tokenizer.convert_ids_to_tokens(i.item()))] = i.item()
        return vocab
    
    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] =  self.pad_token
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab



class Slots (data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, lang, tokenizer, pad_token=0, unk='unk'):
        self.utterances = []
        self.slots = []
        self.unk = unk
        self.tokenizer = tokenizer
        self.text_suddiviso = []
        
        self.pad_token = pad_token
        
        for x in dataset:
            self.utterances.append(x['plain_text'])
            self.slots.append(x['slots'])
            self.text_suddiviso.append(x['text_suddiviso'])

        self.utt_tokenized = tokenizer(self.utterances, padding=True)
        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        
        self.O_slot = lang.slot2id["O"]
        self.length_token_bert = self.getLengthBert(self.text_suddiviso, self.tokenizer)
        
    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        attention_mask = self.utt_tokenized['attention_mask'][idx]
        input_ids = self.utt_tokenized['input_ids'][idx]
        token_type_ids = self.utt_tokenized['token_type_ids'][idx]
        
        frase_testo = self.utterances[idx]
        
        slots = torch.Tensor(self.slot_ids[idx])
        for i in range(len(slots), len(attention_mask) - 1):
            slots = torch.cat((slots, torch.tensor([self.pad_token])))
        
        sample = {'text_suddiviso': self.text_suddiviso[idx],  'length_token_bert': self.length_token_bert, 'frase_testo':frase_testo, 'plain_text': utt, 'slots': slots, 'attention_mask': attention_mask, 'input_ids': input_ids, 'token_type_ids': token_type_ids, 'tokenizedUtterance': self.tokenizer.convert_ids_to_tokens(input_ids)}
        
        return sample
    
    def getLengthBert(self, text_suddiviso, tokenizer):
        lengths = {}
        for txt_line in text_suddiviso:
            for parola in txt_line:
                tokenized = tokenizer(parola)
                lengths[parola] = len(tokenized['input_ids']) - 2  #C'è il cls e il sep
        return lengths
        
    def mapping_seq(self, data, mapper):
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


def collate_fn(data, pad_token=0, device="cpu"):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape 
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(pad_token)
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
    
    new_item["utterances"] = src_utt
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    return new_item