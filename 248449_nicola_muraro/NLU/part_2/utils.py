# Add functions or classes used for data loading and preprocessing
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
from functools import partial
import torch
import json
from sklearn.model_selection import train_test_split
from collections import Counter

class LoadData():
    
    def __init__(self, train_path, test_path, tokenizer, device="cpu", pad_token=0, lang=None):
        self.__device = device
        self.__pad_token = pad_token
        
        self.__test_raw = self.__read_file(test_path)
        
        __original_train_raw_file = self.__read_file(train_path)
        self.__train_raw, self.__dev_raw = self.__createDevSet(__original_train_raw_file)
        

        if lang is not None: #If we are evaluating a saved model we need to pass the lang object (saved during training)
            self.__lang = lang
        else:
            __tokens, __corpus, __intents, __slots = self.__createDatasetInformation(self.__train_raw, self.__dev_raw, self.__test_raw, tokenizer)
            self.__lang = Lang(__tokens, __intents, __slots, tokenizer, pad_token, cutoff=0)
        
        
        self.__train_dataset = IntentsAndSlots(self.__train_raw, self.__lang, tokenizer, self.__pad_token)
        self.__dev_dataset = IntentsAndSlots(self.__dev_raw, self.__lang, tokenizer, self.__pad_token)
        self.__test_dataset = IntentsAndSlots(self.__test_raw, self.__lang, tokenizer, self.__pad_token)


    def __read_file(self, path):
        dataset = []
        with open(path) as f:
            dataset = json.loads(f.read())
        return dataset
    
    def __createDatasetInformation(self, train_raw, dev_raw, dev_test, tokenizer):
        corpus = train_raw + dev_raw + dev_test 
        
        slots = set(sum([line['slots'].split() for line in corpus],[]))
        intents = set([line['intent'] for line in corpus])
        
        phrases_train = [x['utterance'] for x in train_raw]
        tokens = tokenizer(phrases_train, return_tensors='pt', padding=True)
        
        return tokens, corpus, intents, slots 
    
    def __createDevSet(self, original_train_raw, portion_dev=0.10):
        intents = [x['intent'] for x in original_train_raw] # We stratify on intents
        count_y = Counter(intents)

        labels = []
        inputs = []
        mini_train = []

        for id_y, y in enumerate(intents):
            if count_y[y] > 1: # If some intents occurs only once, we put them in training
                inputs.append(original_train_raw[id_y])
                labels.append(y)
            else:
                mini_train.append(original_train_raw[id_y])
        
        X_train, X_dev, _, _ = train_test_split(inputs, labels, test_size=portion_dev, random_state=42, shuffle=True, stratify=labels)
        X_train.extend(mini_train)
        
        return X_train, X_dev
    
    def get_lang(self): #Return the lang object
        return self.__lang
    
    def get_dataset_loaders(self, batch_size_train=64, batch_size_val=128, batch_size_test=128):
        train_loader = DataLoader(self.__train_dataset, batch_size=batch_size_train, collate_fn=partial(collate_fn, pad_token=self.__pad_token, device=self.__device), shuffle=True)
        dev_loader = DataLoader(self.__dev_dataset, batch_size=batch_size_val, collate_fn=partial(collate_fn, pad_token=self.__pad_token, device=self.__device))
        test_loader = DataLoader(self.__test_dataset, batch_size=batch_size_test, collate_fn=partial(collate_fn, pad_token=self.__pad_token, device=self.__device))
    
        return train_loader, dev_loader, test_loader


class Lang(): #This class will be used to handle the vocabulary
    def __init__(self, tokens, intents, slots, tokenizer, pad_token, cutoff=0):
        self.pad_token = pad_token
        
        self.word2id = self.w2id(tokens["input_ids"], tokenizer, cutoff=cutoff)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
        
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
            vocab['pad'] = self.pad_token
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab

class IntentsAndSlots (data.Dataset):  # This class will be used to handle the dataset
    def __init__(self, dataset, lang, tokenizer, pad_token=0, unk='unk'):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk
        self.tokenizer = tokenizer
        self.pad_token = pad_token
        
        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        self.utt_tokenized = tokenizer(self.utterances, padding=True)
        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)
        
        self.O_slot = lang.slot2id["O"]
        
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
        
        intent = self.intent_ids[idx]
        sample = {'frase_testo':frase_testo, 'utterance': utt, 'slots': slots, 'intent': intent, 'attention_mask': attention_mask, 'input_ids': input_ids, 'token_type_ids': token_type_ids, 'tokenizedUtterance': self.tokenizer.convert_ids_to_tokens(input_ids)}
        
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


def collate_fn(data, pad_token=0, device="cpu"): #This function will be used to pad the sequences in the dataloader
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
    data.sort(key=lambda x: len(x['utterance']), reverse=True) 
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
        
    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, _ = merge(new_item['utterance'])
    y_slots, y_lengths = merge(new_item["slots"])
    intent = torch.LongTensor(new_item["intent"])
    
    src_utt = src_utt.to(device) # We load the Tensor on our selected device
    y_slots = y_slots.to(device)
    intent = intent.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)
    
    new_item["utterances"] = src_utt
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    return new_item