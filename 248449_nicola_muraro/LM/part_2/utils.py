from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
from functools import partial
import torch


class LoadData():
    
    def __init__(self, train_path, dev_path, test_path, device="cpu", special_tokens=["<pad>", "<eos>"], lang=None):
        self.__train_raw = self.__read_file(train_path)
        self.__dev_raw = self.__read_file(dev_path)
        self.__test_raw = self.__read_file(test_path)
        
        if lang is not None: #If we are evaluating a saved model we need to pass the lang object (saved during training)
            self.__lang = lang
        else:
            self.__lang = Lang(self.__train_raw, special_tokens)
        
        self.__train_dataset = PennTreeBank(self.__train_raw, self.__lang)
        self.__dev_dataset = PennTreeBank(self.__dev_raw, self.__lang)
        self.__test_dataset = PennTreeBank(self.__test_raw, self.__lang)
        
        self.__device = device
    
    def __read_file(self, path, eos_token="<eos>"):
        output = []
        with open(path, "r") as f:
            for line in f.readlines():
                output.append(line.strip() + " " + eos_token)
        return output
    
    def get_lang(self): #Return the lang object
        return self.__lang
    
    def get_dataset_loaders(self, batch_size_train=64, batch_size_val=128, batch_size_test=128, pad_token="<pad>"):
        train_loader = DataLoader(self.__train_dataset, batch_size=batch_size_train, collate_fn=partial(collate_fn, pad_token=self.__lang.word2id[pad_token], device=self.__device),  shuffle=True)
        dev_loader = DataLoader(self.__dev_dataset, batch_size=batch_size_val, collate_fn=partial(collate_fn, pad_token=self.__lang.word2id[pad_token], device=self.__device))
        test_loader = DataLoader(self.__test_dataset, batch_size=batch_size_test, collate_fn=partial(collate_fn, pad_token=self.__lang.word2id[pad_token], device=self.__device))
    
        return train_loader, dev_loader, test_loader


class Lang(): #This class will be used to handle the vocabulary
    def __init__(self, corpus, special_tokens=[]):
        self.word2id = self.get_vocab(corpus, special_tokens)
        self.id2word = {v:k for k, v in self.word2id.items()}
    def get_vocab(self, corpus, special_tokens=[]):
        output = {}
        i = 0 
        for st in special_tokens:
            output[st] = i
            i += 1
        for sentence in corpus:
            for w in sentence.split():
                if w not in output:
                    output[w] = i
                    i += 1
        return output



class PennTreeBank (data.Dataset): #This class will be used to convert the raw text into a compatible dataset
    def __init__(self, corpus, lang):
        self.source = []
        self.target = []
        
        for sentence in corpus:
            self.source.append(sentence.split()[0:-1]) # We get from the first token till the second-last token
            self.target.append(sentence.split()[1:]) # We get from the second token till the last token
        
        self.source_ids = self.mapping_seq(self.source, lang)
        self.target_ids = self.mapping_seq(self.target, lang)
        

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        src= torch.LongTensor(self.source_ids[idx])
        trg = torch.LongTensor(self.target_ids[idx])
        sample = {'source': src, 'target': trg}
        return sample
    
    # Auxiliary methods
    def mapping_seq(self, data, lang): # Map sequences of tokens to corresponding computed in Lang class
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq:
                if x in lang.word2id:
                    tmp_seq.append(lang.word2id[x])
                else:
                    print('OOV found!')
                    print('You have to deal with that') # PennTreeBank doesn't have OOV but "Trust is good, control is better!"
                    break
            res.append(tmp_seq)
        return res  


def collate_fn(data, pad_token, device="cpu"): #This function will be used to pad the sequences in the dataloader
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
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths
    
    data.sort(key=lambda x: len(x["source"]), reverse=True) 
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    source, _ = merge(new_item["source"])
    target, lengths = merge(new_item["target"])
    
    new_item["source"] = source.to(device)
    new_item["target"] = target.to(device)
    new_item["number_tokens"] = sum(lengths)
    return new_item