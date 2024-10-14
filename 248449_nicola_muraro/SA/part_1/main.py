# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from utils import LoadData
from model import ModelIAS
from torch import optim
from transformers import BertTokenizer, BertModel


if __name__ == "__main__": #Aggiungo argomenti per il main (dalla console)
    #Write the code to load the datasets and to run your functions
    # Print the results
    
    device = "cuda:0" #"cuda:0"
    pad_token = -1
    
    #Load the dataset
    train_path = "./dataset/laptop14_train.txt"
    test_path = "./dataset/laptop14_test.txt"
    
    tokenizer_bert_base = BertTokenizer.from_pretrained("bert-base-uncased") # Download the tokenizer
    model_bert_base = BertModel.from_pretrained("bert-base-uncased") # Download the model
    
    load_data = LoadData(train_path, test_path, tokenizer_bert_base, device=device, pad_token=pad_token)
    lang = load_data.get_lang()
    train_loader, dev_loader, test_loader = load_data.get_dataset_loaders()
    
    
    out_slot = len(lang.slot2id)
    
    #seq_length = 91 #51 it's changed for this assignemnt. WHY?
    hiddenSize_new = 768
    
    criterion_slots = nn.CrossEntropyLoss(ignore_index=pad_token)
    
    
    
    #First experiment
    first_model = ModelIAS(model_bert_base, hiddenSize_new, out_slot, device=device).to(device)
    first_model.apply(init_weights)
    optimizer = optim.Adam(first_model.parameters(), lr=0.00005) #Non so nemmeno che modello devo ottimizzare
    
    first_trained_model = execute_experiment(first_model, train_loader, dev_loader, optimizer, lang, criterion_slots, pad_token, device=device)
    _, _, slot_test_f1 = evaluate_experiment(first_trained_model, train_loader, dev_loader, test_loader, criterion_slots, lang, pad_token, device=device)
    print_results(slot_test_f1)
    