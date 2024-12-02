# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from utils import LoadData
from model import ModelIAS
from torch import optim

if __name__ == "__main__": #Aggiungo argomenti per il main (dalla console)
    #Write the code to load the datasets and to run your functions
    # Print the results
    
    device = "cuda:0"
    pad_token = 0
    
    #Load the dataset
    train_path = "./dataset/train.json"
    test_path = "./dataset/test.json"
    
    load_data = LoadData(train_path, test_path, device=device, pad_token=pad_token)
    lang = load_data.get_lang()
    train_loader, dev_loader, test_loader = load_data.get_dataset_loaders()
    
    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)
    
    runs = 5
    
    
    
    #First experiment
    slot_f1s, intent_acc = [], []
    best_f1_ever = 0
    best_model_ever = None
    print("Starting experiment 1...\n")
    
    for i in tqdm(range(0, runs)):
        first_model = ModelIAS(out_slot, out_int, vocab_len, emb_size=300, hid_size=300).to(device)
        first_model.apply(init_weights)
        
        optimizer = optim.Adam(first_model.parameters(), lr=0.0001)
        criterion_slots = nn.CrossEntropyLoss(ignore_index=pad_token)
        criterion_intents = nn.CrossEntropyLoss()
        
        first_trained_model, new_best_f1 = execute_experiment(first_model, train_loader, dev_loader, optimizer, lang, criterion_slots, criterion_intents, device=device)
        if new_best_f1 >= best_f1_ever:
            best_model_ever = first_trained_model
            best_f1_ever = new_best_f1
            
        _, _, results_test, _, _, intent_test, _, _, _ = evaluate_experiment(first_trained_model, train_loader, dev_loader, test_loader, criterion_slots, criterion_intents, lang)
        
        intent_acc.append(intent_test['accuracy'])
        slot_f1s.append(results_test['total']['f'])
        
    print_results(1, intent_acc, slot_f1s)
    
    
    
    #Second experiment
    slot_f1s, intent_acc = [], []
    best_f1_ever = 0
    best_model_ever = None
    print("Starting experiment 2...\n")
    
    for i in tqdm(range(0, runs)):
        second_model = ModelIAS(out_slot, out_int, vocab_len, emb_size=300, hid_size=300, dropout_value=0.3).to(device)
        second_model.apply(init_weights)
        
        optimizer = optim.Adam(second_model.parameters(), lr=0.0001)
        criterion_slots = nn.CrossEntropyLoss(ignore_index=pad_token)
        criterion_intents = nn.CrossEntropyLoss()
        
        first_trained_model, new_best_f1 = execute_experiment(second_model, train_loader, dev_loader, optimizer, lang, criterion_slots, criterion_intents, device=device)
        if new_best_f1 >= best_f1_ever:
            best_model_ever = first_trained_model
            best_f1_ever = new_best_f1
            
        _, _, results_test, _, _, intent_test, _, _, _ = evaluate_experiment(first_trained_model, train_loader, dev_loader, test_loader, criterion_slots, criterion_intents, lang)
        
        intent_acc.append(intent_test['accuracy'])
        slot_f1s.append(results_test['total']['f'])
        
    print_results(2, intent_acc, slot_f1s)
