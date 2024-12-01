# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from utils import LoadData
from model import LM_LSTM
from torch import optim

if __name__ == "__main__": #Aggiungo argomenti per il main (dalla console)
    #Write the code to load the datasets and to run your functions
    # Print the results
    
    device = "cuda:0"
    
    
    #Load the dataset
    train_path = "./dataset/ptb.train.txt"
    dev_path = "./dataset/ptb.valid.txt"
    test_path = "./dataset/ptb.test.txt"
    
    load_data = LoadData(train_path, dev_path, test_path, device=device)
    lang = load_data.get_lang()
    train_loader, dev_loader, test_loader = load_data.get_dataset_loaders(batch_size_train=64, batch_size_val=128, batch_size_test=128)
    
    
    
    
    
    #First experiment
    first_model = LM_LSTM(emb_size=500, hidden_size=600, output_size=len(lang.word2id), pad_index=lang.word2id["<pad>"]).to(device)
    first_model.apply(init_weights)
    
    optimizer = optim.SGD(first_model.parameters(), lr=1.5)
    
    first_trained_model = execute_experiment(first_model, train_loader, dev_loader, optimizer, lang, experiment_number=1, device=device)
    ppl_train, ppl_dev, ppl_test, loss_train, loss_dev, loss_test = evaluate_experiment(first_trained_model, train_loader, dev_loader, test_loader, lang)
    print_results(1, ppl_train, ppl_dev, ppl_test, loss_train, loss_dev, loss_test)
    
    
    
    
    #Second experiment
    second_model = LM_LSTM(emb_size=500, hidden_size=600, output_size=len(lang.word2id), emb_dropout=0.3, out_dropout=0.3, pad_index=lang.word2id["<pad>"]).to(device)
    second_model.apply(init_weights)
    
    optimizer = optim.SGD(second_model.parameters(), lr=1.5)
    
    second_trained_model = execute_experiment(second_model, train_loader, dev_loader, optimizer, lang, experiment_number=2, device=device)
    ppl_train, ppl_dev, ppl_test, loss_train, loss_dev, loss_test = evaluate_experiment(second_trained_model, train_loader, dev_loader, test_loader, lang)
    print_results(2, ppl_train, ppl_dev, ppl_test, loss_train, loss_dev, loss_test)
    
    
    
    
    #Third experiment
    third_model = LM_LSTM(emb_size=500, hidden_size=600, output_size=len(lang.word2id), emb_dropout=0.1, out_dropout=0.1, pad_index=lang.word2id["<pad>"]).to(device)
    third_model.apply(init_weights)
    
    optimizer = optim.AdamW(third_model.parameters(), lr=0.0005)
    
    third_trained_model = execute_experiment(third_model, train_loader, dev_loader, optimizer, lang, experiment_number=3, device=device)
    ppl_train, ppl_dev, ppl_test, loss_train, loss_dev, loss_test = evaluate_experiment(third_trained_model, train_loader, dev_loader, test_loader, lang)
    print_results(3, ppl_train, ppl_dev, ppl_test, loss_train, loss_dev, loss_test)
    
    
    
    #Saving the best model
    global_results = zip(("first model", "second model", "third model"), (first_trained_model, second_trained_model, third_trained_model))
    best_model = min(global_results, key=lambda x: x[1][1])
    print("\n")
    print("-"*50)
    print(f"The best model is the {best_model[0]}")
    
    torch.save(best_model, f"./models/{best_model[1]}.pt")