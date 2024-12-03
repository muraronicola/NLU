from functions import *
from utils import LoadData
from model import LM_LSTM
from torch import optim
import argparse
import os

if __name__ == "__main__":
    
    #Parse the arguments from the console
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=str, help = "The device on which run the model", default="cuda:0")
    parser.add_argument("-s", "--save", type=bool, help = "Save the best model", default=False)
    parser.add_argument("-e", "--eval_only", type=bool, help = "Whether to evaluate only the best model, without training anything", default=False)
    parser.add_argument("-m", "--model", type=str, help = "The model path to be evaluated (used only with eval_only)", default="./bin/best_model.pt")
    
    args = parser.parse_args()
    eval_only = args.eval_only
    device = args.device
    save_model = args.save
    model_path_eval = args.model
    
    
    #Load the dataset
    train_path = "./dataset/ptb.train.txt"
    dev_path = "./dataset/ptb.valid.txt"
    test_path = "./dataset/ptb.test.txt"
    
    
    if not eval_only: #Train the models and evaluate all of them
        load_data = LoadData(train_path, dev_path, test_path, device=device) #Create the LoadData object
        train_loader, dev_loader, test_loader = load_data.get_dataset_loaders(batch_size_train=64, batch_size_val=128, batch_size_test=128)
        lang = load_data.get_lang() #Get the lang object
        
        #Loss functions
        criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
        criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')
        
        
        
        #First experiment
        first_model = LM_LSTM(emb_size=600, hidden_size=500, output_size=len(lang.word2id), pad_index=lang.word2id["<pad>"]).to(device)
        first_model.apply(init_weights)
        
        optimizer = optim.SGD(first_model.parameters(), lr=1.5)
        
        first_trained_model = execute_experiment(first_model, train_loader, dev_loader, optimizer, lang, experiment_number=1, criterion_train=criterion_train, criterion_eval=criterion_eval, device=device) #Train the model
        ppl_train, ppl_dev_1, ppl_test, loss_train, loss_dev, loss_test = evaluate_experiment(first_trained_model, train_loader, dev_loader, test_loader, criterion_eval, lang) #Evaluate the model
        print_results(ppl_train, ppl_dev_1, ppl_test, loss_train, loss_dev, loss_test, title="Results of experiment 1:")
        first_trained_model.to("cpu") #Offload some of the memory of the GPU
        
        
        
        #Second experiment
        second_model = LM_LSTM(emb_size=600, hidden_size=500, output_size=len(lang.word2id), emb_dropout=0.3, out_dropout=0.3, pad_index=lang.word2id["<pad>"]).to(device)
        second_model.apply(init_weights)
        
        optimizer = optim.SGD(second_model.parameters(), lr=1.5)
        
        second_trained_model = execute_experiment(second_model, train_loader, dev_loader, optimizer, lang, experiment_number=2, criterion_train=criterion_train, criterion_eval=criterion_eval, device=device) #Train the model
        ppl_train, ppl_dev_2, ppl_test, loss_train, loss_dev, loss_test = evaluate_experiment(second_trained_model, train_loader, dev_loader, test_loader, criterion_eval, lang) #Evaluate the model
        print_results(ppl_train, ppl_dev_2, ppl_test, loss_train, loss_dev, loss_test, title="Results of experiment 2:")
        second_trained_model.to("cpu") #Offload some of the memory of the GPU
        
        
        
        #Third experiment
        third_model = LM_LSTM(emb_size=600, hidden_size=500, output_size=len(lang.word2id), emb_dropout=0.1, out_dropout=0.1, pad_index=lang.word2id["<pad>"]).to(device)
        third_model.apply(init_weights)
        
        optimizer = optim.AdamW(third_model.parameters(), lr=0.0005)
        
        third_trained_model = execute_experiment(third_model, train_loader, dev_loader, optimizer, lang, experiment_number=3, criterion_train=criterion_train, criterion_eval=criterion_eval, device=device) #Train the model
        ppl_train, ppl_dev_3, ppl_test, loss_train, loss_dev, loss_test = evaluate_experiment(third_trained_model, train_loader, dev_loader, test_loader, criterion_eval, lang) #Evaluate the model
        print_results(ppl_train, ppl_dev_3, ppl_test, loss_train, loss_dev, loss_test, title="Results of experiment 3:")
        third_trained_model.to("cpu") #Offload some of the memory of the GPU
        
        
        
        #Summary of all the experiments
        summary_results = zip(("first model", "second model", "third model"), (first_trained_model, second_trained_model, third_trained_model), (ppl_dev_1, ppl_dev_2, ppl_dev_3))
        best_model = final_result_summary(summary_results) #Get the best model (of the three trained)
        
        #Saving the best model to disk
        if save_model:
            save_best_model(best_model[1], lang, path="./bin/", device=device)
    
    
    else: #Evaluating only the best model (loaded from the model_path_eval)
        saved_info = torch.load(model_path_eval)
        state_dict = saved_info["state_dict"]
        lang = saved_info["lang"]
        
        load_data = LoadData(train_path, dev_path, test_path, device=device, lang=lang) #Create the LoadData object
        train_loader, dev_loader, test_loader = load_data.get_dataset_loaders(batch_size_train=64, batch_size_val=128, batch_size_test=128)
        
        criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')
        
        best_model = LM_LSTM(emb_size=600, hidden_size=500, output_size=len(lang.word2id), emb_dropout=0.3, out_dropout=0.3, pad_index=lang.word2id["<pad>"])
        best_model.load_state_dict(state_dict)
        best_model.to(device)

        ppl_train, ppl_dev, ppl_test, loss_train, loss_dev, loss_test = evaluate_experiment(best_model, train_loader, dev_loader, test_loader, criterion_eval, lang) #Evaluate the model
        print_results(ppl_train, ppl_dev, ppl_test, loss_train, loss_dev, loss_test, title="Results of the best save model:")