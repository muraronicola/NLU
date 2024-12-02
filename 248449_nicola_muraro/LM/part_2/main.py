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
    
    load_data = LoadData(train_path, dev_path, test_path, device=device)
    train_loader, dev_loader, test_loader = load_data.get_dataset_loaders(batch_size_train=64, batch_size_val=128, batch_size_test=128)
    
    
    
    
    
    if not eval_only: #Train the models and evaluate all of them
        lang = load_data.get_lang() #Get the lang object
        
        #First experiment
        first_model = LM_LSTM(emb_size=600, hidden_size=600, output_size=len(lang.word2id), pad_index=lang.word2id["<pad>"], device=device).to(device)
        first_model.apply(init_weights)
        
        optimizer = optim.SGD(first_model.parameters(), lr=1.5)
        
        first_trained_model = execute_experiment(first_model, train_loader, dev_loader, optimizer, lang, experiment_number=1, device=device)
        ppl_train, ppl_dev_1, ppl_test, loss_train, loss_dev, loss_test = evaluate_experiment(first_trained_model, train_loader, dev_loader, test_loader, lang)
        print_results(ppl_train, ppl_dev_1, ppl_test, loss_train, loss_dev, loss_test, title="Results of experiment 1:")



        #Second experiment
        second_model = LM_LSTM(emb_size=600, hidden_size=600, output_size=len(lang.word2id), variational_dropout=0.05, pad_index=lang.word2id["<pad>"], device=device).to(device)
        second_model.apply(init_weights)
        
        optimizer = optim.SGD(second_model.parameters(), lr=1.5)
        
        second_trained_model = execute_experiment(second_model, train_loader, dev_loader, optimizer, lang, experiment_number=2, device=device)
        ppl_train, ppl_dev_2, ppl_test, loss_train, loss_dev, loss_test = evaluate_experiment(second_trained_model, train_loader, dev_loader, test_loader, lang)
        print_results(ppl_train, ppl_dev_2, ppl_test, loss_train, loss_dev, loss_test, title="Results of experiment 2:")
        
        
        
        #Third experiment
        third_model = LM_LSTM(emb_size=600, hidden_size=600, output_size=len(lang.word2id), variational_dropout=0.05, pad_index=lang.word2id["<pad>"], device=device).to(device)
        third_model.apply(init_weights)
        
        optimizer = optim.SGD(third_model.parameters(), lr=1.5)
        
        third_trained_model = execute_experiment(third_model, train_loader, dev_loader, optimizer, lang, experiment_number=3, nonmono_ASGD=True, ASGD_lr=2.5, device=device)
        ppl_train, ppl_dev_3, ppl_test, loss_train, loss_dev, loss_test = evaluate_experiment(third_trained_model, train_loader, dev_loader, test_loader, lang)
        print_results(ppl_train, ppl_dev_3, ppl_test, loss_train, loss_dev, loss_test, title="Results of experiment 3:")
        
        
        
        #Summary of all the experiments
        summary_results = zip(("first model", "second model", "third model"), (first_trained_model, second_trained_model, third_trained_model), (ppl_dev_1, ppl_dev_2, ppl_dev_3))
        best_model = final_result_summary(summary_results)
        
        #Saving the best model to disk
        if save_model:
            save_best_model(best_model[1], lang, "./bin/")
    
    
    else: #Evaluating only the best model (loaded from the model_path_eval)
        loaded_object = torch.load(model_path_eval)
        best_model = loaded_object["model"] #Get the model object
        lang = loaded_object["lang"] #Get the lang object
        
        best_model.to(device)
        ppl_train, ppl_dev, ppl_test, loss_train, loss_dev, loss_test = evaluate_experiment(best_model, train_loader, dev_loader, test_loader, lang)
        print_results(ppl_train, ppl_dev, ppl_test, loss_train, loss_dev, loss_test, title="Results of the best save model:")