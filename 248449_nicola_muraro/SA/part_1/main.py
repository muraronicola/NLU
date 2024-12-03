from functions import *
from utils import LoadData
from model import ModelIAS
from torch import optim
from transformers import BertTokenizer, BertModel
import argparse


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

    pad_token = -1
    

    #Load the dataset
    train_path = "./dataset/laptop14_train.txt"
    test_path = "./dataset/laptop14_test.txt"
    
    tokenizer_bert_base = BertTokenizer.from_pretrained("bert-base-uncased") # Download the tokenizer
    model_bert_base = BertModel.from_pretrained("bert-base-uncased") # Download the model
    
    criterion_slots = nn.CrossEntropyLoss(ignore_index=pad_token)
    hiddenSize_new = 768


    if not eval_only: #Train the models and evaluate all of them

        load_data = LoadData(train_path, test_path, tokenizer_bert_base, device=device, pad_token=pad_token)
        lang = load_data.get_lang()
        train_loader, dev_loader, test_loader = load_data.get_dataset_loaders()
        
        out_slot = len(lang.slot2id)
        
        
        #First experiment
        first_model = ModelIAS(model_bert_base, hiddenSize_new, out_slot, drop_value=0.1, device=device).to(device)
        first_model.apply(init_weights)
        optimizer = optim.Adam(first_model.parameters(), lr=0.00005) #Non so nemmeno che modello devo ottimizzare
        
        first_trained_model = execute_experiment(first_model, train_loader, dev_loader, optimizer, lang, criterion_slots, pad_token, device=device)
        _, _, slot_test_f1 = evaluate_experiment(first_trained_model, train_loader, dev_loader, test_loader, criterion_slots, lang, pad_token, device=device)
        first_trained_model.to("cpu")
        print_results(slot_test_f1)
        
        if save_model:
            save_best_model(first_trained_model, lang, "./bin/")
        
    else:
        saved_info = torch.load(model_path_eval)
        state_dict = saved_info["state_dict"]
        lang = saved_info["lang"]

        load_data = LoadData(train_path, test_path, tokenizer_bert_base, device=device, pad_token=pad_token, lang=lang)
        train_loader, dev_loader, test_loader = load_data.get_dataset_loaders()
        out_slot = len(lang.slot2id)

        best_model = ModelIAS(model_bert_base, hiddenSize_new, out_slot, drop_value=0.1, device=device).to(device)
        best_model.load_state_dict(state_dict)
        best_model.to(device)

        _, _, slot_test_f1 = evaluate_experiment(best_model, train_loader, dev_loader, test_loader, criterion_slots, lang, pad_token, device=device)
        print_results(slot_test_f1)