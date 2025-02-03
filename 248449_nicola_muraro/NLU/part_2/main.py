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


    pad_token = 0
    seq_length = 51
    hiddenSize_new = 768


    #Load the dataset
    train_path = "./dataset/train.json"
    test_path = "./dataset/test.json"

    tokenizer_bert_base = BertTokenizer.from_pretrained("bert-base-uncased") # Download the tokenizer
    model_bert_base = BertModel.from_pretrained("bert-base-uncased") # Download the model

    criterion_slots = nn.CrossEntropyLoss(ignore_index=pad_token)
    criterion_intents = nn.CrossEntropyLoss()



    if not eval_only: #Train the models and evaluate all of them
        
        load_data = LoadData(train_path, test_path, tokenizer_bert_base, device=device, pad_token=pad_token)
        lang = load_data.get_lang()
        train_loader, dev_loader, test_loader = load_data.get_dataset_loaders()
        
        out_slot = len(lang.slot2id)
        out_int = len(lang.intent2id)
        

        #First experiment
        first_model = ModelIAS(model_bert_base, hiddenSize_new, out_slot, out_int, drop_value=0.1).to(device)
        optimizer = optim.Adam(model_bert_base.parameters(), lr=0.0001)
        
        first_trained_model = execute_experiment(first_model, train_loader, dev_loader, optimizer, lang, criterion_slots, criterion_intents, device=device)
        _, _, slot_test, _, _, intent_test, _, _, _ = evaluate_experiment(first_trained_model, train_loader, dev_loader, test_loader, criterion_slots, criterion_intents, lang, device=device)
        first_trained_model.to("cpu")
        print_results(slot_test, intent_test)
        
        
        if save_model:
            save_best_model(first_trained_model, lang, "./bin/")

    else: #Evaluate only the best model
        
        saved_info = torch.load(model_path_eval)
        state_dict = saved_info["state_dict"]
        lang = saved_info["lang"]

        load_data = LoadData(train_path, test_path, tokenizer_bert_base, device=device, pad_token=pad_token, lang=lang)
        train_loader, dev_loader, test_loader = load_data.get_dataset_loaders()

        best_model = ModelIAS(model_bert_base, hiddenSize_new, len(lang.slot2id), len(lang.intent2id), drop_value=0.1).to(device)
        best_model.load_state_dict(state_dict)
        best_model.to(device)

        _, _, slot_test, _, _, intent_test, _, _, _ = evaluate_experiment(best_model, train_loader, dev_loader, test_loader, criterion_slots, criterion_intents, lang, device=device)
        print_results(slot_test, intent_test)