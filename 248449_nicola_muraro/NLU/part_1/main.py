from functions import *
from utils import LoadData
from model import ModelIAS
from torch import optim
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



    #Load the dataset
    train_path = "./dataset/train.json"
    test_path = "./dataset/test.json"

    runs = 5
    pad_token = 0

    
    criterion_slots = nn.CrossEntropyLoss(ignore_index=pad_token)
    criterion_intents = nn.CrossEntropyLoss()
    
    
    if not eval_only: #Train the models and evaluate all of them

        load_data = LoadData(train_path, test_path, device=device, pad_token=pad_token)
        train_loader, dev_loader, test_loader = load_data.get_dataset_loaders()
        lang = load_data.get_lang()

        out_slot = len(lang.slot2id)
        out_int = len(lang.intent2id)
        vocab_len = len(lang.word2id)
        

        #First experiment
        slot_f1s_1, intent_acc = [], []
        best_f1_ever = 0
        best_model_ever = None
        print("Starting experiment 1...\n")
        
        for i in tqdm(range(0, runs)):
            first_model = ModelIAS(out_slot, out_int, vocab_len, emb_size=300, hid_size=200).to(device)
            first_model.apply(init_weights)
            
            optimizer = optim.Adam(first_model.parameters(), lr=0.0001)
            
            first_trained_model, new_best_f1 = execute_experiment(first_model, train_loader, dev_loader, optimizer, lang, criterion_slots, criterion_intents, device=device)
            if new_best_f1 >= best_f1_ever:
                best_model_ever = first_trained_model
                best_f1_ever = new_best_f1
                
            _, _, results_test, _, _, intent_test, _, _, _ = evaluate_experiment(first_trained_model, train_loader, dev_loader, test_loader, criterion_slots, criterion_intents, lang)
            
            intent_acc.append(intent_test['accuracy'])
            slot_f1s_1.append(results_test['total']['f'])
            
        print_results(intent_acc, slot_f1s_1, title="Results of experiment 1:")
        
        

        #Second experiment
        slot_f1s_2, intent_acc = [], []
        best_f1_ever = 0
        best_model_ever = None
        print("Starting experiment 2...\n")
        
        for i in tqdm(range(0, runs)):
            second_model = ModelIAS(out_slot, out_int, vocab_len, emb_size=300, hid_size=200, dropout_value=0.3).to(device)
            second_model.apply(init_weights)
            
            optimizer = optim.Adam(second_model.parameters(), lr=0.0001)
            
            second_trained_model, new_best_f1 = execute_experiment(second_model, train_loader, dev_loader, optimizer, lang, criterion_slots, criterion_intents, device=device)
            if new_best_f1 >= best_f1_ever:
                best_model_ever = second_trained_model
                best_f1_ever = new_best_f1
                
            _, _, results_test, _, _, intent_test, _, _, _ = evaluate_experiment(second_trained_model, train_loader, dev_loader, test_loader, criterion_slots, criterion_intents, lang)
            
            intent_acc.append(intent_test['accuracy'])
            slot_f1s_2.append(results_test['total']['f'])
            
        print_results(intent_acc, slot_f1s_2, title="Results of experiment 2:")



        #Summary of all the experiments
        summary_results = zip(("first model", "second model"), (first_trained_model, second_trained_model), (round(slot_f1s_1.mean(),3), round(slot_f1s_2.mean(),3)))
        best_model = final_result_summary(summary_results)
        
        #Saving the best model to disk
        if save_model:
            save_best_model(best_model[1], lang, "./bin/")


    else: #Evaluating only the best model (loaded from the model_path_eval)
        
        saved_info = torch.load(model_path_eval)
        state_dict = saved_info["state_dict"]
        lang = saved_info["lang"]

        load_data = LoadData(train_path, test_path, device=device, pad_token=pad_token, lang=lang)
        train_loader, dev_loader, test_loader = load_data.get_dataset_loaders()

        best_model = ModelIAS(len(lang.slot2id), len(lang.intent2id), len(lang.word2id), emb_size=300, hid_size=200, dropout_value=0.3).to(device)
        best_model.load_state_dict(state_dict)
        best_model.to(device)

        _, _, results_test, _, _, intent_test, _, _, _ = evaluate_experiment(best_model, train_loader, dev_loader, test_loader, criterion_slots, criterion_intents, lang)
        print("Results of the best model:")
        print("Intent accuracy: ", intent_test['accuracy'])
        print("Slot F1: ", results_test['total']['f'])