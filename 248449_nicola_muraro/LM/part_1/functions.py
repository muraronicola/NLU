import torch
import torch.nn as nn
import math
import copy
from tqdm import tqdm
import os

def execute_experiment(model, train_loader, dev_loader, optimizer, lang, experiment_number, device="cpu", n_epochs=100, clip=5):
    print("Starting experiment " + str(experiment_number) + "...\n")
    
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')
    
    best_model = copy.deepcopy(model).to('cpu')
    best_ppl = math.inf
    pbar = tqdm(range(1, n_epochs))
    patience = 3
    
    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model, clip)    
        
        if epoch % 1 == 0:
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            pbar.set_description("PPL: %f" % ppl_dev)
            
            if  ppl_dev < best_ppl: # the lower, the better
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                patience = 3
            else:
                patience -= 1
                
            if patience <= 0: # Early stopping with patience
                break # Not nice but it keeps the code clean
            
    return best_model.to(device)


def evaluate_experiment(model, train_loader, dev_loader, test_loader, lang):
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')
    
    ppl_train, loss_train = eval_loop(train_loader, criterion_eval, model)
    ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
    ppl_test, loss_test = eval_loop(test_loader, criterion_eval, model)
    
    return ppl_train, ppl_dev, ppl_test, loss_train, loss_dev, loss_test


def print_results( ppl_train, ppl_dev, ppl_test, loss_train, loss_dev, loss_test, title=""):
    print("\n" + title)
    print("Train:\tPPL "+ str(round(ppl_train, 2))+"\tloss "+str(round(loss_train, 2)))
    print("Dev:\tPPL "+ str(round(ppl_dev, 2))+"\tloss "+str(round(loss_dev, 2)))
    print("Test:\tPPL "+ str(round(ppl_test, 2))+"\tloss "+str(round(loss_test, 2)))
    print("\n")
    print("-"*50)
    print("\n")



def train_loop(data, optimizer, criterion, model, clip=5):
    model.train()
    loss_array = []
    number_of_tokens = []
    
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        output = model(sample['source'])
        loss = criterion(output, sample['target'])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid explosioning gradients
        nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() # Update the weights
        
    return sum(loss_array)/sum(number_of_tokens)


def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []
    # softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            output = model(sample['source'])
            loss = eval_criterion(output, sample['target'])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])
            
    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return


def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)


def final_result_summary(summary_results): #print and return the best model
    best_model = min(summary_results, key=lambda x: x[2])
    print("")
    print("-"*50)
    print("-"*50)
    print(f"\nThe best model is the {best_model[0]}, with a dev PPL of {best_model[2]}\n")
    print("-"*50)
    print("-"*50)
    return best_model


def save_best_model(best_model, lang, path="./bin/"):
    base_filename = "best_model_"
    extension= ".pt"
    new_file = False
    complete_filename = ""
    counter = 0
    
    while not new_file: #Check if the file already exists, if so, generate a new filename
        id = str(counter)
        complete_filename = base_filename + id + extension
        
        if not os.path.exists(f"{path}{complete_filename}"):
            new_file = True
            
        counter+=1
        
    saving_object = {"state_dict": best_model.state_dict(), "lang": lang}
    torch.save(saving_object, f"{path}{complete_filename}") #Save the best model to the bin folder