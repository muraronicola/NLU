# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import torch
import torch.nn as nn
import math
import copy
from tqdm import tqdm
import numpy as np
from torch import optim
from sklearn.metrics import classification_report
import re
import os

def execute_experiment(model, train_loader, dev_loader, optimizer, lang, criterion_slots, criterion_intents, device="cpu", n_epochs=200, clip=5):
    print("Starting experiment...\n")
    
    best_model = copy.deepcopy(model).to('cpu')
    patience = 10
    best_f1 = 0
    pbar = tqdm(range(1, n_epochs))
    
    for x in pbar:
        loss = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model, lang, device=device, clip=clip)
        
        if x % 1 == 0:
            slot_dev, _, _ = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang, device=device)
            f1 = slot_dev['total']['f']
            pbar.set_description("F1: %f" % f1)
            
            if f1 > best_f1:
                best_f1 = f1
                patience = 10
                best_model = copy.deepcopy(model).to('cpu') #save the best model
            else:
                patience -= 1
            if patience <= 0: # Early stopping with patient
                break
            
    return best_model.to(device)



def train_loop(data, optimizer, criterion_slots, criterion_intents, model, lang, device="cpu", clip=5): #Train the model one epoch 
    model.train()
    loss_array = []
    
    for indice, sample in enumerate(data):
        
        optimizer.zero_grad() # Zeroing the gradient
        
        inputs_bert, tokenizedUtterance = get_input_bert(sample, device)
        slots, intent = model(inputs_bert, tokenizedUtterance)
        slots = pad_reshape_slots(slots, tokenizedUtterance, device)
        
        loss_intent = criterion_intents(intent, sample['intents'])
        loss_slot = criterion_slots(slots, sample['y_slots'])
        
        combined_loss = loss_intent*0.1 + loss_slot*0.9 #In joint training we sum the losses. 
        
        loss_array.append(combined_loss.item())
        combined_loss.backward() 
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() # Update the weight
        
    return loss_array



def evaluate_experiment(model, train_loader, dev_loader, test_loader, criterion_slots, criterion_intents, lang, device="cpu"):
    slot_train, intent_train, loss_train = eval_loop(train_loader, criterion_slots, criterion_intents, model, lang, device=device)
    slot_dev, intent_dev, loss_dev = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang, device=device)
    slot_test, intent_test, loss_test = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang, device=device)
    
    return slot_train, slot_dev, slot_test, intent_train, intent_dev, intent_test, loss_train, loss_dev, loss_test



def eval_loop(data, criterion_slots, criterion_intents, model, lang, device="cpu"):
    model.eval()
    loss_array = []
    
    ref_intents = []
    hyp_intents = []
    
    ref_slots = []
    hyp_slots = []
    
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            inputs_bert, tokenizedUtterance = get_input_bert(sample, device)            
            slots, intents = model(inputs_bert, tokenizedUtterance)
            slots = pad_reshape_slots(slots, tokenizedUtterance, device)
            
            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots, sample['y_slots'])
            combined_loss = loss_intent*0.1 + loss_slot*0.9 #In joint training we sum the losses.
            loss_array.append(combined_loss.item())
            
            # Intent inference
            # Get the highest probable class
            out_intents = [lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()] 
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)
            
            # Slot inference 
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = sample['slots_len'].tolist()[id_seq]
                utt_ids = sample['utterance'][id_seq][:length].tolist()
                gt_ids = sample['y_slots'][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                utterance = [lang.id2word[elem] for elem in utt_ids]
                to_decode = seq[:length].tolist()
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots) if id_el < len(utterance) ])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    if id_el < len(utterance):
                        tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)
    
    try:      
        slot_results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        print("Error in slot evaluation")
        slot_results = {"total":{"f":0}}
        
    intent_results = classification_report(ref_intents, hyp_intents, zero_division=False, output_dict=True)
    return slot_results, intent_results, loss_array


def get_input_bert(sample, device):
    tensor_input_ids = torch.Tensor(sample['input_ids']).to(device).to(torch.int64)
    tensor_attention_mask = torch.Tensor(sample['attention_mask']).to(device).to(torch.int64)
    tensor_token_type_ids = torch.Tensor(sample['token_type_ids']).to(device).to(torch.int64)
    tokenizedUtterance = sample['tokenizedUtterance']
    
    inputs_bert = {'input_ids': tensor_input_ids, 'attention_mask': tensor_attention_mask, 'token_type_ids': tensor_token_type_ids}
    return inputs_bert, tokenizedUtterance


def pad_reshape_slots(results_slotFilling, tokenizedUtterance, device):
    #We need to pad the sequences to have the same length
    shapeDim_1 = results_slotFilling[0].shape[1]
    
    for i in range(results_slotFilling.shape[0]):
        frase = []
        contatore = 0
        for j in range(results_slotFilling[i].shape[0]):
            
            token = tokenizedUtterance[i][j]
            if "##" not in token and "'" not in token and "." not in token: #We need to check if the token is a special token
                frase.append(results_slotFilling[i,j]) 
            else:
                contatore += 1 #We need to count how many tokens we need to add at the end of the sequence
        
        for l in range(contatore):
            frase.append(torch.zeros(shapeDim_1).to(device))
            
        newEntry = torch.stack(frase, dim=0)
        results_slotFilling[i] = newEntry
        
    results_slotFilling = results_slotFilling.permute(0,2,1)
    
    return results_slotFilling


def init_weights(mat): # Function to initialize the weights of the model
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


def print_results(slots, intent):
    print("\nResults of the experiment:")
    
    slot_f1s = slots['total']['f']
    intent_acc = intent['accuracy']
    
    print('Slot F1', round(slot_f1s, 3))
    print('Intent Acc', round(intent_acc, 3))
    print("\n")
    print("-"*50)
    print("\n")









#-------------------------------------------------------
#-------------------------------------------------------
#-------------------------------------------------------
#Functions taken from the conn.py script
#-------------------------------------------------------
#-------------------------------------------------------
#-------------------------------------------------------




def stats():
    return {'cor': 0, 'hyp': 0, 'ref': 0}


def evaluate(ref, hyp, otag='O'):
    # evaluation for NLTK
    aligned = align_hyp(ref, hyp)
    return conlleval(aligned, otag=otag)


def align_hyp(ref, hyp):
    # align references and hypotheses for evaluation
    # add last element of token tuple in hyp to ref
    if len(ref) != len(hyp):
        raise ValueError("Size Mismatch: ref: {} & hyp: {}".format(len(ref), len(hyp)))

    out = []
    for i in range(len(ref)):
        if len(ref[i]) != len(hyp[i]):
            raise ValueError("Size Mismatch: ref: {} & hyp: {}".format(len(ref), len(hyp)))
        out.append([(*ref[i][j], hyp[i][j][-1]) for j in range(len(ref[i]))])
    return out


def conlleval(data, otag='O'):
    # token, segment & class level counts for TP, TP+FP, TP+FN
    tok = stats()
    seg = stats()
    cls = {}

    for sent in data:

        prev_ref = otag      # previous reference label
        prev_hyp = otag      # previous hypothesis label
        prev_ref_iob = None  # previous reference label IOB
        prev_hyp_iob = None  # previous hypothesis label IOB

        in_correct = False  # currently processed chunks is correct until now

        for token in sent:

            hyp_iob, hyp = parse_iob(token[-1])
            ref_iob, ref = parse_iob(token[-2])

            ref_e = is_eoc(ref, ref_iob, prev_ref, prev_ref_iob, otag)
            hyp_e = is_eoc(hyp, hyp_iob, prev_hyp, prev_hyp_iob, otag)

            ref_b = is_boc(ref, ref_iob, prev_ref, prev_ref_iob, otag)
            hyp_b = is_boc(hyp, hyp_iob, prev_hyp, prev_hyp_iob, otag)

            if not cls.get(ref) and ref:
                cls[ref] = stats()

            if not cls.get(hyp) and hyp:
                cls[hyp] = stats()

            # segment-level counts
            if in_correct:
                if ref_e and hyp_e and prev_hyp == prev_ref:
                    in_correct = False
                    seg['cor'] += 1
                    cls[prev_ref]['cor'] += 1

                elif ref_e != hyp_e or hyp != ref:
                    in_correct = False

            if ref_b and hyp_b and hyp == ref:
                in_correct = True

            if ref_b:
                seg['ref'] += 1
                cls[ref]['ref'] += 1

            if hyp_b:
                seg['hyp'] += 1
                cls[hyp]['hyp'] += 1

            # token-level counts
            if ref == hyp and ref_iob == hyp_iob:
                tok['cor'] += 1

            tok['ref'] += 1

            prev_ref = ref
            prev_hyp = hyp
            prev_ref_iob = ref_iob
            prev_hyp_iob = hyp_iob

        if in_correct:
            seg['cor'] += 1
            cls[prev_ref]['cor'] += 1

    return summarize(seg, cls)


def parse_iob(t):
    m = re.match(r'^([^-]*)-(.*)$', t)
    return m.groups() if m else (t, None)


def is_boc(lbl, iob, prev_lbl, prev_iob, otag='O'):
    """
    is beginning of a chunk

    supports: IOB, IOBE, BILOU schemes
        - {E,L} --> last
        - {S,U} --> unit

    :param lbl: current label
    :param iob: current iob
    :param prev_lbl: previous label
    :param prev_iob: previous iob
    :param otag: out-of-chunk label
    :return:
    """
    boc = False

    boc = True if iob in ['B', 'S', 'U'] else boc
    boc = True if iob in ['E', 'L'] and prev_iob in ['E', 'L', 'S', otag] else boc
    boc = True if iob == 'I' and prev_iob in ['S', 'L', 'E', otag] else boc

    boc = True if lbl != prev_lbl and iob != otag and iob != '.' else boc

    # these chunks are assumed to have length 1
    boc = True if iob in ['[', ']'] else boc

    return boc


def is_eoc(lbl, iob, prev_lbl, prev_iob, otag='O'):
    """
    is end of a chunk

    supports: IOB, IOBE, BILOU schemes
        - {E,L} --> last
        - {S,U} --> unit

    :param lbl: current label
    :param iob: current iob
    :param prev_lbl: previous label
    :param prev_iob: previous iob
    :param otag: out-of-chunk label
    :return:
    """
    eoc = False

    eoc = True if iob in ['E', 'L', 'S', 'U'] else eoc
    eoc = True if iob == 'B' and prev_iob in ['B', 'I'] else eoc
    eoc = True if iob in ['S', 'U'] and prev_iob in ['B', 'I'] else eoc

    eoc = True if iob == otag and prev_iob in ['B', 'I'] else eoc

    eoc = True if lbl != prev_lbl and iob != otag and prev_iob != '.' else eoc

    # these chunks are assumed to have length 1
    eoc = True if iob in ['[', ']'] else eoc

    return eoc


def score(cor_cnt, hyp_cnt, ref_cnt):
    # precision
    p = 1 if hyp_cnt == 0 else cor_cnt / hyp_cnt
    # recall
    r = 0 if ref_cnt == 0 else cor_cnt / ref_cnt
    # f-measure (f1)
    f = 0 if p+r == 0 else (2*p*r)/(p+r)
    return {"p": p, "r": r, "f": f, "s": ref_cnt}


def summarize(seg, cls):
    # class-level
    res = {lbl: score(cls[lbl]['cor'], cls[lbl]['hyp'], cls[lbl]['ref']) for lbl in set(cls.keys())}
    # micro
    res.update({"total": score(seg.get('cor', 0), seg.get('hyp', 0), seg.get('ref', 0))})
    return res


def read_corpus_conll(corpus_file, fs="\t"):
    """
    read corpus in CoNLL format
    :param corpus_file: corpus in conll format
    :param fs: field separator
    :return: corpus
    """
    featn = None  # number of features for consistency check
    sents = []  # list to hold words list sequences
    words = []  # list to hold feature tuples

    for line in open(corpus_file):
        line = line.strip()
        if len(line.strip()) > 0:
            feats = tuple(line.strip().split(fs))
            if not featn:
                featn = len(feats)
            elif featn != len(feats) and len(feats) != 0:
                raise ValueError("Unexpected number of columns {} ({})".format(len(feats), featn))

            words.append(feats)
        else:
            if len(words) > 0:
                sents.append(words)
                words = []
    return sents


def get_chunks(corpus_file, fs="\t", otag="O"):
    sents = read_corpus_conll(corpus_file, fs=fs)
    return set([parse_iob(token[-1])[1] for sent in sents for token in sent if token[-1] != otag])
