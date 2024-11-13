controllo che i valori ottenuti con il nuovo codice siano comparabili a quelli ottenuti con il codice "scritto male da me"

controllo che i nomi delle variabili/metodi abbiano un senso


controllo di non aver scritto roba a caso in giro per i vari file

1_1: rimetto le epoche al default (ora è 1 per debug)



1_2: secondo esperimento, provo vari valori per il dropout
        terzo esperimento, provo la n_nonmono, e forse il lr


in entrambi 2: forse ho invertito in evaluate e la print l'ordine tra i risultati di slot and intent


2_1 patient?
2_2 dropout size


2_2 different value of combination of the losses (for slots and intent)



3_ in teria c'è già una funzione che converte le stringhe (nella eval) -> da usare (al momento lo sto ancora facendo "a mano")




Modifico l'output in console in modo che sia più carino
Magari printo anche dev e train



INFINE: metto le epoche al valore corretto (al momento sono 1 o 2 per debug)





-> I DATI sono un po' da rivedere. 
AGGIUNGO IL TRAIN NEL FILE CSV