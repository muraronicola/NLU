controllo che i valori ottenuti con il nuovo codice siano comparabili a quelli ottenuti con il codice "scritto male da me" e quelli messi sui reporttt (soprattitto in 3_1, sito che l'evaluation è diversa)
controllo che i nomi delle variabili/metodi abbiano un senso


TOLGO i file txt.tmp (in ./bin/)
TOLGO i file dei latex/report (in _tmp_report)


in entrambi 2: forse ho invertito in evaluate e la print l'ordine tra i risultati di slot and intent


Modifico l'output in console in modo che sia più carino



INFINE: metto le epoche al valore corretto (al momento sono 1 o 2 per debug)



3-12:8:
->Forse avsGD non viene triggerato in Ass1_2; da controllare (la run singola fuziona, da controllare se con anche le prime due abilitate il coso va.)

2_2: c'era un combined_loss = loss_intent+ loss_slot, in eval. Doveve assere combined_loss = loss_intent*0.1 + loss_slot*0.9



3_1: Potremmo sistemare un po' il codice quando chiamo il modello e creo i dizionari (sia qui che in 2_2)
3_1: C'è un try catch cattivissimo nell'eval loop; i dati tra paper e modello credo siano diversi (il result suul test set)

IN tutte controllo il load/save del modello; aggiungo la cartella ./bin/ per il salvataggio dei modelli e metto il best modello eva