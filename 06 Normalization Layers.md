[TOC]

Nelle Deep NN, l'aggiornamento dei pesi nei layer inferiori (più vicini all'input) modifica continuamente la distribuzione degli input (ovvero dei dati che il layer successivo riceve) per i layer successivi.

Questo fenomeno si chiama **Internal Covariate Shift**, **costringe la rete ad adattarsi continuamente a queste nuove distribuzioni**, rallentando la convergenza e l'addestramento.

Il problema può essere alleviato tramite diverse tecniche di normalizzazione.

# Batch Normalization (BN)

La **Batch Normalization (BN)** è un layer (o un metodo) di normalizzazione specifico per le reti neurali.

Gli input di una rete neurale vengono solitamente normalizzati in un range $[0,1]$ o $[-1,1]$, oppure con $mean=0$ e $variance=1$.

La BN applica essenzialmente un'operazione di ***Whitening*** (normalizzazione a media $0$ e varianza $1$) ai dati che fluiscono nei **layer intermedi** della rete.

## Perché l'Approccio Base non Funziona?

L'idea alla base sarebbe:

1. Allenare la rete su un batch.
2. Aggiornare i parametri.
3. Normalizzare gli output.

Questo metodo non funziona perché il processo di ottimizzazione (ovvero la discesa del gradiente) **non tiene conto** del fatto che la normalizzazione avverrà. Il flusso del gradiente viene di fatto interrotto.

Questo può portare a conseguenze indesiderate, come l'esplosione dei bias, senza che i parametri della distribuzione (media e varianza) cambino come dovrebbero.

## Soluzione: Aggiungere un Layer di Normalizzazione

![CNN Baum mit BN](./assets/CNN Baum mit BN.png)

Si inserisce un **nuovo layer di normalizzazione** direttamente nell'architettura della rete. In questo modo, il processo di normalizzazione diventa "visibile" all'algoritmo di backpropagation, e il gradiente può attraversarlo e aggiornare i pesi di conseguenza.

Per non limitare il potere espressivo della rete, questo layer introduce due parametri apprendibili per ogni attivazione:

- $\gamma$: un fattore di scala del valore normalizzato.
- $\beta$: un fattore di traslazione (bias) del valore normalizzato.

L'output del layer è dato da:
$$
y^{(k)} = \gamma^{(k)}\hat{x}^{(k)}+\beta^{(k)}
$$
Dove $\hat{x}^{(k)}$ è l'attivazione normalizzata.

### Riassunto dell'Algoritmo

1. Ogni feature viene normalizzata individualmente.
   + La normalizzazione avviene secondo lo standard z-score (media $0$, varianza $1$).
   + La media $E$ e la varianza $Var$ necessarie per la normalizzazione vengono calcolate su ogni singolo mini-batch.
2. Si aggiunge un nuovo layer, così il gradiente può "vedere" la normalizzazione e fare degli aggiustamenti se necessari.
   + Il nuovo layer di BN ha la capacità di imparare a **invertire la normalizzazione** (imparando la funzione identità) se questo risulta necessario per migliorare per l'apprendimento.

## I Parametri Apprendibili ($\gamma$ e $\beta$)

Se non ci fossero $\gamma$ e $\beta$, l'output di un BN layer sarebbe costretto ad avere sempre una distribuzione con media $0$ e varianza $1$. Questo limiterebbe drasticamente ciò che la rete può imparare.

$\gamma$ e $\beta$ permettono invece alla rete di produrre attivazioni con qualsiasi media e varianza, mantenendo così il suo **potere rappresentativo**.

È importante notare che $\gamma$ e $\beta$ non invertono la normalizzazione. Essendo parametri apprendibili sull'intero training set, sono molto **più stabili** della media e della varianza, che vengono ricalcolate per ogni mini-batch.

## Perché la Batch Normalization Funziona?

La BN era stata proposta per ridurre l'**Internal Covariate Shift**. Tuttavia, questa motivazione è stata messa in discussione e ritenuta **errata** da studi successivi. Secondo questa teoria, con la BN, ogni attivazione dei neuroni diventa più o meno una distribuzione gaussiana. Questo era considerato desiderabile perché i layer successivi non avrebbero più dovuto adattarsi a cambiamenti nel *tipo* di distribuzione in ingresso, ma solo a variazioni dei suoi parametri (come media e varianza), rendendo l'apprendimento più stabile.

In realtà, la BN **riduce gli effetti degli exploding e vanishing gradient**. Poiché la BN **forza le attivazioni** di ogni layer ad avere una distribuzione approssimativamente normale (con media vicina a $0$ e varianza a $1$), **previene** che i valori delle attivazioni diventino estremamente grandi o estremamente piccoli.

Senza la BN, si potrebbe innescare un effetto a catena: attivazioni molto basse in un layer potrebbero causare attivazioni ancora più basse nel layer successivo, portando alla "scomparsa" del gradiente durante la backpropagation. Al contrario, attivazioni molto alte potrebbero portare a un'esplosione dei valori. La BN interrompe questa catena, rendendo l'addestramento molto più stabile.

### Altri Benefici Pratici della Batch Normalization

+ **Addestramento più veloce**: Riduce i tempi di training, come conseguenza della mitigazione dei problemi con i gradienti.

+ **Effetto regolarizzante**: Diminuisce la necessità di altre tecniche di regolarizzazione come il Dropout o la L2 Norm.
  + Questo perché media e varianza calcolate su ogni mini-batch introducono una lieve forma di rumore, che aiuta il modello a generalizzare meglio e non a memorizzare i valori.

+ **Learning rate più alti**: Permette di usare learning rate più elevati in modo sicuro, accelerando ulteriormente la convergenza.
+ **Compatibilità con funzioni saturanti**: Rende possibile l'uso di funzioni di attivazione come la sigmoide anche in reti molto profonde, perché impedisce agli input di finire nelle zone di saturazione dove il gradiente è nullo.

### Posizioni Tipiche dei Layer di Normalizzazione

![image-20250610173824076](./assets/image-20250610173824076.png)

Il layer di normalizzazione è tipicamente inserito **dopo** il layer lineare o convoluzionale e **prima** della funzione di attivazione.

### Altre Operazioni di Normalizzazione

Esistono diverse varianti di normalizzazione (Layer Norm, Instance Norm, Group Norm), che si differenziano per le dimensioni lungo le quali vengono calcolate le statistiche di media e varianza.

![image-20250610173935257](./assets/image-20250610173935257.png)

+ $N$: Numero di immagini.
+ $H$, $W$: Altezza e larghezza delle immagini.
+ $C$: Canali di colore.

### Conclusioni

Sebbene la normalizzazione funzioni bene nella pratica, le ragioni dietro la sua efficacia sono ancora dibattute. In origine, la normalizzazione è stata proposta per ridurre l'internal covariate shift, ma alcuni studiosi hanno dimostrato che ciò è errato grazie agli esperimenti.

---

Tuttavia, la normalizzazione ha chiaramente una combinazione dei seguenti fattori:

- Le reti con layer di normalizzazione sono **più facili da ottimizzare**, consentendo l'uso di learning rate più grandi. La normalizzazione ha un effetto di ottimizzazione che accelera l'addestramento delle reti neurali.
- Le stime di media/deviazione standard sono rumorose a causa della casualità dei campioni nel batch. Questo "rumore" extra si traduce in una **migliore generalizzazione** in alcuni casi. **La normalizzazione ha un effetto di regolarizzazione**.
- La normalizzazione **riduce la sensibilità all'inizializzazione dei pesi**.

Di conseguenza, la normalizzazione permette di essere più "spensierati", combinando quasi tutti i blocchi di costruzione di una rete neurale e avendo una buona possibilità di addestrarla senza dover considerare quanto potrebbe essere mal condizionata.

