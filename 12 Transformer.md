# Transformer

[TOC]

## Attention is All You Need

### Uno Sguardo d'Insieme

![image-20250615194040469](./assets/image-20250615194040469.png)

Nell'esempio, il transformer prende in input una frase in una lingua e produce in output la sua traduzione in un'altra.

![image-20250615194202482](./assets/image-20250615194202482.png)

Ci sono due componenti principali:

+ Un blocco di **Encoder**, che processa la frase di input.

+ Un blocco di **Decoder**, che genera la frase di output.

---

![image-20250616114511140](./assets/image-20250616114511140.png)

Tutti gli encoder nella pila hanno la **stessa struttura** (ma non condividono i pesi), e sono composti da due strati:

+ Un layer di **Self-Attention**.
+ Un layer di **Feed Forward Neural Network**.

Gli input dell'encoder passano prima attraverso il layer di **self-attention**. Questo layer aiuta l'encoder a "guardare" le altre parole nella frase di input mentre sta codificando una parola specifica.

L'output del self-attention viene poi passato a una **rete neurale feed-forward**, che viene applicata indipendentemente a ogni posizione della sequenza.

---

Il decoder ha una struttura simile, con tre sotto-layer:

+ Un layer di **Self-Attention** (mascherato, come vedremo).
+ Un layer di **Encoder-Decoder Attention**, che permette al decoder di "porre attenzione" sulle parti più rilevanti della frase di input.
+ Un layer **Feed Forward Neural Network**.

### Introduzione ai Tensori e al Flusso dei Dati

Ogni parola di input viene trasformata in un **vettore numerico** (un **embedding**) tramite un apposito algoritmo. Nel Transformer, questi vettori hanno tipicamente una dimensione di 512.

![image-20250616114858739](./assets/image-20250616114858739.png)

L'embedding delle parole avviene solo nel **primo encoder** (quello più in basso). Tutti gli encoder ricevono una lista di vettori, ciascuna della dimensione di 512.

La lunghezza di questa lista (cioè il numero di parole) è un iperparametro, solitamente impostato sulla lunghezza della frase più lunga nel dataset.

![image-20250616115042697](./assets/image-20250616115042697.png)

Una proprietà chiave del Transformer è che ogni parola scorre attraverso il proprio percorso all'interno dell'encoder.

Il layer di **self-attention crea delle dipendenze** tra questi percorsi. Il layer **feed-forward, invece, non ha queste dipendenze**. Questo significa che i calcoli per ogni parola possono essere eseguiti **in parallelo**, rendendo i Transformer molto più veloci delle RNN, che devono processare le sequenze una parola alla volta.

## Self-Attention

L'obiettivo è **risolvere le ambiguità contestuali**. Nell'esempio "The animal didn't cross the street because **it** was too tired", un algoritmo deve capire a cosa si riferisce "it". Il self-attention è il meccanismo che permette al modello, mentre processa la parola "it", di **associarla** alla parola "animal", pesando la sua rilevanza rispetto alle altre parole della frase.

### Self-Attention nel Dettaglio

![image-20250616115622927](./assets/image-20250616115622927.png)

Da ogni embedding di input, si generano tre vettori:

+ **Query ($Q$)**: Le rappresentazioni delle parole correnti.
+ **Key ($K$)**: Le label per tutte le parole del segmento.
+ **Value ($V$)**: Le rappresentazioni vere e proprie delle parole, che vengono sommate per produrre l'output.

Questi tre vettori si ottengono moltiplicando l'embedding per tre matrici di pesi $W^Q$, $W^K$ e $W^V$, che vengono apprese durante il training.

---

![image-20250616152827138](./assets/image-20250616152827138.png)

1. Per la parola che stiamo processando (es. "Thinking"), calcoliamo uno score facendo il **prodotto scalare** tra il suo vettore Query ($q_1$) e il vettore Key di ogni parola della frase (incluso se stesso, $k_1$, $k_2$, ...).
   + Lo score determina **quanto bisogna focalizzarsi** su altre parti della frase in input.
2. Si dividono gli score per un fattore di scala (es. 8, la radice quadrata della dimensione dei vettori Key usata nel paper originale, 64) per **stabilizzare i gradienti** durante l'addestramento.
3. Si passa il risultato ad una operazione di softmax, con cui si **normalizzano** gli score in modo tale che siano tutti pesi positivi che si sommano a 1.
   + La softmax determina **quanto ogni parola verrà "espressa" in quella posizione**.
   + Ovviamente la parola corrente avrà il più alto softmax score, ma a volte è utile riferirsi ad un'altra parola rilevante rispetto a quella corrente.
4. Si moltiplica ogni vettore value $v_i$ per il softmax score.
   + L'obiettivo è **mantenere intatti i valori delle parole su cui siamo concentrati**, e lasciare indietro le parole meno rilevanti (moltiplicandole per numeri molti piccoli, come 0.001).
5. Si sommano i vettori value pesati per produrre un output $z_i$ per la parola corrente.

Il vettore risultante si può mandare attraverso la feed-forward neural network.

### Calcolo Matriciale

Nell'implementazione effettiva, questo calcolo è svolto in forma matriciale per essere più veloce.

1. Si calcolano le matrici Query, Key e Value.

   + Lo si fa impacchettando gli embedding in una matrice $X$, e moltiplicandola per le matrici dei pesi che abbiamo allenato ($W^Q$, $W^K$, $W^V$).
   + Ogni riga di $X$ corrisponde ad una parola nella frase di input

   ![image-20250617114128069](./assets/image-20250617114128069.png)

2. Dato che il calcolo è matriciale, si possono condensare tutti i passi in una sola formula per calcolare gli output del layer di self-attention:
   $$
   Attention(Q, K, V) = Z = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V 
   $$
   ![image-20250617114054010](./assets/image-20250617114054010.png)

## Multi-Headed Attention

Il multi-headed attention migliora le performance dell'attention in due modi:

+ **Espande la capacità del modello di focalizzarsi su diverse posizioni.**  Un singolo meccanismo di attention potrebbe essere dominato dalla relazione di una parola con se stessa. Il multi-headed attention permette al modello di **guardare a diverse parti della frase contemporaneamente** per catturare relazioni più ricche
+ **Fornisce al layer di attention molteplici "sottospazi di rappresentazione".**  Invece di avere un solo set di matrici di pesi ($W^Q$, $W^K$, $W^V$), se ne usano molteplici (il paper originale ne usa 8, uno per ogni "testa"). 
  + Ogni set di matrici viene inizializzato casualmente e, dopo l'addestramento, proietta gli embedding di input in un diverso sottospazio di rappresentazione, permettendo al modello di cogliere diversi aspetti delle relazioni tra le parole.
  + Per ogni head, si mantengono matrici dei pesi separate.


---

![image-20250617115124013](./assets/image-20250617115124013.png)

Come per l'attention semplice, per ottenere le matrici $Q$, $K$ e $V$ per una specifica head, si moltiplica la matrice degli embedding $X$ per il set di pesi di quella testa (es. $W_0^Q$, $W_0^K$, $W_0^V$).

![image-20250617115711630](./assets/image-20250617115711630.png)

Il calcolo del self-attention che abbiamo visto prima viene ora eseguito **8 volte diverse, in parallelo**, una per ogni head e con le sue specifiche matrici di pesi. Il risultato di questo processo non è una singola matrice di output $Z$, ma **otto diverse matrici $Z$** ($Z_0$, $Z_1$, $\dots$, $Z_7$), una per ogni head di attention. 

![image-20250617120307389](./assets/image-20250617120307389.png)

C'è però un problema: il layer successivo (la rete feed-forward) si aspetta di ricevere in input una **singola matrice** (un vettore per ogni parola), non otto. È quindi necessario trovare un modo per **condensare queste otto matrici in una sola**:

1. **Concatenazione**: Le otto matrici $Z_i$ vengono concatenate una di seguito all'altra per formare un'unica grande matrice.
2. **Proiezione**: Questa grande matrice viene poi moltiplicata per un'ulteriore matrice di pesi, $W^O$, che viene anch'essa appresa durante l'addestramento.
3. **Output Finale**: Il risultato di questa moltiplicazione è la matrice finale $Z$, che cattura l'informazione da tutte le teste di attention in un'unica rappresentazione, pronta per essere passata al layer feed-forward.

### Schema Riassuntivo del Multi-Headed Attention

![image-20250617120540571](./assets/image-20250617120540571.png)

## Positional Encoding

Il meccanismo di self-attention, di per sé, non ha modo di considerare l'**ordine delle parole**; tratta la frase come una bag of words; ma per un modello linguistico l'ordine è cruciale.

![image-20250617121545737](./assets/image-20250617121545737.png)

Per risolvere questo problema, il Transformer **aggiunge un vettore a ogni embedding di input**. Questi vettori, chiamati **positional encoding**, seguono un pattern specifico che aiuta il modello a determinare la posizione di ogni parola e la distanza relativa tra le parole nella sequenza. 

![image-20250617121637534](./assets/image-20250617121637534.png)

L'intuizione è che, una volta sommati agli embedding, questi valori forniscono distanze significative che il modello può sfruttare durante il calcolo dell'attenzione.

## Residual

![image-20250617122240967](./assets/image-20250617122240967.png)

In ogni encoder e decoder, ogni sub-layer ha una **connessione residua** che lo "circonda", seguita da un passo di **layer-normalization**. La residual connection connette l'output di un layer convoluzionale precedente all'input di un futuro layer convoluzionale, diversi layer dopo.

![image-20250617122546828](./assets/image-20250617122546828.png)

l'input del sub-layer ($X$) viene sommato al suo output ($Z$), e il risultato viene poi normalizzato: $LayerNorm(X + Z)$.

L'esempio seguente mostra un'architettura completa con 2 encoder e 2 decoder:

![image-20250617122912023](./assets/image-20250617122912023-1750156153476-1.png)

## Decoder

![image-20250617123628735](./assets/image-20250617123628735.png)

L'encoder inizia a processare la sequenza di input. L'output dell'encoder in cima viene trasformato in un set di vettori di attention **Key ($K$) e Value ($V$)**. 

Questi vettori verranno usati dal layer "encoder-decoder attention" in ogni decoder per aiutare il decoder a concentrarsi sulle parti rilevanti della sequenza di input. La generazione inizia con un token speciale di inizio (es. `<start>`).

![image-20250617123704167](./assets/image-20250617123704167.png)

Il processo si ripete iterativamente: l'output di ogni passo viene reinserito come input per il passo successivo del decoder seguente. Questo continua finché non viene generato un token speciale di fine sequenza.

Anche gli input del decoder vengono sommati a un positional encoding per mantenere l'informazione sull'ordine.

### Self-Attention e Encoder-Decoder Attention nel Decoder

Ci sono due differenze nei layer di attention nel decoder:

+ **Masked Self-Attention**: Il layer di self-attention nel decoder è autorizzato a "porre attenzione" solo alle **posizioni precedenti** nella sequenza di output. 
  + Questo si ottiene mascherando le posizioni future (impostandole a $-\inf$) prima del passo di softmax, per impedire al modello di "sbirciare" le parole che deve ancora predire.

+ **Encoder-Decoder Attention**: Questo layer funziona come un multi-head self-attention standard, ma con una differenza cruciale: crea la sua matrice di **Query ($Q$) dal layer sottostante del decoder**, mentre prende le matrici di **Key ($K$) e Value ($V$) dall'output finale del blocco degli encoder**. È questo il meccanismo che permette al decoder di "guardare" la frase di input.

### Layer Lineare Finale e la Softmax

![image-20250617124400210](./assets/image-20250617124400210.png)

L'output dello stack di decoder è un vettore di float. Per trasformarlo in una parola, si usano due layer finali:

+ Un **Layer Lineare**: una rete neurale fully-connected che proietta questo vettore in un vettore molto più grande chiamato **logits vector**.
  + La dimensione di questo vettore è pari alla dimensione del vocabolario del modello (es. 10.000 parole). Ogni cella del vettore corrisponde allo score per una parola unica.

+ Un **Layer Softmax**: trasforma questi score in probabilità (tutte positive e la cui somma è 1).

La parola finale viene scelta selezionando la cella con la probabilità più alta (operazione di $argmax$).

## Allenare un Transformer

![image-20250623115058994](./assets/image-20250623115058994.png)

Durante l'addestramento, si usa un dataset etichettato per confrontare l'output del modello con quello corretto. Il vocabolario di output nell'esempio contiene sei parole, compreso il tag di fine.

![image-20250623115553872](./assets/image-20250623115553872.png)

L'output desiderato è una distribuzione di probabilità che assegni il 100% della probabilità alla parola corretta (una rappresentazione **one-hot**). Un modello non addestrato produrrà invece una distribuzione casuale.

![image-20250623115618816](./assets/image-20250623115618816.png)

Supponiamo di essere al primo step della fase di training. Vogliamo che l'output sia una distribuzione di probabilità che indichi la parola corretta. Ma dato che il modello non è ancora allenato, è difficile che accada.

![image-20250623115955483](./assets/image-20250623115955483.png)

![image-20250623120010500](./assets/image-20250623120010500.png)

Per una frase intera, il modello viene addestrato a produrre una sequenza di distribuzioni di probabilità (dove ogni distribuzione è rappresentata da un vettore con la stessa dimensione del vocabolario.), dove ogni distribuzione (per ogni posizione) assegna la probabilità più alta alla parola corretta in quella posizione, fino al token di fine frase `<eos>`.

## BERT (Bidirectional Encoder  Representations from Transformers)

BERT viene utilizzato in applicazioni complesse come il motore di ricerca di Google per comprendere meglio le query degli utenti e svolgere task come:

- Ricerca per similarità di testo.
- Riassunto (Summarization).
- Risposta a domande (Question Answering).

### Come Possiamo Usare BERT?

#### Esempio - Classificazione di Frasi

![image-20250617155856972](./assets/image-20250617155856972.png)

Un input (es. il testo di un'email) viene dato a un modello **BERT pre-addestrato**.

L'output di BERT viene passato a un **classificatore semplice** (es. una rete feed-forward con un layer softmax).

L'intero sistema (BERT + classificatore) viene addestrato sul dataset specifico (es. email di spam) per produrre la predizione finale.

### Architettura del Modello BERT

![image-20250623121826082](./assets/image-20250623121826082.png)

BERT è composto esclusivamente da una pila di **layer di Encoder** del Transformer.

Esistono due versioni principali:

- **BERT Base**: 12 layer di Encoder, 768 unità nascoste, 12 head di attention.
- **BERT Large**: 24 layer di Encoder, 1024 unità nascoste, 16 head di attention.

#### Input e Output del Modello

![image-20250623122530257](./assets/image-20250623122530257.png)

La sequenza di parole in input viene preceduta da un token speciale: **`[CLS]`** (che sta per *Classification*). Questa sequenza completa in input fluisce poi verso l'alto attraverso tutti i layer dell'encoder, proprio come gli encoder di un transformer. Ogni layer applica la self-attention, e passa i risultati attraverso una rete neurale feed-forward, per poi passare al prossimo encoder.

![image-20250617161043994](./assets/image-20250617161043994.png)

BERT produce un vettore di output per ogni token di input. Per i task di classificazione dell'intera frase, si ignora l'output di tutti i token tranne quello corrispondente al primo, il token `[CLS]`. Si assume che questo vettore contenga una **rappresentazione aggregata dell'intera sequenza**.

![image-20250617161131748](./assets/image-20250617161131748.png)

Questo singolo vettore di output (`[CLS]`) viene quindi usato come input per il classificatore finale per ottenere la predizione.

Nel caso di una classificazione multipla, basta avere più neuroni di output, per poi passare attraverso una softmax.

### Primo Task di Pre-Training - Masked Language Model (MLM)

BERT adotta un concetto chiamato **masked language model** per allenare gli encoder.

![image-20250617161759198](./assets/image-20250617161759198.png)

Durante il pre-training, il 15% dei token di input viene mascherato casualmente. L'obiettivo del modello è **predire quali erano i token originali mascherati**.

Per predire la parola mancante, il modello è costretto a usare il contesto sia a **destra** che a **sinistra** del token mascherato. Questo lo rende un modello genuinamente **bidirezionale**.

Dei token selezionati, l'80% viene sostituito con `[MASK]`, il 10% con una parola casuale e il 10% viene lasciato invariato.

### Secondo Task di Pre-training - Next Sentence Prediction (NSP)

BERT viene pre-addestrato anche su un secondo task per comprendere le relazioni tra frasi.

![image-20250617162315978](./assets/image-20250617162315978.png)

Al modello vengono fornite due frasi, A e B. Deve predire se la frase B è quella che effettivamente segue la frase A nel testo originale, oppure se è una frase casuale.

Le due frasi vengono separate da un token speciale `[SEP]`.

La predizione "IsNext" o "NotNext" viene fatta usando l'output del token `[CLS]`.

### BERT per Feature Extraction

![image-20250623123356586](./assets/image-20250623123356586.png)

Invece di addestrare l'intero modello facendo fine-tuning, si può usare un **BERT pre-addestrato** solo per generare **embedding contestualizzati** delle parole. Questi embedding, che sono vettori molto ricchi di informazione, possono poi essere dati in pasto a un altro modello specifico per il proprio task.

![image-20250623123424553](./assets/image-20250623123424553.png)

Quale output dei vari layer di BERT funziona meglio come embedding? Basandoci su un esperimento per un task di Named Entity Recognition, la performance migliore non si ottiene da un singolo layer, ma **concatenando gli output degli ultimi quattro layer nascosti**, suggerendo che layer diversi catturano livelli di informazione semantica differenti e complementari.

## GPT (Generative Pretrained Transformer)

Rispetto a BERT, che usa blocchi di **encoding** dei transformer, GPT è costruito usando i blocchi di **decoding**. Una conseguenza funzionale è che GPT genera l'output **un token alla volta**.

### GPT in Funzione (Autoregressione)

![image-20250617164129729](./assets/image-20250617164129729.png)

Dopo che il modello produce un token, quel token viene aggiunto alla sequenza di input. Questa nuova sequenza, più lunga, diventa l'input per il modello al passo successivo (**autoregressione**). Questo processo iterativo permette di generare testo coerente.

#### Perché Esistono Modelli Non-Autoregressivi?

BERT, non essendo autoregressivo, ha guadagnato la capacità di incorporare il contesto da **entrambi i lati di una parola** (bidirezionalità), ottenendo risultati migliori nei task di comprensione del linguaggio.

GPT, essendo autoregressivo, è intrinsecamente **unidirezionale** (guarda solo al passato).

### Decoder e Masked Self-Attention

![image-20250617164802626](./assets/image-20250617164802626.png)

A differenza del self-attention standard dell'encoder, questa versione "masked" impedisce al modello di "sbirciare" i token futuri. Lo fa bloccando le informazioni provenienti dai token che si trovano a destra della posizione che si sta calcolando. Tuttavia, non cambia la parola in `[mask]`, come fa BERT, ma blocca direttamente l'informazione dai token alla destra della posizione che sta venendo calcolata in quel momento.

![image-20250623123826260](./assets/image-20250623123826260.png)

### Blocco Decoder-Only

![image-20250617165013901](./assets/image-20250617165013901.png)

Dopo il paper originale sui Transformer, altre ricerche hanno proposto di usare un'architettura composta **esclusivamente da una pila di blocchi Decoder**. Questo modello, chiamato "**Transformer-Decoder**", è l'architettura su cui si basa la famiglia di modelli GPT.

### Come GPT Genera Testo

![image-20250617165122399](./assets/image-20250617165122399.png)

Per una generazione non condizionata (senza un prompt), si può semplicemente dare al modello un token di inizio (es. `<s>`).

Il modello processa questo singolo token attraverso tutti i suoi layer. L'output finale (il vettore prodotto) viene usato per calcolare una probabilità per ogni parola del vocabolario. Si può selezionare la parola con la probabilità più alta (es. "The") o campionare dalla distribuzione per avere più varietà, usando parametri come `top-k` (per esempio quando la nostra applicazione della tastiera ci suggerisce altre parole successive, oltre alla più probabile).

Il nuovo token generato ("The") viene aggiunto all'input.

![image-20250623124127923](./assets/image-20250623124127923.png)

È importante notare che GPT **non ricalcola l'interpretazione dei token precedenti**, ma usa le informazioni già elaborate per processare il nuovo token.

### Input Encoding

![image-20250617172307666](./assets/image-20250617172307666.png)

Il modello ha una grande matrice di embedding, dove ogni riga è il vettore che rappresenta un token del vocabolario.

#### Positional Encoding

![image-20250617172615554](./assets/image-20250617172615554.png)

Il modello ha anche una matrice di positional encoding da incorporare agli embedding, dove ogni riga è un vettore che codifica una specifica posizione della parola nella sequenza.

#### Input Encoding

![image-20250617173038021](./assets/image-20250617173038021.png)

Per processare un token in una certa posizione, il modello **somma l'embedding del token al vettore di positional encoding** per quella posizione.

#### Flusso nei Blocchi

![image-20250617173127444](./assets/image-20250617173127444.png)

Questo vettore combinato viene quindi passato attraverso la pila di blocchi decoder.

Il processo si ripete per ogni blocco, ma **ogni blocco ha i propri pesi** per il self-attention e la rete feed-forward.

### Self-Attention per GPT

"*A robot must obey the orders given **it** by human beings except where such orders would conflict with the First Law*". Per comprendere appieno questa frase, un modello deve capire a cosa si riferiscono pronomi e riferimenti come "**it**", "**such orders**" e "**The First Law**". Per un essere umano è semplice, ma per un algoritmo no.

Questo è esattamente il compito del self-attention. Mentre il modello processa una parola, questo meccanismo gli permette di "guardare" le altre parole nella frase per capire il contesto e risolvere queste ambiguità.

Il self-attention raggiunge questo obiettivo **assegnando degli score** che indicano quanto ogni parola della frase è rilevante per la parola corrente, per poi **sommare le rappresentazioni vettoriali** di queste parole in modo pesato. In questo caso, impara ad associare "it" a "robot". 

#### Esempio di Self-Attention per GPT

![image-20250617173904293](./assets/image-20250617173904293.png)

Prendiamo in considerazione questo esempio. Il modello sta processando la nona parola della sequenza, "it". Le percentuali mostrate sopra le parole precedenti (es. 30% su "a", 50% su "robot", 18% su "it" stesso) rappresentano i **pesi di attention** calcolati. Questi sono il risultato della funzione softmax e indicano "quanta attenzione" la parola "it" deve prestare a ciascuna delle altre parole. L'attenzione è alta su "a robot".

Questo significa che, per creare la nuova rappresentazione vettoriale di "it", il modello prenderà una porzione significativa (il 50%) della rappresentazione di "robot". Il vettore che verrà passato al layer successivo (la rete feed-forward) sarà una **somma dei vettori  Value di ogni parola, pesati per i loro rispettivi score di attention**. 

![image-20250617174121719](./assets/image-20250617174121719.png)

Questa è la fase finale, in cui si sommano i vettori *Value* pesati per gli score di attenzione.

**Colonne della Tabella**:

- `Word`: Le parole precedenti nella sequenza.
- `Value vector`: Il vettore *Value* associato a ogni parola.
- `Score`: Lo score di attention (il peso calcolato dalla softmax) per la parola corrente "it" rispetto a ogni parola precedente.
- `Value X Score`: Il risultato della moltiplicazione di ogni vettore Value per il suo score. Questo "smorza" i vettori delle parole meno importanti e "amplifica" quelli delle parole più rilevanti.

La somma di tutti i vettori nella colonna `Value X Score` produce il **vettore di output finale** del layer di self-attention per la posizione #9. Questo vettore è una nuova rappresentazione della parola "it", arricchita con le informazioni contestuali dalle altre parole a cui ha "prestato attenzione". 

### Generazione dell'Output del Modello

![image-20250617174438384](./assets/image-20250617174438384.png)

L'output vettoriale prodotto dal blocco decoder in cima alla pila viene moltiplicato per la **matrice di embedding dei token**.

![image-20250617174554079](./assets/image-20250617174554079.png)

Il risultato di questa moltiplicazione è un vettore di **logits**, con una dimensione pari a quella del vocabolario. Ogni logit è lo score per una specifica parola.

![image-20250617174627691](./assets/image-20250617174627691.png)

Questi logits possono essere trasformati in probabilità (tramite softmax). Si può quindi scegliere la parola con lo score più alto (`top_k=1`) o, per risultati migliori e più vari, si può **campionare** una parola dalla distribuzione di probabilità, magari limitandosi alle `top_k` parole più probabili (es. 40).

### Chiarificazioni

+ **Token vs. Parole**: I termini "parola" e "token" sono stati usati in modo interscambiabile. In realtà, GPT usa un algoritmo chiamato **Byte Pair Encoding (BPE)** per creare i token, che spesso sono parti di parole (sotto-parole).
+ **Inferenza vs. Addestramento**: Il processo di generazione mostrato finora (un token alla volta) è quello che avviene in fase di **inferenza** o valutazione. Durante l'**addestramento**, il modello processa sequenze di testo più lunghe e più token contemporaneamente, con batch di dimensioni maggiori.
+ **Layer Normalization**: Per semplicità, i diagrammi hanno omesso i layer di **Layer Normalization**, che però sono una parte cruciale dell'architettura (i blocchi "Add & Norm").

### Masked Self-Attention in Dettaglio

![image-20250617174934617](./assets/image-20250617174934617.png)

La masked self-attention è identica a quella standard, tranne che nel calcolo degli score. Il suo scopo è **interferire con il calcolo degli score** per impedire al modello di vedere i token futuri.

![image-20250617175853980](./assets/image-20250617175853980.png)

Si considera la sequenza "robot must obey orders". Il modello viene addestrato a predire la parola successiva in ogni passo. Questo significa che il dataset viene implicitamente strutturato come segue:

- Per predire "must", il modello vede solo "robot".
- Per predire "obey", il modello vede "robot must".
- Per predire "orders", il modello vede "robot must obey".

---

Per forzare il modello a comportarsi in questo modo (cioè, a non "barare" guardando le parole future che sta cercando di predire), è necessario un meccanismo che nasconda le informazioni future durante il calcolo del self-attention. Questo meccanismo è l'**attention mask**.

![image-20250617180137888](./assets/image-20250617180137888.png)

Si calcola una matrice di **Score** moltiplicando la matrice delle **Queries** per la matrice delle **Keys** (trasposta). Ogni cella $(i, j)$ di questa matrice contiene lo score (prodotto scalare) tra la parola in posizione $i$ (la query) e la parola in posizione $j$ (la key).

In questa fase, ogni parola sta ancora "guardando" a tutte le altre parole, incluse quelle future.

![image-20250617180321879](./assets/image-20250617180321879.png)

Dopo aver calcolato la matrice degli score, si applica una **attention mask**.  maschera imposta a $-\inf$ (o un numero negativo molto grande) tutti i valori nella matrice degli score che corrispondono a posizioni future. Tutti i valori nella parte superiore della diagonale della matrice degli score vengono sostituiti con $-\inf$. Questo significa, ad esempio, che la parola alla posizione 2 non potrà avere uno score valido per le parole alle posizioni 3 e 4.

![image-20250617181134585](./assets/image-20250617181134585.png)

La funzione softmax viene applicata a ogni riga della matrice degli score mascherati. La funzione esponenziale $e$ calcolata su -infinito dà come risultato $0$ ($e^{−∞}=0$). La matrice degli score finale (dopo la softmax) avrà dei **valori pari a 0** in tutte le posizioni che erano state mascherate. Questo garantisce che, per ogni parola, **i pesi di attention siano distribuiti solo tra la parola stessa e le parole che la precedono nella sequenza**. Il modello è così costretto a operare in modo causale, basando la sua predizione solo sulle informazioni passate, che è esattamente ciò che serve per un compito di generazione di testo.

### GPT Self-Attention

#### Problema dell'Efficienza nella Masked Self-Attention

![image-20250617181433471](./assets/image-20250617181433471.png)

Durante la generazione, il modello aggiunge una parola alla volta. Sarebbe estremamente **inefficiente ricalcolare da capo l'intera self-attention** per tutta la sequenza a ogni nuovo token.

Per evitare questo, GPT **mantiene in memoria (mette in cache) i vettori Key ($K$) e Value ($V$)** per tutti i token che ha già processato.

Ogni layer di self-attention all'interno della pila di decoder memorizza i propri vettori K e V per ogni token precedente.

#### Passaggio 1 (Creazione di $Q$, $K$, $V$ per un nuovo token)

![image-20250617181522512](./assets/image-20250617181522512.png)

Si sta processando un nuovo token (es. la parola "it" in posizione #9). Il suo vettore di input è dato dalla somma del suo embedding e del suo positional encoding.

![image-20250617181611030](./assets/image-20250617181611030.png)

Questo vettore di input viene moltiplicato per le matrici dei pesi del blocco corrente per generare i vettori Query, Key e Value per il token "it".

![image-20250617181634311](./assets/image-20250617181634311.png)

Il risultato di questa moltiplicazione è un unico grande vettore che viene poi suddiviso per ottenere i vettori $q_9$, $k_9$, e $v_9$.

#### Passaggio 1.5 (Suddivisione Multi-Head Attention)

![image-20250617181814987](./assets/image-20250617181814987.png)

I vettori $Q$, $K$, e $V$ appena calcolati non vengono usati così come sono. Vengono **suddivisi e rimodellati** in matrici più piccole, una per ogni head di attention. Ad esempio, nel GPT-2 small ce ne sono 12, quindi i vettori vengono divisi in 12 "sotto-vettori".

![image-20250617181923031](./assets/image-20250617181923031.png)

Il calcolo del self-attention che vedremo ora avviene in parallelo per ognuna di queste 12 teste.

#### Passaggio 2 (Calcolo Score per una Singola Head)

![image-20250617182032837](./assets/image-20250617182032837.png)

![image-20250617182108774](./assets/image-20250617182108774.png)

Il nuovo vettore Query $q_9$ (della parola "it") viene usato per calcolare uno score contro i vettori **Key di tutti i token precedenti ($k_1, \dots , k_8$)**, che erano stati salvati in cache, più il suo stesso vettore Key $k_9$.

#### Passaggio 3 (Calcolo Somma per una Singola Head)

![image-20250617182428317](./assets/image-20250617182428317.png)

Gli score ottenuti vengono usati per calcolare una **somma pesata dei vettori Value** di tutti i token precedenti ($v_1, \dots v_8$), più il suo stesso vettore Value $v_9$. Il risultato è l'output del self-attention per questa specifica testa di attenzione.

#### Passaggio 3.5 (Unione delle Teste)

![image-20250617182522720](./assets/image-20250617182522720.png)

I vettori di output prodotti da tutte le attention head (12 nel caso di GPT-2 small) vengono **concatenati** per formare un unico grande vettore.

#### Passaggio 4 (Proiezione Finale)

![image-20250617182620913](./assets/image-20250617182620913.png)

Questo grande vettore viene poi moltiplicato per una **matrice di pesi di proiezione** ($W^O$). 

![image-20250617182727418](./assets/image-20250617182727418.png)

Il risultato di questa operazione è il vettore di output finale del sub-layer di self-attention, pronto per essere passato al layer successivo.

### GPT Fully-Connected Neural Network

#### Layer 1 (Espansione)

![image-20250617183131441](./assets/image-20250617183131441.png)

Il vettore in uscita dal self-attention entra nel primo layer della rete feed-forward. Questo layer **espande la dimensionalità** del vettore, tipicamente di un fattore 4 (es. da 768 a 3072 unità). Questo aumento di dimensionalità dà al modello maggiore capacità rappresentativa per elaborare l'informazione.

#### Layer 2 (Proiezione)

![image-20250617183154955](./assets/image-20250617183154955.png)

Il secondo layer ri-proietta il risultato alla dimensione originale del modello (es. da 3072 di nuovo a 768). L'output di questa operazione è il risultato finale dell'intero blocco Transformer per quel token.

### Riepilogo dei Pesi del Modello

![image-20250617183241380](./assets/image-20250617183241380.png)

Sono riassunte tutte le matrici di pesi che un vettore incontra passando attraverso **un singolo blocco** Transformer (pesi per $Q$/$K$/$V$, pesi per la proiezione dell'attention, pesi per i due layer feed-forward).

---

![image-20250617183308693](./assets/image-20250617183308693.png)

Un punto fondamentale è che **ogni blocco ha il suo set di pesi unico e indipendente**. Al contrario, il modello ha solo **una matrice di embedding dei token** e **una matrice di positional encoding**, che sono condivise e usate da tutti i blocchi.

## GPT Oltre il Language Modeling

### Traduzione Automatica

![image-20250617183520632](./assets/image-20250617183520632.png)

Un encoder **non è strettamente necessario** per la traduzione.

Lo stesso compito può essere affrontato da un transformer decoder-only, a patto di formattare l'input e l'output come un'unica sequenza. Ad esempio, si può dare al modello un prompt come: `Traduci dal francese all'inglese: Je suis étudiant >>> I am a student` e addestrarlo a completare questo schema.

### Riassunto Automatico

![image-20250617183605796](./assets/image-20250617183605796.png)

Questo è stato il task su cui è stato addestrato il primo transformer solo-decoder. Il modello è stato addestrato a **leggere un articolo di Wikipedia** (privo della sua sezione introduttiva) e a **generare quella sezione introduttiva**, che funge di fatto da riassunto.

### Transfer Learning

Un transformer decoder-only, prima **pre-addestrato** su un task generale di modellazione del linguaggio e poi **fine-tuned** sul compito specifico del riassunto, può ottenere risultati migliori di un'architettura encoder-decoder pre-addestrata, specialmente in contesti con pochi dati etichettati.

### Generazione di Musica

Il modello "Music Transformer" usa un'architettura solo-decoder per generare musica con tempismo e dinamiche espressive.

Il compito viene trattato esattamente come la modellazione del linguaggio: il modello impara la musica in modo non supervisionato e poi la genera campionando dalla distribuzione appresa.

Per dare la musica in pasto alla rete, bisogna rappresentarla numericamente. Oltre alle **note**, è necessario rappresentare anche la **velocity** (la forza con cui un tasto del pianoforte viene premuto) e il tempo.

Una performance musicale può essere convertita in una **sequenza di eventi** (es. nota on, nota off, cambio di velocity, avanzamento del tempo), e ogni evento può essere rappresentato da un **vettore one-hot**. Questa sequenza di vettori diventa l'input per il modello.