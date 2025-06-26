# Recurrent Neural Networks (RNN)

[TOC]

## Modellazione delle Sequenze

Prendiamo in considerazione un esempio visivo: per predire dove andrà una palla, una singola immagine non basta; è necessaria una **sequenza di immagini** per capirne il movimento.

Anche un testo si può interpretare come una sequenza. Un altro esempio che si può prendere in considerazione è la predizione della parola successiva in una frase. L'obiettivo è, date le parole precedenti, prevedere la parola seguente.

### Primi Approcci e Loro Problemi

#### Finestra Fissa

![image-20250621184218025](./assets/image-20250621184218025.png)

Il primo approccio consiste nell'usare una **finestra di dimensione fissa** (es. le ultime due parole) per predire la parola successiva.

Le parole vengono rappresentate numericamente tramite **one-hot encoding**.

Questo approccio però **non può modellare dipendenze a lungo termine**. Nell'esempio "France is where I grew up, but I now live in Boston. I speak fluent ___", l'informazione "France" è cruciale per predire "French", ma si trova troppo indietro nel passato per essere catturata da una finestra piccola.

#### Bag of Words

![image-20250621184351745](./assets/image-20250621184351745.png)

Un'alternativa è usare un modello **Bag of Words**, che rappresenta la frase contando le parole presenti ma ignorandone l'ordine.

Il problema è che **l'ordine delle parole è fondamentale**. Le frasi "The food was good, not bad at all" e "The food was bad, not good at all" hanno lo stesso Bag of Words ma significato opposto. Questo metodo **perde completamente l'informazione sintattica**.

#### Finestra Fissa Grande

![image-20250621184502744](./assets/image-20250621184502744.png)

Si potrebbe usare una finestra fissa molto grande.

Il problema qui è la **mancanza di condivisione dei parametri (parameter sharing)**. Ogni posizione nella finestra ha un suo set di pesi separato. Di conseguenza, ciò che il modello impara su una parola in una certa posizione **non si trasferisce** se la stessa parola appare in un'altra posizione.

## Neuroni con Ricorrenza

![image-20250621184732865](./assets/image-20250621184732865.png)

Le reti feed-forward elaborano ogni input $x_t$ in modo indipendente per produrre un output:
$$
\hat{y_t} = f(x_t)
$$
![image-20250621184838925](./assets/image-20250621184838925.png)

Con la **ricorrenza**, l'output non dipende solo dall'input corrente, ma anche da una **memoria del passato**, chiamata **stato nascosto (hidden state)**, indicato con $h_t$. La formula diventa:
$$
\hat{y_t}=f(x_t,h_{t−1})
$$
Questo stato nascosto viene passato da un passo temporale all'altro attraverso una ricorrenza, permettendo alla rete di mantenere informazioni sugli input precedenti.

## Recurrent Neural Network (RNN)

Si definisce una RNN come una rete che applica una **relazione di ricorrenza** a ogni passo temporale per processare una sequenza. Lo stato $h_t$ è definito come segue:
$$
h_t=f_W(x_t,h_{t−1})
$$
Un punto fondamentale è che la **stessa funzione $f$ e lo stesso set di parametri $W$ sono usati a ogni passo temporale**. Questa è la condivisione dei parametri che mancava nell'idea della finestra fissa grande.

### Intuizione

Un ciclo `for` elabora ogni parola di una frase, aggiornando lo stato nascosto a ogni iterazione per predire la parola successiva.

```python
my_rnn = RNN()
hidden_state = [0, 0, 0, 0]

sentence = ["I", "love", "recurrent", "neural"]

for word in sentence:
	prediction, hidden_state = my_rnn(word, hidden_state)

next_word_prediction = prediction
# >>> "networks!"
```

### Aggiornamento dello Stato Nascosto e Calcolo dell'Output

L'aggiornamento dello **stato nascosto corrente** avviene in questo modo:
$$
h_t = \tanh(W^T_{hh}h_{t-1} + W^T_{xh}x_t)
$$
Lo stato nascosto al tempo $t$ è una funzione non lineare $\tanh$ dello stato nascosto precedente $h_{t-1}$ e dell'input corrente $x_t$.

L'output al tempo $t$ è una funzione dello stato nascosto corrente:
$$
\hat{y_t} = W^T_{hy}h_t
$$

### Il Grafo Computazionale delle RNN nel Tempo

Il diagramma di una RNN viene spesso mostrato in forma compatta, con un "cappio di ricorrenza" che parte dal layer nascosto e torna su se stesso. Questo cappio rappresenta l'idea di memoria: **l'output di un istante di tempo viene riutilizzato come input per l'istante successivo**.

![image-20250621190704496](./assets/image-20250621190704496.png)

Per poter effettivamente eseguire i calcoli e, soprattutto, per addestrare la rete tramite backpropagation, dobbiamo visualizzare questo cappio in modo esplicito. Lo facciamo "srotolando" la rete nel tempo. Questo significa creare una copia della cella RNN **per ogni elemento della sequenza** che stiamo processando.

Per ogni passo temporale ($t-1$, $t$, $t+1$) c'è una copia della struttura della rete. Lo stato $s_t$ dipende dall'input $x_t$ e dallo stato precedente $s_{t-1}$.

Come indicato nel diagramma, le matrici dei pesi $U$, $V$ e $W$ **sono le stesse per ogni passo temporale**. La rete non impara un nuovo set di pesi per ogni parola della frase; usa sempre lo stesso set. Questo rende le RNN estremamente efficienti in termini di parametri e permette loro di generalizzare la conoscenza appresa su diverse posizioni della sequenza.

Come cambia l'addestramento?:

- **Forward Pass**: La propagazione in avanti avviene sequenzialmente. Si calcolano le attivazioni per il tempo $t-1$, poi si usa lo stato risultante per calcolare quelle del tempo $t$, e così via.
- **Backward Pass (BPTT)**: La backpropagation (chiamata Backpropagation Through Time o BPTT) calcola l'errore a ogni passo temporale e lo propaga all'indietro attraverso la catena.
- **Aggiornamento dei Pesi Condivisi**: Poiché un peso come $W$ è usato in ogni passo, il suo gradiente finale è la **somma (o media) dei gradienti calcolati per quel peso in tutti i passi temporali**. In questo modo, il peso viene aggiornato sulla base del suo contributo all'errore su tutta la sequenza.

### Modellazione Sequenziale

![image-20250621191117152](./assets/image-20250621191117152.png)

La rete riceve in input una **sequenza** e produce un **singolo output** alla fine (many-to-one). L'esempio fornito è la **classificazione del sentiment**, dove la rete legge un'intera frase per poi classificarla come "positiva" o "negativa".

La rete riceve in input una **sequenza** e produce in output un'altra **sequenza**. L'esempio è la **generazione di musica**, dove per ogni nota in input viene generata una nota in output. Un altro esempio è la traduzione automatica.

### Embedding

**Le reti neurali non possono interpretare le parole direttamente**. Esse richiedono **input numerici**. Questo pone la necessità di trovare un modo per convertire le parole in una rappresentazione numerica che la rete possa elaborare.

Il processo per convertire il testo in numeri si chiama **embedding**. Il processo si articola in 3 passaggi:

1. **Costruzione del Vocabolario**: Si crea un elenco di tutte le parole uniche presenti nel nostro testo (corpus).
2. **Indicizzazione (Indexing)**: Si assegna un indice numerico univoco a ogni parola del vocabolario.
3. **Embedding**: Si converte ogni indice in un vettore numerico. Ci sono due approcci:
   - **One-hot Embedding**: Si crea un vettore molto lungo, con tutti zeri tranne un $1$ nella posizione corrispondente all'indice della parola. Questo metodo è semplice ma **non cattura alcuna relazione semantica** tra le parole.
   - **Learned Embedding**: Si rappresenta ogni parola con un **vettore denso** (con valori reali) di dimensione fissa. La cosa importante è che questi vettori **vengono appresi** durante l'addestramento. Il risultato è che parole con significato simile (es. "cat", "dog") finiranno per avere vettori simili, ovvero saranno vicine nello "spazio degli embedding".

## Backpropagation Through Time (BPTT)

Si può pensare a una RNN come a una **rete feed-forward a strati con pesi condivisi**. Questa è l'idea fondamentale dello "srotolamento" che abbiamo già visto: trattiamo la RNN come una rete molto profonda, dove ogni "strato" corrisponde a un passo temporale e i pesi sono identici in tutti gli strati.

![image-20250621192201920](./assets/image-20250621192201920.png)

Il **passaggio in avanti** (forward pass) calcola e memorizza le attivazioni di tutte le unità per ogni passo temporale, creando una sorta di "stack" o registro di tutto ciò che è accaduto.

![image-20250621192230168](./assets/image-20250621192230168.png)

Il **passaggio all'indietro** (backward pass) estrae le attività da questo stack, partendo dalla fine e tornando indietro nel tempo, per calcolare le derivate dell'errore a ogni passo temporale.

Dopo aver completato il backward pass, per ogni peso condiviso, si **sommano insieme le derivate calcolate per quel peso a ogni diverso passo temporale**. Questo gradiente totale viene poi utilizzato per aggiornare il peso. Si sommano, in sostanza, i contributi al gradiente da tutti i timestep.

### Vanishing ed Exploding Gradient

![image-20250621192343093](./assets/image-20250621192343093.png)

Il calcolo del gradiente rispetto agli stati nascosti iniziali (es. $h_0$) richiede molte moltiplicazioni successive per la stessa matrice di pesi $W_{hh}$:

+ Se i valori di $W$ sono grandi, il gradiente **esplode**. La soluzione è il **gradient clipping**.
+ Se i valori sono piccoli, il gradiente **svanisce**, impedendo alla rete di apprendere dipendenze a lungo termine. La soluzione richiede un **cambio di architettura**.

C'è inoltre un grande problema: la propagazione in avanti (forward pass) usa funzioni di schiacciamento come $\tanh$ per evitare che le attivazioni esplodano , ma la propagazione all'indietro (backward pass) è **completamente lineare**.

#### Il Problema delle Dipendenze a Lungo Termine

![image-20250621192604079](./assets/image-20250621192604079.png)

A causa del vanishing gradient, gli errori relativi a passi temporali lontani nel passato contribuiscono in modo sempre minore all'aggiornamento dei pesi. Di conseguenza, il modello impara a **dare priorità alle dipendenze a breve termine**, faticando a catturare quelle a lungo raggio.  

### La Soluzione al Vanishing Gradient

La soluzione al vanishing gradient è l'uso di **celle con gate (Gated Cells)**, ovvero unità ricorrenti più complesse che controllano il flusso di informazioni. L'esempio più noto sono le **Long Short Term Memory (LSTM)**, che si affidano alle gated cell per tracciare le informazioni attraverso molti step temporali.

#### Come Funziona una Gated Cell?

La gated cell è una cella di memoria con dei gate (write, keep, read) che decidono quando scrivere, mantenere o leggere l'informazione.

+ L'informazione entra nella cella quando la porta "write" è aperta.
+ L'informazione rimane nella cella finché la porta "keep" è aperta.
+ L'informazione può essere letta dalla cella aprendo la porta "read".

![image-20250621194211646](./assets/image-20250621194211646.png)

Per preservare l'informazione nel tempo si usa un circuito che implementa una cella di memoria analoga. Si tratta di una unità lineare, con un auto-collegamento, con un peso di $1$. L'informazione è conservata nella cella attivando il gate "write", ed è possibile recuperarla attivando il gate "read".

##### Esempio di Backpropagation attraverso Celle di Memoria

![image-20250613153953499](./assets/image-20250613153953499.png)

1. Poiché il `write gate` è attivo, l'informazione proveniente dal resto della rete (in questo esempio, il valore $1.7$) viene **scritta** e memorizzata nello stato interno della cella. Contemporaneamente, il `keep gate` è spento, il che significa che qualsiasi informazione presente in precedenza nella cella viene scartata per far posto a quella nuova. L'informazione viene memorizzata ma non ancora letta, perché il `read gate` è spento.
2. In questo passo temporale, non viene scritta nessuna nuova informazione. Il `keep gate` è attivo, il che, corrisponde a un auto-collegamento con peso 1. Questo fa sì che la cella **mantenga o conservi** il suo stato precedente. Il valore rimane memorizzato nella cella, pronto per essere usato in futuro. Questo è il meccanismo che permette la memoria a lungo termine.
3. Lo stato viene ancora mantenuto grazie al `keep gate` attivo. In più, ora il `read gate` è attivo. Questo permette all'informazione memorizzata di essere **letta** e inviata come output al resto della rete, influenzando i calcoli di quel passo temporale.

## Long Short Term Memory (LSTM) Network

![image-20250613155257805](./assets/image-20250613155257805.png)

Una RNN standard è composta da moduli ripetuti che contengono una semplice operazione (come la funzione della tangente iperbolica).

![image-20250613155522229](./assets/image-20250613155522229.png)

A differenza della cella semplice, i moduli ripetuti di una LSTM contengono **molti layer che interagiscono tra loro** per controllare il flusso di informazioni.

---

Il beneficio principale è che le celle LSTM sono in grado di **tracciare informazioni attraverso molti passi temporali**.

---

![image-20250613155612178](./assets/image-20250613155612178.png)

L'informazione viene aggiunta o rimossa dallo stato della cella attraverso delle strutture chiamate **gate**. I gate permettono o bloccano il passaggio di informazioni in modo selettivo.

Tecnicamente, un gate è implementato da un **layer con funzione sigmoide seguito da una moltiplicazione puntuale**. L'output della sigmoide (tra 0 e 1) agisce come un "rubinetto" che controlla quanta informazione può passare attraverso l'operazione di moltiplicazione.

![image-20250613155758511](./assets/image-20250613155758511.png)

I principali componenti della cella di un LSTM sono i seguenti:

+ **Cell State**: La linea orizzontale che attraversa la parte superiore della cella. Rappresenta la **memoria a lungo termine** della LSTM. L'informazione può scorrere lungo questa linea con poche modifiche.
+ **Hidden State**: È l'output della cella a un dato istante di tempo e viene passato al passo temporale successivo.
+ I Tre Gate Principali: Vengono identificati i tre gate che regolano il flusso di informazioni:
  1. **Forget Gate**
  2. **Input Gate**
  3. **Output Gate**

### Cella LSTM

Il funzionamento della cella è scomponibile in quattro passaggi logici.

#### Passaggio 1: Forget Gate

![image-20250613160140722](./assets/image-20250613160140722.png)

Questo gate decide quali informazioni **dimenticare o buttare via** dallo stato di cella precedente ($c_{t-1}$).

Prende come input lo stato nascosto precedente ($h_{t-1}$) e l'input corrente ($x_t$) e li passa attraverso una funzione **sigmoide**.

L'output è un vettore di valori tra $0$ e $1$. Un valore $0$ significa "dimentica completamente questa informazione", mentre $1$ significa "mantienila completamente".

**Esempio**: Se la frase sta cambiando soggetto, il forget gate potrebbe imparare a "dimenticare" il pronome di genere del soggetto precedente.

#### Passaggio 2: Store (Input) Gate

![image-20250613160444984](./assets/image-20250613160444984.png)

Questo passaggio decide quali nuove informazioni memorizzare nello stato della cella. Si compone di due parti:

1. Un **layer sigmoide** che decide *quali* valori si vogliono aggiornare.
2. Un **layer tanh** crea un vettore di *nuovi valori candidati* ($ĉ_t$) che potrebbero essere aggiunti allo stato.

Entrambi i layer prendono in input $h_{t-1}$ e $x_t$. I loro output vengono poi moltiplicati punto a punto.

La moltiplicazione tra l'output della sigmoide (che decide cosa è importante) e quello della `tanh` (i valori candidati) produce il pezzo di nuova informazione che verrà effettivamente aggiunto.

**Esempio**: Aggiungere il genere del nuovo soggetto per rimpiazzare quello vecchio che è stato dimenticato.

#### Passaggio 3: Update

![image-20250613160803763](./assets/image-20250613160803763.png)

Questo è il momento in cui lo stato della cella viene effettivamente aggiornato da $c_{t-1}$ a $c_t$:

1. Si moltiplica il vecchio stato di cella $c_{t-1}$ per l'output del **forget gate** ($f_t$). Questo **cancella le informazioni** che la rete ha deciso di dimenticare.
2. Si somma il risultato dell'operazione di **store** (vista al passaggio 2). Questo **aggiunge le nuove informazioni** rilevanti.

Il nuovo stato di cella $c_t$ ora contiene una combinazione delle vecchie informazioni rilevanti e delle nuove informazioni rilevanti.

#### Passaggio 4 - Output Gate

![image-20250613161008883](./assets/image-20250613161008883.png)

Questo gate decide quale sarà l'output della cella, ovvero il **nuovo stato nascosto** $h_t$:

1. Un **layer sigmoide** decide quali parti dello stato della cella verranno emesse come output. Prende in input $h_{t-1}$ e $x_t$.
2. Il nuovo stato di cella $c_t$ viene passato attraverso una funzione **tanh** per normalizzarne i valori tra $-1$ e $1$.
3. L'output della sigmoide viene moltiplicato punto a punto con l'output della tanh.

Il risultato di questa moltiplicazione è il nuovo stato nascosto $h_t$. Sia $h_t$ che il nuovo stato di cella $c_t$ vengono passati al passo temporale successivo.

### Flusso del Gradiente in LSTM

![image-20250621200045495](./assets/image-20250621200045495.png)

Questo è il punto di forza delle LSTM.

La backpropagation attraverso lo stato della cella (da $c_t$ a $c_{t-1}$) **richiede solo una moltiplicazione elemento per elemento** (con il forget gate), non una moltiplicazione tra matrici.

Questo crea un **flusso del gradiente ininterrotto**, che risolve il problema del vanishing gradient e permette di modellare dipendenze a lungo termine.

## Gated Recurrent Unit (GRU)

### Cella GRU

![image-20250613162719863](./assets/image-20250613162719863.png)

Una cella GRU ha due gate, rispetto ai tre di quella LSTM.

#### Reset Gate

Questo gate decide **quanto dello stato nascosto precedente deve essere dimenticato**.

Prende come input lo stato nascosto del passo precedente ($h_{t-1}$) e l'input corrente ($x_t$).

Produce un vettore di numeri tra $0$ e $1$. Questo vettore controlla il grado con cui lo stato precedente viene "resettato" o ignorato nel calcolo del nuovo stato candidato.

#### Update Gate

Questo gate determina **quanto del nuovo contenuto candidato deve essere incorporato** nel nuovo stato nascosto. Agisce come un mediatore tra lo stato nascosto precedente e il nuovo stato candidato.

Anch'esso prende come input lo stato nascosto del passo precedente ($h_{t-1}$) e l'input corrente ($x_t$).

Produce un vettore di numeri tra $0$ e $1$ che controlla come il nuovo stato nascosto finale $h_t$ viene calcolato, facendo una sorta di "media pesata" tra lo stato precedente $h_{t-1}$ e il nuovo stato candidato.

### Confronto tra LSTM e GRU

![image-20250621200654577](./assets/image-20250621200654577.png)

+ **LSTM**: È più complessa. Mantiene due stati separati, il **Cell State** (la memoria a lungo termine) e l'**Hidden State** (l'output), e usa tre gate (`forget`, `input`, `output`).
+ **GRU**: È più semplice. Non ha uno stato di cella separato, ma solo l'**Hidden State**. Fonde i gate `forget` e `input` della LSTM in un unico **Update Gate** e usa un **Reset Gate**.

La GRU ha meno parametri, il che la rende computazionalmente più efficiente e talvolta **meno incline all'overfitting su dataset più piccoli**, pur offrendo prestazioni simili alla LSTM in molti task.

## Applicazioni delle RNN

### Generazione di Musica

La rete riceve in input una sequenza di note (o di caratteri da uno spartito) e viene addestrata a predire la nota successiva.

È un modello **many-to-many**, dove una sequenza di input viene usata per generare una sequenza di output.

### Classificazione del Sentiment

La rete legge un'intera sequenza di parole.

È un modello **many-to-one**: l'intera sequenza di input viene processata per produrre un singolo output finale (es. la probabilità che il sentiment sia "positivo").

### Traduzione Automatica

Per la traduzione automatica, si usa una struttura composta da due RNN:

1. **Encoder**: Legge la frase nella lingua di partenza (es. inglese) e la comprime in un vettore di stato nascosto finale.
2. **Decoder**: Prende questo vettore come stato iniziale e genera la frase tradotta nella lingua di destinazione (es. francese), una parola alla volta.

Questa architettura ha un limite significativo. L'Encoder deve comprimere il significato dell'intera frase in un **singolo vettore a dimensione fissa** (il "collo di bottiglia dell'informazione"). Questo rende difficile gestire frasi lunghe o complesse.

La soluzione a questo problema sono i **meccanismi di attention**. L'idea è di permettere al Decoder, a ogni passo della generazione, di "guardare indietro" a tutti gli stati nascosti dell'Encoder e di concentrarsi (**porre attenzione**) sulle parti più rilevanti della frase di input per generare la parola corrente. Questo fornisce alla rete un accesso "apprendibile" alla memoria dell'intera sequenza di input.

## Riepilogo sulle RNN

+ Le RNN sono adatte a compiti di **modellazione di sequenze**.
+ Modellano le sequenze attraverso una **relazione di ricorrenza**.
+ Vengono addestrate con l'algoritmo **Backpropagation Through Time (BPTT)**.
+ Le **celle con gate come le LSTM** permettono di modellare dipendenze a lungo termine, risolvendo il problema del vanishing gradient.
+ Le loro applicazioni includono la generazione di musica, la classificazione e la traduzione automatica.

## Bi-LSTM

![image-20250621200928373](./assets/image-20250621200928373.png)

Questa architettura processa la sequenza in **due direzioni**: una in avanti e una all'indietro.

I risultati delle due passate vengono combinati, fornendo al modello un **contesto sia passato che futuro**, utile per molti task NLP.