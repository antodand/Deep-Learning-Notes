# Graph Machine Learning

## Grafo

Un **grafo** è una struttura dati composta da **nodi** (o vertici) e **archi** (o *edges*) che li collegano. Viene usata per rappresentare informazioni che non hanno un inizio o una fine definiti.

Le frecce sugli archi indicano il tipo di relazione (es. unilaterale o reciproca). Questo distingue tra:

- **Grafi diretti**: la direzione della connessione conta.
- **Grafi non diretti**: la connessione è simmetrica.

I tre tipi di informazioni (attributi o embedding) che un grafo può contenere sono:

- **V (Vertex/Node)**: Attributi associati ai singoli **nodi**, come la loro identità o il numero di connessioni.
- **E (Edge/Link)**: Attributi associati agli **archi**, come il tipo di relazione, la sua forza (peso) o la direzione.
- **U (Global/Master Node)**: Attributi che descrivono il **grafo nel suo complesso**, come il numero totale di nodi o la lunghezza del percorso più lungo.

![image-20250623172751905](./assets/image-20250623172751905.png)

Un'immagine può essere rappresentata anche come un grafo dove ogni **pixel è un nodo**, connesso ai suoi vicini adiacenti da un arco. Di conseguenza, ogni pixel non ai bordi ha 8 vicini, e l'attributo di ogni nodo è il suo vettore di colori (es. RGB).

La connettività può essere visualizzata tramite una **matrice di adiacenza**, dove una entry indica se due pixel/nodi sono connessi.

![image-20250623172929013](./assets/image-20250623172929013.png)

Rappresentare immagini e testi come grafi è spesso **ridondante**, poiché hanno già una struttura molto regolare. Un testo infatti può essere visto come un grafo lineare, dove ogni parola è un nodo connesso solo alla parola precedente e a quella successiva.

I grafi sono più potenti quando si applicano a dati con connessioni complesse e irregolari.

### Esempi di Dati a Grafo

- Le molecole sono un esempio perfetto di dati a grafo, come astrazione dell'oggetto 3D. Gli **atomi sono i nodi** e i **legami covalenti sono gli archi**. Inoltre, diverse coppie di atomi e di archi hanno diverse distanze.

  ![image-20250619100825184](./assets/image-20250619100825184.png)

- I social network sono un altro esempio classico. Le **persone sono i nodi** e le loro **relazioni (amicizia, follower) sono gli archi**.

  ![image-20250619100926326](./assets/image-20250619100926326.png)

### Task Graph-Level

In un task di questo tipo, l'obiettivo è predire una proprietà che riguarda l'**intero grafo**.

Per esempio, data una molecola rappresentata come un grafo, bisogna predire se è tossica o quale odore ha.

È un task analogo alla **classificazione di immagini** (assegnare un'etichetta a un'intera immagine) o alla **sentiment analysis** (assegnare un'etichetta a un intero testo).

![image-20250619101731281](./assets/image-20250619101731281.png)

### Task Node-Level

In un task di questo tipo, l'obiettivo è predire una proprietà o un'etichetta per **ogni singolo nodo** del grafo.

Prendiamo come esempio il dataset **Zach's Karate Club**. Dopo un litigio, il club di karate si è diviso in due fazioni. Il task consiste nel predire, per ogni membro (nodo), a quale delle due fazioni si unirà. In questo caso, la distanza tra un nodo e quello dell'istruttore è altamente correlato all'etichetta che verrà assegnata.

È analogo alla **segmentazione di immagini** (etichettare ogni pixel) o al **Part-of-Speech tagging** nel testo (etichettare ogni parola come nome, verbo, ecc.).

![image-20250619101758403](./assets/image-20250619101758403.png)

### Task Edge-Level

In un task di questo tipo l'obiettivo è predire l'esistenza o il tipo di relazione tra due nodi.

Un esempio è la comprensione di una scena. Oltre a identificare gli oggetti (nodi), si vogliono predire le relazioni tra di essi (es. "Daniel *sta combattendo* Johnny", "Daniel *è in piedi su* tappeto").

![image-20250619103648125](./assets/image-20250619103648125.png)

A volte si parte da un grafo completamente connesso e si addestra il modello a predire quali archi esistono veramente e qual è la loro etichetta, "sfoltendo" il grafo. Questo task si chiama **Link Prediction**.

![image-20250619101830389](./assets/image-20250619101830389.png)

## Le Sfide del Calcolo sui Grafi

### Mancanza di una Struttura Coerente

A differenza delle immagini o del testo, i grafi non hanno una struttura fissa e regolare, perché sono modelli matematici estremamente flessibili. 

Per un task come predire la tossicità di una molecola, ogni molecola (grafo) può avere:

- Un numero diverso di atomi (nodi). 
- Tipi di atomi diversi. 
- Un numero di connessioni (legami) diverso per ogni atomo. 

Rappresentare questi dati variabili in un formato che un modello di machine learning possa processare non è un compito banale.

### L'Equivarianza all'Ordine dei Nodi

Lo stesso grafo può essere descritto da molte matrici di adiacenza diverse, semplicemente cambiando l'ordine dei nodi. 

![image-20250619110009322](./assets/image-20250619110009322.png)

Di conseguenza, un buon algoritmo per grafi deve essere **equivariante rispetto all'ordine dei nodi** (node-order equivariant). Questo significa che se si permuta l'ordine dei nodi nell'input, **l'output delle rappresentazioni dei nodi deve essere permutato esattamente nello stesso modo**. In parole povere, il risultato non deve dipendere da come etichettiamo o ordiniamo i nodi.

![image-20250619110208635](./assets/image-20250619110208635.png)

### Scalabilità

I grafi del mondo reale possono essere enormi (es. social network con miliardi di utenti). 

Fortunatamente, i grafi naturali sono tipicamente ***sparse***, cioè il numero di archi cresce in modo lineare con il numero di nodi, non quadratico. 

Questa sparsità permette di usare metodi computazionalmente efficienti. Inoltre, le GNN hanno un numero di parametri che non dipende dalla dimensione del grafo, il che le rende scalabili.

### Come Rappresentare i Grafi

Un modo efficiente in termini di memoria per rappresentare grafi sparsi sono le **adjacency list**. 

Invece di una grande matrice, si memorizza solo un **elenco di coppie di nodi ($n_i, n_j$) che sono connessi da un arco $e_k$**, nella entry $k$-esima della lista.

![image-20250619163113065](./assets/image-20250619163113065.png)

Dato che ci aspettiamo un numero di archi molto più piccolo del numero di entry per una adjacency matrix, evitiamo calcoli e memorizzazione per le parti non connesse del grafo. 

Va notato che sebbene i diagrammi usino valori scalari per semplicità, nella pratica le GNN lavorano con vettori di feature per ogni nodo, arco e per il contesto globale (per esesempio, un tensore di nodi avrà dimensioni $[\text{numero nodi}, \text{dimensione feature nodo}]$). 

## Graph Neural Network

Una **Graph Neural Network (GNN)** è una trasformazione ottimizzabile che agisce su tutti gli attributi di un grafo (nodi, archi, globali) e che **preserva le simmetrie del grafo** (come l'invarianza alla permutazione dei nodi). 

La presentazione segue il framework delle **"message passing neural network" (MPNN)**. 

Le GNN prendono in input un grafo e restituiscono in output un grafo con la stessa struttura (**graph-in, graph-out**), ma con gli attributi (embedding) trasformati e arricchiti di informazione, senza cambiare la connettività del grafo di input.

### La GNN più Semplice

La GNN più semplice apprende nuovi embedding per ogni componente del grafo (nodi, archi, globale) ma non usa ancora la connettività del grafo. 

![image-20250619170922082](./assets/image-20250619170922082.png)

Questa GNN applica una rete neurale separata (un **MultiLayer Perceptron - MLP**) su ogni componente del grafo, e lo chiamiamo **layer GNN**. 

Per ogni nodo, applichiamo il MLP e otteniamo un embedding per ogni nodo. Lo stesso viene fatto per gli archi e per il vettore che riguarda il contesto globale (che restituisce un singolo embedding dell'intero grafo).

### Predizioni con le GNN tramite Pooling

![image-20250623175138216](./assets/image-20250623175138216.png)

Consideriamo il caso di una classificazione binaria (ma il caso è estendibile ad un caso di regressione multi-classe). Se cerchiamo di classificare i nodi, e il grafo contiene già informazioni su di essi, l'approccio sembrerebbe diretto (ovvero, per ogni nodo applicare un classificatore). In realtà non è così.

Spesso le informazioni di cui abbiamo bisogno non si trovano dove ci servirebbero (es. abbiamo feature sugli archi ma in realtà vogliamo classificare i nodi). 

Serve un modo per raccogliere delle informazioni da una parte del grafo e passarle a un'altra.

La soluzione è il **pooling**, che si svolge in due passi:

1. Per ogni elemento, si raccolgono gli embedding di interesse (es. gli archi connessi a un nodo).
2. Si aggregano questi embedding, di solito con una somma. 

---

![image-20250619173042579](./assets/image-20250619173042579.png)

Si denota l'operazione di pooling con la lettera $\rho$, e si denota che si stanno raccogliendo informazioni dagli archi ai nodi come $\rho_{ E_n \rightarrow V_n}$. Quindi, se abbiamo solamente feature edge-level e stiamo cercando di prevedere informazioni binarie sui nodi, possiamo usare il pooling per indirizzare (o passare) l'informazione dove ha bisogno di andare.

---

![image-20250619173117551](./assets/image-20250619173117551.png)

Contrario è il caso in cui abbiamo solamente feature node-level e stiamo cercando di prevedere informazioni binarie edge-level: $\rho_{V_n \rightarrow E_n}$.

---

![image-20250619173133134](./assets/image-20250619173133134.png)

Se abbiamo feature node-level e vogliamo predire una proprietà globale, ci serve raccogliere tutta l'informazione disponibile dai nodi e aggregarla. Questo procedimento è simile a quello dei **Global Average Pooling Layer** nelle CNN. Lo stesso può essere fatto per gli archi.

![image-20250619173159858](./assets/image-20250619173159858.png)

Il modello di classificazione $c$ può tranquillamente venire sostituito con un altro modello, o adattato a classificazioni multi-classe.

---

Questa tecnica di pooling servirà a costruire GNN sofisticate. Nel caso di nuovi attributi, sarà sufficiente definire come passare le informazioni da un attributo all'altro.

**Non stiamo sfruttando la connettività** in questo caso semplice. Ogni nodo è **processato indipendentemente** dagli altri, e lo stesso vale per gli archi e per il contesto globale. Usiamo la connettività solamente quando facciamo il pooling dell'informazione per la predizione.

### Passare Messaggi tra Parti del Grafo

Per creare modelli più sofisticati, si usa la connettività del grafo all'interno del layer GNN stesso, così da rendere i nostri embedding "consapevoli" della connettività.

Questo avviene tramite il **message passing**, dove i nodi vicini si scambiano informazioni per aggiornare i propri embedding.

![image-20250620113847732](./assets/image-20250620113847732.png)

Il processo si articola in 3 fasi:

1. Per ogni nodo nel grafo, **si radunano tutti gli embedding** dei nodi vicini (detti anche messaggi).
1. Si **aggregano** tutti i messaggi con una funzione (es. una somma).
1. Tutti i messaggi processati vengono passati ad una **funzione di aggiornamento** (di solito, una rete neurale).

Così come il pooling, anche il message passing si può applicare sia ai nodi che agli archi.

---

Questa sequenza di operazioni, quando applicata una volta, è il tipo più semplice di layer GNN basato su message-passing.

Questo ricorda la convoluzione standard: il message passing e la convoluzione sono operazioni per aggregare ed elaborare l'informazione dei vicini di un elemento per aggiornare il valore dell'elemento stesso. Nei grafi, l'elemento è un nodo, e nelle immagini, l'elemento è un pixel.

Tuttavia, il numero di nodi vicini in un grafo può essere variabile, a differenza di un'immagine in cui ogni pixel ha un numero fisso di elementi vicini.

---

Impilando insieme i layer GNN di message passing, un nodo può alla fine incorporare informazione da tutto il grafo: dopo tre layer, un nodo ha informazione sui nodi a tre passi di distanza da esso. 

![image-20250620120255105](./assets/image-20250620120255105.png)

Il diagramma dell'architettura si può quindi aggiornare con il message passing. Si possono passare messaggi da nodi a nodi, da archi a nodi, e si può decidere l'ordine di aggiornamento.

---

Non sempre i dataset dei grafi contengono tutti i tipi di informazioni (nodi, archi, globali). Se dobbiamo fare una predizione sui nodi, ma il nostro dataset ha feature solo sugli archi, come possiamo trasferire l'informazione?

Abbiamo visto che si può usare il pooling alla fine, prima della classificazione.

Il problema può però essere risolto meglio, condividendo l'informazione tra nodi e archi **all'interno del layer GNN stesso**, usando il message passing.

![image-20250620120633623](./assets/image-20250620120633623.png)

Il diagramma del GNN layer ora include un'operazione di pooling **dagli archi ai nodi** ($ρ_{E_n \rightarrow V_n}$).

1. Per ogni nodo, si fa il **pooling delle feature** degli archi a esso connessi.
2. Questa informazione aggregata dagli archi viene usata dalla funzione di aggiornamento $f_{Vn}$ per calcolare il nuovo embedding del nodo.

In questo modo, l'informazione contenuta negli archi influenza direttamente l'apprendimento delle rappresentazioni dei nodi a ogni layer.

---

Tuttavia, i vettori di embedding dei nodi e quelli degli archi non hanno necessariamente la stessa dimensione. Pertanto, non possono essere combinati direttamente con operazioni come la somma.

Le possibili soluzioni sono due:

1. **Mappatura Lineare**: Si può apprendere una trasformazione lineare (una matrice di pesi) per proiettare gli embedding degli archi nello stesso spazio dimensionale di quelli dei nodi (o viceversa), rendendoli così compatibili per la somma.
2. **Concatenazione**: Un'alternativa più semplice è concatenare i vettori degli archi e dei nodi prima di passarli alla funzione di aggiornamento.

---

L'ordine delle operazioni di message passing è una scelta di progettazione dell'architettura. Bisogna decidere quali attributi del grafo aggiornare e in quale sequenza.

Ad esempio, si potrebbe prima aggiornare i nodi usando le informazioni dei vicini, e poi usare questi nuovi embedding dei nodi per aggiornare gli archi. Oppure si potrebbe fare il contrario.

Esistono anche schemi più complessi, chiamati "weave" (intreccio), che combinano diversi percorsi di aggiornamento.

![image-20250620121028024](./assets/image-20250620121028024.png)

Ci sono due possibili ordini di aggiornamento:

+ **Node Then Edge Learning**: Prima si aggiornano i nodi (usando pooling da altri nodi e/o archi), e poi si usano i nodi aggiornati per calcolare i nuovi embedding degli archi.
+ **Edge Then Node Learning**: Prima si aggiornano gli archi (usando i nodi a cui sono connessi), e poi si usano gli archi aggiornati per calcolare i nuovi embedding dei nodi.

![image-20250620121007025](./assets/image-20250620121007025.png)

### Rappresentazioni Globali e Conditioning

I nodi che sono lontani da ogni altro nodo nel grafo non potranno mai efficacemente trasferire l'informazione ad un altro nodo. Per un nodo, se abbiamo $k$ layer, l'informazione si propagherà almeno $k$ passi più distante.

Questo può essere un problema nelle situazioni in cui il task di predizione dipende da nodi o da un gruppo di essi, **troppo lontani**.

Una soluzione sarebbe quella di avere tutti i nodi capaci di passare informazioni. Tuttavia, per grafi molto grandi, questo diventa computazionalmente costoso.

Una soluzione è quella di **usare la rappresentazione globale del grafo ($U$)**, detta anche *master node* o *context vector*. Essa è connessa a tutti i nodi e archi, e agisce da "ponte" per lo scambio di informazioni a lunga distanza. 

![image-20250620121540960](./assets/image-20250620121540960.png)

---

I modelli più potenti usano il **conditioning**.

![image-20250620121757670](./assets/image-20250620121757670.png)

Per un nodo, per esempio, possiamo considerare l'informazione dei vicini, degli archi connessi e del contesto globale. Per considerare tutte queste informazioni da tutte queste sorgenti, **basta concatenarle**. In aggiunta, si potrebbero anche mappare nello stesso spazio.

### Riepilogo dei Problemi

![image-20250620122451494](./assets/image-20250620122451494.png)

- **Node Classification**: assegnare un'etichetta a ogni nodo.
- **Graph Classification**: assegnare un'etichetta all'intero grafo.
- **Node Clustering**: raggruppare nodi simili.
- **Link Prediction**: predire se esiste un arco tra due nodi.
- **Influence Maximization**: identificare i nodi più influenti.

![image-20250620122535871](./assets/image-20250620122535871.png)

## Filtri Polinomiali sui Grafi

Dato un grafo $G$ con $n$ nodi ordinati in un modo preciso:

+ $A$ è la **adjacency matrix 0-1** .

  + Contiene $1$ se i due nodi sono connessi, $0$ altrimenti.

+ $D$ è la **diagonal degree matrix**.

  + E' una matrice diagonale in cui ogni elemento $D_v$ sulla diagonale corrisponde al grado del nodo $v$ (cioè al suo numero di connessioni).
    $$
    D_v = \sum_u A_{vu}
    $$

  + Il grado del nodo $v$ è il numero di archi incidenti a $v$.

  + $A_{vu}$ denota la entry nella riga corrispondente a $v$ e la colonna corrispondente a $u$ nella matrice $A$.

+ $L$ è il **laplaciano del grafo**.

  + E' una matrice $n \times n$ definita come:
    $$
    L = D- A
    $$

Il Laplaciano del Grafo è l'**analogo discreto** dell'operatore Laplaciano usato in fisica e analisi matematica.

Codifica la stessa informazione come della adjacency matrix $A$, nel senso che, data una delle due matrici $A$ o $L$, è possibile costruire l'altra.

Sebbene contenga la stessa informazione di connettività della matrice $A$, ha molte proprietà matematiche utili che lo rendono fondamentale in diversi campi dell'analisi dei grafi, come i random walks e lo spectral clustering.

![image-20250620124743017](./assets/image-20250620124743017.png)

Dato un grafo, la sua matrice Laplaciana avrà sulla diagonale il grado di ogni nodo e un $-1$ in corrispondenza delle connessioni esistenti.

### Polinomi del Laplaciano

Una volta definito il Laplaciano, possiamo costruire dei polinomi della matrice $L$. La forma generale del polinomio di grado $d$ è:
$$
p_w(L) = w_0 I_n + w_1 L + w_2 L^2 + \dots +w_d L^d = \sum^d_{i=0} w_i L^i
$$
Questi polinomi sono l'equivalente dei filtri  nelle CNN. I coefficienti del polinomio ($w_0$, $w_1$, ...) sono i **pesi del filtro**, ovvero i parametri che la rete neurale dovrà apprendere.

![image-20250620130754665](./assets/image-20250620130754665.png)

Per poter applicare un filtro, abbiamo bisogno di un "segnale" sul grafo. Assumiamo che ogni nodo $v$ abbia una feature (un valore numerico) $x_v$. Possiamo impilare tutte queste feature $x_v$ in un unico vettore $x \in R^n$.

La **convoluzione** del segnale $x$ con un filtro polinomiale $p_w(L)$ è definita come il prodotto matrice-vettore:
$$
x' = p_w(L)x
$$
Dove l'output $x'$ è un nuovo vettore di feature, con ogni feature filtrata in base alla struttura del grafo.

Considerando il caso semplice di un polinomio di grado $1$ dove solo $w_1=1$ (e gli altri $w$ sono $0$), la convoluzione diventa $x' = Lx$, dunque:
$$
x'_v = (Lx)_v = L_vx \\
= \sum_{u \in G} L_{vu}x_u \\
= \sum_{u \in G}(D_{vu} - A{vu}) x_u \\
= D_v x_v - \sum_{u \in N(v)}x_u
$$
Notiamo che le feature ad ogni nodo $v$ sono combinate con le feature dei loro vicini $u \in N(v)$. 

A questo punto, come il grado $d$ di un polinomio Laplaciano influenza il comportamento della convoluzione? E' semplice notare che:
$$
dist_G(v,u) > 1 \Rightarrow L^i_{vu} = 0
$$
Questo implica che, quando facciamo convoluzione su $x$ con $p_w(L)$ di grado $d$ per ottenere $x'$:

EQ

Efficientemente, la convoluzione ...

### ChebNet

ChebNet è stata una svolta nell'apprendimento di filtri localizzati sui grafi, e ha motivato molti a pensare alle convoluzioni su grafo da una prospettiva diversa. 

Torniamo al risultato della convoluzione di $x$ con il kernel polinomiale $p_W(L)=L$, concentrandoci su un particolare vertice $v$:
$$
 (Lx)_v & =Lvx \\
 & =∑_{u∈G}L_{vu}x_u \\
& =∑_{u∈G}(D_{vu}−A_{vu})x_u \\
& =D_vx_v−∑_{u∈N(v)}x_u
$$
La formula mostrata descrive l'effetto di un **filtro Laplaciano di base**. In pratica, per calcolare il nuovo valore di un nodo $v$, si prende il suo valore originale ($x_v$) moltiplicato per il numero dei suoi vicini $D_v$), e si sottrae la somma dei valori di tutti i suoi vicini immediati. 

Questa operazione misura quanto un nodo è diverso dalla media del suo vicinato, agendo come un rilevatore di "bordi" o "contorni" nel grafo.

---

ChebNet perfeziona questa idea di filtri polinomiali considerando filtri polinomiali della forma:
$$
p_w(L) = \sum^d_{i=1} w_i T_i (\bar{L})
$$
Dove $T^i$ è il polinomio di Chebyshev di primo tipo di grado $i$ e $\bar{L}$ è il Laplaciano normalizzato, definito usando l'autovalore più grande di L:
$$
\bar{L} = \frac{2L}{\lambda_{max}(L)}- I_n
$$


## Modern Graph Neural Network

