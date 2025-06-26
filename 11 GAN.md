# Generative Adversarial Network (GAN)

[TOC]

## Concetti Preliminari

E' possibile generare variabili casuali complesse partendo da variabili casuali semplici, come le distribuzioni uniformi.

### Variabili Casuali Uniformi

I computer sono macchine **fondamentalmente deterministiche**. Tuttavia, è possibile creare algoritmi (generatori di numeri pseudo-casuali) che producono sequenze di numeri le cui proprietà sono molto simili a quelle di sequenze casuali teoriche.

Usando questi generatori, si può ottenere una sequenza di numeri che segue approssimativamente una **distribuzione uniforme tra 0 e 1**.  Questo è il nostro punto di partenza.

### Variabili Casuali come Risultato di un Processo

Esistono diverse tecniche per generare variabili casuali più complesse a partire da quelle semplici. Tra queste troviamo il metodo della trasformazione inversa, il rejection sampling, l'algoritmo di Metropolis-Hasting, ecc. 

L'idea comune a tutti questi metodi è che la variabile casuale complessa che vogliamo generare può essere rappresentata come il **risultato di un'operazione o di un processo** applicato a variabili casuali più semplici.

#### Il Metodo della Trasformazione Inversa

Il **metodo della trasformazione inversa** è una tecnica che può essere generalizzata: si generano variabili casuali complesse applicando una "funzione di trasformazione" a variabili più semplici.

Lo scopo di questa funzione di trasformazione è **deformare/rimodellare** la distribuzione di probabilità iniziale per **farla assomigliare** a quella desiderata (target). 

![image-20250622183326847](./assets/image-20250622183326847.png)

Per esempio, la distribuzione uniforme viene trasformata in una distribuzione Gaussiana attraverso una mappatura che corrisponde all'inversa della funzione di distribuzione cumulativa (CDF). 

## Modelli Generativi

Spieghiamo con un esempio pratico cosa fa un modello generativo.

Supponiamo di voler generare immagini in bianco e nero e quadrate di cani.  Ogni immagine $n\times n$ può essere "srotolata" e rappresentata come un **vettore ad alta dimensionalità $N = n \times n$**.

Il punto cruciale è che **non tutti i vettori in questo spazio rappresentano un cane**, una volta ritrasformati in un quadrato.  I vettori che effettivamente assomigliano a cani seguono una **distribuzione di probabilità molto specifica** all'interno di questo enorme spazio.  Esisteranno distribuzioni simili ma diverse per gatti, uccelli, ecc.

Il problema di generare una nuova immagine di un cane diventa quindi equivalente al problema di **generare un nuovo vettore seguendo questa specifica "distribuzione di probabilità dei cani"**. Questo corrisponde a generare una variabile casuale rispetto ad una specifica distribuzione di probabilità.

### Le Sfide

Ci sono due problemi principali.

+ La "distribuzione dei cani" è **estremamente complessa** e definita su uno spazio a dimensioni molto grandi. 

+ Anche se sappiamo che questa distribuzione esiste (perché esistono immagini di cani), **non sappiamo come esprimerla con una formula matematica esplicita**.

### Diagramma di un Modello Generativo

![image-20250615172615285](./assets/image-20250615172615285.png)

Il modello generativo prende in input una variabile casuale semplice (da una distribuzione uniforme) e la trasforma in una variabile più complessa. Dopo l'addestramento, l'output di questa rete dovrebbe seguire la distribuzione desiderata (es. una volta rimodellato, dovrebbe assomigliare a un cane).

L'idea sarebbe addestrare la rete confrontando direttamente la distribuzione generata con quella vera.  Tuttavia, questo non è possibile perché, come detto prima, non conosciamo l'espressione esplicita di queste distribuzioni complesse.  La soluzione è quindi basarsi sui **campioni**: abbiamo un campione di dati veri (immagini reali di cani) e possiamo, ad ogni iterazione del training, generare un campione di dati generati dalla nostra rete.

---

![image-20250615172929597](./assets/image-20250615172929597-1750001370955-11.png)

Questo diagramma visualizza il processo di addestramento. Si parte da variabili casuali uniformi, si passano attraverso la rete generativa e si confronta la distribuzione generata con quella vera. L'"errore di corrispondenza" viene poi usato per la backpropagation per addestrare la rete.

### Riassunto del Ciclo di Addestramento

Data una variabile casuale con una distribuzione uniforme di probabilità come input, vogliamo che la distribuzione di probabilità dell'output generato sia la "distribuzione di probabilità del cane".

L'idea è quella di ottimizzare la rete ripetendo questi passaggi:

1. Generare degli input uniformi.
2. Passarli attraverso la rete per ottenere degli output generati.
3. Confrontare i campioni di dati veri (la distribuzione di probabilità del cane) e quelli generati (es. calcolando una distanza tra le due distribuzioni di campioni).
4. Usare la backpropagation per fare un passo di discesa del gradiente e ridurre questa distanza.

## Generative Adversarial Network

L'idea brillante delle GAN consiste nel sostituire il **confronto diretto** tra distribuzioni di probabilità (che abbiamo visto essere difficile) con un **confronto indiretto**. Questo confronto indiretto prende la forma di un **task secondario (downstream task)** eseguito su entrambe le distribuzioni. 

La rete generativa viene quindi addestrata rispetto a questo task, in un modo che la costringe ad avvicinare la sua distribuzione a quella reale. 

Il task è un compito di **discriminazione** tra campioni veri e campioni generati. Noi vogliamo che questa discriminazione **fallisca** il più possibile.

In una GAN, ci sono due reti in competizione:

+ Un **discriminatore**, che cerca di classificare correttamente i dati come "veri" o "falsi" (generati). 
+ Un **generatore**, che viene addestrato per ingannare il più possibile il discriminatore.

### Caso Ideale (Metodo Diretto)

![image-20250622185429871](./assets/image-20250622185429871.png)

Supponiamo di avere due distribuzioni, e che vogliamo generare quei sample da queste distribuzioni di probabilità.

![image-20250622185447758](./assets/image-20250622185447758.png)

Quello che chiamiamo metodo di apprendimento "diretto" consiste nell'aggiustare iterativamente il generatore (tramite gradient descent) per correggere l'errore tra la distribuzione generata e quella vera.

![image-20250622185513378](./assets/image-20250622185513378.png)

Infine, supponendo che il processo di ottimizzazione sia perfetto, dovremmo ritrovarci con una distribuzione generata che corrisponde esattamente a quella vera.

### Approccio GAN (Metodo Indiretto)

Per l'approccio indiretto, introduciamo un **discriminatore**. Per ora, immaginiamo che sia un "oracolo" perfetto che conosce entrambe le distribuzioni.  Se le due distribuzioni sono molto diverse, il discriminatore potrà classificarle facilmente e con alta confidenza. 

Per ingannare il discriminatore, il generatore deve rendere la sua distribuzione più simile a quella reale.  Il discriminatore si troverà nella massima difficoltà quando le due distribuzioni saranno identiche: a quel punto, non potrà fare meglio che tirare a indovinare (accuratezza del 50%).

---

![image-20250615182645793](./assets/image-20250615182645793.png)

Questi grafici visualizzano il processo. Le curve blu (reale) e arancione (generata) sono le distribuzioni. La curva grigia rappresenta la probabilità, secondo il discriminatore, che un campione sia "vero".

Quando le distribuzioni sono separate, il discriminatore è molto sicuro.

![image-20250615182750424](./assets/image-20250615182750424.png)

Man mano che le distribuzioni si sovrappongono, l'incertezza del discriminatore aumenta (la curva grigia si appiattisce).

![image-20250615182913433](./assets/image-20250615182913433.png)

Quando le distribuzioni coincidono, il discriminatore è completamente incerto e assegna una probabilità del 50% a ogni punto.

---

A questo punto è lecito chiedersi se il metodo indiretto è davvero una buona idea.

Sembra più complicato del metodo diretto. Inoltre, richiede un discriminatore che abbiamo immaginato come un oracolo perfetto, ma che in realtà non conosciamo.

Per il primo problema, la difficoltà pratica di confrontare direttamente le distribuzioni tramite campioni controbilancia la difficoltà più alta dell'approccio indiretto. Per il secondo, è vero che il discriminatore non è noto, ma **può essere appreso**.

### L'Approssimazione: Adversarial Neural Network

Il  **generatore** è una rete neurale che prende in input una variabile casuale semplice e genera (una volta allenata) dati complessi di variabili casuali che seguono la distribuzione target.

Anche il **discriminatore** è modellato con un'altra rete neurale.  Prende in input un campione (es. un'immagine) e restituisce la probabilità che quel campione sia "vero".

Le due reti vengono addestrate **congiuntamente**, ma con **obiettivi opposti**:

+ **Obiettivo del Generatore**: Ingannare il discriminatore.
  + Viene addestrato, aggiornando i pesi ad ogni iterazione di training, per **massimizzare** l'errore di classificazione finale (usando la "salita del gradiente" o *gradient ascent*).

+ **Obiettivo del Discriminatore**: Riconoscere i dati falsi.
  + Viene addestrato, aggiornando i pesi ad ogni iterazione di training, per **minimizzare** l'errore di classificazione (usando la classica discesa del gradiente o *gradient descent*).


Il generatore prende come input variabili casuali semplici e genera nuovi dati. Il discriminatore, invece, prende dati veri e generati, cercando di discriminarli, tramite un classificatore.

#### Diagramma Riassuntivo delle GAN

![image-20250615184216114](./assets/image-20250615184216114-1750005737762-13.png)

1. La rete generativa viene addestrata per **massimizzare** l'errore di classificazione. 
1. La rete discriminativa viene addestrata per **minimizzare** lo stesso errore. 
1. **L'errore di classificazione** è la metrica che guida l'addestramento di entrambe le reti. 
1. Le distribuzioni **non vengono mai confrontate direttamente**.

Il nome "reti avversarie" deriva dal fatto che le due reti cercano di "battere" l'una l'altra, e così facendo, entrambe migliorano. 

Dal punto di vista della teoria dei giochi, questo è un **gioco minimax a due giocatori**. Lo **stato di equilibrio** si raggiunge quando il generatore produce dati indistinguibili da quelli reali e il discriminatore non può fare altro che tirare a indovinare (probabilità 1/2). 

### Architettura Pratica delle GAN

![image-20250615184943077](./assets/image-20250615184943077.png)

1. Si parte da un **rumore casuale $z$**.
2. Il **Generatore $G$** prende in input questo rumore e produce dei dati **falsi $G(z)$**.
3. Il **Discriminatore $D$** riceve in input sia i dati **reali $x$** provenienti dal dataset, sia i dati falsi $G(z)$.
4. L'output del Discriminatore è una classificazione: decide se l'input che ha ricevuto è **reale o falso**.

### Discriminatore

Il Discriminatore è un **classificatore binario**. Il suo unico scopo è distinguere se l'input $x$ è reale (proveniente dal dataset) o falso (prodotto dal generatore).

Tipicamente, il Discriminatore produce un output scalare $o \in R$, che viene poi passato attraverso una **funzione sigmoide** per ottenere una probabilità $D(x)$ compresa tra $0$ e $1$. Questo valore rappresenta la probabilità predetta che $x$ sia un dato reale. Per convenzione, ai dati veri viene assegnata l'etichetta $y=1$, mentre ai dati falsi viene assegnata l'etichetta $y=0$.

Il Discriminatore viene addestrato per minimizzare la cross-entropy loss, una funzione di costo standard per la classificazione binaria. La formula è:
$$
\min_D \{−y \log D(x)−(1−y)\log(1−D(x)) \}
$$
Questo significa che quando l'input è reale ($y=1$), la loss è $-\log{D(x)}$, e il discriminatore impara a rendere $D(x)$ vicino a $1$. Quando l'input è falso ($y=0$), la loss è $- \log(1-D(x))$, e impara a rendere $D(x)$ vicino a $0$.

### Generatore

Il Generatore $G$ parte da un vettore casuale $z$ (la **variabile latente**), solitamente campionato da una distribuzione semplice come una normale $N(0,1)$.

Applica una funzione (la sua rete neurale) per generare un nuovo dato $x' = G(z)$.

Lo scopo del Generatore è **ingannare il Discriminatore**, facendogli classificare i dati generati come reali. Vuole quindi che $D(G(z))$ sia il più vicino possibile a $1$.

Questo si traduce nell'addestrare il Generatore per **massimizzare la loss del Discriminatore** quando l'etichetta è $y=0$. La sua funzione obiettivo è quindi:
$$
\max_G \{−\log(1−D(G(z))) \}
$$

#### Problema con la Loss del Generatore

L'obiettivo del generatore:
$$
\min_G \{\log(1−D(G(z))) \}
$$
Soffre del problema dei **gradienti che svaniscono (vanishing gradients)**, specialmente all'inizio dell'addestramento. Quando il generatore è ancora scarso, il discriminatore riconosce facilmente i suoi output come falsi, quindi $D(G(z))$ è vicino a $0$.

In questa regione, la funzione $\log(1-x)$ è molto piatta, il che significa che i gradienti sono quasi nulli e il generatore impara molto lentamente.

Per risolvere questo problema, in pratica si modifica l'obiettivo del generatore. Invece di minimizzare $\log(1-D(G(z)))$, lo si addestra a **massimizzare $\log(D(G(z)))$**.

Il grafico di $-\log(x)$ (che è l'obiettivo che si massimizza) mostra che questa nuova funzione ha gradienti molto più forti quando l'input è vicino a $0$. Quindi, quando il generatore produce dati palesemente falsi ($D(G(z))$ vicino a $0$), riceve un segnale di errore molto più forte che lo aiuta a migliorare rapidamente. Questo trucco equivale a "ingannare" il generatore facendogli credere che il suo obiettivo sia produrre dati che il discriminatore classifichi come veri ($y=1$).

La funzione obiettivo completa è:
$$
\min_D \max_G \{ -E_{x \sim Data} \log D(x) - E_{z \sim Noise} \log (1 - D(G(z))) \}
$$