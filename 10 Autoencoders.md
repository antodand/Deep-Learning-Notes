# Autoencoder

[TOC]

L'Autoencoder è una rete addestrata per produrre in output il suo stesso input. In altre parole, l'obiettivo è imparare la **funzione identità**, in modo che l'output $h_θ(x)$ sia il più simile possibile all'input $x$:
$$
h_\theta(x) \approx x
$$
![image-20250622165558001](./assets/image-20250622165558001.png)

È composto da due parti:

1. Un **Encoder** che mappa l'input a una rappresentazione interna (le attivazioni del Layer 2).
2. Un **Decoder** che cerca di ricostruire l'input originale a partire da questa rappresentazione interna.

Imparare la funzione identità è un compito banale se non si impongono dei vincoli al layer nascosto:

- **Numero di unità**: Si può costringere il layer nascosto ad avere meno unità del layer di input. Questo forza la rete a imparare una **rappresentazione compressa** dei dati. Questo è un **Autoencoder standard**.
- **Sparsità**: Si può costringere il layer nascosto a essere "sparso", ovvero ad avere solo un piccolo numero di neuroni attivi per ogni dato input. Questo è uno **Sparse Autoencoder**.

---

Per ottenere una rappresentazione nascosta sparsa, **si penalizzano i valori delle attivazioni dei neuroni nascosti**.

La funzione di costo ($\underset{\theta}\min$) è una combinazione di due termini:

1. **Errore di ricostruzione**: $∣∣h_θ(x)−x∣∣^2$, che misura *quanto l'output è diverso dall'input*.
2. **Termine di sparsità L1**: $λ∑_i∣a_i∣$, che penalizza la somma dei valori assoluti delle attivazioni $a_i$ nel layer nascosto, *spingendoli verso lo zero* e favorendo così la sparsità.

## Stacked Autoencoders

1. Si addestra un primo autoencoder con un singolo strato nascosto per imparare a ricostruire l'input $x$.

   ![image-20250614162749646](./assets/image-20250614162749646.png)

2.  Una volta addestrato, si "butta via" la parte del decoder e si usa l'output del layer nascosto ($a_i$) come una nuova rappresentazione, più compressa e significativa, dell'input originale.

   ![image-20250614162833131](./assets/image-20250614162833131.png)

3. Si prende la rappresentazione appresa ($a_i$) come nuovo input e si addestra un **secondo autoencoder** su di essa. L'obiettivo di questo secondo autoencoder è imparare a ricostruire $a_i$, generando nel suo strato nascosto una rappresentazione ancora più astratta e compressa ($b_i$).
   ![image-20250614163037509](./assets/image-20250614163037509.png)

4. Anche in questo caso, una volta completato l'addestramento, si conserva solo la parte dell'encoder, e $b_i$ diventa la nuova rappresentazione.

   ![image-20250614163245719](./assets/image-20250614163245719.png)

5. Il processo viene ripetuto un'altra volta: si usa $b_i$ come input per addestrare un terzo autoencoder e ottenere una rappresentazione finale $c_i$. Questa rappresentazione finale $[c1, c2, c3]$ è quella che viene poi utilizzata come input.

   ![image-20250614163345060](./assets/image-20250614163345060.png)

### Impilare Autoencoder Multipli

L'input (es. un'immagine 28x28) passa attraverso una serie di layer di encoding, ognuno dei quali riduce la dimensionalità della rappresentazione (es. da 784 a 1000, a 500, a 250, fino a una rappresentazione finale di 30 neuroni).

Il decoder ha una struttura speculare. A partire dalla rappresentazione compressa a 30 dimensioni, espande i dati attraverso una serie di layer di decoding per ricostruire l'input originale 28x28. Spesso i pesi del decoder sono una versione "legata" (tied weights) a quelli dell'encoder (es. $W_1^T$).

![image-20250614163634233](./assets/image-20250614163634233.png)

> [!NOTE]
>
> L'autoencoder profondo e non lineare si dimostra superiore nel catturare le caratteristiche essenziali dei dati per una compressione e ricostruzione efficaci (es. immagini più nitide e fedeli alle originali rispetto a PCA).

## U-Net

La U-Net è un'architettura di deep learning molto popolare, utilizzata per la **segmentazione semantica**. È stata sviluppata originariamente per l'analisi di immagini mediche, ambito in cui ha riscosso grande successo.

### Cos'è la Segmentazione Semantica?

La segmentazione semantica è il task di **assegnare una classe a ogni singolo pixel** di un'immagine.

I modelli vengono addestrati utilizzando delle **mappe di segmentazione come output desiderato**. Queste mappe separano i pixel dell'immagine in categorie target (es. "cellula") e non-target.

![image-20250614164457584](./assets/image-20250614164457584-1749912302366-7.png)

### Architettura U-Net

L'architettura della U-Net è composta da tre componenti:

+ **Encoder**: Il percorso a sinistra, che "contrae" l'immagine per estrarre le feature.
+ **Decoder**: Il percorso a destra, che "espande" la rappresentazione per ricostruire la mappa.
+ **Skip Connections**: Le frecce orizzontali che collegano l'encoder al decoder.

![image-20250614164628877](./assets/image-20250614164628877.png)

#### Encoder

L'encoder è simile a una tipica rete convoluzionale, composto da blocchi di layer di **convoluzione** e **pooling**. Il suo scopo è **l'estrazione di feature**, comprimendo progressivamente l'informazione spaziale e astraendo le caratteristiche dell'immagine.

![image-20250614164723462](./assets/image-20250614164723462.png)

![image-20250614164935243](./assets/image-20250614164935243.png)

#### Decoder

Il decoder ricostruisce la **mappa di segmentazione** a piena risoluzione a partire dalla rappresentazione compressa delle feature. A differenza del pooling, che riduce le dimensioni con metodi predefiniti, il decoder aumenta le dimensioni usando layer di **upsampling** o **deconvoluzione**, la cui funzione **viene appresa** durante l'addestramento.

![image-20250614165038715](./assets/image-20250614165038715-1749912641425-9.png)

![image-20250614165105618](./assets/image-20250614165105618.png)

> [!NOTE]
>
> C'è differenza funzionale tra un autoencoder standard e una U-Net: il primo mira a ricostruire l'input originale (compressione), il secondo a produrre una mappa di segmentazione (classificazione per pixel).
>
> ![image-20250614165240171](./assets/image-20250614165240171.png)

#### Skip Connection

Per un autoencoder classico usato per la compressione, encoder e decoder devono essere separati. Altrimenti, si perde l'intero senso della compressione. Per la segmentazione semantica, questa restrizione non è necessaria.

Le **skip connection** passano informazione direttamente dai layer dell'encoder ai layer corrispondenti del decoder.

L'informazione passata è la **localizzazione delle feature** in quel layer di convoluzione. Questo permette al decoder di recuperare i dettagli spaziali ad alta risoluzione che vengono persi durante il downsampling nell'encoder, migliorando drasticamente la precisione della segmentazione finale.

![image-20250614165436559](./assets/image-20250614165436559.png)

---

La **mappa di attivazione** proveniente da un layer dell'encoder viene **concatenata** con la mappa corrispondente nel decoder dopo l'operazione di upsampling.

In questo modo, il decoder riceve due tipi di informazione:

+ Quella **astratta** di **feature extraction** che risale dal layer precedente del decoder.
+ Quella **precisa** di **feature localization** che arriva dall'encoder.

![image-20250614171409239](./assets/image-20250614171409239.png)

## Variational Autoencoder (VAE)

### Limitazioni degli Autoencoder

![image-20250622174858976](./assets/image-20250622174858976.png)

È possibile campionare un punto a caso dallo spazio latente per generare un nuovo contenuto realistico? La risposta, per un autoencoder standard, è generalmente no.

![image-20250614172445660](./assets/image-20250614172445660.png)

L'encoder impara a **mappare gli input in regioni specifiche e spesso isolate** dello spazio latente. Vaste aree di questo spazio rimangono "vuote" o "senza significato". Se si campiona un punto da una di queste aree vuote, il decoder produrrà un output **insensato**. Inoltre, non c'è garanzia che lo spazio sia organizzato in modo regolare. Interpolare tra due punti validi (es. l'embedding di un "3" e quello di un "5") non produce necessariamente un'immagine coerente di un "4".

### Introduzione ai Variational Autoencoder

Un VAE è un autoencoder che viene addestrato con una forma di **regolarizzazione** per garantire che lo spazio latente abbia proprietà desiderabili per la **generazione di nuovi dati**.

Invece di codificare un input come un **singolo punto** nello spazio latente, un VAE lo codifica come una **distribuzione di probabilità** (tipicamente una Gaussiana) su quello spazio.

**Processo di addestramento**:

1. L'encoder mappa l'input in una distribuzione.
2. Viene **campionato** un punto da questa distribuzione.
3. Il decoder ricostruisce l'input a partire dal punto campionato.
4. L'errore di ricostruzione viene usato per la backpropagation.

![image-20250614173113618](./assets/image-20250614173113618.png)

Mentre l'autoencoder semplice ha un mapping deterministico $z = e(x)$, il VAE ha un passaggio di campionamento $z \sim p(z|x)$.

### La Funzione di Costo (Loss) del VAE

La loss è composta da due termini:

1. **Termine di ricostruzione** (sul layer finale): Misura quanto bene l'output del decoder ricostruisce l'input originale. Questo spinge il modello a essere un buon autoencoder.
2. **Termine di regolarizzazione** (sul layer latente): Misura quanto la distribuzione prodotta dall'encoder si discosta da una distribuzione normale standard (media 0, varianza 1). Questo termine è la **Divergenza di Kullback-Leibler (KL)**. Forza il modello a organizzare lo spazio latente in modo strutturato.

#### Diagramma della VAE Loss

![image-20250614173347514](./assets/image-20250614173347514.png)

1. L'**encoder** prende $x$ e produce i parametri di una distribuzione gaussiana: la media $μ_x$ e la deviazione standard $σ_x$.
2. Un punto latente $z$ viene **campionato** da questa distribuzione $N(μ_x, σ_x)$.
3. Il **decoder** prende $z$ e produce la ricostruzione $\hat{x} = d(z)$.
4. La **loss finale** è la somma dell'**errore di ricostruzione** ($∣∣x−\hat{x}∣∣^2$) e della **divergenza KL**, che agisce come regolarizzatore sullo spazio latente.

### Intuizione

Un buon spazio latente generativo deve avere due proprietà:

- **Continuità**: Punti vicini nello spazio latente devono decodificare in output simili.
- **Completezza**: Qualsiasi punto campionato dallo spazio latente dovrebbe produrre un output sensato.

Senza la regolarizzazione (il termine di loss KL), l'encoder potrebbe imparare a **produrre distribuzioni molto strette** (varianza quasi zero) e molto distanti tra loro, creando uno spazio latente "irregolare" con buchi, simile a quello di un autoencoder standard. La regolarizzazione **forza le distribuzioni** a essere vicine a una normale standard e a sovrapporsi, creando uno spazio latente regolare, continuo e completo.

![image-20250614173746866](./assets/image-20250614173746866.png)

![image-20250614173820834](./assets/image-20250614173820834.png)

### Dettagli Matematici e Visione Probabilistica

Vengono definite le due **variabili** chiave del modello:

- $x$: Rappresenta i dati che osserviamo (es. le immagini).
- $z$: È la variabile latente, ovvero la rappresentazione codificata di $x$ in uno spazio a dimensionalità inferiore. Non è direttamente osservata.

Sappiamo che i dati nel modello VAE si generano così:

1. Si campiona un vettore latente $z$ da una distribuzione a priori, $p(z)$.
2. Si campiona un dato $x$ da una distribuzione di likelihood condizionata, $p(x|z)$.

---

Encoder e Decoder in Termini Probabilistici:

- **Decoder Probabilistico**: È definito dalla verosimiglianza $p(x|z)$. Rappresenta la distribuzione della variabile decodificata ($x$) data quella codificata ($z$).
- **Encoder Probabilistico**: È definito dalla distribuzione a posteriori $p(z|x)$. Rappresenta la distribuzione della variabile codificata ($z$) data quella decodificata ($x$).

#### Teorema di Bayes

Il teorema lega la probabilità a posteriori $p(z|x)$ , alla verosimiglianza $p(x|z)$ , e alla probabilità a priori $p(z)$:

![image-20250614174851393](./assets/image-20250614174851393.png)

Il denominatore $p(x)$, chiamato **marginalizzazione**, richiede il calcolo di un integrale su tutte le possibili variabili latenti ($p(x)=∫p(x∣u)p(u)du$), che è quasi sempre computazionalmente intrattabile. 

Il teorema viene riformulato in un contesto più generale del machine learning, usando i parametri del modello $\theta$ e i dati $D$:
$$
P(\theta | D) = \frac{P(D|\theta) P(\theta)}{P(D)}
$$
I termini sono così definiti:

- **Posterior $P(θ|D)$**: La nostra credenza aggiornata sui parametri $θ$ dopo aver visto i dati $D$.
  - Rappresenta le nostre assunzioni iniziali, come una distribuzione Gaussiana sui pesi di una rete neurale.
- **Likelihood $P(D|θ)$**: La probabilità di osservare i dati $D$ dato un modello con parametri $θ$.
  - Descrive quanto bene i dati si adattano a un modello con dati parametri. 
- **Prior $P(θ)$**: La nostra credenza sui parametri $θ$ prima di vedere i dati $D$.
  - È la nostra credenza aggiornata sui parametri dopo aver visto i dati. Questo concetto è legato alla stima Maximum a Posteriori (MAP).
- **Marginalization $P(D)$**: La probabilità totale di osservare i dati $D$.

Una volta trovati i parametri ottimali, li usiamo per fare previsioni. 

#### Assunzioni dei VAE

Facciamo due assunzioni fondamentali che semplificano il modello VAE:

1. **Assunzione sul Prior**: Si assume che la distribuzione a priori delle variabili latenti, $p(z)$, sia una distribuzione Gaussiana standard (media $0$, deviazione standard $1$), indicata come $N(0,I)$:
   $$
   p(z) = N(0,I)
   $$

1. **Assunzione sulla Likelihood**: Si assume che anche la verosimiglianza, $p(x|z)$ (il decoder), sia una **distribuzione Gaussiana**.  La sua media è definita da una funzione deterministica $f(z)$ (la rete neurale del decoder), e la sua matrice di covarianza è una semplice matrice identità scalata da una costante $c$:
   $$
   p(x|z) = N(f(x), cI) \quad f \in F \quad c>0
   $$

#### Il Problema dell'Intrattabilità

In teoria, avendo definito $p(z)$ e $p(x|z)$, potremmo calcolare la distribuzione a posteriori $p(z|x)$ usando il teorema di Bayes. 

Tuttavia, in pratica, questo calcolo è spesso troppo complesso o impossibile a causa dell'integrale intrattabile al denominatore. 

Per questo motivo, si ricorre a metodi di approssimazione come l'**inferenza variazionale**. 

#### Inferenza Variazionale

La VI è un metodo statistico usato per approssimare distribuzioni complesse. 

L'idea è di definire una famiglia di distribuzioni più semplici e parametrizzate (es. Gaussiane) e poi trovare, all'interno di questa famiglia, la distribuzione che più si avvicina a quella complessa che vogliamo stimare. 

Nel VAE, si decide di approssimare la vera (ma intrattabile) distribuzione a posteriori $p(z|x)$ con una distribuzione Gaussiana $q_x(z)$. 

La media e la covarianza di questa distribuzione approssimata sono determinate da due funzioni, $g(x)$ e $h(x)$, che sono di fatto la **rete neurale dell'encoder**.
$$
q_x(z) = N(g(x), h(x)) \quad g \in G \quad h \in H
$$

#### Derivazione della Funzione di Costo

L'obiettivo è trovare la migliore approssimazione $q_x(z)$ minimizzando la divergenza KL tra $q_x(z)$ e la vera posteriore $p(z|x)$. 

Attraverso una serie di passaggi algebrici, minimizzare questa divergenza KL è matematicamente equivalente a **massimizzare** un'altra espressione. 

Questa espressione, chiamata **Evidence Lower Bound (ELBO)**, è composta da due termini:

1. $Ez∼q_x(\log p(x∣z))$: Il termine di ricostruzione.
2. $−KL(q_x(z),p(z))$: Il termine di regolarizzazione.

#### Funzione Obiettivo Finale del VAE

Poiché anche la funzione del decoder $f$ è sconosciuta, deve essere appresa insieme a $g$ e $h$ (l'encoder). 

L'obiettivo completo è quindi trovare le funzioni ottimali $f*$ ,$g*$ e $h*$ che massimizzano la funzione ELBO, che contiene i due termini chiave:

1. **Errore di ricostruzione**: Misura quanto bene il modello può ricreare i dati originali. 
2. **Termine di regolarizzazione (KL)**: Incoraggia la distribuzione approssimata a rimanere vicina a una Gaussiana standard, prevenendo l'overfitting. 

### Reparameterization Trick

Il processo di **campionamento** di $z$ da una distribuzione è un'operazione stocastica (casuale) e **non è differenziabile**. Questo significa che non è possibile propagare all'indietro il gradiente attraverso il campionamento, rendendo impossibile l'addestramento dell'encoder.

![image-20250614182132160](./assets/image-20250614182132160.png)

Il **reparametrization trick** risolve questo problema. Invece di campionare $z$ direttamente, il processo viene ristrutturato:

1. Si campiona un vettore di rumore $ζ$ da una distribuzione fissa e semplice, come una normale standard $N(0, 1)$.
2. Si calcola $z$ in modo deterministico usando i parametri prodotti dall'encoder: $z=μ_x+σ_x⋅ζ$.

La casualità è ora "esternalizzata" in $ζ$. Il percorso che va dai parametri $μ_x$ e $σ_x$ alla loss finale è ora completamente differenziabile, permettendo alla backpropagation di funzionare e di aggiornare i pesi dell'encoder.

![image-20250614182147403](./assets/image-20250614182147403.png)