# Learning Representation

[TOC]

## Linear Classifiers e Limitazioni

![image-20250620174319274](./assets/image-20250620174319274.png)

I **classificatori lineari** sono modelli che suddividono lo spazio degli input in regioni separate da un **iperpiano**.

Tuttavia:

+ Non possono gestire dati **non-linearmente separabili**.
+ La probabilità che una distribuzione casuale di punti $P$ in $N$ dimensioni sia separabile linearmente diminuisce nel momento in cui $P \ge N$ (**Teorema di Cover**)
  + A causa del teorema di Cover bisogna fare attenzione alla **Curse of Dimensionality**.

---

Per risolvere questi problemi, invece di lavorare sui dati grezzi, l'idea è usare un **Feature Extractor** per calcolare una nuova e migliore **rappresentazione** dei dati. Questo estrattore deve essere **non-lineare**.

L'obiettivo è trasformare i dati in un nuovo spazio in cui siano linearmente separabili, per poi applicare un classificatore lineare su questa nuova rappresentazione.

## Metodi di Feature Extraction

Il principio comune a tutti questi metodi è **espandere la dimensionalità della rappresentazione** per rendere i dati più facilmente separabili linearmente.

I metodi sono:

+ **Tiling dello spazio**: Suddivisione dello spazio in regioni più piccole.
+ **Proiezioni casuali**: Trasformazioni casuali per aumentare la dimensione dello spazio.
+ **Classificatori polinomiali**: Combinazioni di variabili di input per migliorare la separabilità.
+ **SVMs**: Mappatura non lineare dei dati in uno spazio a dimensionalità maggiore.

> [!NOTE]
>
> Usiamo solo le feature in maniera lineare perché altrimenti tutto il processo diventerebbe complesso a livello di costo computazionale.

### Feature Monomiali

L'estrattore di feature calcola i **prodotti incrociati** delle variabili di input (es.$ x_1 \cdot x2$, $x_1^2$, ecc.) per creare nuove feature.

Un classificatore lineare applicato a queste nuove feature calcola di fatto una funzione polinomiale degli input originali.

Questo approccio diventa impraticabile per polinomi di grado $d$ elevato, poiché il numero di feature cresce in modo esponenziale.

## Shallow Neural Networks

Per far fronte a questi problemi, si possono utilizzare diverse strategie:

+ **SVM e metodi a kernel** che utilizzano uno strato con funzioni di base non lineari, seguito da uno strato lineare
+ **Reti neurali con due strati**, che diventano degli approssimatori universali, ma che richiedono un numero troppo grande di neuroni per rappresentare funzioni complesse
  + Una rete neurale con due strati (di cui uno nascosto) è un **approssimatore universale**, ovvero può approssimare una funzione continua $g(u_1, u_2, \dots . u_d)$ pur prendendo un numero sufficiente di neuroni.

Tuttavia, poche funzioni complesse possono essere rappresentate in modo **efficiente** con una rete "shallow" (poco profonda) di dimensioni ragionevoli. Spesso richiederebbero un numero di neuroni esponenzialmente grande.

### Ci Servono Davvero le Deep Architecture?

Se le reti shallow sono universali, perché ne servono di profonde?

Le architetture profonde sono **più efficienti** nel rappresentare certe classi di funzioni. Possono rappresentare funzioni più complesse con **meno "hardware"** (meno neuroni totali).

In sostanza, una rete profonda **scambia lo spazio per il tempo** (o l'ampiezza per la profondità): richiede più calcoli sequenziali (più layer) ma meno risorse in parallelo (meno unità per layer).

### Idea di Base per l'Apprendimento di Feature Invarianti

![image-20250620175906073](./assets/image-20250620175906073.png)

1. **Embedding Non-Lineare**: Si proietta l'input in uno spazio a dimensionalità più alta.
2. **Pooling/Aggregazione**: Si aggregano regioni di questo nuovo spazio per creare feature stabili e invarianti, raggruppando elementi semanticamente simili.

## Ipotesi del Manifold

L'uomo riesce a riconoscere una persona in foto anche se le due foto non sono identiche. Questo è perché la nostra mente si trova nello stesso spazio della nostra realtà: a tre dimensioni, che non è la stessa grandezza della figura che la rappresenta.

Un umano può riconoscere la faccia di una persona contando su meno di 56 variabili, mentre alla macchina necessitano $1000 \cdot 1000 = 1 000 000$ pixel.

---

L'**ipotesi del Manifold** sostiene che i **dati naturali vivono su una manifold (una superficie) non-lineare a bassa dimensionalità**, che è immersa nello spazio ad alta dimensionalità che usiamo per rappresentarli (es. lo spazio dei pixel). In sostanza, la realtà vive in una dimensione nettamente minore rispetto alla dimensione che usiamo per rappresentarla

---

Un estrattore di feature ideale (Ideal Disentangling Feature Extractor) sarebbe in grado di districare questi fattori di variazione, mappando l'immagine a uno spazio in cui ogni dimensione corrisponde a un fattore significativo (es. una dimensione per la posa, una per l'espressione).

## Rappresentazioni Gerarchiche dei Dati

Le architetture multi-strato riflettono la **natura composizionale dei dati del mondo reale**, creando una gerarchia di rappresentazioni con un livello di astrazione crescente.

Analizzando le parti di esse di volta in volta, queste possono farci tendere verso un estrattore ideale, permettendo un’efficiente rappresentazione delle informazioni, ad esempio:

+ Visione: pixel $\rightarrow$ bordi $\rightarrow$ texture $\rightarrow$ oggetti
+ Testo: caratteri $\rightarrow$ parole $\rightarrow$ frasi $\rightarrow$ discorso
+ Audio: campioni $\rightarrow$ bande spettrali $\rightarrow$ fonemi $\rightarrow$ parole
