# Modi per Migliorare la Generalizzazione dei Modelli

[TOC]

## Recap sull'Overfitting

L'overfitting è il principale ostacolo alla generalizzazione.

I dati di addestramento contengono informazioni riguardo le **regolarità reali** nel mapping tra input e output, e l'**errore di campionamento**. L'errore di campionamento introduce delle **regolarità accidentali**, che esistono solo a causa del particolare sottoinsieme di dati scelto per il training.

Quando addestriamo un modello, esso **non può distinguere** tra le regolarità reali e quelle accidentali, quindi cerca di adattarsi a entrambe. Se il modello è molto flessibile (ha un'alta capacità), può finire per modellare molto bene anche l'errore di campionamento. Questo fenomeno è l'**overfitting**.

## Prevenire l'Overfitting

- **Approccio 1: Ottenere più dati!** Questa è quasi sempre la soluzione migliore, se si ha la possibilità di ottenerli e la potenza di calcolo per usarli.
- **Approccio 2: Usare un modello con la giusta capacità**. Il modello deve essere abbastanza potente da catturare le regolarità vere, ma non così potente da adattarsi anche a quelle accidentali.
- **Approccio 3: Fare la media di molti modelli diversi**. Si possono usare modelli con architetture diverse o addestrare lo stesso tipo di modello su sottoinsiemi diversi dei dati di training.

###  Modi per Limitare la Capacità di una Rete Neurale

La capacità si può controllare in molti modi:

+ **Architettura**: Limitare il numero di layer nascosti e il numero di unità per layer.
+ **Early Stopping**: Iniziare l'addestramento con pesi piccoli e fermarlo prima che il modello inizi a fare overfitting.
+ **Weight-decay**: Penalizzare i pesi di grande valore. Le tecniche più comuni sono la penalità L2 (sui quadrati dei pesi) e L1 (sui valori assoluti).
+ **Noise (Rumore)**: Aggiungere rumore ai pesi o alle attivazioni durante l'addestramento.

Di solito si usa una **combinazione** di questi metodi.

#### Prevenire l'Overfitting con l'Early Stopping

All'inizio dell'addestramento, quando i pesi sono molto piccoli, ogni unità nascosta si comporta in modo quasi **lineare**. La rete ha quindi una capacità bassa.

Man mano che i pesi crescono durante l'addestramento, le unità nascoste iniziano a usare le loro regioni **non lineari**, e la capacità del modello aumenta.

Interrompere il training al momento giusto (prima che l'errore sul validation set inizi a risalire) "congela" il modello in uno stato di capacità inferiore, prevenendo l'overfitting.

#### Limitare la Dimensione dei Pesi

La penalità L2 standard consiste nell'aggiungere un termine alla funzione di costo che **penalizza i quadrati dei pesi**.

Questo meccanismo forza i pesi a rimanere piccoli, a meno che non siano strettamente necessari per correggere grandi errori (ovvero, a meno che non abbiano derivate dell'errore molto grandi).

#### Usare il Rumore come Regolarizzatore

Aggiungere rumore agli input può avere un effetto simile a una penalità L2 sui pesi.

Se si aggiunge del **rumore Gaussiano** agli input $x_i$, la sua varianza viene **amplificata dal quadrato del peso** $w_i^2$ quando il segnale passa al layer successivo.

In una rete semplice con un'unità di output lineare, questo rumore amplificato si aggiunge direttamente all'output, contribuendo all'errore quadratico totale.

Di conseguenza, quando il modello cerca di minimizzare l'errore, è incentivato a **mantenere i pesi piccoli** per ridurre l'impatto di questo rumore amplificato.

### Perché Combinare Modelli Aiuta

Fare la media delle predizioni di molti modelli diversi è un ottimo modo per **ridurre l'overfitting**, specialmente se i modelli producono predizioni molto diverse tra loro:

- **Bias**: È grande se il modello ha troppo poca capacità e non riesce a catturare i dati (underfitting).
- **Varianza**: È grande se il modello ha così tanta capacità che si adatta all'errore di campionamento specifico di ogni training set (overfitting).

Combinando i modelli (**ensembling**), si può fare la **media della varianza**, riducendola. Questo ci permette di usare modelli individuali con alta capacità (basso bias ma alta varianza), ottenendo il meglio da entrambi i mondi.

#### Confronto tra Predittore Combinato e Individuale

Su un singolo caso di test, un modello individuale potrebbe essere migliore del modello combinato. Tuttavia, **modelli individuali diversi saranno migliori su casi di test diversi**.

Se i modelli individuali sono sufficientemente in disaccordo tra loro, il predittore combinato, quando si fa la media su tutti i casi di test, è **tipicamente migliore di tutti i singoli predittori**.

L'obiettivo è quindi creare modelli che siano il più possibile in disaccordo, senza che diventino individualmente scarsi.

#### Modi per Rendere i Predittori Diversi

+ **Minimi Locali Diversi**: Sperare che l'algoritmo di apprendimento si blocchi in minimi locali diversi. È un "trucco dubbio", ma vale la pena provare.
+ **Usare Tipi di Modelli Diversi**: Combinare reti neurali con altri modelli come Support Vector Machines, Decision Trees, ecc.
+ **Usare Reti Neurali Diverse**: Variare l'architettura (numero di layer/unità), il tipo di unità, il tipo o l'intensità della penalizzazione sui pesi, o l'algoritmo di apprendimento.

#### Rendere i Modelli Diversi Modificando i Dati di Training

Ci sono due tecniche per creare diversità:

- **Bagging: Si addestrano modelli diversi su sottoinsiemi diversi dei dati, creati tramite **campionamento con rimpiazzo. Un esempio famoso sono le **Random Forest**. Applicare il bagging alle reti neurali è possibile, ma molto costoso computazionalmente.
- **Boosting**: Si addestra una **sequenza** di modelli a **bassa capacità**. Ogni modello della sequenza dà un peso maggiore ai casi di training che i modelli precedenti hanno sbagliato. Questo concentra le risorse computazionali sui casi "difficili".

#### Due Modi per Fare la Media dei Modelli

- **Mixture (Misto)**: Si fa la **media aritmetica** delle probabilità di output dei modelli.
- **Product (Prodotto)**: Si fa la **media geometrica** delle probabilità di output.

## Dropout

Si considera una rete neurale. Ogni volta che le viene presentato un esempio di training, si **omette casualmente ogni unità nascosta** con una certa probabilità (es. $0.5$).

Questo equivale a campionare casualmente una diversa architettura di rete da un totale di $2^H$ possibili architetture (dove $H$ è il numero di unità nascoste).

Il punto cruciale è che tutte queste sotto-reti campionate **condividono i pesi**.

### Dropout come Forma di Model Averaging

Questo processo è una forma **estrema di bagging**: solo una piccola parte delle $2^H$ possibili reti viene effettivamente addestrata, e spesso solo su un singolo esempio di training.

La **condivisione dei pesi** agisce come una regolarizzazione molto forte, spesso più efficace delle penalità L1 o L2.

### Cosa Fare in Fase di Test?

Sarebbe possibile campionare molte architetture e fare la media geometrica delle loro predizioni, ma è inefficiente.

La soluzione pratica è usare l'intera rete (con tutte le unità attive), ma **moltiplicare i pesi in uscita da ogni unità per la probabilità di mantenimento** $p$ usata durante il training (es. dimezzarli se $p=0.5$).

Questa procedura calcola **esattamente la media geometrica** delle predizioni di tutti i $2^H$ modelli possibili.

###  E se ci sono più Layer Nascosti?

La tecnica si estende semplicemente applicando il **dropout in ogni layer**.

In fase di test, si usa la "mean net", la rete completa con i pesi scalati per la probabilità di mantenimento. Questo è un'ottima e veloce approssimazione dell'averaging di tutti i modelli.

Un'alternativa è eseguire più volte il modello "stocastico" (con dropout attivo) sullo stesso input per avere un'idea dell'incertezza della predizione.

### E per il Layer di Input?

Il dropout può essere applicato anche al **layer di input**.

Tuttavia, solitamente si usa una **probabilità di mantenimento più alta** (es. $p=0.8$ invece di $p=0.5$), per evitare di perdere troppa informazione dall'input originale.

Questa tecnica è simile a quella usata nei "denoising autoencoders".

### Un Altro Modo di Pensare al Dropout

Se un'unità nascosta "sa" quali altre unità sono presenti, può **co-adattarsi** a loro, imparando dipendenze complesse che potrebbero essere valide solo per i dati di training.

Il Dropout forza ogni unità nascosta a funzionare bene con un insieme casuale di "colleghi".

Questo la spinge a imparare feature che sono **utili individualmente** e non solo in combinazione con altre specifiche unità, rendendo il modello più robusto.