# Loss Functions

[TOC]

Le **funzioni di costo** calcolano **quanto è sbagliata la predizione di un modello rispetto ai valori reali**. L'obiettivo è quello di minimizzarla, in modo tale che il modello faccia previsioni più accurate.

## MSE Loss

$$
L = \frac{1}{N} \sum^N_{i = 1} (x_i - y_i)^2
$$

La MSE calcola l'errore quadratico medio tra le predizioni $x_i$ e i valori target reali $y_i$. Se si processa un mini-batch, si ottiene un vettore di questi errori, uno per ogni campione.

La MSE utilizza la **L2 Normalization**.

L'operazione di elevamento al quadrato ha due effetti importanti: 

+ Rende l'errore **sempre positivo**.
  + Ciò la rende molto **sensibile** agli outlier.

+ **Penalizza gli errori grandi in modo molto più significativo** rispetto agli errori piccoli.

Si può presentare in due forme distinte:
$$
l(x,y) =
\begin{cases}
mean(L), & \text{if reduction = 'mean'} \\
sum(L), & \text{if reduction = 'sum'}
\end{cases}
$$

- **Non Ridotta ($\text{reduction='none'}$)**: Quando si processa un mini-batch, si può scegliere di ottenere un vettore $L$ che contiene l'errore quadratico per ogni singolo campione del batch.
- **Ridotta ($\text{reduction='mean'}$ o $\text{'sum'}$)**: Solitamente, questi errori individuali vengono aggregati in un unico valore scalare per la backpropagation. L'impostazione predefinita è $mean$, che calcola la **media** di tutti gli errori quadratici nel batch.

## MAE Loss

$$
L = \frac{1}{N} \sum^N_{i=1}\mid{x_i - y_i}\mid
$$

La MAE calcola, invece, l'errore assoluto medio e utilizza la **L1 Normalization**. Per un singolo campione $n$, la loss è calcolata come il **valore assoluto della differenza** tra la predizione $x_n$ e il target $y_n$. A differenza della L2, la L1 non eleva l'errore al quadrato.

Il vantaggio principale della L1 loss è la sua **maggiore robustezza agli outlier** (valori anomali). Poiché non eleva al quadrato gli errori, un singolo outlier con un errore molto grande avrà un impatto molto meno sproporzionato sulla loss totale rispetto a quanto accadrebbe con la L2 loss.

Il problema è che la funzione valore assoluto ha un "angolo" appuntito in zero e in quel punto **non è differenziabile**. Questo può creare instabilità per gli algoritmi di ottimizzazione basati sul gradiente e motiva l'introduzione della prossima loss.

## Smooth L1 Loss

$$
L = \frac{1}{N} \sum_i z_i \quad z_i = 
\begin{cases}
0.5(x_i - y_i)^2, & \mid x_i - y_i \mid < 1 \\
\mid x_i - y_i \mid - 0.5, & otherwise
\end{cases}
$$

Questa funzione di costo usa la L2 Loss se l'errore è inferiore a $1$ (così è anche **differenziabile** vicino allo $0$), mentre usa L2 Loss se l'errore è maggiore di $1$ (il che la rende **robusta** agli outlier perché è lineare).

È molto usata in computer vision proprio perché è meno sensibile agli outlier rispetto alla MSE, ma non ha i problemi di differenziabilità della L1 pura.

Ha però un problema: introduce una "scala" (il punto di transizione, che qui è fissato a 1) che potrebbe essere un iperparametro da regolare.

> [!NOTE]
>
> In computer vision:
>
> + **L2 Loss (MSE)**: Quando il modello deve predire un'immagine e ci sono più risultati plausibili, minimizzare l'errore quadratico lo porta a predire l'**average (la media)** di tutte le possibilità. Nelle immagini, la media di tanti dettagli nitidi è una **sfocatura**.
> + **L1 Loss (MAE)**: Minimizzare l'errore assoluto, invece, spinge il modello a trovare la **median (la mediana)** delle possibilità. La mediana di un insieme di immagini nitide è ancora un'immagine nitida, non una media sfocata. Per questo motivo, la L1 loss tende a produrre **immagini più definite (sharper)**.

## NLL Loss

$$
L =
\begin{cases}
\sum^N_{n=1} \frac{1}{\sum^N_{n=1}w_{yn}}l_n, & reduction = mean \\
\sum^N_{n=1} l_n, & reduction = sum
\end{cases}
$$

Questa è la **Negative Log Likelihood Loss**, usata quando ci si allena per un problema di classificazione multi-classe.

L'input $x$ di questa funzione dovrebbe essere costituito da **log-probabilità**. Questo significa che, in una rete neurale, l'output dei neuroni finali (i logits) dovrebbe essere processato da un layer LogSoftmax prima di essere passato a questa funzione di costo.

La formula per un singolo campione $n$ è $l_n=−w_{yn}x_{n,y_n}$. In questa formula, $x_{n,y_n}$ è la log-probabilità predetta per la classe corretta $y_n$. Poiché le log-probabilità sono numeri negativi, la loss è il loro opposto. Minimizzare questa loss equivale a rendere la log-probabilità della classe corretta il più grande possibile (cioè, il più vicino possibile a zero), spingendo la probabilità verso $1$.

L'argomento opzionale $w_{y_n}$ permette di assegnare un peso a ogni classe. Questo è estremamente utile per gestire dataset sbilanciati, come vedremo nelle prossime slide.

---

Se un dataset è sbilanciato (l'esempio è l'influenza comune, molto più frequente del cancro ai polmoni), un modello addestrato normalmente tenderà a predire sempre la classe maggioritaria per minimizzare l'errore, ignorando di fatto la classe minoritaria.

Anche se si possono usare i pesi nella loss, un approccio migliore è **equalizzare la frequenza delle classi durante il training**. Questo si fa mettendo i campioni di ogni classe in buffer separati e creando ogni mini-batch campionando lo stesso numero di esempi da ogni buffer. Quando un buffer più piccolo si esaurisce, si ricomincia a campionare da esso.

Questo metodo ha un contro: il modello non impara le reali frequenze delle classi nel mondo reale. La soluzione è effettuare un **fine-tuning** finale: si addestra la rete per alcune epoche conclusive usando i dati con la loro frequenza originale, in modo da adattare i bias del layer di output per favorire le classi più frequenti. Inoltre è molto importante non buttare mai via i dati della classe maggioritaria per bilanciare il dataset.

## Cross Entropy Loss

Questa è la funzione di costo più comune per i problemi di classificazione multi-classe. La sua utilità principale è che **combina LogSoftmax e NLL Loss in un'unica classe**. Questo le permette di prendere in input gli score non normalizzati (chiamati **logits**) direttamente dall'ultimo layer della rete.

Il motivo principale per cui si combinano le due funzioni è per garantire la **stabilità numerica**. Se si applicasse la softmax e poi il logaritmo separatamente, i valori di probabilità molto vicini a $0$ o $1$ potrebbero diventare $-∞$ o $0$ dopo il logaritmo. La derivata del logaritmo vicino a zero tende a infinito, causando problemi di instabilità numerica durante la backpropagation. L'implementazione combinata gestisce questi calcoli in modo più stabile. Quando le due funzioni sono combinate, i gradienti saturano ad un valore ragionevole alla fine.

---

La Cross-Entropy Loss è matematicamente legata alla **Divergenza di Kullback-Leibler (KL)**. Essa misura la "distanza" o divergenza tra due distribuzioni di probabilità: la distribuzione reale dei dati $p$ (che in un problema di classificazione è un vettore one-hot) e la distribuzione predetta dal modello $q$ (l'output della softmax):
$$
D_{KL}(P||Q) = \sum_i P(i)\log{\frac{P(i)}{Q(i)}}
$$


---

Questa funzione si definisce come:
$$
L = - \sum^N_{i=1}\sum^C_{j=1} y_{i,j} \log(\hat{y}_{i,j})
$$
Dove $C$ è il numero delle classi, $y_{i,j}$ è $1$ se l'osservazione $i$ appartiene alla classe $j$, altrimenti è $0$, e $\hat{y}_{i,j}$ è la probabilità predetta per la classe $j$.

## Adaptive LogSoftmax with Loss

Questa funzione di costo è una versione efficiente della softmax, progettata specificamente per problemi con un **numero di classi estremamente grande** (es. milioni, come nei modelli linguistici con vocabolari enormi).

Implementa dei "trucchi" per approssimare la softmax e accelerare la computazione.

## Binary Cross Entropy Loss

Questa è un'applicazione della Cross-Entropy per il caso specifico della **classificazione binaria**. Per esempio, si utilizza per calcolare l'errore di ricostruzione in un autoencoder.

Questa funzione si aspetta che sia l'input $x$ che il target $y$ siano già delle **probabilità**, quindi valori strettamente compresi tra $0$ e $1$.

Questa funzione penalizza maggiormente le previsioni errate con alta confidenza, risultando più efficace rispetto a MSE per la classificazione.

## Kullback-Leibler Divergence Loss

Questa funzione calcola direttamente la **Divergenza KL**.

È utile quando il **target $y$ non è una singola classe (one-hot), ma è esso stesso una distribuzione di probabilità**.

Non è fusa ad una Softmax o a LogSoftmax, perciò potrebbe avere dei problemi di stabilità numerica.

## Binary Cross Entropy Loss with Logits

Questa è la versione di Binary Cross Entropy Loss **fortemente raccomandata** nella pratica.

La sua caratteristica principale è che prende in input i punteggi grezzi (**logits**), che non sono ancora stati passati attraverso una funzione di attivazione.

Applica la funzione Sigmoide internamente, prima di calcolare la BCE loss.

Questo approccio combinato è molto più stabile numericamente rispetto ad applicare un layer Sigmoide e poi una BCELoss separatamente.

## Hinge Embedding Loss

$$
l_n =
\begin{cases}
x_n, & y_n = 1 \\
\max\{ 0, \delta - x_n\}, & y_n = -1
\end{cases}
$$

Questa funzione di costo è usata per task di **apprendimento semi-supervisionato** in cui l'obiettivo è misurare se due input sono **simili o dissimili**, enfatizzandone la distanza. L'idea è di "tirare" vicini gli elementi simili e "spingere" lontani quelli dissimili.

+ $x_n$ è una misura di distanza tra due campioni.
+ $y_n$ è un'etichetta che vale $1$ se sono simili, $-1$ se sono dissimili.
+ Se i campioni sono **simili ($y_n=1$)**, la loss è semplicemente la loro distanza $x_n$. Minimizzare la loss significa spingere la loro distanza a zero.
+ Se i campioni sono **dissimili ($y_n=-1$)**, la loss è positiva solo se la loro distanza$x_n$ è inferiore a un margine $δ$. Questo spinge la loro distanza a essere almeno pari al margine.

## Margin Ranking Loss

$$
L(x,y) = \max(0, -y \cdot (x_1 - x_2) + margin)
$$

Questa loss è usata in compiti di **ranking**, dove si vuole che lo score di un item corretto ($x_1$) sia superiore a quello di un item scorretto ($x_2$) di almeno un certo margine.

$y$ è una label che indica quale score dovrebbe essere più alto. Se $y=1$, la loss è zero solo se $x_1$ è maggiore di $x_2$ di almeno il valore del $margin$. Altrimenti, la loss è positiva e l'addestramento spingerà gli score nella direzione giusta.

---

In un task di classificazione, si può usare questa loss per forzare lo score della classe corretta a essere più alto dello score della migliore tra le classi sbagliate.

## Triplet Margin Loss

$$
L(a,p,n) = \max\{ 0, d(a_i, p_i) - d(a_i, n_i) + margin\}
$$

Questa è una delle loss più importanti per il **metric learning** (es. riconoscimento facciale), perché misura la **similarità relativa** tra campioni, massimizzando la distanza per campioni di categorie diverse e minimizzandola per campioni della stessa categoria.

La loss opera su una "tripletta" di campioni:

+ **Anchor ($a$)**: L'esempio di riferimento.
+ **Positive ($p$)**: Un esempio della stessa classe dell'anchor.
+ **Negative ($n$)**: Un esempio di una classe diversa dall'anchor.

L'obiettivo è minimizzare la distanza tra l'ancora e il positivo ($d(a,p)$) e allo stesso tempo massimizzare la distanza tra l'ancora e il negativo ($d(a,n)$), assicurandosi che la seconda sia più grande della prima di almeno un $margin$. Questo "spinge via" gli esempi negativi, raggruppando quelli positivi.

## Soft Margin Loss

Questa è descritta come la **versione "softmax" della margin loss**.

Invece di usare un margine "duro" (hard margin) che una volta soddisfatto dà loss zero, questa funzione usa una **penalità logistica**.

Questo crea un effetto più "morbido" e continuo, che continua a incoraggiare un margine maggiore anche quando quello minimo è già stato raggiunto.

## Multi-Class Hinge Loss

Questa funzione estende il concetto della Hinge Loss a problemi di **classificazione multi-label**, dove un singolo input può avere più di un'etichetta corretta.

In sostanza, somma la Hinge Loss per tutte le categorie, spingendo verso il basso gli score delle categorie corrette e verso l'alto quelli delle categorie non corrette.

## Cosine Embedding Loss

$$
l_n =
\begin{cases}
1 - \cos(x_1, x_2), & y=1 \\
\max\{ 0, \cos(x_1, x_2) - margin \}, & y = -1
\end{cases}
$$

Questa Loss viene usata per misurare quanto due input sono simili o dissimili, usando la distanza coseno. Tipicamente viene utilizzata per il semi-supervised learning o per **l'apprendimento di embedding non lineari**.

![image-20250607190014176](./assets/image-20250607190014176.png)

La loss viene calcolata in modo diverso a seconda che i due input $x_1$ e $x_2$ debbano essere considerati simili o dissimili. Questa informazione è data da una variabile $y$ che può essere $1$ o $-1$:

+ **Caso positivo**: I due vettori devono essere il più possibile allineati, e la funzione coseno deve essere $1$. Quindi, la loss tende a $0$ quando i vettori sono perfettamente allineati (cioè, hanno la stessa direzione), minimizzando la loss
+ **Caso negativo**: I due vettori devono essere dissimili. La loss function cerca di far sì che il coseno del loro angolo sia inferiore a un certo valore di margine. Se il coseno è già inferiore al margine, la loss diventa $0$, altrimenti è positiva e spinge i vettori ad allontanarsi fino a quando la loro similarità non scende sotto la soglia del margine

---

Il vantaggio principale di usare la distanza coseno è che si concentra sulla **direzione** dei vettori piuttosto che sulla loro **magnitudine**. Calcolare il coseno equivale a normalizzare i vettori (scalarli a lunghezza unitaria) e poi calcolarne la distanza Euclidea normalizzata.

Se usassimo una distanza semplice come quella Euclidea, una rete neurale potrebbe imparare a rendere due vettori molto distanti semplicemente aumentandone la lunghezza (magnitudine) all'infinito. Questo è un modo "pigro" di minimizzare la loss. Normalizzando i vettori, la rete è costretta a imparare a ruotarli nella direzione corretta per renderli simili o dissimili, che è l'obiettivo desiderato.

---

Dopo la normalizzazione, tutti i nostri embedding si trovano sulla superficie di un'ipersfera.

L'obiettivo è posizionare i campioni semanticamente simili **vicini** tra loro su questa sfera.

Per i campioni dissimili, l'obiettivo non è renderli opposti, ma **ortogonali** (con un angolo di 90°, quindi un coseno di 0), perché in uno spazio ad alta dimensione c'è molto più "spazio" per l'ortogonalità.