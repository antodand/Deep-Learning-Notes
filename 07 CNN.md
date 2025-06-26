# Convolutional Neural Networks (CNN)

[TOC]

## Limiti delle Reti Neurali Standard

Le reti neurali standard **non scalano bene** su immagini intere.

In CIFAR-10, le immagini sono piccole (32x32x3 pixel). Anche con queste dimensioni, un singolo neurone nel primo strato nascosto di una rete fully-connected avrebbe **$3072$ pesi** ($32\times 32\times3$). Questo numero è ancora gestibile.

Se prendiamo un'immagine di dimensioni più comuni, come $200\times200\times3$, un singolo neurone avrebbe **$120.000$ pesi**. Se volessimo usare più neuroni, il numero di parametri crescerebbe vertiginosamente.

Questa struttura fully-connected è uno **spreco** e un numero così elevato di parametri porterebbe quasi certamente a **overfitting**.

Le **Reti Neurali Convoluzionali (CNN)** sfruttano il fatto che l'input è un'immagine e quindi **impongono dei vincoli** alla loro architettura.

### Volumi di Unità 3D

A differenza di una rete tradizionale, i layer di una CNN hanno le unità disposte in **3 dimensioni: larghezza, altezza e profondità**.

In questo contesto, "profondità" si riferisce alla terza dimensione di un volume di attivazione (es. i canali di colore), non al numero totale di layer che compongono l'intera rete neurale. 

### Connettività Locale

Quando si lavora con input ad alta dimensionalità come le immagini, è impraticabile connettere ogni neurone a tutti i neuroni del volume precedente.

La soluzione è la **Connettività Locale**: ogni neurone è connesso solo a una **regione locale** del volume di input.

L'estensione spaziale di questa connessione locale è un iperparametro chiamato **Campo Recettivo (Receptive Field)** del neurone, che è equivalente alla **dimensione del filtro (filter size)**.

C'è un'importante asimmetria nel modo in cui vengono trattate le dimensioni. Le connessioni sono:

- **Locali** nello spazio 2D (larghezza e altezza).
- **Complete** lungo l'intera profondità del volume di input.

## Architettura delle CNN

### Fully Connected Layer

L'immagine $32\times32\times3$ viene "appiattita" in un unico vettore di 3072 elementi. Ogni neurone di output calcola un prodotto scalare tra questo enorme vettore e una riga della matrice dei pesi.

![image-20250610182115005](./assets/image-20250610182115005.png)

### Convolution Layer

![img](./assets/1b47o1Xttq8aQz4WAjxQSEw.png)

Un layer CONV, invece, usa un **filtro** (in questo esempio, $5\times5\times3$). Questo filtro ha sempre la stessa profondità del volume di input.

![img](./assets/imagesrc=https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F9b4f4334-532f-43ba-87e0-49e1ec524e0c%2FUntitled.png)

L'operazione di convoluzione consiste nel "far scorrere" questo filtro sull'immagine e calcolare dei prodotti scalari tra il filtro e piccole porzioni ("chunk") dell'immagine. Ogni prodotto scalare (in questo caso, di dimensione $5\times5\times3 = 75$) più un bias produce **un singolo numero**:
$$
w^Tx + b
$$
![img](./assets/image.png)

Facendo scorrere il filtro su tutte le posizioni spaziali dell'immagine, si ottiene una griglia 2D di numeri chiamata **mappa di attivazione (activation map)**. Ogni numero in questa mappa rappresenta la risposta di quel filtro specifico in quella posizione dell'immagine.

![img](./assets/image-1749573345338-10.png)

In pratica, si usano **molti filtri**. Ogni filtro è specializzato nel rilevare una feature diversa (es. un bordo verticale, un colore specifico, ecc.).

![img](./assets/image-1749573360035-13.png)

Ognuno dei 6 filtri produce la sua mappa di attivazione $28\times 28$. Queste mappe di attivazione vengono impilate lungo la dimensione della profondità per creare il **volume di output** finale, che in questo caso avrà dimensioni $6\times 28\times 28$.

Viene anche aggiunto un **vettore di bias**, con un valore di bias per ogni filtro (quindi 6 bias in totale).

![image-20250621175931919](./assets/image-20250621175931919.png)

Il volume di output di un layer CONV (es. $6\times 28\times 28$) diventa il volume di input per il layer successivo. I filtri del secondo layer devono avere una profondità che corrisponde alla profondità del suo input (in questo caso, 6). Ad esempio, si usano 10 filtri di dimensione $5\times 5\times 6$.

![image-20250621175859266](./assets/image-20250621175859266.png)

Tipicamente, dopo ogni layer CONV viene applicata una funzione di attivazione come ReLU.

### Cosa Imparano i Filtri Convoluzionali?

I filtri del primo layer convoluzionale imparano a riconoscere **"template" locali e semplici**. Spesso imparano a rilevare **bordi orientati** e **colori opposti**.

### Analisi delle Dimensioni Spaziali e Stride

La **dimensione dell'output** si calcola con la formula:
$$
\frac{(N - F)}{stride} + 1
$$
Dove $N$ è la dimensione dell'input, $F$ quella del filtro e $stride$ è il passo con cui il filtro scorre sull'input.

Il risultato deve essere intero, altrimenti il filtro **non si "adatterà" correttamente**.

### Zero Padding

La convoluzione tende a rimpicciolire i volumi spazialmente, e questo, se avviene troppo in fretta, non è positivo per l'apprendimento. 

È pratica comune aggiungere un bordo di zeri attorno all'input, operazione chiamata **zero-padding**.

Con un padding $P$, la dimensione dell'output diventa:
$$
\frac{N + 2P - F}{stride} + 1
$$
Per preservare le dimensioni spaziali dell'input, si usa uno $stride=1$ e un padding $P = (F - 1) / 2$. Ad esempio, per un filtro $3 \times 3$ ($F=3$), si usa un padding di $1$ pixel. 

Il padding aiuta anche a **dare la stessa importanza ai pixel sui bordi** e a non perdere informazioni.

### Convoluzioni $1\times 1$

Una convoluzione con un filtro $1\times 1$ può sembrare inutile, ma ha perfettamente senso.

![image-20250621180640805](./assets/image-20250621180640805.png)

Essa agisce come un prodotto scalare lungo tutta la profondità del volume di input. Ad esempio, su un input $56\times 56 \times 64$, un filtro $1\times 1\times 64$ esegue un prodotto scalare a $64$ dimensioni per ogni pixel.

![image-20250621180704574](./assets/image-20250621180704574.png)

Il suo scopo principale è **modificare la profondità del volume di attivazione** (riducendola o aumentandola), agendo come una rete $1\times 1$ che combina le informazioni tra i canali.

### Campo Recettivo e Downsampling

Il **Campo Recettivo** è la porzione dell'immagine di input originale che influenza l'attivazione di un singolo neurone in un dato layer.

Per una convoluzione con filtri di dimensione $K$, ogni elemento dell'output dipenderà da un campo recettivo $K \times K$ nell'input.

ad ogni convoluzione successiva si aggiunge $K-1$ alla dimensione del campo recettivo. Impilando i layer, quindi, il campo recettivo si espande. Dopo $L$ layer con filtri di dimensione $K$, il campo recettivo ha dimensione $1 + L  (K - 1)$.

#### Downsampling

C'è un problema: per immagini grandi, però, servirebbero moltissimi layer perché ogni neurone di output possa "vedere" l'intera immagine.

La soluzione è eseguire un ***downsampling*** (sottocampionamento) all'interno della rete.

##### Strided Convolution

Un modo per farlo è usare una **convoluzione con stride > 1**.

##### Pooling Layer

È un altro modo per fare downsampling. La sua funzione è rendere le rappresentazioni **più piccole e più gestibili**. Opera indipendentemente su ogni mappa di attivazione.

Il **Max Pooling** è un tipo comune di pooling che, data una finestra (es. max pool con un filtro $2\times 2$ e con stride di $2$), seleziona solo il valore massimo al suo interno.

Le sue proprietà sono che:

- Non ha parametri da apprendere.
- Introduce **invarianza spaziale**, ovvero una certa robustezza a piccole **traslazioni** delle feature nell'immagine.

---

Per concludere, alla fine della catena di layer CONV e POOL, un layer Fully Connected si connette all'intero volume di feature (appiattito) per eseguire la classificazione finale.