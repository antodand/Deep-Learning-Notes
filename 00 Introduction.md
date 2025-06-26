# Introduzione al Deep Learning

[TOC]

Il **Deep Learning** è una **sotto-disciplina** del Machine Learning che utilizza le **Reti Neurali**.

---

![image-20250620152542574](./assets/image-20250620152542574.png)

Diversamente dal ML tradizionale, che necessita di una estrazione delle feature **manuale**, il DL **impara automaticamente rappresentazioni multi-layer**, direttamente dai dati grezzi.

## Grafi Computazionali

I grafi computazionali rappresentano le operazioni tra variabili. Vengono usati per modellare i calcoli svolti dalle reti neurali.

![image-20250605101226361](./assets/image-20250605101226361.png)

Ogni nodo rappresenta un'operazione matematica, mentre gli archi rappresentano il flusso di dati.

## Loss e Cost Function

Per ogni coppia di input e output desiderato ($x[0], y[0]$), il modello $G(x,w)$ produce una predizione $\bar{y}$. Questa predizione viene confrontata con il valore reale $y$ all'interno di una funzione di costo, che calcola l'errore per quel singolo campione:
$$
L(x,y,w) = C(y,\bar{y}) = C(y, G(x,w))
$$
Il modello cerca di minimizzare questa funzione aggiustando i pesi della rete attraverso un algoritmo di ottimizzazione.

Per valutare il modello su un intero set di dati $S$, si calcola la **loss media** $L(S,w)$. Questa non è altro che la media di tutte le loss calcolate per ogni singolo campione nel set:
$$
L(S,w) = \frac{1}{P}\sum_{(x,y)}L(x,y,w) = \frac{1}{P} \sum_{p=0}^{P-1} L(x[p], y[p], w)
$$
Una delle funzoni di costo più comuni è il MSE:
$$
C(y, \hat{y}) = \frac{1}{n}\sum^{n}_{i = 1}(y_i - \hat{y}_i)^2
$$
## Neural Network Tradizionale

![image-20250620154555194](./assets/image-20250620154555194.png)

Una rete neurale è una pila di blocchi funzionali. Questi blocchi sono di due tipi:

- **Blocchi Lineari**: Calcolano somme pesate dei loro input ($\Sigma$).
  - Matematicamente, è un prodotto matrice-vettore.
- **Blocchi Non-Lineari**: Applicano una funzione non-lineare (la funzione di attivazione) all'output del blocco lineare, elemento per elemento.

---

![image-20250620155623129](./assets/image-20250620155623129.png)

Per ogni neurone $i$:

+ La somma pesata è:
  $$
  s[i] = \sum_{j \in U P(i)}w[i,j] \cdot z[j]
  $$

+ L'attivazione è:
  $$
  z[i] = f(s[i])
  $$

### Backpropagation

#### Backpropagation Attraverso una Funzione Non-Lineare

![image-20250620155737893](./assets/image-20250620155737893.png)

La **Chain Rule** è il principio matematico alla base della backpropagation. Per calcolare la derivata della funzione di costo $c$ rispetto all'input $s$ del blocco non-lineare, si usa la chain rule:
$$
\frac{dc}{ds}= \frac{dc}{dz} \cdot \frac{dz}{ds}
$$
Poiché $z = h(s)$ (funzione di attivazione), la derivata $\frac{dz}{ds}$ è semplicemente la derivata della funzione di attivazione $h'(s)$. 

Quindi, il gradiente che fluisce all'indietro da $z$ verso $s$ (ovvero $\frac{dc}{dz}$), viene moltiplicato per la derivata della funzione di attivazione.

#### Backpropagation attraverso una Somma Pesata

Un valore $z$ contribuisce a più somme pesate ($s[0]$, $s[1]$, $s[2]$) attraverso i pesi $w[0]$, $w[1]$, $w[2]$.

Per calcolare il gradiente rispetto a $z$ (ovvero $\frac{dc}{dz}$), si devono **sommare tutti i percorsi** attraverso cui $z$ influenza il costo finale. Il gradiente che torna indietro da ogni $s[i]$ ($\frac{dc}{ds[i]}$) viene moltiplicato per il peso $w[i]$ della connessione corrispondente.

Il gradiente totale $\frac{dc}{dz}$ è la somma di questi contributi:
$$
\frac{dc}{dz} = \sum_i \frac{dc}{ds[i]}\cdot w[i]
$$

### Diagramma a Blocchi di una Rete Neurale Tradizionale

![image-20250620161004695](./assets/image-20250620161004695.png)

La rete è una catena di due tipi di blocchi:

+ **Blocco Lineare**: Esegue l'operazione:
  $$
  s_{k+1} = w_k z_k
  $$

+ **Blocco Non-Lineare**: Esegue l'operazione:
  $$
  z_k=h(s_k)
  $$