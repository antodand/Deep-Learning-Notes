# Basic Architectures

[TOC]

## Multiplicative Modules

I **multiplicative modules** sono i blocchi costitutivi delle reti neurali. L'interazione tra gli input può essere sia additiva (somme pesate), sia moltiplicativa.

---

![image-20250620161648091](./assets/image-20250620161648091.png)

Un esempio può essere il modulo di **Attention**:
$$
s_i = \sum_{j} w_j x_{ij}
$$
Qui l'output $s_i$ è una somma di input $x_{ij}$ pesati da coefficienti $w_j$:
$$
w_j = \frac{e^{z_j}}{\sum_k e^{z_k}}
$$
I pesi $w_j$, invece, sono calcolati dinamicamente, applicando una funzione **softmax** ad un altro input $z$.

Il vettore $z$ controlla quali input $x$ ricevono più "attenzione" (ovvero un peso $w_j$ più alto, influenzando maggiormente l'output finale.

---

![image-20250620161746214](./assets/image-20250620161746214.png)

Un altro esempio è un modulo che funge da **switch o multiplexer**.

Usando un vettore di controllo $z$ con valori $[1,0]$ o $[0,1]$, si può selezionare, rispettivamente, l'input $x_1$ o $x_2$. Il vettore $z$ di fatto si comporta come un vettore dei pesi $w$.

Un problema è che l'operazione **non è differenziabile rispetto a $z$**, perché $z$ assume valori discreti ($0$ o $1$). Questo impedirebbe di addestrare la parte di rete che produce $z$ tramite backpropagation. La soluzione è usare un "interruttore morbido" come la funzione softmax, i cui output sono continui e differenziabili.

## Mixture of Experts

![image-20250620162441925](./assets/image-20250620162441925.png)

Si usa un modulo di attenzione, chiamato **Gater**, per "attivare" o "pesare" l'output di diverse reti neurali specializzate su uno specifico input, chiamate **Expert**:

1. L'input di controllo $z$ entra nel $Gater$ (che usa una softmax) per produrre i pesi $w_1$ e $w_2$.
2. L'$Expert_1$ processa l'input $x_1$, mentre l'$Expert_2$ processa $x_2$.
3. L'output finale $S$ è una somma pesata degli output dei due expert: $(w_1\cdot x_1)+(w_2\cdot x_2)$.

Questa architettura è usata nei moderni modelli di linguaggio per attivare dinamicamente solo le parti della rete più adatte a un certo compito.

## Parameter Transformation

![image-20250620163559790](./assets/image-20250620163559790.png)

I parametri $w$ di un modello $G(x,w)$ possono essere essi stessi l'output di un'altra funzione (o rete) $H(u)$.

Un input $u$ viene trasformato da $H$ per generare i pesi $w$. Questi pesi vengono poi usati dal modello principale $G$ per processare l'input $x$.

Per addestrare i parametri di $H$ (cioè $u$), il gradiente dell'errore deve essere propagato all'indietro non solo attraverso $G$, ma anche attraverso $H$, usando la sua matrice Jacobiana ($\frac{\partial H}{\partial u}$).

### Weight Sharing

Un caso specifico di trasformazione dei parametri è il Weight Sharing, una tecnica in cui **lo stesso parametro viene usato in più punti di una rete**.

La funzione $H$ semplicemente replica i suoi input. Per esempio, il parametro $u_1$ viene usato sia come $w_1$ che come $w_2$. Dunque, $w_1$ e $w_2$ sono "legati" e avranno sempre lo stesso valore.

Il punto cruciale è che, durante la backpropagation, i gradienti calcolati per $w_1$ e $w_2$ vengono **sommati insieme** per calcolare il gradiente finale per $u_1$.

#### Motif Detection

![image-20250620164049642](./assets/image-20250620164049642.png)

Un'applicazione classica del weight sharing è la **Motif Detection**.

L'obiettivo è **rilevare un pattern specifico** in qualsiasi posizione di una sequenza di input.

Si usa un unico "rilevatore di motivi" $(x, w)$ con un set di pesi $w$ condivisi. Questo rilevatore viene fatto "scorrere" lungo tutta la sequenza di input.

Si ottiene un output per ogni posizione, che indica la presenza del pattern. Un'operazione di **MAX pooling** può poi essere usata per determinare se il motivo è stato trovato almeno una volta.

Questa è l'idea alla base delle **Reti Neurali Convoluzionali (CNN)**, dove un filtro (con pesi condivisi) scorre su un'immagine per rilevare una feature in ogni sua parte.