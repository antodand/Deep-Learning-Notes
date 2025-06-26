# Activation Functions

[TOC]

Le **activation function** introducono la non-linearità nel modello, permettendo di apprendere relazioni complesse nei dati. Senza le activation function, una rete neurale sarebbe equivalente ad una semplice trasformazione lineare.

## Problema dei Vanishing Gradient

Questo problema si verifica quando il gradiente diventa molto piccolo negli strati più profondi, e il modello apprende molto lentamente o addirittura smette di imparare.

Questo problema si verifica con funzioni di attivazione come la **sigmoide** o la **tangente iperbolica** proprio a causa della polarizzazione eccessiva dei valori (compresi fra $0$ e $1$), per la maggior parte molto vicini tra loro:

![image-20250605171852739](./assets/image-20250605171852739.png)

Il valore massimo che la derivata della sigmoide può raggiungere è solo $0.25$. Per la maggior parte dei valori di input (grandi o piccoli), la derivata è molto vicina a zero.

Durante la backpropagation, l'errore viene propagato all'indietro moltiplicandolo per la derivata della funzione di attivazione ad ogni layer. Moltiplicare ripetutamente per numeri molto più piccoli di $1$ fa sì che il gradiente diventi esponenzialmente piccolo. Se i valori dei gradienti tendono ad avvicinarsi a $0$, il modello smette di apprendere perché i pesi smettono di aggiornarsi:
$$
\theta_t = \theta_{t - 1} - \alpha \frac{\partial L}{\partial \theta}
$$
In questo modo i  primi layer della rete ricevono un gradiente quasi nullo e quindi **imparano molto lentamente o smettono del tutto di imparare**.

## Non-Saturating Activation Functions

Per risolvere questo problema, vengono utilizzate le cosiddette funzioni di attivazione **non-saturanti**:

![image-20250605172530671](./assets/image-20250605172530671.png)

Queste funzioni devono avere due importanti caratteristiche:

+ Devono essere **non-saturanti**: non devono saturare ad un valore specifico (rispetto all'asse $y$). In altre parole, non "schiacciano" l'output in un piccolo intervallo.
+ Devono essere **non-lineari**, ovvero hanno derivate che non si annullano (almeno per input positivi)

> [!NOTE]
>
> Proprio in virtù di questa seconda caratteristica, estendere la pendenza (*working slope*) della funzione sigmoide non basta, poiché rimane lineare.

### ReLU (Rectified Linear Unit)

$$
ReLU(x) = max(0,x)
$$

![image-20250605173408120](./assets/image-20250605173408120.png)

ReLU elimina i valori negativi rendendo i calcoli più efficienti. Ha una derivata costante pari a 1 per tutti gli input positivi, risolvendo efficacemente il problema del vanishing gradient.

Tuttavia, non risolve la problematica relativa al gradiente per i valori negativi (la derivata rimane 0).

Inoltre, ha un problema legato alla non-derivabilità del punto zero. Il vantaggio della sigmoide era proprio la sua facile derivabilità. In questo caso, invece, abbiamo due diverse derivate (per $0^-$ e $0^+$), dunque calcolare quella in $0$ è impossibile.

### LeakyReLU

$$
LeakyReLU(x) =
\begin{cases}
x, & x \ge 0 \\
a_{negative\_slope} \times x, & otherwise
\end{cases}
$$

![image-20250605175446682](./assets/image-20250605175446682.png)

È quasi identica alla ReLU, ma per i valori negativi ha una piccola pendenza, invece di essere piatta a zero.

Evita il problema delle "dying ReLUs" (neuroni che si "spengono" e non imparano più) perché garantisce che ci sia sempre un piccolo gradiente anche per input negativi.

### PReLU (Parametric ReLU)

$$
PReLU(x) =
\begin{cases}
x, & x \ge 0 \\
ax & otherwise
\end{cases}
$$

![image-20250605175749606](./assets/image-20250605175749606.png)

Il coefficiente `a` della pendenza negativa non è un iperparametro fisso, ma un **parametro che viene appreso** durante l'addestramento.

### RReLU (Randomized LeakyReLU)

$$
RReLU(x) =
\begin{cases}
x, & x \ge 0 \\
ax & otherwise
\end{cases}
$$

![image-20250605180047933](./assets/image-20250605180047933.png)

Qui, il coefficiente $a$ è una **variabile casuale** che viene campionata da un intervallo predefinito durante la fase di training, e rimane fissa durante la fase di testing.

### ELU (Exponential Linear Unit)

$$
ELU(x) = \max(0, x) + \min(0, \alpha \cdot (\exp(x) - 1))
$$

![image-20250605180630233](./assets/image-20250605180630233.png)

Per input positivi si comporta come una ReLU. Per input negativi, assume valori negativi seguendo una curva esponenziale che tende a $−\alpha$.

Permettendo output negativi, può aiutare a portare la media delle attivazioni del layer vicino a zero, il che può accelerare la convergenza.

## Altre Funzioni di Attivazione

### Softplus

È una versione liscia e differenziabile della ReLU. Utile per forzare l'output di una rete a essere sempre positivo.

### CELU e SELU

Sono varianti della ELU con diverse parametrizzazioni. La SELU (Scaled ELU) è progettata per indurre proprietà di auto-normalizzazione nelle reti profonde.

### GELU

È un'altra approssimazione liscia della ReLU, basata sulla funzione di distribuzione cumulativa Gaussiana. È non-monotona ed è molto usata nei modelli Transformer.

### ReLU6

È una ReLU con un limite superiore a 6. Utile nelle reti per dispositivi mobili per mantenere le attivazioni in un range limitato.

### Sigmoide

La sigmoide è sconsigliata per i layer nascosti delle reti profonde a causa della saturazione dei gradienti. 

### Tanh

Identica alla sigmoide ma centrata in zero, il che aiuta la convergenza.

### Softsign

### Hardtanh

$$
Hardtanh(x) =
\begin{cases}
1, & x > 1 \\
-1, & x < -1\\
x, & otherwise
\end{cases}
$$

Una versione limitata della tangente iperbolica, con due valori costanti agli estremi e una retta obliqua che li collega centrata nell'origine.

### Threshold

$$
Threshold(x) =
\begin{cases}
x, & x > threshold \\
v, & otherwise
\end{cases}
$$

Usata raramente perché non è possibile propagare all'indietro il gradiente.

E' stata la prima funzione di attivazione.

### Funzioni Shrink (Tanshrink, Softshrink, Hardshrink)

Riducono progressivamente i valori verso lo $0$.

## Activation Function Probabilistiche

### Softmax

$$
Softmax(x_i) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}
$$

La funzione di attivazione softmax **trasforma un vettore di numeri in una distribuzione di probabilità** all'interno di un range. 

La somma di tutti i valori è pari a $1$​, **amplificando la differenza tra i valori ed enfatizzando i valori più grandi**. A valori più grandi una probabilità più alta, mentre quelli piccoli vengono "schiacciati" vicino allo zero.

> [!NOTE]
>
> La sigmoide risulta essere un caso speciale della softmax, quando ci sono solo due classi.
>
> La softmax non è altro che una generalizzazione della funzione sigmoide, poiché basta considerare il caso in cui ci sono due $x_i$, dei quali il secondo si annulla, e si avrà come esito proprio la funzione sigmoide.

### Softmin

$$
Softmin(x_i) = \frac{\exp(-x_i)}{\sum_j \exp(-x_j)}
$$

Differisce dalla precedente per l'utilizzo del segno negativo all'esponente. Questo fa sì che i valori più piccoli abbiano un esponenziale più grande (quindi una probabilità maggiore), mentre i valori più grandi hanno probabilità più piccole. 

E' simmetrica rispetto alla softmax, quindi **enfatizza i valori più piccoli**.

### LogSoftmax

$$
LogSoftmax(x_i) = \log \left( \frac{\exp(x_i)}{\sum_j \exp(x_j)} \right)
$$

Variante della softmax in cui si applica il logaritmo per **migliorare la stabilità numerica** e **semplifica il calcolo della loss** nelle reti neurali.

Non è usata come attivazione nei layer intermedi, ma è comunemente usata come **ultimo layer prima di una funzione di costo** come la NLLLoss per migliorare la stabilità numerica.

---

La funzione di costo da usare in combinazione con un output softmax è la **Cross-Entropy Loss**. Minimizzare questa loss equivale a minimizzare il $-\log(\text{predizione per la classe corretta})$, ovvero, massimizzare quella probabilità predetta $p_i$:
$$
CEL = -\sum_iq_i \log(p_i)
$$