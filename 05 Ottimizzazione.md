# Ottimizzazione

[TOC]

## Superficie di Errore per un Neurone Lineare

La **superficie di errore** è una rappresentazione geometrica della funzione di costo, che mostra come cambia l'errore della rete per ogni possibile combinazione dei suoi pesi. L'altezza del grafico rappresenta l'errore, mentre gli assi orizzontali rappresentano i valori dei pesi. 

Per un **neurone lineare con una funzione di costo quadratica**, la superficie di errore ha la forma di una semplice **conca quadratica**. Le sezioni trasversali verticali di questa conca sono delle parabole, mentre quelle orizzontali sono delle ellissi.

Nelle reti neurali profonde e non lineari, la superficie è molto più complessa (piena di valli, picchi e punti di sella), ma **localmente, in una piccola regione, è quasi sempre ben approssimata da una porzione di conca quadratica**. Questa approssimazione locale è ciò che permette a molti algoritmi di funzionare.

### Velocità di Convergenza

In una superficie di errore a forma di ellisse allungata (una "valle"), la **direzione di massima pendenza (il gradiente) non punta direttamente verso il minimo**.

Il gradiente è **molto grande** nelle direzioni di alta curvatura (i "lati ripidi" della valle), dove in realtà vorremmo muoverci di poco per non scavalcare il fondo, ed è **molto piccolo** nelle direzioni di bassa curvatura (il "fondo piatto" della valle), dove invece vorremmo muoverci velocemente per raggiungere il minimo.

Questo squilibrio causa le famose "oscillazioni" della discesa del gradiente.

### Modalità di Apprendimento

I grandi dataset sono spesso **altamente ridondanti**. Il gradiente calcolato sulla prima metà del dataset è quasi identico a quello calcolato sulla seconda metà. Aggiornare i pesi più frequentemente con stime "rumorose" ma veloci del gradiente è più efficiente che calcolare un unico gradiente perfetto su tutti i dati.

#### Full-Batch

Si usa l'intero dataset per calcolare un unico, preciso gradiente e fare un solo aggiornamento dei pesi. Per questa modalità esistono algoritmi di ottimizzazione molto avanzati (es. L-BFGS, gradiente coniugato).

#### Mini-Batch

Invece di calcolare il gradiente sull'intero dataset, lo si calcola su piccoli sottoinsiemi (mini-batch) e si aggiornano i pesi più frequentemente. Per le reti grandi, questo approccio è **quasi sempre il migliore**.

##### Vantaggi del Mini-Batch

+ Sono molto efficienti dal punto di vista computazionale, specialmente su GPU, perché sfruttano le moltiplicazioni matrice-matrice.
+ È importante che i mini-batch siano bilanciati per le classi che contengono.

## Trucchi per il Mini-Batch Gradient Descent

### Attenzione a Non Ridurre il Learning Rate Troppo Presto

Abbassare il learning rate riduce le fluttuazioni casuali dell'errore dovute ai diversi mini-batch, dando l'impressione di un **miglioramento immediato**, detto *quick win* (la linea rossa nel grafico inizialmente scende più velocemente).

Tuttavia, dopo questo "guadagno rapido", l'**apprendimento successivo diventa molto più lento**.

![image-20250608115444054](./assets/image-20250608115444054.png)

### Inizializzazione dei Pesi

L'inizializzazione è fondamentale:

+ Se due neuroni nascosti partono con pesi e bias identici, otterranno sempre lo stesso gradiente e resteranno identici per sempre, non imparando mai a specializzarsi. La soluzione è inizializzare i pesi con **valori piccoli e casuali**.
+ Un neurone con molti input (un grande *fan-in*) è molto sensibile. La somma di tanti piccoli aggiornamenti ai suoi pesi d'ingresso può portare a un cambiamento enorme del suo output, rendendo l'apprendimento instabile (*overshooting*).
  + Per neuroni con grande fan-in, i pesi iniziali dovrebbero essere più piccoli, ad esempio inizializzandoli con una distribuzione casuale scalata per $1/\sqrt{(\text{fan-in})}$.

#### Inizializzazione dei Pesi (Xavier/Glorot)

Un metodo migliore considera l'idea di rendere la varianza dei pesi una funzione sia il numero di input (fan-in, $\text{n-in}$) che di output (fan-out, $\text{n-out}$):

+ **Inizializzazione Uniforme**: I pesi sono campionati da una distribuzione uniforme $U(−a,a)$ dove $a=gain$⋅
  $$
  \sqrt{\frac{6}{n_{in} + n_{out}}}
  $$

+ **Inizializzazione Normale**: I pesi sono campionati da una distribuzione normale $N(0,σ2)$ dove $σ=gain$⋅
  $$
  \sqrt{\frac{2}{n_{in} + n_{out}}}
  $$

### Pre-elaborare gli Input

Rendere i dati più "digeribili" per la rete può accelerare drasticamente l'apprendimento. L'obiettivo è trasformare gli input per **decorrelare** le loro componenti.

Un metodo consigliato è quello della **PCA (Principal Components Analysis)**:

1. Si calcolano le componenti principali dei dati.
2. Si scartano le componenti con autovalori più piccoli per ridurre la dimensionalità.
3. Si esegue il ***whitening***: si divide ogni componente rimanente per la radice quadrata del suo autovalore.

Per un neurone lineare, questo trasforma una superficie di errore ellittica in una **circolare**, rendendo la discesa del gradiente molto più diretta ed efficiente. In questo modo il gradiente punta direttamente al minimo, risolvendo alla radice il problema delle oscillazioni.

## Il Metodo del Momentum

Come abbiamo discusso, in superfici di errore complesse simili a valli strette, la discesa del gradiente (SGD) è inefficiente: oscilla tra le pareti della valle invece di procedere speditamente verso il minimo. Il metodo del **momentum** è stato introdotto proprio per risolvere questo problema.

### Intuizione

Si può immaginare il processo di ottimizzazione come una palla pesante che rotola sulla superficie di errore. La posizione della palla rappresenta il vettore dei pesi.

* All'inizio, la palla segue la direzione del gradiente (la massima pendenza), ma una volta che acquista velocità, la sua **inerzia (momento)** la fa proseguire nella direzione precedente.
* Il gradiente agisce come una forza esterna che spinge la palla, correggendone la rotta.

Questo approccio porta a due vantaggi cruciali:
+ **Smorza le oscillazioni:** Nelle direzioni di alta curvatura (le pareti della valle), i gradienti cambiano segno rapidamente. Il momento, combinando gradienti con segni opposti, smorza queste oscillazioni.
+ **Accumula velocità:** Nelle direzioni in cui il gradiente è debole ma punta costantemente nella stessa direzione (il fondo della valle), il momento si accumula e la "palla" accelera.

### Le Equazioni del Momento Standard

L'aggiornamento del momento prevede due iterazioni, una per il momento $p$ e una per i pesi $w$.

$$
p_{k+1}=\beta p_{k}+\eta\nabla L(X,y,w_{k})
$$

A ogni passo, il vecchio momento $p_k$ viene smorzato da un fattore $β$ (tra $0$ e $1$) e poi viene aggiunto il gradiente corrente. Se $β$ è $0$, si ritorna alla discesa del gradiente semplice. Valori più grandi di $β$ significano più momentum e curve più lente.

$p$ è il **parametro del momento o momento SGD**, che può essere visto come una media mobile dei gradienti.

---

$$
w_{k+1}=w_{k}-\gamma p_{k+1}
$$

I pesi $w$ vengono aggiornati muovendosi nella direzione del nuovo momento $p$.

---

![image-20250621102848846](./assets/image-20250621102848846.png)

Il momento SGD è simile al concetto fisico di inerzia. Il momento mantiene uguale la direzione della palla rispetto a quella verso cui sta già andando.

---

![image-20250621103019325](./assets/image-20250621103019325.png)

Il momentum **smorza le oscillazioni**. Il parametro $β$ (chiamato *Dampening Factor*) controlla la rapidità con cui la direzione può cambiare: valori più piccoli permettono cambi di direzione più rapidi, valori più grandi mantengono la traiettoria più a lungo. Se $β$ è zero, si ritorna alla discesa del gradiente standard.

All'aumentare di $\beta$, il percorso diventa progressivamente più liscio, le oscillazioni si riducono e la traiettoria verso il minimo diventa più diretta

### Nesterov Accelerated Gradient (NAG)

Ilya Sutskever ha proposto una versione migliorata, ispirata al metodo di Nesterov, che spesso funziona meglio.

Il momento standard calcola prima il gradiente nella posizione corrente e poi fa un salto nella direzione del momento aggiornato.

L'idea di Nesterov è più "previdente":
1.  **Salto di previsione:** Per prima cosa, fa un salto nella direzione del momento *precedente*. Questo ci porta in una posizione approssimativa di dove saremo.
2.  **Correzione:** Calcola il gradiente in questa nuova posizione prevista e lo usa per correggere la direzione finale del salto.

![image-20250609125048814](./assets/image-20250609125048814.png)

L'intuizione è che si fa un primo balzo (vettore marrone), si calcola una correzione nel punto di arrivo (vettore rosso) e si ottiene lo spostamento finale (vettore verde). Questo approccio "guarda avanti" e previene di superare il minimo, portando a una convergenza più stabile e veloce.

### Perché il Momento Funziona Davvero?

Ci sono due spiegazioni principali per l'efficacia del momento.

1. **Accelerazione:** Sebbene il NAG possa garantire una convergenza accelerata per problemi convessi, questa prova non si applica direttamente alle reti neurali (che non sono convesse). L'accelerazione, inoltre, non funziona bene con il rumore intrinseco della SGD. Pertanto, l'accelerazione da sola non è una spiegazione completa.

2. **Smorzamento del Rumore:** Questa è probabilmente la ragione più pratica e importante. Il momento calcola una **media mobile dei gradienti**. Questo processo di media **smorza il rumore** causato dalla stima del gradiente su mini-batch diversi. ![image-20250621111619319](./assets/image-20250621111619319.png)

   Mentre la SGD tende a "rimbalzare" caoticamente vicino al minimo, il momento smorza questi rimbalzi, permettendo passi più stabili e diretti verso la soluzione.

## Learning Rate Separati e Adattivi

Finora abbiamo usato un unico learning rate globale (magari con momentum) per tutti i pesi. Tuttavia, in una rete complessa, non tutti i pesi sono uguali, per due ragioni:

+ **Gradienti diversi:** I gradienti possono avere magnitudini molto diverse tra i vari layer. 
+ **Fan-in diversi:** Unità con un grande fan-in sono più soggette a effetti di *overshooting*, ovvero cambiamenti simultanei dei pesi in ingresso, e potrebbero beneficiare di un learning rate più piccolo.

L'idea è quindi di usare un learning rate globale $ε$, ma moltiplicarlo per un **guadagno locale** $g_{ij}$ che viene determinato automaticamente per ogni singolo peso.

### Determinare i Learning Rate Individuali

Un modo semplice per adattare i guadagni è basarsi sulla coerenza del segno del gradiente. 

Si parte con un guadagno locale di $1$ per ogni peso. Poi:

+ **Se il segno del gradiente per un peso non cambia per due passi consecutivi**, significa che stiamo andando nella direzione giusta, quindi aumentiamo il suo guadagno locale (es. $g = g + 0.05$). 
+ **Se il segno cambia**, significa che abbiamo "saltato" il minimo e stiamo oscillando, quindi diminuiamo drasticamente il suo guadagno in modo moltiplicativo (es. $g = g \cdot 0.95$). 
  + Questo meccanismo di "Additive Increase, Multiplicative Decrease" (AIMD) è simile al controllo di congestione TCP. L'uso di un piccolo aumento additivo e una grande diminuzione moltiplicativa fa sì che i guadagni elevati decadano rapidamente quando iniziano le oscillazioni.


 ### Trucchi per i Learning Rate Adattivi

Per far funzionare meglio questi metodi, si possono usare alcuni accorgimenti:

+ **Limitare i guadagni** in un range ragionevole, ad esempio tra $[0.1, 10]$.
+ Usare il **full-batch o mini-batch molto grandi**. Questo assicura che un cambio di segno nel gradiente sia dovuto a un reale *overshooting* e non al rumore di campionamento del mini-batch.
+ **Combinare** i learning rate adattivi con il momentum.

È importante notare che questi metodi gestiscono solo effetti allineati agli assi, mentre il momentum non ha questa limitazione.

## rmsprop

### rprop: Usare solo il Segno del Gradiente

Il metodo **rprop (resilient propagation)** affronta il problema che la magnitudine del gradiente può variare molto, rendendo difficile la scelta di un learning rate globale, usando **solo il segno del gradiente**.

Questo fa sì che tutti gli aggiornamenti dei pesi abbiano la stessa grandezza. Questo permette di uscire rapidamente da zone di plateau con gradienti molto piccoli.

+ Se i segni degli ultimi due gradienti per un peso sono uguali, la dimensione del passo **aumenta moltiplicativamente** (es. $\times 1.2$).
+ Altrimenti, la dimensione del passo **diminuisce moltiplicativamente** (es. $\times 0.5$).
+ Limita la dimensione del passo a meno di $50$ e a maggiore di un milionesimo.

#### Perché rprop non Funziona con i Mini-Batch

La SGD si basa sull'idea che, con un learning rate piccolo, si fa una media dei gradienti su mini-batch successivi.

rprop rompe questo meccanismo perché **ignora la magnitudine** dei gradienti.

Per esempio, un peso riceve un gradiente di $+0.1$ per 9 mini-batch e $-0.9$ per il decimo, vorremmo che il peso rimanesse circa dov'è. rprop, usando solo il segno, incrementerebbe il peso 9 volte e lo decrementerebbe solo una volta (e non $9$ volte, facendolo tornare a $0$), facendolo crescere molto e ignorando le magnitudini relative dei gradienti.

### rmsprop: rprop per Mini-Batch

**rmsprop** è la risposta al problema precedente.

L'idea è forzare a dividere il gradiente per un valore che sia simile tra mini-batch adiacenti:

1. Si tiene una **media mobile del quadrato del gradiente** per ogni peso:
   $$
   MeanSquare(w,t)=0.9MeanSquare(w,t−1)+0.1 \left( \frac{\partial E}{\partial w} \right)^2
   $$

2. L'aggiornamento del peso viene quindi fatto dividendo il gradiente per la radice quadrata di questa media mobile ($\sqrt{(MeanSquare)}$). Questo normalizza efficacemente il gradiente.

### Riassunto dei Metodi di Apprendimento

+ Per **dataset piccoli** o **grandi ma senza molta ridondanza**, usare metodi **full-batch** come rprop.
+ Per **dataset grandi e ridondanti**, usare **mini-batch**, provando la discesa del gradiente con momentum e rmsprop.

Non c'è una ricetta universale, perché le reti neurali e i task sono molto diversi tra loro.

## Altri Metodi di Gradiente Adattivo

### Adagrad

**Adagrad** adatta il learning rate per ogni parametro, eseguendo grandi aggiornamenti per i parametri con gradienti aggiornati poco e piccoli aggiornamenti per quelli con gradienti aggiornati frequentemente:
$$
g_{t,i} = \nabla_\theta J(\theta_{t,i})
$$
La regola di aggiornamento dello SGD standard per ogni parametro $\theta_i$ ad ogni passo $t$, è:
$$
\theta_{t+1,i} = \theta_{t,i} - \eta \cdot g_{t,i}
$$
Adagrad modifica il learning rate globale $\eta$ ad ogni passo $t$ per ogni parametro $\theta_i$ basandosi sui gradienti passati calcolati sempre per $\theta_i$:
$$
\theta_{t+1,i} = \theta_{t,i} - \frac{\eta}{\sqrt{G_{t,ii} + \epsilon}} \cdot g_{t,i}
$$
Qui, $G_t$ è una matrice diagonale dove ogni elemento $i,i$ è la **somma dei quadrati di tutti i gradienti passati** per il parametro $\theta_i$ fino al tempo $t$. $\epsilon$, invece, è un termine di smorzamento per evitare divisioni per zero.

---

Il learning rate effettivo per ogni parametro viene ridotto dividendo per la radice quadrata della somma di tutti i suoi gradienti passati. Questo significa che i parametri che ricevono spesso grandi gradienti vedranno il loro learning rate ridursi più velocemente.

#### Motivazioni di Adagrad

Spesso abbiamo degli spazi di feature a grandi dimensioni. Solo poche feature sono davvero informative, mentre molte altre sono irrilevanti. Altre invece possono essere rare ma comunque informative. Adagrad aiuta a dare a queste feature rare aggiornamenti più significativi.

---

Il beneficio principale di Adagrad è che **elimina la necessità di regolare manualmente il learning rate**, fornendo un adattamento specifico per una determinata feature incorporando la conoscenza delle precedenti osservazioni.

$G_t$ adatta $\eta$ allo spazio nel quale bisogna operare.

#### Limiti di Adagrad

La sua debolezza principale è l'**accumulo dei quadrati dei gradienti** nel denominatore. Poiché ogni termine è positivo, la somma continua a crescere, causando una riduzione continua del learning rate fino a diventare infinitesimale, bloccando di fatto l'apprendimento.

Il prossimo algoritmo mira a risolvere questa debolezza.

### Adadelta

**Adadelta** è un'estensione di Adagrad che cerca di risolvere il problema del learning rate che si riduce in maniera aggressiva, fino ad annullarsi.

Invece di accumulare tutti i gradienti passati, **restringe la finestra di accumulo dei gradienti a una dimensione fissa**. Lo fa in modo efficiente usando una **media mobile a decadimento esponenziale** dei quadrati dei gradienti, identica a quella di rmsprop. La media mobile al tempo $t$ dipende solo dalla media precedente e dal gradiente corrente, secondo un fattore di decadimento $\gamma$
$$
E[g^2]_t = \gamma E[g^2]_{t-1} + (1 - \gamma)g^2_t
$$
---

Partendo dalla regola di Adagrad, si sostituisce la matrice $G_t$ con la nuova media a decadimento esponenziale. Il denominatore, inoltre, non è altro che la **Root Mean Squared (RMS)** del gradiente:
$$
\theta_{t+1,i} & = \theta_{t,i} - \frac{\eta}{\sqrt{ E[g^2]_{t} + \epsilon}} \cdot g_{t,i} \\
&= \theta_{t,i} - \frac{\eta}{RMS[g]_t} \cdot g_{t,i}
$$
---

Gli autori di Adadelta hanno notato che, in tutti gli algoritmi di aggiornamento visti finora (SGD, Momentum, Adagrad), **le unità di misura non corrispondono**.  L'aggiornamento del parametro ($\Delta\theta$) dovrebbe avere le stesse unità ipotetiche del parametro stesso ($\theta$).

Per risolvere questo, **definiscono un'altra media a decadimento esponenziale**, ma questa volta non dei quadrati dei gradienti, bensì dei **quadrati degli aggiornamenti dei parametri**. La formula è:
$$
E[\Delta \theta^2]_t = \gamma E [ \Delta \theta^2]_{t-1} + (1 - \gamma) \Delta \theta^2_t
$$
Da questa si calcola la RMS degli aggiornamenti dei parametri:
$$
RMS[\Delta \theta]_t = \sqrt{ E[\Delta \theta^2]_t + \epsilon }
$$
---

La regola di aggiornamento finale di Adadelta **elimina completamente il learning rate ** $\eta$ dalla regola di aggiornamento. Viene sostituito con la RMS degli aggiornamenti dei parametri calcolata al passo precedente, $RMS[\Delta \theta]_{t−1}$, poiché quella al tempo $t$ non è ancora nota.

La regola di aggiornamento finale di Adadelta diventa la seguente.

1. Si calcola l'aggiornamento dei parametri come:

$$
\Delta \theta_t = - \frac{RMS[\Delta \theta]_{t-1}}{RMS[g]_t} g_t
$$

2. Si applica l'aggiornamento:

$$
\theta_{t + 1} = \theta_t + \Delta \theta_t
$$

In questo modo, con Adadelta, non **c'è più un learning rate da impostare manualmente**.

#### Analogie con RMSprop

Rmsprop e Adadelta sono stati sviluppati indipendentemente nello stesso periodo per risolvere il problema del learning rate decrescente di Adagrad. 

RMSprop è di fatto identico alla prima regola di aggiornamento di Adadelta:
$$
E[g^2]_t = 0.9E[g^2]_{t-1} + 0.1g^2_t \\
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}}g_t
$$
Come Adadelta, RMSprop divide il learning rate per una media a decadimento esponenziale dei quadrati dei gradienti.

Geoffrey Hinton suggerisce di impostare il coefficiente di decadimento $\gamma =  0.9$ e un buon valore di default per il learning rate $\eta = 0.001$. 

### Adam (Adaptive Moment Estimation)

**Adam** è un altro metodo che calcola il learning rate adattivo per ciascun parametro.

Adam combina due concetti già visti:

+ La media mobile a decadimento esponenziale dei **gradienti passati** (il "primo momento" $m_t$).
  + Non è altro che l'implementazione del concetto di Momentum all'interno di Adam.
  + Il suo ruolo nell'aggiornamento finale di Adam è quello di determinare la **direzione** e l'**accelerazione** del passo. È la componente che spinge l'ottimizzazione in avanti e smorza le oscillazioni.
+ La media mobile dei **quadrati dei gradienti passati** (il "secondo momento" $v_t$), in modo simile ad Adadelta e RMSprop.
  + Non è altro che l'implementazione del concetto di RMSprop all'interno di Adam.
  + Il suo ruolo nell'aggiornamento finale di Adam è quello di **adattare la dimensione del learning rate per ogni singolo parametro**. Agisce come un fattore di normalizzazione: se un parametro ha avuto gradienti molto variabili (e quindi $v_t$ è grande), la dimensione del passo per quel parametro viene ridotta per stabilizzare l'apprendimento.

Le formule per l'aggiornamento di questi due momenti sono le seguenti (la media e la varianza dei gradienti):
$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g^2_t
$$
---

Poiché $m_t$ e $v_t$ sono inizializzati a zero (come vettori di zeri), le loro stime sono "distorte" (biased) verso lo zero all'inizio dell'addestramento. Adam li corregge per rimuovere questo bias iniziale, calcolando delle stime più corrette:
$$
\hat{m_t} = \frac{m_t}{1 - \beta^t_1} \\
\hat{v_t} = \frac{v_t}{1 - \beta^t_2} \\
$$
L'aggiornamento dei parametri $\theta$ usa queste stime corrette, combinando l'effetto del momentum e la normalizzazione del learning rate tramite la varianza:
$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v_t} + \epsilon}} \hat{m_t}
$$
Gli autori propongono valori di default di $\beta_1=0.9$, $\beta_2=0.999$ e $\epsilon=10^{−8}$.

Adam si dimostra empiricamente molto efficace, convergendo in modo più diretto.

### SGD vs RMSprop vs Adam

![image-20250621121404834](./assets/image-20250621121404834.png)

- **SGD**: Mostra un percorso con molte oscillazioni, che fatica a scendere nella "valle" dell'errore.
- **RMSprop**: Mostra un percorso molto più diretto e stabile, grazie all'adattamento del learning rate che smorza le oscillazioni.
- **Adam**: Mostra un percorso spesso ancora più diretto, che beneficia sia dell'adattamento del learning rate sia dell'effetto di accelerazione del momentum.

#### Limitazioni di Adam

Nonostante la sua popolarità, Adam non è perfetto:

+ In alcuni semplici problemi, è stato dimostrato che il metodo può **non convergere**.
+ Spesso produce **errori di generalizzazione peggiori** rispetto a un SGD con momentum ben calibrato, specialmente in problemi di computer vision.
+ Richiede **più memoria** per tenere traccia dei buffer per $m$ e $v$.
+ Ha **più iperparametri** da regolare ($β_1$, $β_2$).

#### Analisi Approfondita di Adam

![image-20250621124021135](./assets/image-20250621124021135.png)

Adam tiene traccia di due medie mobili esponenziali: $m$ (il primo momento, la media dei gradienti) e $v$ (il secondo momento grezzo, la media dei quadrati dei gradienti). Il problema nasce dal fatto che entrambi i vettori sono **inizializzati a zero**. Questa distorsione iniziale, specialmente nelle prime iterazioni, rende necessario l'uso di una **correzione del bias**, che verrà spiegata nella slide successiva.

Al primo passo ($t=1$), la stima del primo momento $m(1)$ sarebbe $0.9 \cdot 0 + 0.1 \cdot g_1 = 0.1 \cdot g_1$. Questo valore è una stima "distorta" (biased) verso lo zero, perché la vera media al primo passo dovrebbe essere semplicemente $g_1$.

Per correggere la stima $m(1) = 0.1 \cdot g(1)$ e ottenere il valore corretto $g(1)$, dobbiamo dividerla per $0.1$. Le linee 9 e 10 dello pseudo-codice, implementano esattamente questa correzione. Al tempo $t=1$, il correttore $1 - β_1^t$ vale $1 - 0.9^1 = 0.1$, che è esattamente il fattore di cui abbiamo bisogno.

Man mano che $t$ aumenta, il termine $β_1^t$ tende a zero, e il fattore di correzione tende a $1$, annullando di fatto il suo effetto quando le stime dei momenti si sono "riscaldate" e sono diventate più accurate.

Infine, la regola di aggiornamento (linea 12) usa queste stime corrette $m̂$ e $v̂$ per adattare la dimensione del passo.

##### Intuizione di $v_t$

Ricordando la definizione statistica di varianza:
$$
Var(x) = E[x^2] - (E[x])^2
$$
Il termine $v(t)$ (la media mobile dei quadrati dei gradienti, $E[g^2]$) è chiamato **varianza non centrata** perché non viene sottratto il quadrato della media dei gradienti ($E[g]^2$).

---

La varianza quantifica quanto i gradienti variano attorno alla loro media.

Se i gradienti rimangono approssimativamente costanti, la loro varianza è quasi $0$. In questo caso, la varianza non centrata $v_t$ è circa uguale al quadrato della media $m_t^2$.

+ Se i gradienti sono stabili, $v_t$ è circa $m_t^2$. Di conseguenza, il rapporto $\frac{m_t}{\sqrt{v_t}}$ è circa $1$. La dimensione del passo di aggiornamento sarà quindi dell'ordine del learning rate $\alpha$. Il modello fa un passo grande e sicuro.
+ Se invece i gradienti cambiano rapidamente, la varianza $v_t$ sarà molto più grande di $m_t^2$. Questo rende il rapporto $\frac{m_t}{\sqrt{v_t}}$ molto più piccolo di $1$.

La dimensione del passo di aggiornamento diventa quindi molto più piccola di $α$. Adam adatta la dimensione del passo per ogni singolo peso: quando i gradienti sono stabili è "coraggioso", quando sono instabili diventa "cauto".

#### Problema della Regolarizzazione L2

Il termine di weight decay viene aggiunto al gradiente **prima** che vengano calcolate le medie mobili $m$ e $v$.

Di conseguenza, anche il weight decay viene **normalizzato** (diviso) per $\sqrt{v_t}$.

Questo porta a un effetto controintuitivo: i pesi con gradienti grandi (che hanno un $v_t$ alto e che vorremmo regolarizzare di più) vengono in realtà **regolarizzati di meno**, rendendo la L2 regularization inefficace.

#### AdamW

**AdamW** (Adam with Weight Decay) risolve questo problema.

**Disaccoppia** il weight decay dall'aggiornamento adattivo del gradiente.

Come mostrato nello pseudo-codice (termine in verde), l'aggiornamento di Adam viene calcolato prima, e solo dopo, in un passaggio separato, viene applicato il weight decay.

In questo modo la regolarizzazione agisce come previsto, migliorando la generalizzazione del modello.

### Lion

Lion è più **efficiente in termini di memoria** rispetto ad Adam, perché tiene traccia solo del momentum (il primo momento).

Il suo aggiornamento ha la stessa magnitudine per ogni parametro, poiché non è un ottimizzatore adattivo ma usa l'operazione sign (segno).

Le sue performance migliorano con batch size più grandi e richiede un learning rate più piccolo rispetto ad Adam.

---

Lion è stato scoperto tramite un processo di program search.

## Ottimizzazione Hessian-Free

Di quanto possiamo ridurre l'errore muovendoci in una data direzione?

La massima riduzione dell'errore dipende dal rapporto tra il **gradiente** e la **curvatura** in quella direzione. Quindi, una buona direzione in cui muoversi è quella con un alto rapporto gradiente/curvatura, anche se il gradiente stesso è piccolo.

### Metodo di Newton e Matrice Hessiana

Il metodo di Newton risolve il problema della discesa del gradiente su una superficie ellittica moltiplicando il vettore gradiente per l'**inversa della matrice di curvatura (l'Hessiano $H$)**:
$$
\Delta w = - \epsilon H (w)^{-1} \frac{dE}{dw}
$$
Su una superficie perfettamente quadratica, questo metodo salta direttamente al minimo in un solo passo.

Il problema è che, per una rete con milioni di pesi, l'Hessiano ha un numero spropositato di termini (trilioni) ed è **computazionalmente impossibile** calcolarla e invertirla.

---

I metodi **Hessian-Free (HF)** sono la soluzione. Essi approssimano la matrice di curvatura e poi usano una tecnica efficiente chiamata **gradiente coniugato** per minimizzare l'errore su quella approssimazione.

### Metodo del Gradiente Coniugato

Invece di fare un unico grande passo, si usa una sequenza di passi. Ogni passo trova il minimo lungo una specifica direzione. La caratteristica chiave è che ogni nuova direzione è **"coniugata"** alle precedenti, ovvero non "rovina" la minimizzazione già effettuata.

![image-20250621130608500](./assets/image-20250621130608500.png)

Il diagramma mostra che, dopo aver minimizzato lungo la prima direzione (linea rossa), il passo successivo avviene lungo una direzione coniugata (linea verde), dove il gradiente nella direzione della prima è zero.

Dopo $N$ passi, il gradiente coniugato è garantito di trovare il minimo di una superficie quadratica $N$-dimensionale. È un modo efficiente di approssimare il metodo di Newton senza mai formare o invertire esplicitamente l'enorme matrice Hessiana.