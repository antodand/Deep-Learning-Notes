# Modelli di Diffusion

[TOC]

## Stable Diffusion

Possibili implementazioni di Stable Diffusion sono:

+ text2img
+ text+img

![image-20250618094801547](./assets/image-20250618094801547.png)

Le componenti dell'architettura sono le seguenti:

+ **Text Encoder**: Il modulo che processa l'input testuale.
+ **Image Generator**: Il cuore del sistema, a sua volta è composto da:
  - **Image Information Creator**: Il modulo che genera **l'informazione latente** dell'immagine.
  - **Image Decoder**: Il modulo che **trasforma** l'informazione latente in un'immagine visibile in pixel

L'Image Information Creator è l'elemento che rappresenta il principale miglioramento prestazionale rispetto ai modelli precedenti. Il suo funzionamento si articola in **steps**, un parametro che nelle interfacce di Stable Diffusion è tipicamente impostato a 50 o 100.

L'obiettivo di questo componente è generare l'informazione dell'immagine operando interamente nello **spazio latente**. Questo approccio lo rende molto più veloce rispetto ai modelli di diffusion precedenti che lavoravano direttamente nello spazio dei pixel.

La sua implementazione tecnica consiste in una rete **UNet** e un **algoritmo di scheduling**.

---

![image-20250618102833626](./assets/image-20250618102833626.png)

I modelli usati per ogni blocco sono i seguenti:

- **Text Encoder**: Usa **CLIPText**.
  - Prende in input del testo e restituisce **77 vettori di embedding**, ognuno di dimensione 768.
- **Image Information Creator**: Usa una **UNet + Scheduler**.
  - Prende in input gli embedding del testo e un **tensore di rumore casuale**, e restituisce un tensore di informazione processata.
- **Image Decoder**: Usa un **Decoder di un Autoencoder**.
  - Prende in input il tensore di informazione processata (es. di dimensioni 4x64x64) e restituisce l'immagine finale in pixel (es. di dimensioni 3x512x512).

### Step di Diffusion

![image-20250618102926760](./assets/image-20250618102926760.png)

Il processo inizia con un **tensore di informazione casuale di una immagine**, che è puro rumore nello spazio latente.

![image-20250618103056911](./assets/image-20250618103056911.png)

Si distinguono uno **spazio delle informazioni (spazio latente)**, dove opera l'Image Information Creator, e il **mondo visivo (spazio dei pixel)**, dove opera l'Image Decoder per creare l'immagine finale.

![image-20250618103146364](./assets/image-20250618103146364.png)

La diffusion avviene in **passi multipli**. A ogni passo, la UNet elabora il tensore latente per produrre un nuovo tensore latente che assomiglia sempre di più al prompt testuale e alle conoscenze visive apprese durante l'addestramento.

![image-20250618103231460](./assets/image-20250618103231460.png)

Partendo da rumore puro, dopo ogni passo del processo di diffusione nella UNet, se si decodificasse il tensore latente si otterrebbe un'immagine via via più definita e coerente, che emerge progressivamente dal caos di pixel.

### Come Funziona la Diffusion

![image-20250618104518998](./assets/image-20250618104518998.png)

Gli esempi di training sono creati generando rumore e aggiungendone una quantità alle immagini nel dataset di training, chiamato **forward diffusion**, funziona così:

1. Si prende un'immagine reale dal training set.
2. Si genera un pattern di rumore casuale.
3. Si sceglie una quantità di rumore (un numero di "passi").
4. Si corrompe l'immagine originale aggiungendo quella quantità di rumore.

![image-20250618104713931](./assets/image-20250618104713931.png)

Questo processo viene ripetuto per creare un grande dataset. 

![image-20250618104917689](./assets/image-20250618104917689.png)

Il dataset sarà costituito da un input composto dall'immagine con il rumore e dalla quantità di rumore, e un output/label composto dal rumore originale che è stato aggiunto. Il modello da addestrare su questo dataset è il **Noise Predictor**, che è una **UNet**.

![image-20250618105056378](./assets/image-20250618105056378.png)

Durante uno step di training, si prende un esempio dal dataset, si chiede alla UNet di predire il rumore, si confronta la predizione con il rumore reale per calcolare la loss, e si aggiornano i pesi del modello tramite backpropagation.

### Generazione di Immagini Rimuovendo il Rumore

![image-20250618105513608](./assets/image-20250618105513608.png)

Una volta addestrato, il Noise Predictor è in grado di prendere un'immagine rumorosa e il numero del passo di denoising (la quantità di rumore) e predire quale rumore è presente.

---

![image-20250618112800951](./assets/image-20250618112800951.png)

Il processo di **reverse diffusion** funziona così:

1. Si parte da un'immagine di puro rumore casuale.
2. Si dà questa immagine rumorosa in input al Noise Predictor.
3. Si **sottrae il rumore predetto** dall'immagine rumorosa.
4. Il risultato è un'immagine leggermente meno rumorosa e più strutturata.

![image-20250618112950260](./assets/image-20250618112950260-1750238994663-1.png)

Questo processo viene ripetuto per un certo numero di passi. Iterando la predizione e la sottrazione del rumore, un'immagine coerente emerge dal rumore iniziale. Il tipo di immagine generata dipende interamente dai dati su cui il modello è stato addestrato (se alleniamo con immagini di loghi, il modello imparerà a generare loghi).

### Noise Schedule

Il processo di de-noising è graduale. Il **noise schedule** è un piano predefinito che stabilisce **la quantità di rumore attesa a ogni passo del campionamento**.

L'algoritmo che esegue la reverse diffusion sottrae a ogni passo la giusta quantità di rumore predetto per raggiungere il livello di rumore previsto dallo schedule per il passo successivo.

Lo schedule può essere lineare (sottraendo la stessa quantità a ogni passo) o non lineare (es. sottraendo più rumore all'inizio).

### Diffusion su Dati Latenti

Per accelerare il processo di generazione delle immagini, la diffusion **non avviene nello spazio dei pixel**, ma su una **versione compressa dell'immagine**. Questo spazio compresso è chiamato **spazio latente**.

La compressione (e la successiva decompressione) viene effettuata tramite un **Variational Autoencoder (VAE)**. L'encoder del VAE comprime l'immagine nello spazio latente, e il decoder la ricostruisce alla fine del processo.

## Training di Stable Diffusion

Per fare reverse diffusion, bisogna sapere quanto rumore è stato aggiunto ad un'immagine. La risposta sta nell'addestrare la rete neurale a predire il rumore aggiunto (con il Noise Predictor, modello U-Net). 

![image-20250623154759873](./assets/image-20250623154759873.png)

![image-20250623154822614](./assets/image-20250623154822614.png)

Il training funziona come segue:

1. Si seleziona un'immagine di training.
2. SI genera un rumore casuale.
3. Si aggiunge il rumore generato per un certo numero di passi. Il Noise Predictor stima il rumore totale aggiunto per ciascun passo.
4. Si insegna il Noise Predictor a determinare quanto rumore è stato aggiunto. Questo viene fatto adattando i pesi e mostrando la soluzione corretta.

Si genera prima un'immagine completamente casuale e poi si chiede al Noise Predictor di prevedere il rumore. Dopodichè si sottrae quel rumore predetto dall'immagine originale (questo processo viene ripetuto più volte). Infine, otterremo l'immagine desiderata.

C'è però un limite in questo processo: è **non condizionato**. Non abbiamo alcun controllo su ciò che viene generato. La soluzione è il **conditioning**.

### Spazio di Reverse Diffusion

Il processo di diffusion descritto finora, se eseguito nello spazio dei pixel, sarebbe **computazionalmente molto lento**. Lo spazio dei pixel è enorme: un'immagine 512x512 con 3 canali di colore ha 786.432 dimensioni. Altri modelli come Imagen di Google e DALL-E di OpenAI operano nello spazio dei pixel, ma richiedono enormi risorse computazionali.

### Latent Diffusion

Stable Diffusion è un **modello di latent diffusion**. Opera in uno spazio compresso che è circa 48 volte più piccolo, rendendolo molto più veloce.

Il componente che gestisce questa compressione/decompressione è il **Variational Autoencoder (VAE)**, con la sua struttura a Encoder e Decoder.

Il processo di diffusion (sia in avanti durante il training, sia inverso durante la generazione) **avviene interamente in questo spazio latente**. Invece di aggiungere rumore ai pixel di un'immagine, si aggiunge "rumore latente" alla rappresentazione latente di quella immagine. Questo aumenta drasticamente la velocità.

### Risoluzione dell'Immagine

La dimensione del tensore latente dipende dalla risoluzione dell'immagine di output. Per un'immagine 512x512, il tensore latente è 4x64x64.

Poiché Stable Diffusion v1 è stato addestrato principalmente su immagini 512x512, generare immagini molto più grandi può causare errori come la duplicazione di oggetti.

### Perché lo Spazio Latente è Possibile?

Il motivo per cui un'immagine può essere compressa in uno spazio latente molto più piccolo senza perdere troppa informazione è che le **immagini naturali non sono casuali**, ma hanno una grande regolarità (es. i volti hanno una struttura specifica, i cani hanno 4 zampe, ecc.).

Questa idea è nota come **manifold hypothesis** nel machine learning, con cui si definisce che le immagini naturali si possono comprimere in uno spazio latente ristretto senza perdita di informazione.

### Reverse Diffusion nello Spazio Latente

1. Si genera una **matrice casuale nello spazio latente**.
2. Il Noise Predictor stima il **rumore della matrice latente**.
3. Il rumore stimato viene **sottratto** dalla matrice latente.
4. I passi 2 e 3 sono ripetuti.
5. Il decoder del VAE **converte** la matrice latente nell'immagine finale.

Il processo di forward (usando l'encoder dell'AE) è il modo di generare dati per allenare il Noise Predictor.

Una volta allenato, si possono generare le immagini eseguendo il processo inverso (usando il decoder dell'AE).

### File VAE

I file VAE sono dei **decoder VAE fine-tuned**, usati in Stable Diffusion v1 per migliorare dettagli come occhi e volti. 

I VAE originali perdono dell'informazione nello spazio latente, ma i VAE decoder sui quali viene effetuato fine-tuning possono ricreare certi dettagli visivi.

### Conditioning

Il problema finora è che la generazione è casuale. Non sappiamo come il prompt testuale ci permetta di generare un'immagine. Il **conditioning** è il processo per **guidare il Noise Predictor a generare ciò che vogliamo**, basandosi su un input esterno come un testo.

![image-20250618120932951](./assets/image-20250618120932951.png)

Il **Prompt Testuale** viene prima trasformato in numeri dal **Tokenizer**, poi convertito in vettori dall'**Embedding**, e infine processato da un **Text Transformer** prima di essere passato al predittore di rumore.

Il Text Transformer non solo elabora gli embedding, ma fornisce anche un meccanismo per **includere diverse modalità di conditioning**.

Il modello di linguaggio usato come Text Encoder è fondamentale. Stable Diffusion usa **ClipText**. Il paper di Imagen ha dimostrato che usare un modello di linguaggio più grande e potente migliora la qualità dell'immagine generata più che aumentare le dimensioni del generatore di immagini stesso.

Stable Diffusion v1 usava CLIP, mentre la v2 è passata a **OpenCLIP**, una variante molto più grande e trasparente, per migliorare ulteriormente la qualità e la comprensione del testo.

### Training di CLIP

CLIP viene addestrato su un enorme dataset di 400 milioni di coppie immagine-alt, raccolte dal web.

L'architettura di CLIP è composta da due parti: un **Image Encoder** e un **Text Encoder**.

1. Durante il training, un'immagine e la sua didascalia vengono codificate simultaneamente per poi produrre due vettori di embedding separati.

   ![image-20250623160107209](./assets/image-20250623160107209.png)

2. I due embedding vengono confrontati, tipicamente usando la **cosine similarity**. All'inizio dell'addestramento, anche se il testo descrive l'immagine, la loro similarità sarà bassa.

   ![image-20250623160144331](./assets/image-20250623160144331.png)

3. Attraverso la backpropagation, i pesi di entrambi gli encoder vengono aggiornati per **massimizzare la similarità** tra gli embedding di coppie immagine-testo corrispondenti. Questo spinge le rappresentazioni di immagini e testi semanticamente simili a essere vicine nello spazio degli embedding.

---

![image-20250618122217042](./assets/image-20250618122217042.png)

![image-20250618122239512](./assets/image-20250618122239512.png)

Per rendere la generazione controllabile, il Noise Predictor deve essere modificato per usare il prompt testuale come input aggiuntivo. Il dataset di addestramento, quindi, non conterrà solo immagini rumorose e la quantità di rumore, ma anche il testo codificato associato.

### Layer del Noise Predictor di Unet (Senza Testo)

![image-20250618122627367](./assets/image-20250618122627367.png)

![image-20250618122656726](./assets/image-20250618122656726.png)

La Unet è una serie di layer che lavorano per trasformare l'array latente. I layer operano sull'output del layer precedente. Alcuni output, inoltre, sono forniti (via residual connection) per essere processati più avanti nella rete.

### Layer del Noise Predictor di Unet (Con Testo)

![image-20250618122931248](./assets/image-20250618122931248.png)

![image-20250618122953656](./assets/image-20250618122953656.png)

Per includere il conditioning, viene aggiunto un **layer di Attention** tra i blocchi ResNet. I blocchi ResNet continuano a processare l'informazione dell'immagine, mentre i layer di Attention hanno il compito di **fondere le rappresentazioni testuali** all'interno del tensore dell'immagine. Questo viene effettuato a vari stadi del processo di de-noising.

### Cross-Attention

La Unet utilizza gli embedding del testo attraverso un meccanismo di **cross-attention**. È qui che il prompt testuale "incontra" l'immagine.

Prendamo per esempio "A man with blue eyes":

1. Il self-attention **all'interno del prompt** lega "blue" a "eyes".
2. La cross-attention **tra prompt e immagine** guida il processo di reverse diffusion verso immagini che contengono "blue eyes".

Tecniche di fine-tuning come Hypernetwork e LoRA agiscono proprio modificando questo modulo di cross-attention per inserire stili personalizzati.

#### Cross-Attention nell'Architettura Transformer

![image-20250618130746867](./assets/image-20250618130746867.png)

La cross-attention è un meccanismo che **mescola due sequenze di embedding diverse**. Le sequenze possono essere di modalità diverse (testo, immagine, ecc.) ma devono avere la stessa dimensione di embedding.

Una sequenza fornisce la **Query $Q$** (la lunghezza dell'output), mentre l'altra fornisce le **Key ($K$) e i Value ($V$)**.

![image-20250618130900312](./assets/image-20250618130900312.png)

La differenza fondamentale con la self-attention è che quest'ultima usa una singola sequenza di input per Q, K e V, mentre la cross-attention ne usa due.

#### Algoritmo di Cross-Attention

1. Si hanno due embedding $S_1$ ed $S_2$.
2. Calcolare $K$ e $V$ da $S_1$.
3. Calcolare $Q$ da $S_2$.
4. Calcolare la matrice di attention dalle $K$ e dalle $Q$.
5. Applicare le $Q$ sulla matrice di attention.
6. Produrre le sequenze con dimesione e lunghezza di $S_2$.

$$
softmax((W_QS_2)(W_KS_1)^T)W_VS_1
$$

La formula è identica a quella della self-attention, ma evidenzia che $Q$ proviene da una sequenza $S_2$, mentre $K$ e $V$ provengono da una sequenza $S_1$.

---

![image-20250618131253408](./assets/image-20250618131253408.png)

La cross-attention era già presente nel decoder del Transformer originale (nel layer "Encoder-Decoder Attention").

### Altre Forme di Condizionamento

Il testo non è l'unico modo per condizionare un modello di diffusion.

#### Image-to-Image

Questo metodo usa sia un'**immagine di input** che un prompt testuale per guidare la generazione.

##### Passaggi

1. L'immagine di input viene "encodata" nello spazio latente.
2. Si aggiunge il rumore all'immagine latente.
   + La forza di de-noising determina quanto rumore aggiungere.
   + Se 0, non si aggiunge rumore.
   + Se 1, si aggiunge il massimo quantitativo di rumore (l'immagine diventa un tensore completamente casuale).
3. Il Noise Predictor prende l'immagine latente con rumore e il prompt testuale come input e produce il rumore (sempre nello spazio latente).
4. Si sottrae il rumore latente dall'immagine latente, producendo una nuova immagine latente.

Gli step 3 e 4 sono ripetuti per un certo numero di passi.

5. Infine, il decoder del VAE converte l'immagine latente nello spazio di pixel.

#### Inpainting

Questo è un caso particolare di image-to-image, dove il rumore viene aggiunto solo alle aree dell'immagine che si vogliono modificare.

#### Depth-to-Image

Questa è un'evoluzione dell'image-to-image che aggiunge un ulteriore segnale di condizionamento: una **mappa di profondità**, stimata dall'immagine di input con un modello come MiDaS.

La U-Net viene quindi condizionata dal prompt, dall'immagine rumorosa e dalla mappa di profondità, permettendo di generare immagini che rispettano la struttura 3D della scena originale.

##### Passaggi

1. L'immagine di input viene "encodata" nello spazio latente.

   ![image-20250623162704723](./assets/image-20250623162704723.png)

2. MiDaS (un modello di AI) stima la mappa di profondità dall'immagine di input.

   ![image-20250623162643135](./assets/image-20250623162643135.png)

3. Si aggiunge il rumore all'immagine latente.
   + La forza di de-noising determina quanto rumore aggiungere.
   + Se 0, non si aggiunge rumore.
   + Se 1, si aggiunge il massimo quantitativo di rumore (l'immagine diventa un tensore completamente casuale).

   ![image-20250623162625154](./assets/image-20250623162625154.png)

4. Il Noise Predictor stima il rumore dello spazio latente, condizionato dal prompt testuale e dalla mappa di profondità.

   ![image-20250623162601770](./assets/image-20250623162601770.png)

5. Si sottrae il rumore latente dall'immagine latente, producendo una nuova immagine latente.

   ![image-20250623162729757](./assets/image-20250623162729757.png)

Gli step 4 e 5 sono ripetuti per un certo numero di passi.

6. Infine, il decoder del VAE converte l'immagine latente nello spazio di pixel.

   ![image-20250623162746531](./assets/image-20250623162746531.png)

### Classifier Guidance

La **Classifier Guidance** è un modo per incorporare etichette di immagini nei modelli di diffusione per guidare il processo di generazione. Il parametro **classifier guidance scale** controlla quanto strettamente il processo debba seguire l'etichetta fornita.

Supponiamo di avere tre gruppi di dati ("gatto", "cane", "umano"). **Senza guida**, il modello campiona liberamente e può generare immagini ambigue che si trovano a metà tra le categorie. **Con guida**, il campionamento viene "spinto" verso il centro della distribuzione della classe desiderata. Una guidance scale più alta produce esempi più "puri" e inequivocabili di quella classe.

Con una classifier guidance scale elevata, se si chiede un "gatto", il modello genererà un'immagine che è inequivocabilmente un gatto e nient'altro, evitando ambiguità.

Il problema è che la Classifier Guidance classica richiede l'addestramento di un **modello classificatore extra** (che impari a classificare le immagini rumorose), il che aggiunge complessità.

Esiste però la **Classifier-Free Guidance (CFG)**, che ottiene lo stesso risultato "senza un classificatore". Al posto di usare delle label e un modello separato per la guida, propone di usare le didascalie delle immagini e allenare un diffusion model condizionale, esattamente come per il task di text-to-image.

Durante l'addestramento, il modello impara a fare predizioni sia in modalità **condizionata** (usando il prompt testuale) sia in modalità **non condizionata** (ignorando il prompt). In fase di inferenza, le due predizioni vengono combinate per controllare la generazione.

#### CFG Scale

La **Classifier-Free Guidance (CFG) scale** è un valore che controlla quanto il prompt testuale debba influenzare il processo di diffusione:

+ Se la scala è 0, il prompt viene ignorato e la generazione è casuale (non condizionata).
+ Valori più alti "forzano" il modello a seguire più fedelmente il prompt.

### Confronto tra Stable Diffusion v1 e v2

La differenza principale risiede nel Text Encoder:

- La v1 usava **CLIP** di OpenAI, addestrato su dati proprietari.
- La v2 usa **OpenCLIP**, un modello open-source molto più grande e trasparente. Il passaggio è stato fatto per migliorare la qualità delle immagini (text encoder più grandi aiutano) e per avere maggiore trasparenza nello sviluppo.

La v2 è stata addestrata su un dataset (LAION-5B) filtrato per rimuovere materiale NSFW (Not Safe For Work), mentre per la v2.1 questo filtro è stato successivamente allentato.

Con la v2, gli utenti hanno trovato **più difficile generare stili artistici specifici e volti di celebrità**. Questo è probabilmente dovuto alla differenza nei dati di addestramento: il dataset proprietario di CLIP (usato per la v1) conteneva probabilmente più opere d'arte e foto di celebrità rispetto al dataset LAION, più orientato al web generico.