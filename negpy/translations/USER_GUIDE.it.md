# Guida utente di NegPy

NegPy trasforma le scansioni di negativi e diapositive in immagini finite. Funziona come una camera oscura: scegli una carta, poi imposti luminosità, contrasto e colore. I tuoi file originali non cambiano mai; NegPy conserva le tue modifiche in un proprio database.

Per la matematica dietro ogni passaggio, leggi [PIPELINE.md](PIPELINE.md).

---

## 1. Il quadro generale

### Termini usati in questa guida

*   **Rullino**: un gruppo di fotogrammi, di solito una pellicola. Ciò che un rullino condivide si trova nella scheda Rullino.
*   **Rullino-cartella**: un rullino formato dai file immagine di una cartella. I nuovi file della cartella vi entrano.
*   **Rullino virtuale**: un rullino formato da un elenco fisso di fotogrammi, da qualsiasi cartella. I nuovi file non vi entrano mai.
*   **Densità**: quanto è scura una parte del negativo o della stampa. Un negativo denso è scuro; su una stampa, più densità significa più scuro.
*   **Densità di stampa**: quanto è chiara o scura la stampa, come il tempo di esposizione sotto l'ingranditore.
*   **Gradazione**: il contrasto della stampa, come la gradazione di contrasto della carta da camera oscura. NegPy usa la scala ISO-R, in cui un numero più basso dà più contrasto.
*   **Stop**: un raddoppio della luce. +1 stop è il doppio della luce, −1 stop la metà.
*   **Supporto di stampa**: la carta su cui viene stampata l'immagine. Ogni carta ha il proprio contrasto e i propri colori.
*   **Filtratura**: correzione del colore, come i filtri colorati di un ingranditore.
*   **Schermatura e bruciatura**: rendere un'area della stampa più chiara (schermatura) o più scura (bruciatura).
*   **Ombre, mezzitoni, alte luci**: le parti scure, intermedie e chiare dell'immagine.
*   **Piede e spalla**: l'estremità scura e quella chiara della curva della carta, dove i toni si avvicinano tra loro.
*   **Bordo pellicola**: il bordo della pellicola attorno all'immagine.

### Disposizione dello schermo

*   **Area di lavoro** (al centro): l'immagine. Scorri o pizzica per lo zoom, trascina per spostarti e fai doppio clic per passare da adatta a 100%. Il clic destro offre opzioni di visualizzazione, copia e incolla delle impostazioni, reimpostazioni e **Chiudi**.
*   **Striscia pellicola** (sotto l'area di lavoro): i fotogrammi aperti in una riga. Trascinane il bordo superiore per ridimensionarla; `Ctrl+[` la ripiega.
*   **Barra degli strumenti** (in basso): **Foto** contiene navigazione, zoom, rotazione, **Strumenti** (ritaglio, prelievi, maschere, ritocco), annulla, il menu **Stampa** (Prima / Dopo e le sbirciate), **Media del rullino**, **Esporta** e **Sovrapposizioni**. **Esporta** fa ciò che fa il pulsante del pannello Esporta, e la sua freccia sceglie i fotogrammi. **Rullino** contiene importazione, filtro, ordinamento e azioni del rullino della Striscia pellicola. Il menu **⋯** contiene tutte le azioni, comprese le **Preferenze…**.
*   **Controlli** (a destra): le schede **Libreria**, **Rullino**, **Fotogramma**, **Metadati**, **Esporta** e **Scansione**. Solo Rullino e Fotogramma cambiano l'immagine. Trascina il pannello dal bordo superiore per renderlo mobile.
*   **Info fotogramma** (`I`): la striscia sopra l'area di lavoro. Mostra a sinistra nome file, pellicola e dimensioni, a destra la ricetta di stampa e al centro i messaggi. Un'operazione in corso riempie la barra lungo il bordo inferiore, un segmento per fotogramma per un lotto breve; **Interrompi** la ferma.

**Maiusc+H** nasconde i pannelli, e **Reimposta la disposizione dei pannelli** (Preferenze → Interfaccia, oppure `Ctrl+Maiusc+L`) li ripristina. Quando non è aperto nulla, l'area di lavoro offre **Carica scansioni…** e **Fai la visita guidata**.

### Il flusso di lavoro

1.  Nella scheda **Libreria**, importa una cartella come rullino e aprila.
2.  Nella scheda **Rullino**, controlla **Configurazione** (tipo di pellicola e scanner).
3.  In **Fotogramma › Geometria**, ritaglia e raddrizza. Poi esegui l'**Analisi del rullino** nella scheda Rullino, così i fotogrammi si uniformano.
4.  Nelle altre schede di **Fotogramma**, imposta luminosità, contrasto e colore (**Esposizione**), l'aspetto (**Aspetto**) e rimuovi la polvere (**Finitura**).
5.  Esporta nella scheda **Esporta**.

| Scheda | Riquadri |
|---|---|
| **Rullino** | Configurazione · Decodifica RAW · Ottica · Calibrazione · Misurazione · Analisi del rullino · Composizione del fotogramma |
| **Fotogramma › Geometria** | Ritaglio · Raddrizza |
| **Fotogramma › Esposizione** | Filtratura · Tono · Risposta della carta · Schermatura e bruciatura |
| **Fotogramma › Aspetto** | Lab · Processi alternativi · Viraggio |
| **Fotogramma › Finitura** | Ritocco · Rifinitura |
| **Fotogramma › Preferiti** (stella) | I cursori che hai scelto · Preimpostazioni |
| **Fotogramma › Cronologia** (orologio) | Stampe di lavoro · Cronologia delle modifiche |
| **Metadati** | Anteprima dei metadati · Scatto · Processo · Preimpostazioni · La mia attrezzatura |
| **Esporta** | File · Prova · Preimpostazioni · Accanto alle esportazioni |
| **Scansione** | Scanner · i riquadri dello scanner · Uscita |

### Usare i controlli

Su un cursore, fai clic sul valore per digitare un numero, trascina il nome per regolarlo (**Maiusc** per passi fini) e fai doppio clic per reimpostarlo. Una freccia nasconde i controlli di regolazione fine; un punto su una sezione chiusa, un riquadro o una scheda segnala un valore modificato. In un menu a tendina, **‹** e **›** scorrono tra le scelte. Ogni riquadro ha una freccia di reimpostazione e una **ⓘ** che apre la sua parte di questa guida.

Una scheda con più riquadri ha una barra come **3 riquadri su 5 modificati**, con reimposta, **Reimposta al rullino**, **Applica** e comprimi per l'intera scheda. Nelle schede Rullino e Metadati, **Applica** dà al rullino ogni riquadro che questo fotogramma ha modificato. In una scheda Fotogramma, copia la scheda su altri fotogrammi.

### Fotogramma o rullino

Ogni riquadro delle schede **Rullino** e **Metadati** ha **Fotogramma** (ambra) e **Rullino** (rosso). Quello acceso indica dove si trovano i valori del riquadro.

*   Il riquadro segue il rullino finché non lo modifichi su questo fotogramma. **Rullino** dà al rullino i valori di questo fotogramma; **Fotogramma** li mantiene solo su questo fotogramma.
*   **Reimposta al rullino** riporta sul riquadro i valori del rullino.
*   Un riquadro di una scheda **Fotogramma** contiene i valori propri di questo fotogramma. Il suo pulsante di copia, **Applica ai fotogrammi**, li copia una volta sulla selezione o sull'intero rullino.

Fai clic destro sull'intestazione di un riquadro per **Copia le impostazioni del riquadro** e **Annulla la modifica a più fotogrammi**. Dopo un Applica, Incolla, Reimposta o un invio al rullino su più fotogrammi, la striscia dei messaggi sopra l'immagine offre anche **Annulla**, che riporta indietro tutti i fotogrammi insieme. Funziona fino alla tua modifica successiva. Un'impostazione cambiata di nuovo dopo l'invio mantiene il suo nuovo valore. I fotogrammi che non formano un unico rullino mostrano solo **Fotogramma** finché non li salvi come rullino.

### Cosa passa al fotogramma successivo

Un nuovo fotogramma riceve il tipo di pellicola del rullino, e il ritaglio, il supporto di stampa e le impostazioni Lab dell'ultimo fotogramma che hai modificato. L'aspetto (luminosità, colore, schermatura e bruciatura, viraggio) parte da zero. **Preferenze → Sessione e archiviazione → Impostazioni persistenti…** stabilisce cosa passa. Ottica e Recupero passano solo ai fotogrammi RAW di fotocamera.

**Reimposta impostazioni** riporta un fotogramma ai valori predefiniti. Non cambia le scelte di scansione del riquadro **Configurazione**.

### Visualizzazione

*   **Prima / Dopo** (`\`, oppure il menu **Stampa**): divide l'area di lavoro tra la conversione predefinita e la tua modifica. Tieni premuto `\` per vedere solo quella predefinita. Per confrontare con una versione precedente, fai clic destro su di essa in **Cronologia**.
*   **Sbirciate**: tieni premuto `N` per la scansione così come è stata caricata (il negativo), `M` per il master piatto e `P` per il JPEG della fotocamera. Si trovano anche nel menu **Stampa**.
*   **Media del rullino** (`U`, oppure il menu **Stampa**): stampa questo fotogramma sulla media e non cambia nulla. La barra ha **Luma**, **Colore** e **Dominante**; fai clic destro su una per attivarle o disattivarle tutte. Le medie che il fotogramma usa ora compaiono dopo **Applicato**. Su un fotogramma in una scena, **Scena** e **Rullino** scelgono quale media. **Applica** le salva nel fotogramma. La vista resta aperta quando passi al fotogramma successivo.
*   Pulsante **Media del rullino** (icona a pellicola sulla barra degli strumenti): apre la barra **Media del rullino**. È acceso mentre il fotogramma usa una media. Il clic destro attiva o disattiva tutte le medie in una volta, come un'unica modifica. Un'icona ambra segnala un fotogramma lontano dal colore del rullino.
*   **Dimensioni di esportazione** (menu **Stampa**): mostra il fotogramma come viene esportato, alle dimensioni impostate dal pannello **Esporta**, al 100%. Usalo per valutare nitidezza e grana. Si aggiorna un attimo dopo ogni modifica.
*   **Focometro** (`Maiusc+L`, menu **Sovrapposizioni**): una lente sotto il puntatore che mostra i pixel esportati alla loro dimensione reale, con un valore di nitidezza.
*   **Vista di riferimento** (`Maiusc+R`): fissa questo fotogramma accanto all'area di lavoro, per adeguarvi gli altri fotogrammi.
*   **Sonda sul cursore** (`Maiusc+I`, menu **Sovrapposizioni**): mostra la densità e la zona sotto il puntatore.
*   **Trova** (`Ctrl+K`): trova per nome qualsiasi cursore, riquadro o azione, oppure con la parola usata da un'altra app (*contrasto* trova **Gradazione ISO-R**).

`Esc` chiude qualsiasi sbirciata, divisione o striscia di prova.

### Scegli il migliore

**Scegli il migliore** (la bilancia sulla barra della scheda Esposizione) aiuta quando non sei sicuro di quale cursore muovere. Mostra la tua modifica al centro con due versioni ai lati. Fai clic su quella migliore, oppure su quella centrale per saltare. Ogni turno regola un controllo, e i turni successivi si ripetono con passi più fini. **Differenza** stabilisce quanto sono distanti le due versioni, e **Instradamento personalizzato** stabilisce l'ordine dei controlli. **OK** o `Esc` termina e mantiene le tue scelte.

### La visita guidata

**Fai la visita guidata** (⋯ → **Aiuto**) percorre NegPy in brevi capitoli. Quando non è aperto nulla, **Carica negativo dimostrativo** apre un esempio. `Esc` termina la visita; la successiva offre **Riprendi da dove avevi lasciato**.

### macOS e Windows

`Ctrl` in questa guida corrisponde a `⌘` su macOS, dove i menu **NegPy**, **Finestra** e **Aiuto** contengono Preferenze, comandi delle finestre, la visita guidata e il controllo degli aggiornamenti.

Se Windows blocca la cartella dei dati predefinita, NegPy suggerisce `%LOCALAPPDATA%\NegPy\data`. Fai clic sul percorso per sceglierne un altro, poi su **Usa questa cartella**. Non viene copiato nulla. `NEGPY_USER_DIR` ha la precedenza sulla scelta.

### Riga di comando e automazione

Script e altri strumenti possono avviare NegPy dalla riga di comando, per esempio per aprire una scansione da modificare o per creare anteprime.

`NegPy <file> [<file>…]` apre i file e mostra il primo. Una cartella apre tutte le sue immagini. Se NegPy è già aperto, i file si aprono in quella finestra e la finestra passa in primo piano. Su Linux, **Apri con NegPy** in un gestore di file apre anch'esso i file.

Il comando può anche stabilire come NegPy esporta questi file, così il passaggio successivo del flusso di lavoro trova il risultato. Le impostazioni valgono finché non chiudi NegPy, e solo per questi file:

| Opzione | Valore |
|------|-------|
| `--preset <name>` | Una preimpostazione di esportazione salvata. Le altre opzioni ne cambiano i valori. |
| `--intent` | `print`, `flat` o `linear` |
| `--format` | `jpeg`, `tiff`, `png`, `jxl` o `webp` |
| `--bit-depth` | `8` o `16` |
| `--quality` | da 1 a 100, per JPEG e WebP |
| `--color-space` o `--profile` | Uno spazio colore come `srgb` o `adobe-rgb`, oppure il percorso di un file ICC |
| `--original-size`, `--long-edge <px>` o `--print-size <cm>` con `--dpi` | Le dimensioni |
| `--ratio` | Le proporzioni della carta, come `3:2` |
| `--output-dir <folder>` e `--filename <pattern>` | Dove va il file e il suo nome |
| `--no-overwrite` | Chiede prima che un'esportazione sostituisca un file. Senza questa opzione, un'esportazione sostituisce quella precedente. |
| `--quit-after-export` | Dopo un'esportazione riuscita, NegPy si chiude. Se NegPy era già aperto, la sua finestra viene ridotta nella barra delle applicazioni. |

Per un fotogramma di questo tipo, la scheda Esporta nasconde le impostazioni fissate dal comando e le elenca in alto. Le preimpostazioni sono disattivate per il fotogramma. Se un'opzione ha un valore errato, NegPy non si avvia e scrive l'errore in `negpy.log`.

#### Creare anteprime senza finestra

`NegPy --render <file or folder>… --format jpeg --long-edge 1600 --output-dir <folder>` scrive un positivo di ogni scansione ed esce. Una cartella fornisce tutte le sue scansioni, escluse le sottocartelle. Usa le opzioni di esportazione indicate sopra, tranne `--intent linear` e `--quit-after-export`.

*   Una scansione che hai modificato in NegPy usa la sua modifica. Le altre scansioni usano le impostazioni standard.
*   `--auto-crop` ritaglia sull'immagine una scansione che non ha un ritaglio. Usalo per le scansioni che mostrano il supporto o la sorgente luminosa.
*   Il codice di uscita è 0 quando tutti i file vengono scritti, 1 quando un file non riesce e 2 per un'opzione errata.
*   Questo comando non cambia la tua libreria NegPy né le tue modifiche. Funziona anche mentre NegPy è aperto.

---

## 2. Libreria e Striscia pellicola

<!-- panel:library src:401ca42f -->
### Libreria

La scheda **Libreria** (`Ctrl+L`) elenca i tuoi **rullini**. Un rullino è un gruppo di fotogrammi con un nome, di solito una pellicola: un **rullino-cartella** o un **rullino virtuale**. L'importazione registra la cartella; NegPy non sposta, rinomina né elimina mai i tuoi file.

*   **+**: **Importa cartella come rullino…** crea un rullino da una cartella e lo apre. **Importa sottocartelle come rullini…** crea un rullino da ogni sottocartella che contiene immagini. Anche trascinare delle cartelle sulla Libreria le importa.
*   **Fai doppio clic** su un rullino (o premi Invio) per aprirlo.
*   **⋮** o clic destro su un rullino: **Apri**, **Chiudi tutti…** (il rullino aperto), **Analisi del rullino**, **Rinomina…**, **Individua cartella…**, **Rimuovi fotogrammi mancanti…**, **Salva nel rullino-cartella**, **Ripristina fotogrammi rimossi…**, **Reimposta rullino…** (con i suoi fotogrammi) ed **Elimina…**. Ripristina fotogrammi rimossi riporta i fotogrammi che Rimuovi dal rullino ha tolto. Elimina conserva i file e le modifiche condivise. Perde le impostazioni del rullino, il riferimento, la base della pellicola, le scene, i blocchi dei riquadri e le modifiche proprie, e Aggiorna non aggiunge di nuovo un rullino-cartella eliminato.
*   **Filtra i rullini…** e **Ordina** ti aiutano a trovare un rullino. **⋮ → Aggiorna** trova nuove sottocartelle e riconta i fotogrammi. **⋮ → Filtri di ricerca…** elenca i nomi di cartella da saltare (inizialmente `export`).

Un rullino la cui cartella o i cui fotogrammi sono stati spostati fuori da NegPy mostra **cartella mancante** o **file mancanti**. **Individua cartella…** trova la nuova cartella. Un nome di cartella che corrisponde alla tua attrezzatura ("penf" per "Pen F") apre le Impostazioni del rullino già compilate.

Un rullino-cartella contiene solo i file della propria cartella. Un file che aggiungi da un'altra cartella resta solo per questa sessione.

Per salvare come rullino i fotogrammi della Striscia pellicola, usa la Striscia pellicola. **Salva nel rullino-cartella** rende i fotogrammi che provengono tutti da una cartella il rullino di quella cartella; gli altri suoi file risultano rimossi. **Salva come rullino virtuale…** conserva esattamente questi fotogrammi, da qualsiasi cartella. Salva nel rullino-cartella trasforma anche un rullino virtuale di una sola cartella nel rullino di quella cartella. Il nuovo rullino parte con le impostazioni, i blocchi dei riquadri e le modifiche fatte solo nel rullino aperto. Un fotogramma ha un'unica modifica in ogni rullino che lo contiene; clic destro → **Modifica in modo indipendente in questo rullino** gli dà una modifica propria in un rullino.

<!-- panel:frames src:c1f54a29 -->
### Striscia pellicola

La **Striscia pellicola** sotto l'area di lavoro mostra i fotogrammi aperti in una riga. Trascinane il bordo superiore per ingrandire i fotogrammi.

La riga **Rullino** della barra degli strumenti contiene:

*   Il nome del rullino e il numero di fotogrammi, di quelli da tenere e di quelli scartati.
*   La **casella del filtro**: digita parte di un nome file, oppure termini come `film:portra iso:>=400` (vedi sotto). Il pulsante di ricerca, o **Invio**, cerca nell'intera libreria.
*   **Ordina**: nome, data, scena o numero di fotogramma.
*   **Filtro foglio**: tutti i fotogrammi, solo da tenere, nascondi scartati, oppure solo non contrassegnati. Il pulsante è evidenziato mentre un filtro nasconde dei fotogrammi.
*   **Impostazioni del rullino** (l'etichetta): attrezzatura e dettagli della pellicola per questo fotogramma, per la selezione o per il rullino.
*   **Applica**: copia le impostazioni scelte di questo fotogramma sulla selezione o sul rullino.
*   **⋮**: **Aggiungi file…**, **Aggiungi cartella…**, **Salva nel rullino-cartella**, **Salva come rullino virtuale…**, **Aggiorna miniature**, **Cartella monitorata** (carica i nuovi file che uno scanner salva nella cartella del fotogramma più recente), **Chiudi…**, **Chiudi tutti…** (svuota la Striscia pellicola; il rullino resta nella Libreria), **Reimposta i fotogrammi ai valori predefiniti…**, **Ripristina fotogrammi rimossi…** e **Guida alla Striscia pellicola…**.

Il pulsante a griglia apre il **Tavolo luminoso** (`Maiusc+G`): l'intero rullino in righe, per lo smistamento. Un clic seleziona un fotogramma, un doppio clic o `Invio` lo apre. La barra degli strumenti agisce sui fotogrammi selezionati: **Tieni**, **Scarta**, ruota e rifletti, **Ritaglio automatico** ed **Esporta**, mentre ⋯ contiene il resto del menu del clic destro. Il cursore **Fotogrammi per riga**, `+` e `-`, oppure `Ctrl` con la rotellina cambiano la dimensione dei fotogrammi. `Esc` torna indietro, e così uno strumento dell'area di lavoro o una sbirciata.

#### Filtrare

Una parola semplice corrisponde al nome del file. I termini `campo:valore` corrispondono ai dati del fotogramma, e un `-` iniziale esclude:

| Termine | Trova |
|---|---|
| `film:portra` | pellicola che contiene "portra" |
| `camera:"Nikon F3"` | virgolette per un valore con spazi |
| `iso:>=400` | i numeri accettano anche `>`, `>=`, `<`, `<=` |
| `date:2024-03` · `shot:>=1998` | data del file · data di ripresa |
| `place:tokyo` · `lens:` · `roll:` | altri campi dei Metadati |
| `scene:beach` | fotogrammi in una scena |
| `keeper:` · `rejected:` · `edited:` | fotogrammi contrassegnati o modificati |

I termini si combinano: `film:portra iso:>=400 -rejected:`. I termini dei metadati corrispondono solo ai fotogrammi di cui hai compilato i Metadati. **Ricerca per significato** (attivala in Preferenze → Prestazioni) trova le immagini tramite una descrizione, come "un cane su una spiaggia".

### Contrassegnare i fotogrammi

Fai clic destro su un fotogramma, o usa le scorciatoie, per **Tenerlo** (un segno di spunta) o **Scartarlo** (una croce; il fotogramma si attenua e le esportazioni in lotto lo saltano). I tuoi file non cambiano mai.

Un segno a forma di pellicola nell'angolo in alto a destra indica che il fotogramma stampa sulla media del rullino; è ambra quando il colore del fotogramma è lontano da quella media. Un segno a forma di immagine accanto indica che il fotogramma mantiene un valore proprio in un riquadro della scheda Rullino, quindi una modifica al rullino non lo raggiunge; passaci sopra per vedere quali riquadri. Un segno nell'angolo in basso a sinistra indica che il fotogramma è costruito da più file (un'unione, una fusione HDR, una tricromia o un mezzo formato), oppure che è stato scattato con un'acquisizione a Miscela o a Esposizione singola; passaci sopra per il nome. Un punto esclamativo indica che il file non si è aperto (fai clic per riprovare); un piccolo punto ambra indica che la miniatura non è aggiornata.

Il menu del clic destro inoltre copia, incolla e reimposta le impostazioni, le applica ad altri fotogrammi, rianalizza i fotogrammi ([§10.3](#103-misurazione-negativo--positivo)), mette la selezione sulla media del rullino o la riporta sulla propria misura, mostra il file nella sua cartella e sposta i file nel Cestino (la modifica resta).

### Unione e HDR

*   **Unisci**: seleziona scatti sovrapposti di un unico negativo grande, poi clic destro → **Unisci i fotogrammi selezionati**. **Separa unione** li separa di nuovo.
*   **HDR** (diapositive): seleziona una forcella di esposizioni distanti uno stop, poi clic destro → **Fondi esposizioni (HDR)**. Clic destro sul fotogramma fuso → **Esposizione di resa** per scegliere l'esposizione che mostra, oppure usa il cursore **Esposizione di resa** nel riquadro Misurazione. **Separa esposizioni** le separa. Fai la forcella in entrambe le direzioni attorno allo scatto misurato.
*   **Fondi fotogramma in negativo TIFF…** salva un'unione come un unico file ([§10.8](#108-composizione-del-fotogramma)).

### Scene

Una scena è un gruppo di fotogrammi scattati con la stessa luce. Ha un proprio riferimento dell'Analisi del rullino, così i suoi fotogrammi corrispondono tra loro. **Rileva scene** in Analisi del rullino trova questi gruppi per te. Per crearne una tu stesso, seleziona i fotogrammi, poi clic destro → **Scena → Raggruppa come scena…**. Lo stesso menu aggiunge e rimuove fotogrammi, esegue **Analizza scena…**, e rinomina o elimina la scena. **Ordina → Scena** li raggruppa nella striscia. Le scene richiedono un rullino; senza, il menu **Scena** è disattivato.

---

<!-- panel:analysis src:7afa1351 -->
## 3. Lettura dell'analisi

Questa lettura descrive il fotogramma corrente mentre lo modifichi.

#### Grafico

*   **Curva**: la curva della carta, dal negativo (in orizzontale) alla stampa (in verticale). Più è ripida, più contrasto. L'istogramma grigio è il negativo, quello colorato la stampa.
*   **Forma d'onda**: la luminosità della stampa lungo il fotogramma.
*   **Parata**: rosso, verde e blu affiancati; un soggetto grigio si trova alla stessa altezza in tutti e tre. **Sovrapposizione Parata** mette i tre in un unico grafico.
*   **Vettorscopio**: i colori della stampa attorno a un punto centrale. Un gruppo decentrato è una dominante di colore.

**Log** mostra le aree piccole, come la tosatura.

**Sovrapposizione analisi**: il grafico sull'immagine mentre il riquadro è chiuso.

#### Tosatura

I triangoli negli angoli superiori avvisano quando le ombre diventano nero puro (a sinistra) o le alte luci bianco puro (a destra). **Sovrapposizione tosatura** (il pulsante a semicerchio) segna quelle parti sull'immagine. **Evidenziazione zona** (il pulsante a pennarello) mostra solo la zona sotto il cursore sul fotogramma.

#### Zone e collocazione delle zone

La **scala di grigi a gradini** stampa 21 gradini di grigio; i gradini che si fondono nel nero o nel bianco sono toni persi. La **striscia delle zone** mostra quanta parte del fotogramma cade in ogni zona, da 0 (nero) passando per V (grigio medio) fino a X (bianco). Fai doppio clic su di essa (o premi `Maiusc+Z`) per la **Sovrapposizione delle zone**, che colora l'immagine per zona.

Fai clic su una zona della striscia, poi su un punto dell'immagine, e NegPy imposta la stampa in modo che quel punto stampi in quella zona. Un segnaposto imposta la Densità di stampa; un secondo imposta anche la Gradazione ISO-R. **Colloca le zone** (o Invio) applica il risultato; `Esc` annulla.

#### Letture

La sezione **Letture** contiene numeri sulla scansione. Un valore ambra è un avviso; passa sopra una riga per i dettagli.

**Base della pellicola** (il mirino): fai clic sulla pellicola trasparente tra i fotogrammi per impostare la base da cui misurano le letture.

*   **Negativo**: la gamma di densità del negativo (Δ).
*   **Esposizione**: fotogramma chiaro (positivo) o scuro (negativo), in stop.
*   **Tosatura**: quota della stampa che è nero puro, poi bianco puro.
*   **Tosatura della scansione**: quota di ogni canale al massimo dello scanner. Nessuna modifica la corregge; esponi la scansione di meno.
*   **Ripara**: quota dell'immagine riscritta dalla rimozione della polvere.
*   **Gamma cromatica**: quota che la stampante di prova non può stampare, durante la prova a video.

---

## 4. Scheda Geometria

Due riquadri: **Ritaglio** stabilisce la parte dell'immagine che conservi, e **Raddrizza** la ruota e la livella. Entrambi contengono i valori propri di questo fotogramma. Il loro pulsante **Applica ai fotogrammi** li copia sulla selezione o sull'intero rullino ([§1](#fotogramma-o-rullino)).

<!-- panel:autocrop src:3047f00d -->
### 4.1 Ritaglio

Scegli la parte dell'immagine da conservare, a mano o con il Ritaglio automatico.

*   **Disegna**: trascina un ritaglio sull'immagine. Trascina un angolo per ridimensionarlo, l'interno per spostarlo o fuori dal riquadro per ruotarlo. `Invio` o **Fatto** conserva il ritaglio, e `Esc` ripristina il ritaglio precedente. La **×** rimuove un ritaglio che hai disegnato. Disegnare disattiva il Ritaglio automatico.
*   **Ritaglio automatico** (anche sulla barra della scheda Geometria): trova i bordi dell'immagine e ritaglia su di essi.
*   **Proporzioni**: la forma del ritaglio, come `3:2` o `1:1`, ruotata per adattarsi al fotogramma. **Libero** usa la forma che trova NegPy, e puoi trascinare un lato alla volta. **Rileva** sceglie la forma standard più vicina.
*   **Guida**: linee di composizione, come i terzi, mentre disegni. Il pulsante di rotazione le ruota.
*   **Modalità** (sezione Impostazioni del ritaglio automatico): **Solo immagine** ritaglia sull'immagine. **Bordo della pellicola** conserva l'intera pellicola con il suo bordo e le perforazioni.
*   **Scostamento del ritaglio** e **Rifinitura del bordo pellicola** (sezione sotto Modalità): spostano il ritaglio all'interno (positivo) o all'esterno (negativo) del bordo trovato, e stabiliscono quanto entra nel bordo della pellicola (Solo immagine). 0% si ferma al bordo della pellicola e 100% al bordo dell'immagine; oltre il 100% entra nell'immagine per eliminare un bordo bianco.
*   **Prima squadra**: raddrizza il fotogramma sui bordi della pellicola prima che il Ritaglio automatico lo ritagli.
*   **Ritaglia tutti i fotogrammi** (Solo immagine): ritaglia tutti i fotogrammi insieme. I fotogrammi nitidi aiutano quelli incerti, e i ritagli che hai disegnato restano.

Il Ritaglio automatico richiede che la luce dello scanner o il supporto sia la parte più luminosa della scansione. Lascia stare un fotogramma quando l'immagine è luminosa quanto la luce, o quando il fotogramma successivo occupa gran parte di un lato. Ritaglia a mano un fotogramma del genere, oppure usa **Bordo della pellicola** e rifila verso l'interno.

<!-- panel:geometry src:ad12a4c6 -->
### 4.2 Raddrizza

Ruota l'immagine e mettila in bolla.

*   **Ruota** e **Rifletti**: ruotano l'immagine di un quarto di giro o la specchiano. La barra degli strumenti ha gli stessi pulsanti.
*   **Livella**: traccia una linea lungo l'orizzonte o un bordo verticale, e l'immagine ruota per metterla in bolla.
*   **Allinea alla pellicola** (anche sulla barra della scheda Geometria): allinea l'immagine ai bordi della pellicola. Ignora il contenuto dell'immagine, quindi una foto scattata inclinata resta inclinata.
*   **Rotazione fine**: ruota l'immagine di piccole quantità. Positivo è in senso orario.
*   **Basculaggio** e **Brandeggio** (sezione ripiegabile): raddrizzano un edificio ripreso dal basso (Basculaggio) o un muro ripreso di lato (Brandeggio). **Traccia linee** imposta entrambi da linee lungo i quattro bordi della pellicola.
*   **Ritaglia gli angoli vuoti**: elimina gli angoli vuoti lasciati da rotazione, Basculaggio e Brandeggio.

Ritaglia prima di un Basculaggio o Brandeggio ampio, altrimenti il bordo della pellicola può entrare nell'inquadratura e scurire la stampa. La distorsione dell'obiettivo si corregge in **Ottica** ([§10.6](#106-ottica)).

---

## 5. Scheda Esposizione

Questi quattro riquadri impostano luminosità, contrasto e colore della stampa.

<!-- panel:color src:23d03265 -->
### 5.1 Filtratura: bilanciamento del colore

Corregge il colore della stampa, come fanno i filtri colorati di un ingranditore da camera oscura. **Globale / Ombre / Alte luci** applica i controlli del colore all'intera immagine, oppure soprattutto alle sue parti scure o chiare.

*   **Prelievo BB**: fai clic su qualcosa nell'immagine che dovrebbe essere grigio, e NegPy corregge il colore per renderlo grigio.
*   **Temperatura**: rende l'immagine più calda o più fredda.
*   **Ciano / Magenta / Giallo**: spostano il colore tra ciano e rosso, magenta e verde, oppure giallo e blu.
*   **Ventaglio di prove** (`Maiusc+F`): mostra l'immagine come una griglia di colori leggermente diversi. Fai clic su quello che sembra giusto. `Esc` lo chiude.
*   **Rimozione della dominante**: rimuove una dominante di colore diversa tra le parti scure e chiare dell'immagine. Parte da 0,5 (metà) sui negativi a colori e da 0 sulle diapositive. Foglie, cielo e acqua non contano come grigio, quindi mantengono il loro colore.
*   **Filtro**: un filtro colorato, come sull'obiettivo della fotocamera per la pellicola in bianco e nero. Compare solo su un negativo a colori o una diapositiva a colori in B/N, e sostituisce i controlli precedenti. Ogni filtro scurisce i colori opposti al proprio. Giallo n. 8, Giallo intenso n. 15, Arancione n. 21 e Rosso n. 25 scuriscono un cielo blu, passo dopo passo. Il rosso rende anche scure le foglie e pallida la pelle. Giallo-verde n. 11 e Verde n. 58 schiariscono le foglie e scuriscono la pelle. Nessuno dà l'aspetto di una normale pellicola in bianco e nero.

<!-- panel:tone src:c29af454 -->
### 5.2 Tono: luminosità e contrasto

Quanto è chiara o scura la stampa, e quanto contrasto ha.

*   **Densità di stampa**: quanto è chiara o scura la stampa (il tempo di esposizione della camera oscura). Più basso è più chiaro.
*   **Gradazione ISO-R**: contrasto (la gradazione della carta in camera oscura). Un numero più basso dà più contrasto; circa 110 è normale.
*   **Auto** (la bacchetta, anche sulla barra della scheda Esposizione): **Densità automatica** e **Gradazione automatica** impostano luminosità e contrasto per ogni fotogramma, così un rullino stampa in modo uniforme. Un cursore mostra poi il valore automatico: spostalo per regolarlo, fai doppio clic per tornare indietro. Fai clic destro sulla bacchetta per attivarle o disattivarle entrambe. Il menu Auto contiene anche:
    *   **Imposta obiettivi…**: la luminosità e il contrasto a cui punta Auto.
    *   **Fissa densità/gradazione automatiche**: mantiene i valori automatici di questo fotogramma e disattiva Auto per esso.
    *   **Mantieni il nero**: quando schiarisci le ombre, i neri più profondi restano neri.
*   **Striscia di prova** (`Maiusc+T`): una griglia di versioni a diverse luminosità e contrasti. Fai clic su quella che preferisci. `Esc` la chiude.
*   **Densità delle ombre** / **Densità delle alte luci** (sezione sotto Densità di stampa): scuriscono o schiariscono solo le ombre o le alte luci.
*   **Gradazione delle ombre** / **Gradazione delle alte luci** (sezione sotto Gradazione ISO-R): contrasto solo nelle ombre o solo nelle alte luci.
*   **Aiuti alla stampa** (sezione ripiegabile):
    *   **Prelampeggiatura**: recupera il dettaglio nelle zone più luminose.
    *   **Diffusione**: un alone morbido; le zone scure si espandono in quelle chiare.
    *   **Maschera di contrasto**: apre le grandi zone scure e trattiene quelle grandi e chiare, conservando il dettaglio fine. In camera oscura, uno stampatore ottiene questo effetto con una maschera morbida e sfocata posta sul negativo. Nella stampa digitale, gli scanner Fujifilm Frontier ottengono un effetto simile con HyperTone. Le zone lontane dal grigio medio cambiano di più, quindi un fotogramma a basso contrasto cambia poco.
*   **Globale / R / G / B**: R, G o B applica le gradazioni a un solo colore, per una dominante diversa tra ombre e alte luci.

**Punto del bianco** e **Punto del nero** si trovano nel riquadro **Misurazione** della scheda Rullino ([§10.3](#103-misurazione-negativo--positivo)).

<!-- panel:paper src:1e3bc30a -->
### 5.3 Risposta della carta: la carta

La carta su cui viene stampata l'immagine, e il modo in cui trasforma la luce in toni dal nero al bianco.

*   **Supporto di stampa**: la carta. Una carta reale stabilisce la curva di contrasto e i colori in base ai dati del produttore. **Digitale (Adobe RGB)**, quella predefinita, è una carta ideale. **Digitale (Pakon)** ha i colori di uno scanner da laboratorio Pakon. **Digitale (Frontier)** ha i colori e la curva di contrasto di una scansione da laboratorio Fujifilm Frontier. **Digitale (Nikon Scan)** ha quelli di Nikon Scan, il software degli scanner per pellicola Nikon Coolscan.
*   **Striscia di prova della carta** (il pulsante a righe): stampa il fotogramma su ogni carta. Fai clic su una carta per vederla in grande, scorri con i tasti freccia e premi **Tieni** per usarla.
*   **Separazione dei coloranti**: intensità del colore nella stampa. 1 è invariato. **Smorzamento della separazione** applica più modifica ai colori spenti e meno a quelli intensi. **Croma** nella scheda Aspetto cambia invece tutti i colori in modo uniforme.
*   **Bianco carta**: mostra il bianco di una carta reale, un po' più scuro del bianco puro.
*   **Nero carta**: mostra il nero di una carta reale, un po' più chiaro del nero puro.
*   **Forma della curva** (sezione ripiegabile):
    *   **Incisività**: contrasto dei mezzitoni. Il nero e il bianco non si spostano.
    *   **Piede**: modella il passaggio delle ombre al nero. Positivo lo ammorbidisce e apre le ombre, negativo lo rende più duro. Il nero non si sposta.
    *   **Spalla**: modella il passaggio delle alte luci al bianco. Positivo lo ammorbidisce, negativo lo rende più duro e può perdere dettaglio. Il bianco non si sposta.
*   **Globale / R / G / B**: R, G o B applica i controlli della curva a un solo strato di colore.

<!-- panel:local src:7d85677b -->
### 5.4 Schermatura e bruciatura: schiarire o scurire alcune aree

Disegna una forma, poi rendi solo quell'area più chiara (schermatura) o più scura (bruciatura). Una diapositiva non usa questo riquadro, ma le sue maschere restano.

*   **Disegna**: fai clic per collocare i punti. Fai doppio clic, premi Invio o fai clic vicino al punto iniziale per chiudere la forma. Trascina un punto per spostarlo, fai clic su un "+" di un lato per aggiungerne uno, e clic destro su un punto per eliminarlo.
*   **Ovale**: trascina un ovale. Le sue maniglie lo spostano, lo allungano e lo ruotano.
*   **Cartoncino**: una bruciatura che sfuma, come un cartoncino mosso sopra la carta. Trascina dal bordo a piena intensità (linea continua) fino al punto in cui finisce (linea tratteggiata).
*   **Elenco delle maschere**: fai clic su una maschera per selezionarla. Le sue icone la attivano o disattivano, la invertono (agisce ovunque tranne al suo interno), ne mostrano il contorno o la eliminano. I cursori sotto modificano la maschera selezionata.
*   **Bruciatura**: positivo scurisce l'area, negativo la schiarisce. L'unità è lo stop: +1 raddoppia la luce.
*   **Sfumatura**: quanto è morbido il bordo.
*   **Gradazione**: contrasto all'interno della maschera. Negativo dà più contrasto.
*   **Lampo**: recupera il dettaglio nelle zone chiare all'interno della maschera, come la Prelampeggiatura.
*   **Limite tonale**: **Alte luci** o **Ombre** fa agire la maschera solo sui toni più chiari o più scuri di **Zona tonale** (0 è il nero, V il grigio medio, X il bianco). Una bruciatura del cielo si ferma così alla linea dell'orizzonte. **Morbidezza tonale** stabilisce quanto gradualmente inizia.

**Note di stampa** (`Maiusc+N`) segna le maschere sull'immagine con i loro numeri e valori, come fanno gli appunti di uno stampatore.

---

## 6. Scheda Aspetto

<!-- panel:lab src:e8e4a89c -->
### 6.1 Lab: colore e dettaglio

Le regolazioni che uno scanner da laboratorio fa dopo la stampa.

*   **Croma**: intensità del colore. 1 è invariato, 0 è bianco e nero.
*   **Protezione dell'incarnato**: evita che gli incarnati diventino troppo intensi di colore.
*   **Riduzione rumore cromatico**: rimuove i puntini colorati. La grana della pellicola resta. È attiva a 2,5 per la pellicola a colori.
*   **Nitidezza**: rende più nitidi i dettagli senza bordi colorati. La sua sezione contiene **Metodo** (Maschera di nitidezza rende i contorni più netti; Deconvoluzione annulla la sfocatura dello scanner), **Raggio** (la dimensione del dettaglio, in pixel esportati; valutalo in **Dimensioni di esportazione** o con il **Focometro**) e **Mascheratura** (rende nitidi solo i contorni, così cielo e pelle restano uniformi).
*   **Mascheratura automatica** (la bacchetta accanto a Metodo, anche nell'intestazione della scheda): imposta la Mascheratura in base alla grana di ogni fotogramma. Mascheratura mostra il valore in uso, e un trascinamento lo sposta in su o in giù da lì. Un'esportazione misura di nuovo la grana alle proprie dimensioni.
*   **CLAHE**: contrasto locale, per far emergere i dettagli. Valori forti possono sembrare innaturali.
*   **Bagliore** (sezione Effetti): un alone morbido attorno alle zone luminose, come la messa a fuoco morbida di uno scanner da minilab.
*   **Alonatura** (sezione Effetti): un alone rosso attorno alle zone luminose, come danno alcune pellicole.

<!-- panel:altproc src:1b810792 -->
### 6.2 Processi alternativi

Processi di stampa storici per i negativi B/N. Scegli **Nessuno**, **Lith**, **Cianotipia** o **Sabattier**, e compaiono solo i controlli di quel processo.

#### Lith

Alte luci cremose e calde e neri duri e profondi. Il supporto di stampa in Risposta della carta stabilisce il colore. Su una stampa lith funzionano solo i viraggi al selenio e all'oro.

*   **Esposizione**: di più dà alte luci più calde e colorate e toni più morbidi.
*   **Punto di estrazione**: quando la stampa esce dal rivelatore. Più alto dà neri più profondi e freddi.
*   **Bruschezza**: quanto bruscamente le ombre diventano nere.

#### Cianotipia

Una stampa blu con alte luci verdi. I viraggi non funzionano su di essa; usa Sbianca e Tannino.

*   **Sensibilizzante**: **Classica** dà un azzurro chiaro con alte luci verdi. **Nuova** dà blu più profondi e puliti.
*   **Esposizione**: di più rende la stampa più blu.
*   **Scala di esposizione**: contrasto. Una scala più corta dà più contrasto.
*   **Sbianca**: rimuove il blu, a partire dalle alte luci.
*   **Tannino**: rende marroni le zone sbiancate. Usa prima Sbianca per un marrone pieno.

#### Sabattier

Un lampo di luce durante lo sviluppo. I toni chiari si invertono verso il grigio, e sottili linee chiare (linee di Mackie) compaiono lungo i contorni tra scuro e chiaro.

*   **Intensità**: la durata del lampo. Sopra 1 i toni chiari si invertono.
*   **Riesposizione**: dove inizia l'inversione. I toni più scuri restano come sono.
*   **Agitazione**: meno agitazione dà linee di Mackie più larghe.

---

<!-- panel:toning src:8aa1a282 -->
### 6.3 Viraggio

I viraggi sono bagni chimici che cambiano il colore di una stampa B/N. Il Viraggio diviso aggiunge colore a qualsiasi stampa.

**Viraggio chimico** (negativi B/N e diapositive B/N) agisce nell'ordine mostrato:

*   **Selenio**: neri più profondi e più freddi.
*   **Seppia**: bruno caldo, a partire dalle alte luci.
*   **Oro**: nero-bluastro. Sopra il seppia, dà alte luci rosso-arancio.
*   **Blu di ferro**: ombre blu e neri blu notte.
*   **Rame**: dal rosa al rosso mattone. I neri si indeboliscono.
*   **Vanadio**: mezzitoni e alte luci verdi. I neri restano neri.

Su una stampa lith funzionano solo Selenio e Oro. Su una cianotipia non funziona alcun viraggio.

**Viraggio diviso** (tutte le stampe): **Intensità delle ombre** e **Intensità delle alte luci** aggiungono colore alle ombre e alle alte luci. **Tinta delle ombre** e **Tinta delle alte luci** scelgono il colore, e compaiono quando l'intensità è sopra 0.

---

## 7. Scheda Finitura

<!-- panel:retouch src:85e58a6a -->
### 7.1 Ritocco: polvere, peli, graffi

Rimuove polvere, peli e graffi. Ogni riparazione viene ricostruita dalla pellicola pulita circostante.

*   **Sovrapposizione**: **Contrassegnati** mostra cosa ha trovato la rimozione della polvere, **IR** il canale infrarosso della scansione.
*   **Rimozione ottica** (anche sulla barra della scheda Finitura): trova da sola polvere e peli. **Soglia delle macchie** e **Soglia dei peli**: più basso ne trova di più ma può rimuovere dettagli reali; 1 è disattivato. **Dimensioni**: la polvere più grande da rimuovere. Per proteggere un'area, trascina con il tasto destro sopra di essa o fai clic destro su un segno; il pulsante del cursore salta il menu del clic destro.
*   **Rimozione IR** (scansioni con un canale infrarosso, anche sulla barra della scheda Finitura): trova la polvere nell'infrarosso, dove l'immagine non compare. **Soglia IR**: più basso ne trova di più. B/N e Kodachrome vengono saltate: la loro immagine compare anche nell'infrarosso. Il canale proviene da scansioni TIFF e DNG a 4 canali oppure da un file `_IR.tif` accanto alla scansione.
*   **Correttivo**: fai clic su un segno per ripararlo; un clic ripara sempre. Dipingi sopra un'area per riparare solo i segni che risaltano, quindi dipingi con generosità.
*   **Graffio**: fai clic su alcuni punti lungo un graffio o un pelo, poi premi Invio.
*   **Linea**: fai clic una volta su un graffio lungo e dritto, e NegPy trova e ripara l'intera linea. **Sensibilità della linea**: più basso trova linee più deboli.
*   **Dimensione pennello**: anche `Alt`+rotellina sull'immagine.
*   **Clona**: copia la pellicola da un'altra area sopra un difetto che il Correttivo non può riparare. Il primo clic (o `Alt`+clic) sceglie la sorgente; poi dipingi. **Imposta sorgente** ne sceglie una nuova. **Adegua il tono** prende la luminosità e il colore attorno al difetto. **Intensità** stabilisce quanto copre, **Sfumatura** quanto è morbido il suo bordo.
*   **Annulla l'ultimo** e **Cancella tutto** rimuovono le riparazioni manuali; fai clic destro su una di esse sull'immagine per eliminarla.

<!-- panel:finish src:0e98fdde -->
### 7.2 Rifinitura: vignettatura, portanegativi, bordo

Come viene presentata la stampa. Gli altri cursori di ogni effetto compaiono quando il suo primo cursore non è a 0.

*   **Vignettatura**: **Bruciatura** scurisce (positivo) o schiarisce (negativo) i bordi. **Dimensioni** stabilisce fin dove si estende. **Arrotondamento** va dalla forma rotonda (0) alla forma della stampa (1).
*   **Portanegativi limato**: **Larghezza** disegna una cornice nera irregolare attorno all'immagine, come fa un portanegativi da camera oscura limato per allargarlo. **Ruvidità**, **Riflesso** (un alone grigio fuori dalla cornice) e **Angoli** la modellano.
*   **Bordo**: **Larghezza** mette un passe-partout attorno alla stampa. **Peso inferiore** rende più largo il lato inferiore, come in un passe-partout da cornice. **Colore**: **Bianco carta** si adegua alla carta, **Personalizzato** usa un colore che scegli.

---

## 8. Scheda Preferiti

I cursori che usi di più, in un unico posto. È vuota finché non la riempi: fai clic su **Modifica preferiti**, attiva i cursori a sinistra e trascinali nell'ordine voluto a destra, poi fai clic su **Applica**. Sono gli stessi controlli presenti nei loro riquadri.

<!-- panel:presets src:32c2813a -->
### Preimpostazioni

Impostazioni salvate che puoi applicare ad altri fotogrammi.

*   **Applica** (o doppio clic su una preimpostazione): applica la preimpostazione selezionata a questo fotogramma, ai fotogrammi selezionati o all'intero rullino.
*   **Salva…**: scegli quali delle impostazioni attuali conservare come nuova preimpostazione. Una preimpostazione può contenere le impostazioni di Ritaglio e Ottica con il profilo di Campo piatto, ma mai il ritaglio, la rotazione o i riflessi di un singolo fotogramma.
*   **Penna** e **Cestino**: modificano o eliminano la preimpostazione selezionata.

---

## 9. Scheda Cronologia

<!-- panel:work_prints src:6d4f4720 -->
### Stampe di lavoro

Una **stampa di lavoro** è una versione di questo fotogramma che conservi con un nome, come le stampe di prova lungo la strada verso quella finale.

*   **Salva stampa di lavoro…** (`Ctrl+Maiusc+S`) conserva la modifica attuale.
*   **Fai clic** su una stampa di lavoro per usarla. `Ctrl+Z` torna indietro.
*   **⋮** o clic destro: **Confronta con questa versione**, **Esporta questa versione…**, **Rinomina…** o **Elimina…**. **Confronta con questa versione** apre **Prima / Dopo** con la stampa di lavoro a sinistra.
*   **Carica metadati** (inizialmente disattivato): ripristina anche i metadati salvati con la stampa di lavoro.

<!-- panel:edit_history src:59f9ee38 -->
### Cronologia delle modifiche

Ogni modifica che hai fatto, la più recente in alto (le ultime 100). **Ora** indica lo stato attuale; dopo un annullamento, i passaggi successivi compaiono come **Annullato** fino alla tua modifica successiva. Fai clic su un passaggio per tornarci. **⋮** o clic destro → **Esporta questa versione…** esporta uno stato passato. **Confronta con questo passaggio** apre **Prima / Dopo** con quel passaggio a sinistra, sul tuo ritaglio attuale. Premi `\` per chiuderlo.

---

## 10. Scheda Rullino

Questi riquadri contengono ciò che i fotogrammi di un rullino condividono: la pellicola, come è stata scansionata e le misurazioni del rullino. Ogni riquadro ha la coppia Fotogramma / Rullino ([§1](#fotogramma-o-rullino)). La riga sopra i riquadri elenca i riquadri che questo fotogramma modifica per conto proprio.

<!-- panel:film src:60044f56 -->
### 10.1 Configurazione

Che pellicola è e come l'hai scansionata. Queste scelte decidono quali riquadri e controlli compaiono.

*   **Colore / B/N / Diapositiva**: negativo a colori (C-41), una stampa in bianco e nero, oppure diapositiva (E-6 e simili). B/N stampa qualsiasi negativo in bianco e nero. Su un negativo a colori, il riquadro Filtratura offre allora un Filtro di contrasto.
*   **Positivo** (solo Diapositiva): per un file che è già un positivo, come una stampa scansionata o l'esportazione di un'altra app. NegPy salta allora inversione e misurazione. Una diapositiva che non è un RAW di fotocamera, come una scansione TIFF, JPEG, JXL, DNG di scanner o Pakon, Noritsu, Coolscan o Flextight, parte come Positivo. Disattivalo per dare a una scansione sbiadita la curva tonale delle diapositive.
*   **Diapositiva B/N** (solo Diapositiva): rende la diapositiva in bianco e nero, per pellicole invertibili B/N come la Scala. Una diapositiva a colori riceve allora il Filtro.
*   **Codifica** (TIFF positivo): come i campioni codificano la luce. **Auto** legge il profilo ICC del file, altrimenti presuppone lineare sopra gli 8 bit e sRGB a 8 bit. **Lineare** è l'uscita lineare di uno scanner, come Nikon Scan con la gestione del colore disattivata. **2,2** è una gamma 2,2 con primari Adobe RGB. Scegli Lineare o 2,2 in base al software dello scanner.
*   **Fotocamera / Scanner per pellicola**: con cosa hai eseguito la scansione. NegPy lo legge da ogni file (un RAW di fotocamera è una fotocamera) finché non ne scegli uno. La tua scelta vale per ogni fotogramma e imposta lo scanner della scheda Scansione.
*   **Luce bianca / Luce a banda stretta** (non per un negativo B/N): la luce dietro il negativo. La luce bianca è un visore luminoso o un pannello LED. La banda stretta è una luce con rosso, verde e blu puri, come uno Scanlight.

Le scelte di scansione sono un'unica impostazione per tutta l'app. L'elenco sotto di esse mostra che cosa NegPy ha letto dal file e quale supporto di stampa viene usato.

Una fotocamera mostra i riquadri Decodifica RAW e Ottica. Una fotocamera con luce a banda stretta mostra anche Tricromia e Calibrazione. Una modifica aggiorna ogni fotogramma aperto (Ctrl+Z la annulla fotogramma per fotogramma), ma un supporto di stampa che hai scelto tu resta. Una Scansione con fotocamera imposta la decodifica e il supporto di stampa di ogni fotogramma in base alla propria luce.

<!-- panel:baseline src:6773897f -->
### 10.2 Analisi del rullino

Misura l'intero rullino una volta e condivide il risultato, così i fotogrammi di una pellicola stampano in modo simile. NegPy misura il rullino e ogni scena man mano che arrivano i fotogrammi, e di nuovo dopo una modifica del ritaglio o della Misurazione. Nessun fotogramma usa un riferimento finché non attivi una media.

*   **Usa la media**: fai clic destro su uno qualsiasi dei tre per attivare o disattivare tutte le medie. **Luma** usa i punti del nero e del bianco del rullino, **Colore** il bilanciamento del colore del rullino, e **Dominante** (negativi a colori) il bilanciamento del grigio del rullino per la Rimozione della dominante. Con Dominante attiva, un fotogramma scattato sotto una luce diversa conserva un po' di quella luce, come su una stampa Frontier. Una dominante magenta resta, una dominante calda o verde resta in parte, e una dominante blu scompare. Con Luma e Colore attivi, il rullino stampa in modo uniforme; con entrambi disattivati, ogni fotogramma viene misurato da solo.
*   **Riferimento**: indica da dove proviene il riferimento (rullino, scena o fotogramma). Avvisa quando non ce n'è uno, quando non è aggiornato, o quando il colore di questo fotogramma è lontano dalla media.
*   **Rianalizza**: misura ogni fotogramma visibile al di fuori di una scena, salva la media come riferimento del rullino e attiva le medie per il rullino e i suoi nuovi fotogrammi. I fotogrammi lontani dagli altri mantengono i propri valori. Prima ritaglia i fotogrammi in attesa del Ritaglio automatico.
*   **Usa fotogramma**: rende la misura propria di questo fotogramma il riferimento del rullino; Dominante si disattiva.
*   **Da un altro rullino** (sezione ripiegabile): scegli il riferimento salvato di un altro rullino. Un rullino spuntato ne ha uno.
*   **Scene**: gruppi di fotogrammi con un riferimento proprio, come un soggetto sotto una certa luce ([Scene](#scene)). **Analizza** misura la scena, **Seleziona** ne seleziona i fotogrammi ed **Elimina** rimuove il gruppo.
*   **Rileva**: trova i gruppi di fotogrammi sotto una luce diversa da quella del rullino, in qualsiasi ordine di fotogrammi, e li mostra sul Tavolo luminoso. Ogni gruppo resta in attesa sotto **Trovati**. Per cambiare un gruppo, trascina i fotogrammi su un altro gruppo, oppure su un fotogramma senza scena per escluderli; clic destro › **Scena** fa lo stesso. Dagli un nome, poi **Accettalo** come scena con un riferimento proprio, oppure **Scartalo**.

<!-- panel:process src:2c179de9 -->
### 10.3 Misurazione: negativo → positivo

Per trasformare un negativo in un positivo, NegPy ne misura le parti più scure e più chiare (i punti del nero e del bianco) e il colore della pellicola. Questi controlli stabiliscono cosa misura.

*   **Misura da solo**: toglie questo fotogramma, o ogni fotogramma selezionato, dalla media del rullino o della scena e lo misura da solo, dal suo ritaglio attuale.
*   **Disegna regione**: disegna un'area, e NegPy misura solo quell'area. Fai doppio clic al suo interno o premi `Invio` per confermare, e `Esc` ripristina la regione precedente. **×** la cancella.
*   **Margine di analisi**: ignora un bordo attorno al fotogramma durante la misurazione, così i bordi della pellicola e i supporti non contano.
*   **Tosatura**: quanta parte della stampa è nero puro o bianco puro.
*   **Blocca limiti**: mantiene i punti del nero e del bianco di questo fotogramma, anche quando cambiano il ritaglio o l'Analisi del rullino.
*   **Tosatura della gamma di luma** (sezione ripiegabile): quanto strettamente vengono impostati il nero e il bianco. Positivo dà più contrasto, per negativi densi o velati. Negativo conserva più dettaglio nelle ombre e nelle alte luci.
*   **Tosatura colore** (sezione ripiegabile): quanto strettamente viene impostato il bilanciamento del colore.
*   **Dmin** / **Dmax** (sezione Scegli a mano): per un fotogramma che NegPy misura male. Fai clic su **Dmin**, poi sulla base della pellicola o sull'ombra più profonda, per impostare il punto del nero. Fai clic su **Dmax**, poi sull'alta luce più brillante, per impostare il punto del bianco. Ogni **×** cancella il proprio punto. Entrambi sono disattivati mentre il fotogramma stampa sulla media del rullino.
*   **Punto del bianco** / **Punto del nero** (sezione ripiegabile): spostano i punti del bianco e del nero misurati. Un punto del bianco positivo schiarisce la stampa; un punto del nero positivo solleva i neri. **R / G / B** sposta un solo colore.

Una **diapositiva** non richiede inversione: i controlli tonali modellano direttamente l'immagine. Densità automatica e Gradazione automatica partono disattivate; attivale per una diapositiva sbiadita. Su una diapositiva, Disegna regione stabilisce l'area misurata da Stima, Rimuovi macchia, Rimozione della dominante, Densità automatica e Gradazione automatica. Disegnala prima di fare clic su Stima. Nei termini di Lightroom, Esposizione è la Densità di stampa (più basso è più chiaro), Contrasto è la Gradazione ISO-R, e Ombre e Luci sono la Densità delle ombre e delle alte luci.

<!-- panel:fade src:df17e8d2 -->
### 10.4 Ripristino dello sbiadimento: diapositive sbiadite

Restituisce a una diapositiva sbiadita i suoi strati di colorante. Il riquadro compare solo per le diapositive.

*   **Stima**: imposta Miscela di grigio e le Persistenze.
*   **Ripristina intervallo** (icona accanto a Stima): se attivo, imposta Persistenza del rosso dall'intervallo del fotogramma; Stima la mantiene.
*   **Persistenza del rosso**: quanto colorante rosso è rimasto. Abbassala finché il contrasto non ritorna. Con Ripristina intervallo attivo, la tua modifica resta ogni volta che reimposta il rosso.
*   **Persistenza del verde** / **Persistenza del blu**: lo stesso, misurato rispetto al rosso.
*   **Rimuovi macchia**: sbianca le zone trasparenti ingiallite e stima di nuovo. Attivalo quando Stima lo suggerisce.
*   **Miscela di grigio**: cambia di quali grigi si fida Stima. Trascinala se i colori sembrano sbagliati.
*   **Incrocio**: rimuove una variazione di colore tra ombre e alte luci. Trascinalo su ogni fotogramma, oppure condividi quello di un fotogramma con Condividi incrocio.

Il gruppo Rullino alla fine del riquadro agisce su ogni fotogramma del rullino:

*   **Stima rullino**: legge ogni diapositiva del rullino, o quelle selezionate, e dà a ogni fotogramma le stesse Persistenze e lo stesso Ripristina intervallo. Le diapositive che si leggono in modo diverso, come con luce interna, restano fuori dalla lettura ma ricevono comunque il risultato. Ogni fotogramma mantiene il proprio Incrocio.
*   **Condividi incrocio**: misura l'incrocio su questo fotogramma e lo dà a ogni fotogramma, con l'intensità di questo fotogramma. Usa un fotogramma con grigi netti nelle zone chiare e scure, come una strada e vestiti bianchi. Il suggerimento del rullino può proporne uno.

<!-- panel:demosaic src:5627a0ef -->
### 10.5 Decodifica RAW

Come un file RAW di fotocamera viene trasformato in pixel. Compare per una fotocamera.

*   **Bil. bianco fotocamera / RAW lineare**: Bil. bianco fotocamera usa il bilanciamento del bianco della fotocamera. RAW lineare decodifica in modo neutro, per la luce a banda stretta. Configurazione lo imposta per te.
*   **Anteprima** / **Esportazione**: il metodo di demosaicizzazione, che ricostruisce i colori che ogni pixel del sensore non vede. **Auto** è veloce a schermo e usa **AHD** per l'esportazione. **VNG** è morbido e **PPG** veloce. **DCB** e **DHT** conservano più dettaglio fine ma possono trasformare la grana in motivi. **AAHD** ammorbidisce i contorni. Sono elencati solo i metodi presenti nella tua versione di LibRaw.
*   **Recupero** (diapositive): ricostruisce le alte luci bruciate in un RAW di fotocamera. **Miscela** è adatto a sole e cielo. **Ricostruisci** è più forte ma può sbagliare il colore. Un nuovo fotogramma da scanner per pellicola parte con Recupero disattivato. Sotto Recupero compare un avviso quando il fotogramma ha alte luci tosate, che possono diventare rosa.

<!-- panel:optics src:33b2d330 -->
### 10.6 Ottica

Correzioni per l'obiettivo e la luce, uguali per ogni fotogramma che scansioni con questa configurazione. Compare per una fotocamera.

#### Correzione dell'obiettivo

*   **Incorporata**: usa la correzione dell'obiettivo propria del file (DNG, Sony ARW, Panasonic RW2). **Distorsione** raddrizza le linee curve e **AC** rimuove le frange di colore. Impostala prima di ritagliare o ritoccare.
*   **Correzione della distorsione**: raddrizza le linee che si incurvano verso l'esterno (positivo) o verso l'interno (negativo). Usa il bordo della pellicola come guida diritta.

#### Correzione del campo piatto: uniformare la luce

Rimuove la luce non uniforme, come gli angoli più scuri, con uno scatto della sola sorgente luminosa.

*   **Profilo**: **+** salva uno scatto della sola luce come profilo con un nome. Il pulsante del cestino lo elimina, e ogni fotogramma che lo usa perde la sua correzione.
*   **Applica Campo piatto**: applica il profilo selezionato a questo rullino.
*   **Verifica Campo piatto**: mostra quanto bene il profilo uniforma il proprio scatto. Un grigio uniforme è buon segno; una fascia o una macchia indica dove sbaglia.

Un fotogramma mantiene il proprio profilo a meno che non segua il suo rullino, così ogni luce e ogni formato di pellicola può averne uno. Un nuovo fotogramma da fotocamera, o un fotogramma da fotocamera il cui profilo manca, prende l'ultimo profilo che hai scelto su un fotogramma da fotocamera. Un nuovo fotogramma da scanner per pellicola parte senza correzioni di Ottica. Con Scanner per pellicola impostato, questo riquadro compare solo su un fotogramma che ne ha una.

<!-- panel:sensor src:5722a58e -->
### 10.7 Calibrazione: i colori della tua configurazione di scansione

Corregge gli errori di colore dovuti alla luce, alla fotocamera e alla pellicola.

**Profilo di diafonia del sensore** (un'Esposizione singola sotto luce a banda stretta, RAW lineare): rimuove il colore che si disperde tra i canali del sensore. Esegui di nuovo l'**Analisi del rullino** dopo una modifica.

*   **Profilo**: il profilo di diafonia del sensore. I tuoi profili `.toml` vanno in `<Documents>/NegPy/sensor/`.
*   **Metodo**: **A due scale** (predefinito) rimuove la dispersione senza puntinatura. **Lineare** è esatto, ma i colori intensi possono puntinarsi. **Densità** non ha puntinatura, ma spegne un poco i colori intensi.
*   **Calibra**: crea un profilo da tre scatti della sola luce (rossa, verde, blu), da file o da una fotocamera collegata (vedi CAMERA_SCANNING.md).

**Diafonia** (pellicola a colori): i coloranti della pellicola, la luce e il sensore mescolano un poco i colori, e una matrice li separa. Esegui di nuovo l'**Analisi del rullino** dopo una modifica.

*   **Matrice**: le matrici per questo tipo di pellicola. Una matrice *(approssimativa)* proviene da una scheda tecnica. I tuoi file `.toml` vanno in `<Documents>/NegPy/crosstalk/` (vedi CROSSTALK.md). Il pulsante con i cursori apre l'editor; **+** crea una nuova matrice. NegPy non fornisce alcuna matrice per diapositive.
*   **Intensità**: quanto intensamente si applica la matrice.
*   Il pulsante con il campione di colore apre **Misura dalla tabella** (negativi a colori): misura una matrice da una ColorChecker 24, una SpyderCheckr 24 o una DKC-Pro fotografata sulla pellicola. Trascina i quattro punti sulle tacche d'angolo.

**Rifinitura della tinta** (fotocamera con luce a banda stretta): ruota tutte le tinte, per una luce che rende arancioni i gialli o oliva i verdi. Lascia 0 per la luce bianca. A differenza del profilo del sensore, funziona anche sulle diapositive.

<!-- panel:assembly src:96b2e550 -->
### 10.8 Composizione del fotogramma

Come i file diventano fotogrammi. Ogni rullino conserva qui le proprie impostazioni.

#### Tricromia

Per una fotocamera con luce a banda stretta che scatta tre foto di ogni fotogramma: rossa, verde e blu.

*   **Modalità tricromia**: unisce i tre scatti in un unico fotogramma a colori. Gli scatti vengono raggruppati per ora di ripresa (o per nome file), quindi scatta i tre scatti di ogni fotogramma uno dopo l'altro. Gli scatti che non si raggruppano restano separati.
*   **Fondi esposizioni (tricromia)** (icona di gruppo, o clic destro): seleziona i tre scatti di un fotogramma per unirli. NegPy capisce quale scatto è rosso, verde e blu.
*   **Modifica tricromia…**: scegli a mano i tre scatti del fotogramma corrente. **Allinea canali** li allinea per rimuovere le frange di colore. **Separa tricromia** (icona di separazione, o clic destro) li separa di nuovo.
*   **Fondi fotogramma in negativo TIFF…** (clic destro): salva una terna o un'unione come un unico TIFF accanto al primo file sorgente, con la sua modifica. Un'opzione sposta i file sorgente nel Cestino.

#### Mezzo formato

**Modalità mezzo formato** divide ogni scansione in due fotogrammi, per le fotocamere a mezzo formato. Ogni metà ha la propria modifica. Ogni rullino conserva la propria impostazione. Senza un rullino aperto, si applica ai fotogrammi che non appartengono ad alcun rullino fino a **Chiudi tutti**, e **Riaprire l'ultima sessione** la mantiene. È disattivata mentre è caricato un fotogramma di un rullino, perché quel fotogramma segue il proprio rullino.

*   **Regola…**: trascina il riquadro bianco per ritagliare e la linea arancione per impostare la divisione. **Direzione della divisione** e **Spessore del taglio** impostano lo spazio, **Rilevamento automatico** li trova di nuovo, e **Applica** salva su questo fotogramma, sulla selezione o su tutti i fotogrammi.
*   **Rileva tutto**: trova di nuovo le divisioni su ogni scansione.
*   **Ricomponi** (su un fotogramma diviso): lo rende di nuovo un unico fotogramma ed elimina le modifiche di entrambe le metà.

Fai clic destro su una metà per **Regola la divisione per questo fotogramma…** e **Reimposta la divisione al valore predefinito del rullino**. Disattivare il Mezzo formato mantiene le metà modificate affiancate come **dittico**; riattivalo per modificarle.

---

## 11. Scheda Metadati

I dettagli su come hai scattato e sviluppato l'immagine: fotocamera, obiettivo, pellicola e processo. NegPy li scrive in ogni esportazione, così le app di fotografia come Lightroom mostrano la tua attrezzatura analogica, non il tuo scanner.

**Anteprima dei metadati** è in alto. Sotto, **Scatto** contiene Attrezzatura analogica, Ripresa ed Esposizione, e **Processo** contiene Processo e Scansione. Le due schede a icona sono **Preimpostazioni** e **La mia attrezzatura**. Questi riquadri si applicano inizialmente all'intero rullino ([§1](#fotogramma-o-rullino)); il numero di fotogramma è sempre per fotogramma.

<!-- panel:metadata_gear src:208e1498 -->
### Attrezzatura analogica

*   **Fotocamera / Obiettivo / Pellicola**: scegli dalla tua attrezzatura (la scheda **La mia attrezzatura**). **Altro…** apre il catalogo integrato completo.
*   **Cancella**: svuota tutti e tre.
*   **Deduci dal nome della cartella**: compila i campi vuoti (fotocamera, pellicola, ISO, data) dal nome della cartella.

<!-- panel:metadata_capture src:a84922a9 -->
### Ripresa

*   **Data**: quando hai scattato l'immagine, come `1998`, `1998-07`, `1998-07-14` o `1998-07-14 16:30`.
*   **Luogo**: il pulsante del segnaposto apre una mappa. Puoi anche incollare coordinate o un link di mappa. La mappa usa OpenStreetMap.

<!-- panel:metadata_process src:9dd71862 -->
### Processo

Come è stata sviluppata la pellicola.

*   **Processo salvato**: una ricetta de La mia attrezzatura che compila i campi sottostanti. Scrivere sopra un campo lo scollega.
*   **Formato della pellicola**: le dimensioni della pellicola, come `35mm`, `120` o `4×5`.
*   **Rivelatore** e **Diluizione**: per esempio `D-76` e `1+1`.
*   **Sovra/Sottosviluppo**: da `Sovrasviluppo +3` a `Sottosviluppo -3`.
*   **Tempo** e **Temperatura (°C)**: il tempo come `9:30` o in minuti.
*   **Cancella**: svuota i campi del processo.

<!-- panel:metadata_scanning src:eab4bcdf -->
### Scansione

*   **Configurazione salvata**: una configurazione di scansione de La mia attrezzatura.
*   **Scansione**: come hai eseguito la scansione, oppure delle note.
*   **Rullino / Fotogramma**: il nome del rullino e il numero di fotogramma. I nomi dei file possono usarli (`{{ roll }}`, `{{ frame }}`).
*   **Sequenza…**: numera i fotogrammi selezionati, o l'intero rullino, nell'ordine della pellicola. Imposta il numero iniziale, il passo e la direzione; **Salta i fotogrammi scartati** esclude gli scartati.
*   **Cancella**: svuota la configurazione e la nota.

<!-- panel:metadata_exposure src:9c0029fe -->
### Esposizione

Il tempo di otturazione, il diaframma, l'ISO e la compensazione dell'esposizione della fotocamera (EV) dello scatto originale, se li conosci. Fai clic sul lucchetto per digitare testo libero come `1/125s f/2.8 ISO 400 -1/3ev`. Una compensazione è il numero di stop prima di `ev`, e somme come `1+2/3ev` vengono sommate.

<!-- panel:metadata_preview src:3cf26bec -->
### Anteprima dei metadati

Mostra ciò che entra nel file esportato. Il pulsante con lo scudo, **Proteggi i metadati originali**, esporta invece invariati i metadati propri del file sorgente. **Descrizione…** sceglie i campi che compongono la descrizione dell'immagine (inizialmente fotocamera, obiettivo, pellicola e ISO).

Fotocamera, obiettivo e pellicola entrano nei campi EXIF standard, e il tuo scanner nei campi XMP propri di NegPy. Una valutazione a stelle impostata nella fotocamera viene mantenuta.

<!-- panel:gear_items src:1c45c908 -->
### La mia attrezzatura

Il tuo elenco dell'attrezzatura, usato da ogni selettore di attrezzatura. **Categoria** sceglie **Fotocamere**, **Obiettivi**, **Pellicole**, **Processi** (ricette di sviluppo) o **Configurazioni di scansione**. Viene salvato in `~/NegPy/gear/`.

*   **+**: copia un elemento dal catalogo integrato, oppure **Aggiungi personalizzato** per inserirne uno.
*   **Catalogo**: mostra anche i modelli integrati.
*   **Copia / Cestino**: duplica o elimina l'elemento selezionato. Gli elementi integrati non possono essere eliminati.

<!-- panel:gear_presets src:17a221ff -->
### Preimpostazioni

Insiemi salvati di metadati, come una fotocamera, un obiettivo e una pellicola. Vengono salvati in `~/NegPy/presets/metadata/`.

*   **Applica a questo fotogramma**: scrive i campi della preimpostazione su questo fotogramma; gli altri campi restano.
*   **+**: salva con un nome i metadati del fotogramma corrente.
*   **⋮** o clic destro su una preimpostazione: applicala, duplicala, modificala o eliminala.
*   **Dettagli**: modifica i campi della preimpostazione. **Note** è testo libero.

Una pellicola imposta il formato, quindi imposta un formato come `6×7` dopo la pellicola.

---

## 12. Scheda Esporta

Il pulsante **Esporta** è in fondo alla scheda. Sopra di esso, **File** contiene le impostazioni di uscita e **Prova** la prova a video. Le due schede a icona sono **Preimpostazioni** e **Accanto alle esportazioni**.

### Intento di uscita

*   **Stampa** (predefinito): l'immagine come la vedi.
*   **Piatto**: un master piatto a basso contrasto da modificare in un'altra app. Salta l'aspetto di stampa e gli effetti, ma conserva il colore dei coloranti di un supporto di stampa a colori, e scrive un file a 16 bit. **Anteprima piatta** lo mostra nell'area di lavoro. In un rullino, esegui prima l'**Analisi del rullino** così i master corrispondono. Senza un rullino, NegPy misura ogni fotogramma da solo, quindi i master possono differire.
*   **Lineare**: i dati grezzi della scansione prima dell'elaborazione di NegPy, come TIFF a 16 bit o JPEG XL, per un flusso di lavoro grezzo in un'altra app. Legge RAW di fotocamera, DNG LinearRaw, NEF di Coolscan, file Flextight, Noritsu e Pakon e scansioni TIFF. Le opzioni per infrarosso, bilanciamento del bianco, campo piatto, correzione del sensore e dell'obiettivo sono inizialmente disattivate. I nomi dei file ricevono `_linear`, così una sorgente non viene mai sovrascritta.

### Impostazioni del file

*   **Formato**: JPEG, TIFF, PNG, JPEG XL o WebP, con opzioni di qualità. **Profondità di bit**: 8 o 16 bit (TIFF, PNG, JPEG XL). La compressione di TIFF e PNG è sempre senza perdita.
*   **Profilo di esportazione**: lo spazio colore del file, come sRGB per il web o Adobe RGB, oppure un profilo di stampante o di carta. **Importa ICC** aggiunge un profilo. **ICC di ingresso** imposta lo spazio colore di una scansione che non ne ha.
*   **Proporzioni della carta**: la forma della stampa finale, oppure **Originale**.
*   **Risoluzione**: **Originale** (piena risoluzione), **Stampa** (dimensioni in cm e DPI) o **Pixel** (il lato lungo in pixel). Un'esportazione Stampa o Pixel applica la Nitidezza di Lab dopo il ridimensionamento. **Anteprima delle dimensioni di esportazione** mostra il fotogramma a queste dimensioni nell'area di lavoro, per valutare nitidezza e grana.
*   **Destinazione**: il **Modello del nome file** (vedi [TEMPLATING.md](TEMPLATING.md)), **Sovrascrivi i file esistenti**, e la cartella: una sottocartella `export` accanto a ogni file sorgente (predefinito), la cartella sorgente, oppure un percorso che scegli.

### Pulsante Esporta

La sua freccia sceglie cosa esportare: questo fotogramma (`Ctrl+E`), i fotogrammi selezionati o tutti i fotogrammi visibili. NegPy chiede prima quando un lotto contiene fotogrammi non modificati.

<!-- panel:export_presets src:b7511145 -->
### Preimpostazioni

Ricette di esportazione salvate: formato, dimensioni, colore, cartella e nome file. Attiva quelle che vuoi, e **Preimpostazioni di esportazione** scrive ogni fotogramma una volta per preimpostazione. **Gestisci** le modifica. Le preimpostazioni non funzionano con l'uscita Lineare.

<!-- panel:export_alongside src:ef285087 -->
### Accanto alle esportazioni

File aggiuntivi accanto alla stampa. Ognuno ha un interruttore a sinistra e un'azione una tantum a destra.

*   **Note di stampa**: una copia annotata del fotogramma con la sua schermatura e bruciatura e le impostazioni di stampa, come quella che conserva uno stampatore in camera oscura. **Anteprima** la mostra (`Maiusc+N`); **Esporta note** la salva.
*   **Densità / gradazione automatica**: **Fissa all'esportazione** fissa la luminosità e il contrasto automatici di ogni fotogramma esportato, così non cambiano in seguito. **Fissa sul rullino** lo fa subito per ogni fotogramma visibile.
*   **File collaterali di modifica**: **Salva all'esportazione** scrive accanto a ogni sorgente un file `.negpy` con la modifica. **Esporta file collaterali** li scrive per tutti i fotogrammi visibili. Clic destro su un fotogramma → **Carica modifica da file collaterale…** ne rilegge uno.

<!-- panel:contact_sheet src:cc84fd86 -->
### Provino a contatto

Nella scheda **File**, accanto a **Intento di uscita**. Un provino a contatto mostra l'intero rullino come strisce di pellicola su un unico foglio di carta, come fa una stampa a contatto in camera oscura.

*   **Provino a contatto…**: apre il foglio. Trascina un lato della carta per ridimensionarla.
*   **Formato della pellicola**: 35mm, 35mm mezzo formato o 120 (con **Dimensioni del fotogramma** per il 120).
*   **Carta**, **Larghezza**, **Altezza** e **DPI**: dimensioni e risoluzione del foglio. La carta predefinita contiene un rullino da 36 pose.
*   **Ordine**: per nome file, per data, per rullino e numero di fotogramma, oppure per scena con una nuova striscia o riga per ogni scena.
*   **Stampa**: **Come modificato**, oppure **Provino diretto**: l'intero rullino con un'unica esposizione e contrasto normale, così vedi quali negativi sono sottili o densi.
*   **Scegli fotogrammi**: fai clic su un fotogramma per escluderlo o reinserirlo. Gli scartati partono esclusi.
*   **Scritte del bordo**: le marcature del produttore lungo i bordi, come il nome della pellicola, i numeri dei fotogrammi e le linee colorate che una pellicola stampa accanto alle perforazioni.
*   **Etichetta del rullino**: nome del rullino, pellicola, fotocamera e data sopra le strisce. Scrivi il tuo testo in **Testo dell'etichetta** per stamparlo al loro posto.
*   **Carta bianca**: stampa su bianco, il che fa risparmiare inchiostro su una stampante domestica.
*   **Bordi della pellicola**: se disattivato, stampa i fotogrammi dritti, ognuno con la propria forma, in righe come sul Tavolo luminoso. Il foglio non ha allora né perforazioni né **Scritte del bordo**.
*   **Cartella**: la cartella per i fogli. Se vuoto usa la cartella di esportazione.

I fogli vengono salvati come `contact_sheet.jpg` (`contact_sheet_1of2.jpg` e così via per un rullino lungo).

<!-- panel:soft_proof src:2ec0a049 -->
### Prova a video

La scheda **Prova** mostra sullo schermo come apparirà l'immagine stampata su una data stampante e carta. Cambia solo l'anteprima, mai l'esportazione. La carta non può mostrare colori luminosi come quelli di uno schermo, quindi una prova corretta appare più spenta.

*   **Prova a video** (`Maiusc+P`): attiva o disattiva la prova. Mentre è attiva, le info del fotogramma sopra la foto mostrano **Prova** e il profilo.
*   **Preimpostazione**: una configurazione salvata di stampante e carta.
*   **Profilo**: il profilo della stampante o della carta. Segue il profilo di esportazione finché non ne scegli uno. Un profilo in scala di grigi rende grigia l'anteprima, quindi un fotogramma a colori mostra un avviso ambra sopra la foto.
*   **Intento**: **Colorimetrico relativo** mantiene esatti i colori stampabili e tosa gli altri. **Percettivo** comprime tutti i colori, il che si adatta alle immagini molto colorate. **Saturazione** è per grafici e illustrazioni.
*   **Compensazione del punto del nero**: fa corrispondere il tono più scuro al nero della carta.
*   **Simula il bianco della carta** e **Simula il nero dell'inchiostro**: mostrano il bianco e il nero reali della carta.
*   **Avviso fuori gamma**: mostra in grigio i colori che la stampante non può stampare.
*   **Visualizzazione**: il profilo del tuo monitor, trovato automaticamente.

---

## 13. Scheda Scansione

Esegui la scansione della pellicola direttamente in NegPy. Il riquadro **Scanner** sceglie lo scanner; la scheda mostra poi i riquadri di quello scanner, il riquadro condiviso **Uscita** e il pulsante **Scansiona**.

<!-- panel:scan_source src:939ed7bb -->
### Scanner

*   **Scanner**: **Scanner per pellicola** per uno scanner dedicato alle pellicole (i riquadri Dispositivo, Pellicola e qualità e Inquadratura), oppure **Fotocamera** per una fotocamera su uno stativo di riproduzione (i riquadri Fotocamera e Preimpostazione).

<!-- panel:scan_device src:a18b5f39 -->
### Dispositivo

*   **Backend**: **SANE** (Linux e macOS), **Nikon Coolscan (nkscan)** o **pyOpticfilm (Plustek)**. I controlli seguono ciò che lo scanner riporta.
*   **Dispositivo**: lo scanner; le frecce aggiornano l'elenco. Il menu di espulsione ha **Espelli ora** e, su un alimentatore di strisce, **Espelli al termine**, che restituisce la striscia dopo un lotto.
*   **Registro di debug** (Coolscan): scrive `nkscan.log` nella cartella di NegPy. Per una segnalazione di bug, imposta **Trace**, ripeti il problema e allega il file.

NegPy tiene svegli macOS e Windows durante la scansione. Su Linux, disattiva la sospensione automatica prima di un rullino lungo.

**Plustek**: funzionano OpticFilm 8200i SE, 8100 V2, 8300i SE, 7600i v1 e 7600i v2; per altri modelli, prova SANE. Su Windows, installa prima il driver USB ([PLUSTEK_WINDOWS.md](PLUSTEK_WINDOWS.md)). **Coolscan**: non richiede SANE. Su Linux, l'USB richiede una regola udev per il produttore `04b0`.

<!-- panel:scan_quality src:1f841943 -->
### Pellicola e qualità

*   **Pellicola**: Negativo a colori, Negativo B/N, Diapositiva o Kodachrome. Stabilisce il rilevamento dei fotogrammi, la misurazione e se viene offerto l'infrarosso.
*   **Formato della pellicola**: la lunghezza del fotogramma (135, 66, 645 e altri). **Auto** funziona dove la imposta il supporto.
*   **Risoluzione** e **Profondità di bit**: i dpi della scansione e i bit per canale.
*   **Modalità di scansione** (Plustek): **Multipassata** fa la media di più scansioni per meno rumore; **Multiesposizione adattiva** combina un'esposizione breve e una lunga per una gamma maggiore, con l'esposizione lunga impostata in base a ogni fotogramma; **Multiesposizione fissa** usa l'unica esposizione lunga dello scanner su ogni fotogramma, come fa SilverFast (8200i SE, 8100 V2 e 8300i SE). Il 7600i v2 offre la Multiesposizione adattiva ma non la Multipassata.
*   **Campioni** (Coolscan): legge ogni riga più volte, per meno rumore nelle ombre. Più lento.
*   **Infrarosso**: **IR** scansiona un canale infrarosso, per la rimozione della polvere nel riquadro Ritocco. **ICE** (Coolscan) rimuove invece la polvere nello scanner. Uno scanner senza ICE mostra un unico interruttore **IR**. Non per B/N o Kodachrome.
*   **Superfine** (Coolscan): scansiona una riga per passata. Più lento.
*   **Autofocus**, **Esposizione automatica** ed **Esposizione**: messa a fuoco ed esposizione sullo scanner, prima della scansione.
*   **Guadagno analogico** (Coolscan): più o meno esposizione dello scanner rispetto a quella misurata, in stop. La freccia apre **Rosso**, **Verde** e **Blu**, uno per canale.

<!-- panel:scan_framing src:945dad86 -->
### Inquadratura

*   **Fotogrammi**: `1-6`, `1,2,5`, oppure vuoto per tutti. L'anteprima della striscia lo compila.
*   **Anteprima della striscia…** (alimentatore di strisce): mostra l'anteprima di ogni fotogramma, per posizionare i fotogrammi e sceglierli. **Anteprima…** (supporto manuale): imposta una finestra di ritaglio. **Cancella** la reimposta. **Rileva fotogrammi** trova di nuovo i fotogrammi dopo che la pellicola si è spostata.
*   **Misura fotogramma…** (Coolscan): misura l'esposizione su un fotogramma e la usa per ogni scansione successiva, così tutti i fotogrammi di un rullino corrispondono. **Sblocca** la termina. Misura di nuovo per ogni rullino.
*   **Prescansione…** (Plustek): una rapida anteprima completa. Trascina un ritaglio, e la scansione successiva legge solo quell'area.

<!-- panel:scan_camera src:0145e5d1 -->
### Fotocamera

Esegue la scansione con una fotocamera su uno stativo di riproduzione, collegata via USB in modalità **PC Remote** (macOS e Linux; richiede `python-gphoto2`). Con uno **Scanlight** di NegPy, NegPy controlla anche la luce. CAMERA_SCANNING.md spiega la configurazione.

*   **Fotocamera** / **Luce**: lo stato della connessione. La luce diventa ambra quando si scalda. Su Linux, lo Scanlight richiede che il tuo utente sia nel gruppo `dialout`.
*   **Live View**: un'immagine dal vivo dalla fotocamera. Fai clic per collocare la lente di messa a fuoco, e metti a fuoco finché l'**Indicatore di fuoco** non segna **al massimo**. Imposta ISO, otturatore e diaframma qui o sulla fotocamera. Quando si chiude, lo Scanlight torna com'era prima.
*   **Istogramma** e **Zebre**: l'istogramma del fotogramma, e strisce sulle parti che stanno per essere tosate, nei colori di tosatura impostati da Visione dei colori.
*   **Esposimetro**: imposta l'otturatore in modo che la parte più luminosa che non deve essere tosata (la pellicola trasparente di un negativo, le alte luci di una diapositiva) cada appena sotto la tosatura. Misura prima della prima scansione. Senza una preimpostazione calibrata, una scansione usa l'esposizione impostata sulla fotocamera, e NegPy chiede conferma prima di eseguire una scansione non misurata.
*   **Scansiona** scatta il fotogramma successivo e lo importa; **Riprendi** (la freccia di ripetizione) scatta di nuovo l'ultimo fotogramma. Quando una scansione viene tosata, la riga di stato indica quanta parte dell'immagine è tosata. Nella finestra del Live View, `Spazio` o `S` esegue la scansione e `R` riprende. Con uno Scanlight, **R**, **G**, **B** e **IR** scelgono quali esposizioni Riprendi ripete.
*   **Ritardo di acquisizione**: attende prima della prima esposizione, così un tavolo traballante può assestarsi.
*   **Luce sempre accesa** (la lampadina): tiene acceso lo Scanlight dopo ogni scansione, così puoi inquadrare lo scatto successivo.
*   **Pellicola** (con un Lento Dock): cosa fa il caricatore quando fai clic su Scansiona. **Fotogramma successivo** fa avanzare la pellicola di un fotogramma dopo ogni scansione. **Intera striscia** scansiona ogni fotogramma, poi espelle la striscia; Ferma la termina. Riprendi non sposta mai la pellicola. **Espelli** fa uscire la striscia. Il pulsante S del caricatore funziona come Scansiona.
*   **Scatta Campo piatto**: dopo l'espulsione di un'intera striscia, fotografa il caricatore vuoto e lo salva come profilo di Campo piatto del rullino.

<!-- panel:scan_light src:6e486298 -->
### Preimpostazione

Compare mentre è collegato uno Scanlight. Una preimpostazione contiene i livelli della luce e l'esposizione della fotocamera per un tipo di pellicola.

*   **Preimpostazione**: la luce e l'esposizione per ogni scansione. **+** ne aggiunge una in tre modi:
    *   **Preimpostazioni integrate**: pronte all'uso, senza calibrazione: **Luce bianca**, **Luce bianca + IR**, **Tricromia** e **Tricromia + IR**. Usano l'esposizione della fotocamera, impostata dall'esposimetro. Luce bianca richiede uno Scanlight con LED bianco (v4, Big Scanlight).
    *   **Calibra sulla base della pellicola…**: posiziona il rettangolo su pellicola trasparente ed eseguila. NegPy trova i livelli della luce e l'otturatore.
    *   **Imposta a mano**: imposta tu i cursori, poi salvali come preimpostazione. Senza fotocamera collegata, non salva alcuna esposizione, e una scansione usa quella della fotocamera.
*   **Rosso**, **Verde**, **Blu**, **Bianco**: i livelli della luce (da 0 a 255). **Luce accesa** accende la luce per inquadrare la pellicola, o la spegne.
*   **Modalità di acquisizione**: **Tricromia** esegue un'esposizione per colore e le unisce; **Esposizione singola** esegue un'esposizione con tutti e tre i colori accesi. **Miscela** tiene aperta la fotocamera in posa B e accende a turno rosso, verde e blu, per un unico file. Una preimpostazione Tricromia o Miscela può passare dall'una all'altra.
*   **Scatto dell'otturatore** (Miscela): cosa tiene aperto l'otturatore. **USB** usa il controllo della posa B della fotocamera. **Cavo** usa un cavo di scatto nella presa jack da 3,5 mm dello Scanlight, per esposizioni fino a 2,55 s.
*   **Profilo del sensore** (preimpostazione Miscela senza un profilo proprio): il profilo di diafonia del sensore per le sue scansioni. Scansiona attende che ce ne sia uno. Il pulsante con le provette ne crea uno.
*   **Crea profilo di diafonia del sensore** (Esposizione singola o Miscela): salva anche un profilo di diafonia del sensore usato dalle scansioni di questa preimpostazione (vedi Calibrazione, §10.7).
*   **ISO**, **Otturatore**, **Diaframma**: l'esposizione della preimpostazione. Una modifica vale finché non scegli di nuovo la preimpostazione. Il pulsante con l'indicatore misura.
*   **Otturatore IR** e **Infrarosso** (luci con LED a infrarossi): aggiunge a ogni scansione un'esposizione a infrarossi per la rimozione della polvere, salvata come file `_IR`. Richiede una fotocamera che veda l'infrarosso (a spettro completo o convertita) e la messa a fuoco manuale. Solo pellicola a colori.
*   **Ritardo tra i canali**: una pausa tra le esposizioni rossa, verde e blu, per le fotocamere che ne hanno bisogno.

<!-- panel:scan_output src:3015443a -->
### Uscita

Condiviso da entrambi gli scanner.

*   **Cartella**: dove vengono salvate le scansioni.
*   **Scansiona come rullino**: la cartella si apre come rullino, e i fotogrammi si caricano man mano che vengono salvati.
*   **Cartella come rullino**: le scansioni vanno nella cartella stessa. Se disattivato, scrive in una sottocartella **Rullino**, e un nuovo nome di Rullino avvia un nuovo rullino. **+** accanto a Rullino passa al nome successivo (da `Roll001` a `Roll002`).
*   **Formato** e **Nome file** (scanner per pellicola): TIFF, oppure TIFF (mono) per il B/N, e il modello del nome file (`{{ date }}`, `{{ seq }}`).

<!-- panel:scan_strip src:135ca0e1 -->
### Anteprima della striscia

Mostra l'intera striscia, per posizionare e scegliere i fotogrammi prima della scansione. **Applica inquadratura** conserva l'inquadratura e **Scansiona** esegue la scansione. **Ferma anteprima** termina in anticipo un'anteprima.

*   **Ritaglio**: trascina su un fotogramma. **Cancella ritagli** rimuove tutti i ritagli.
*   **Scostamento**: sposta ogni fotogramma lungo la pellicola per allinearlo alle immagini. Il riquadro bianco mostra il fotogramma trovato da NegPy.
*   **Deriva**: uno scostamento che cresce da un fotogramma all'altro.
*   Il cursore sotto ogni tessera sposta solo quel fotogramma.
*   **Dimensioni** imposta la dimensione delle tessere. **Tutti**, **Nessuno** e l'interruttore di ogni fotogramma scelgono i fotogrammi. **Anteprima del fotogramma** scansiona di nuovo un fotogramma.

---

## 14. Preferenze

Menu **⋯** → **Preferenze…** (`Ctrl+,`). Le modifiche si applicano subito; una riga indica quando serve un riavvio.

### Interfaccia

*   **Lingua**: inglese, cinese semplificato, giapponese, coreano, tedesco, vietnamita, norvegese bokmål o italiano, dopo un riavvio.
*   **Scala dell'interfaccia**: dall'80% al 120%, dopo un riavvio.
*   **Sfondo dell'area di lavoro**: dal nero al bianco. Il grigio medio è il migliore per valutare una stampa.
*   **Zoom persistente**: mantiene lo zoom quando cambi fotogramma. **Zoom con rotellina invertito**: la rotellina in su riduce lo zoom.
*   **Suggerimenti delle scorciatoie** (inizialmente attivo): dopo che usi un controllo con il mouse, un avviso nell'area di lavoro mostra la sua scorciatoia da tastiera, una volta per sessione.
*   **Personalizza scorciatoie…**, **Modifica barra degli strumenti…** e **Reimposta la disposizione dei pannelli**.

### Accessibilità

*   **Visione dei colori**: i colori dei segni che si distinguono solo per colore (sovrapposizione della polvere, maschere, sovrapposizione delle zone, tosatura). Scegli l'impostazione per protanopia o deuteranopia, tritanopia o acromatopsia.

### Prestazioni

*   **Accelerazione GPU**: esegue la resa sulla scheda grafica. Se disattivata usa la CPU, più lenta, con lo stesso risultato. Se la GPU non funziona all'avvio, una riga ambra qui lo segnala.
*   **Ricerca per significato**: trova i fotogrammi tramite una descrizione (§2).
*   **Aggiorna automaticamente le miniature**: rigenera in background le miniature non aggiornate, quando si apre un rullino e dopo una modifica a molti fotogrammi. Se disattivato (predefinito), i fotogrammi ricevono un punto ambra finché non usi **Aggiorna miniature**.
*   **Dimensioni dell'anteprima**: la risoluzione dell'area di lavoro. Più alta è più nitida e usa più memoria.
*   **Cache delle anteprime**, **Limite della cache delle anteprime**, **Buffer HQ** e **Fotogrammi resi**: quanta memoria usa NegPy. Ogni riga mostra la raccomandazione per questo computer. Puoi impostare un valore più alto, e il suggerimento diventa allora ambra.
*   **Limite delle texture GPU**: l'immagine più grande che la GPU carica in una volta. A 0, una GPU integrata usa 6144 px e le altre GPU non hanno limite. Un valore qui sostituisce quel limite. Abbassalo se le esportazioni esauriscono la memoria della GPU.
*   **Riduci l'uso di memoria in esportazione**: esporta in porzioni più piccole, per un computer in cui le esportazioni esauriscono la memoria della GPU.
*   **Mostra avviso di memoria GPU**: ti avvisa quando un'anteprima HQ viene ridotta per rientrare.

### Sessione e archiviazione

*   **Trasferisci le impostazioni tra i fotogrammi** e **Impostazioni persistenti…**: cosa prende un nuovo fotogramma dall'ultimo (§1).
*   **Gestisci database…**: dimensioni, cancellazione delle modifiche salvate e cartelle della libreria.

### Sostituzione all'avvio (`override.toml`)

Se NegPy si arresta all'avvio o disegna in modo errato, modifica `Documents/NegPy/override.toml` (creato al primo avvio) e riavvia. I suoi valori prevalgono sulle Preferenze.

| Impostazione | Valori | Effetto |
|---------|--------|--------|
| `rendering.backend` | `"auto"`, `"vulkan"`, `"dx12"`, `"metal"`, `"cpu"` | Come vengono elaborate le immagini. `"cpu"` disattiva la GPU. |
| `display.qt_rhi_backend` | `"auto"`, `"vulkan"`, `"d3d12"`, `"metal"`, `"opengl"`, `"software"` | Come viene disegnata la finestra. |
| `display.qt_platform` | `"auto"`, `"xcb"`, `"wayland"` | Sistema di finestre (solo Linux). |
| `performance.force_hq_preview` | `true` / `false` | Attiva o disattiva l'Anteprima HQ. |
| `logging.level` | `"debug"`, `"info"`, `"warning"`, `"error"` | Quanto registra NegPy. Usa `"debug"` per una segnalazione di bug. |

Anche le righe sulla memoria delle Preferenze si possono impostare qui (`max_texture_size`, `preview_render_size`, `preview_cache_max_bytes`, `preview_cache_max_entries`, `preview_cache_max_full_res_entries`, `render_memo_max_entries`).

**Soluzioni comuni:**

*   **Arresto immediato su Linux**: `backend = "cpu"` oppure `qt_rhi_backend = "opengl"`.
*   **Anteprima nera o vuota su Windows**: `backend = "dx12"` oppure `qt_rhi_backend = "software"`.
*   **Problemi di disegno su Wayland**: `qt_platform = "xcb"`.
*   **Memoria GPU esaurita durante l'esportazione**: `max_texture_size = 4096`.

---

## 15. Aggiornare NegPy

NegPy controlla all'avvio se c'è una nuova versione. Una nuova versione mette un punto verde su **⋯** e aggiunge **Aggiorna alla vX.Y.Z…** al suo menu **Aiuto**. **Controlla aggiornamenti…** controlla manualmente.

**Installa aggiornamento** scarica la nuova versione, chiude NegPy, la installa e lo riapre. Su Windows, approva la richiesta di amministratore *prima* che NegPy si chiuda. Se l'aggiornamento non riesce, la tua installazione resta com'era. Le tue modifiche, preimpostazioni e impostazioni non vengono toccate.

Quando NegPy non può aggiornarsi da solo, per esempio quando viene eseguito dal codice sorgente, il pulsante indica **Apri la pagina delle versioni**.

---

## Informazioni aggiuntive

*   I NEF di Nikon Z 8 e Z 9 con compressione **HE** o **HE\*** non possono essere aperti. Scatta in **Compressione senza perdita**, oppure convertili con Adobe DNG Converter.
*   Le **modifiche** sono salvate in un database locale in base al contenuto del file, quindi i file possono essere spostati o rinominati. **File collaterali di modifica** (scheda Esporta) può anche scriverle come file `.negpy`.
*   NegPy salva una modifica quando cambi fotogramma o esporti. Chiudere NegPy prima di allora può far perdere le ultime modifiche.
*   [KEYBOARD.md](KEYBOARD.md) elenca tutte le scorciatoie, [TEMPLATING.md](TEMPLATING.md) spiega i modelli dei nomi file e [PIPELINE.md](PIPELINE.md) spiega in dettaglio l'elaborazione.
