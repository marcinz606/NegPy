# Scansione con fotocamera

La scheda **Scansione**, con **Scanner** impostato su **Fotocamera**, acquisisce i negativi con una fotocamera collegata e li importa
in NegPy. Ha due modalità e ne sceglie una in base all'hardware che trova.

## Che cosa fa

**Scansione normale con fotocamera.** Un'esposizione del fotogramma sotto qualsiasi luce, importata come un
normale RAW. Richiede solo una fotocamera supportata.

**Scansione tricromica.** Con uno [Scanlight](https://github.com/jackw01/scanlight)
collegato, la luce lampeggia in rosso, poi in verde, poi in blu, e la fotocamera esegue un'esposizione
per canale. La fusione **Tricromia** di NegPy allinea al sottopixel i tre RAW in un unico fotogramma
prima dell'inversione. Una banda stretta alla volta elimina la diafonia tra gli strati di colorante, e
ogni canale ottiene l'intera gamma dinamica del sensore.

**Scansione a Esposizione singola.** Una preimpostazione a banda stretta può invece accendere rosso, verde e blu
insieme per un'unica esposizione: uno scatto dell'otturatore per fotogramma e nessuna registrazione. Il fotogramma
viene importato come un normale RAW, e il suo rullino viene registrato con la Modalità tricromia disattivata. I canali
si sovrappongono sul sensore, quindi abbinala a un profilo di [calibrazione del sensore](#calibrazione-del-sensore).

**Scansione a Miscela.** Una preimpostazione Miscela scatta un'esposizione in posa B per fotogramma. L'otturatore
si apre al buio, la luce mostra rosso, poi verde, poi blu, e l'otturatore
si chiude al buio. Ogni canale riceve la luce della sua esposizione tricromica, quindi il fotogramma
non richiede registrazione e non ha scostamento tra i canali. Viene importato come un unico RAW, come un fotogramma
a Esposizione singola. Ogni LED raggiunge comunque tutti i canali del sensore, quindi la Miscela richiede un
profilo di [calibrazione del sensore](#calibrazione-del-sensore). Vedi [Miscela](#miscela).

## Che cosa serve

| | |
|---|---|
| **Fotocamera** | Qualsiasi corpo che [libgphoto2 può pilotare](http://gphoto.org/proj/libgphoto2/support.php) con Live View e scatto remoto. Un corpo assente da quell'elenco funziona spesso con il driver PTP generico (una a7C II sì). |
| **python-gphoto2** | Dipendenza facoltativa, LGPL. `pip install gphoto2`. Nessun SDK del produttore è incluso, collegato o richiesto. |
| **Scanlight** *(facoltativo)* | Serve solo per la luce a banda stretta. |
| **Lento Dock** *(facoltativo)* | Si usa al posto di uno Scanlight. Vedi [Lento Dock](#lento-dock). |

Senza python-gphoto2, lo scanner Fotocamera della scheda Scansione mostra un suggerimento di configurazione di una riga e il resto
di NegPy funziona come al solito.

> **Solo macOS e Linux.** libgphoto2 non ha una versione per Windows, quindi non ci sono pacchetti
> per Windows e la scheda lì non può collegarsi.

## Configurazione

```bash
uv sync --group camera     # oppure: pip install gphoto2
```

libgphoto2 è contenuto nel pacchetto wheel. Se crei tu il pacchetto del programma, esegui lo stesso comando
prima di `make build` per includere i driver delle fotocamere di libgphoto2; senza di esso, il programma impacchettato
mostra il suggerimento di configurazione.

Metti la fotocamera in modalità **PC Remote** e collegala via USB. NegPy la rileva.

> Su macOS il demone di sistema delle fotocamere si appropria di una fotocamera PTP libera e la tiene per minuti, quindi
> NegPy prende il corpo non appena lo scanner Fotocamera della scheda Scansione lo vede. Finché quello scanner è
> aperto, Anteprima, Foto, Acquisizione Immagine e altri programmi di scatto remoto non possono usare la fotocamera.
> Scollega il cavo per liberarla. Se la fotocamera risulta in uso senza che nulla di visibile sia
> in esecuzione, vedi [Risoluzione dei problemi](#risoluzione-dei-problemi).

## Scansione

**Inquadratura e messa a fuoco.** Apri il **Live View**. Fai clic sull'immagine per puntare la lente di ingrandimento
*hardware* della fotocamera su quel punto; fai di nuovo clic per il fotogramma intero. L'**Indicatore di fuoco** sotto
l'immagine confronta la nitidezza in tempo reale con il valore migliore da quando la vista è cambiata l'ultima volta. Un clic
cambia la vista, e così fa la lente di ingrandimento del corpo: l'MF Assist di Sony ingrandisce quando
si gira la ghiera di messa a fuoco, e **Durata ingrand. MF** torna indietro dopo 2 s o 5 s (impostala su
**Nessun limite** per mantenere l'ingrandimento). Gira la ghiera di messa a fuoco oltre il fuoco migliore, poi torna indietro finché
l'indicatore non segna **al massimo**. Funziona su ogni corpo con Live View.

Nella scansione a luce bianca e con la sola fotocamera, imposta ISO, otturatore e diaframma in tempo reale dalla
barra degli strumenti. Una preimpostazione a banda stretta calibrata li nasconde e li blocca (vedi **Preimpostazioni**). Un controllo che il corpo
non può offrire è disattivato, di solito il diaframma su un obiettivo senza diaframma elettronico (la maggior parte delle ottiche
da ingranditore e macro). Un bordo sottile indica il limite del fotogramma acquisito.

**Istogramma** e **Zebre** sulla barra degli strumenti leggono il fotogramma di anteprima. L'istogramma mostra R, G
e B. Le zebre tratteggiano la parte alta (rosso, dal 99% in su) e la parte bassa (blu, dall'1% in giù) della
gamma dell'anteprima. L'anteprima è il JPEG del corpo, che viene tosato prima del RAW, quindi una zebra
lascia un po' di margine al RAW. Su un negativo la base trasparente deve restare sotto la parte alta; su una diapositiva
devono restarci le alte luci dell'immagine.

**Esposimetro.** Premi **Esposimetro** nel Live View, oppure l'indicatore **Esposimetro a priorità di diaframma** accanto a
ESPOSIZIONE DELLA FOTOCAMERA nel riquadro, per scattare un unico RAW di prova all'esposizione attuale. L'indicatore del riquadro
apre il Live View quando è chiuso. Con **Luce bianca, negativo B/N**, **Luce bianca, negativo
a colori**, oppure un negativo a colori con la preimpostazione AUTO, la base trasparente (letta sul bordo pellicola attorno
all'immagine) va al 90% della gamma nel canale più luminoso, lo stesso obiettivo di Calibra.
Un negativo senza base trasparente nell'inquadratura, o con una base che la prova non riesce a leggere senza tosatura, misura invece
la parte meno densa dell'immagine e lo segnala. Un fotogramma a luce bianca viene importato con il bilanciamento del bianco
della fotocamera, quindi quel guadagno è compreso nel suo obiettivo. Con **Luce bianca, diapositiva** ci vanno
le alte luci più brillanti dell'immagine. AUTO classifica la prova come farebbe un'importazione, quindi
misura come negativo una diapositiva interpretata male finché non scegli la preimpostazione Diapositiva. Senza bordo pellicola, o con un
bordo non più luminoso dell'immagine, l'Esposimetro usa le alte luci dell'immagine.

L'otturatore passa al valore del corpo più vicino uguale o inferiore a quell'esposizione, come scrittura di esposizione
conteggiata: Scansiona attende finché il corpo non la conferma. La riga di stato riporta, per esempio,
`Esposimetro: negativo, base al 62% → 1/15 (+0,5 stop)` (gli stop sono lo spostamento compiuto da quel valore),
e indica la quota dell'immagine tosata. Una base che la prova non riesce a leggere senza tosatura
misura l'estremo meno denso dell'immagine e la riga dice `(base illeggibile)`. Un'immagine con il due
per cento o più dei pixel saturati scende di due stop e la riga chiede un'altra
prova; un numero inferiore di pixel saturati viene riportato ed escluso dal livello. Una prova sotto lo 0,05% della
gamma in ogni canale viene segnalata come troppo scura, e non viene scritto nulla. Per un obiettivo
oltre il valore più rapido del corpo, la riga dice di chiudere il diaframma o abbassare l'ISO, e
non scrive nulla.

Sotto lo Scanlight, l'Esposimetro richiede un otturatore tra 1/250 s e 2 s, l'intervallo su cui Calibra
calcola. La scansione con la sola fotocamera
misura sull'intera scala del corpo e lascia che sia la prova a decidere il tipo di pellicola. L'Esposimetro
attende mentre è in corso una scansione, una calibrazione o un'altra prova. Una lettura vale per un rullino di
negativi; misura ogni diapositiva.

Dopo una prova le zebre passano al RAW: tratteggiano dove il RAW verrà tosato con l'otturatore
misurato, previsto dalla prova lineare (un pixel che la prova ha tosato resta tratteggiato
finché una prova non lo legge). Restano valide finché ISO, otturatore, diaframma e luce sono quelli
misurati e l'anteprima mostra la stessa immagine. Un avanzamento della pellicola, un cambio di inquadratura o la lente di ingrandimento
riportano alle zebre dell'anteprima. Il tratteggio blu delle ombre proviene sempre
dall'anteprima. La mappa si usa solo quando la prima anteprima all'esposizione misurata ha le stesse
proporzioni della prova e le corrisponde per correlazione di rango delle miniature. Un corpo il cui Live
View ritaglia o reinquadra il RAW non supera questo controllo, e la riga di stato dice `zebre RAW disattivate:`
con il motivo. Con una preimpostazione a banda stretta calibrata la prova scatta sotto la luce di scansione, e
l'otturatore misurato sostituisce quello della preimpostazione finché non scegli di nuovo la preimpostazione.

**Calibra (luce a banda stretta).** Imposta l'ISO e il diaframma con cui esegui la scansione. Premi **+** accanto al
menu delle preimpostazioni, scegli **Calibra sulla base della pellicola…**, posiziona il piccolo rettangolo sulla base trasparente della pellicola (la striscia di bordo tra
i fotogrammi è un buon bersaglio), dai un nome alla preimpostazione ed eseguila. La calibrazione calcola un unico
otturatore condiviso più un livello dei LED per canale, così ogni canale cade appena sotto la tosatura, e
registra l'ISO e il diaframma. La base trasparente diventa il *punto del nero* dopo l'inversione,
quindi una protezione dalla tosatura controlla i fotositi Bayer grezzi e riduce l'esposizione se un canale
si satura. Salva la preimpostazione una volta per pellicola e riutilizzala.

Il menu accanto al nome imposta la modalità di acquisizione. **Tricromia** calcola ogni canale sotto
il proprio LED. **Esposizione singola** calcola insieme i tre livelli per un'unica esposizione con tutti e
tre i LED accesi, misurando quanto ogni canale del sensore legge i LED vicini. Se nessun
livello bilancia i canali, l'esecuzione si ferma e lo segnala; calibra invece una preimpostazione Tricromia.

Con **Crea profilo di diafonia del sensore** attivo, un'esecuzione a Esposizione singola salva anche un
profilo di [calibrazione del sensore](#calibrazione-del-sensore) con il nome della preimpostazione, dalle stesse
esposizioni. Un rullino scansionato con la preimpostazione prende quel profilo e attiva il RAW lineare. Disattiva
l'interruttore per mantenere un profilo creato da te.

Se l'obiettivo non è raggiungibile, l'esecuzione si ferma alla prova e una finestra a comparsa indica cosa fare:
sovraesposto, chiudi il diaframma o abbassa l'ISO; sottoesposto, apri o alza l'ISO.
Non viene salvata alcuna preimpostazione. Correggi e calibra di nuovo nella finestra, che resta aperta.

**Preimpostazioni.** Una preimpostazione selezionata compare in sola lettura, con i suoi livelli RGB, ISO, otturatore e
diaframma. La scansione imposta quell'esposizione sul corpo prima di ogni fotogramma. Il menu contiene le tue
preimpostazioni e quelle integrate che hai aggiunto da **+** › **Preimpostazioni integrate**; una integrata resta
nell'elenco finché il cestino non la rimuove. Per creare una preimpostazione a mano, premi **+** accanto a
PREIMPOSTAZIONE e scegli **Imposta a mano** per sbloccare i cursori e i selettori dell'esposizione. Impostali insieme a **Modalità
di acquisizione** (Tricromia, Esposizione singola o Miscela); Scansiona li usa così come sono, e il pulsante di salvataggio (il dischetto)
li conserva come preimpostazione. Solo le
preimpostazioni a luce bianca usano il canale Bianco, perché lo Scanlight non può accenderlo insieme ai LED rosso, verde e blu. Le preimpostazioni a luce bianca differiscono solo per il tipo di pellicola: AUTO classifica ogni acquisizione,
**Negativo B/N**, **Negativo a colori** e **Diapositiva** lo fissano, per l'importazione e per l'Esposimetro. **Luce
bianca + IR, negativo a colori**, **Luce bianca + IR, diapositiva**, **Tricromia, negativo a colori** e **Tricromia + IR, negativo a colori**
non richiedono neanche loro calibrazione: piena potenza dei LED con l'otturatore della fotocamera, impostato dall'Esposimetro, e
una preimpostazione **+ IR** termina ogni fotogramma con un'esposizione a infrarossi con l'**Otturatore IR**.
L'interruttore **Infrarosso** accanto a Otturatore IR fa lo stesso per qualsiasi preimpostazione.

### Miscela

Miscela è la terza **Modalità di acquisizione**, accanto a Tricromia ed Esposizione singola. Sceglila nella
finestra di calibrazione, oppure a mano con **Imposta a mano**. Miscela si calibra come Tricromia, quindi una
preimpostazione Tricromia o Miscela selezionata può passare dall'una all'altra. Una preimpostazione a Esposizione singola non può,
perché i suoi livelli valgono per tre LED accesi insieme. Imposta la fotocamera su RAW.

Una calibrazione Miscela, con **Crea profilo di diafonia del sensore** attivo (il valore predefinito), salva anche un
profilo di diafonia del sensore con il nome della preimpostazione. Legge tutti i canali del sensore nell'ultimo
scatto di ogni LED, vicino all'obiettivo di esposizione. Il rullino di ogni fotogramma a Miscela prende quel profilo, con
il RAW lineare.

Una preimpostazione Miscela senza un profilo proprio (una impostata a mano, o una preimpostazione Tricromia passata a
Miscela) mostra invece **Profilo del sensore**, e Scansiona attende finché non ne scegli uno. Il pulsante con le provette
apre **Calibra sensore**, dove **Acquisisci dalla fotocamera…** crea un profilo dalla sola luce.

**Scatto dell'otturatore** stabilisce cosa tiene aperto l'otturatore:

- **USB.** NegPy imposta l'otturatore su posa B e preme il comando della posa B della fotocamera tramite il
  cavo USB: `eosremoterelease` su Canon, l'interruttore `bulb` su Nikon e Sony. Un corpo
  che non ha nessuno dei due non può usare l'USB.
- **Cavo.** Un cavo di scatto va dalla presa jack da 3,5 mm dello Scanlight alla fotocamera. La luce
  tiene attiva la linea dell'otturatore per tutta l'esposizione, che il firmware limita a 2,55 s. NegPy
  imposta la posa B via USB se il corpo la elenca. Altrimenti, imposta la posa B sulla fotocamera. La fotocamera
  resta collegata via USB, perché il file arriva comunque via USB.

Ogni canale resta acceso per il tempo di otturazione della preimpostazione al livello della preimpostazione. NegPy temporizza
i canali, quindi un canale più breve di 0,2 s viene attenuato e acceso più a lungo, con la stessa
luce. Tutti e tre i livelli cambiano dello stesso fattore, quindi i canali mantengono il loro equilibrio. Un livello
non scende mai sotto 40, il minimo della calibrazione. Con Cavo, un'esposizione lunga viene schiarita per
rientrare nell'impulso; se non rientra ancora, la scansione si ferma e ti chiede di aprire il diaframma
o alzare l'ISO e calibrare di nuovo. Dopo l'esposizione, il corpo torna al suo valore di otturazione. Una scansione dell'intera striscia tiene la
fotocamera in posa B da un fotogramma all'altro, e ripristina l'otturatore prima degli scatti del Campo piatto. Una preimpostazione
integrata **Tricromia** usa come tempo il valore di otturazione della fotocamera, quindi misura prima.

La luce resta spenta per un breve anticipo dopo la pressione della posa B, perché una fotocamera apre l'otturatore
un po' di tempo dopo la pressione.

**Scansiona.** Scegli una cartella di destinazione e una preimpostazione, poi premi **Scansiona** per ogni fotogramma. I file vanno
in una sottocartella per rullino numerata automaticamente e vengono importati e fusi automaticamente. Una scansione a Esposizione singola o a Miscela scrive un file per fotogramma. **Riprendi** scatta di nuovo il fotogramma corrente
senza far avanzare il contatore. **Ritardo di acquisizione** trattiene entrambi fino a 10 s prima della prima
esposizione, così un tavolo che vibra quando lasci la fotocamera può assestarsi; **Ferma** termina
l'attesa. **Ritardo tra le esposizioni** aggiunge una pausa tra le acquisizioni rossa, verde e blu, per
i corpi più vecchi che bloccano il collegamento USB/PTP quando un nuovo comando di acquisizione arriva troppo presto. Scansiona
scatta l'intera terna nel fotogramma successivo e non tocca mai un file esistente. Nel Live
View, **R**, **G** e **B** scelgono quali esposizioni Riprendi scatta di nuovo sopra quelle del fotogramma
corrente, così si può riscattare un solo canale di una terna; Riprendi sostituisce quei file. **IR**, su una
luce con LED a infrarossi, è l'impostazione Infrarosso e accompagna Scansiona e Riprendi. **Otturatore
IR** nel riquadro imposta l'otturatore dell'esposizione a infrarossi, che un filtro taglia-IR rende
molto più lenta dei canali visibili; un valore che la fotocamera non offre ricade
sull'otturatore della fotocamera. L'infrarosso richiede una luce con LED a infrarossi e una fotocamera che lasci passare gli 850 nm,
un corpo a spettro completo o convertito all'IR: il filtro taglia-IR di una fotocamera di serie lascia il fotogramma
a infrarossi al livello del rumore. Imposta l'obiettivo su messa a fuoco manuale: l'autofocus cerca senza successo sul fotogramma a infrarossi scuro e
va in timeout. Un file `_IR` resta un fotogramma a sé nella Striscia pellicola. Ogni fotogramma scansionato
prende la decodifica implicata dalla sua luce: la luce bianca mantiene il bilanciamento del bianco della fotocamera, Tricromia
ed Esposizione singola decodificano in RAW lineare; un fotogramma già modificato mantiene la propria.

**Luce a banda stretta.** I LED a banda stretta separano gli strati di colorante in modo più pulito della luce bianca, quindi
i colori risultano più saturi. Il supporto di stampa *Digitale (Adobe RGB)* li stampa attraverso coloranti
ideali; una carta o una pellicola da stampa in **Supporto di stampa** li stampa attraverso i coloranti propri di quel supporto.

### Lento Dock

Un Lento Dock può fare da luce al posto di uno Scanlight.
NegPy pilota la luce continua del caricatore, quindi l'otturatore della fotocamera stabilisce ogni esposizione, come con
uno Scanlight. Live View, Esposimetro, Calibra, preimpostazioni, Tricromia ed Esposizione singola funzionano come
descritto sopra.

- Chiudi Dock Connect prima di aprire la scheda Scansione. Un solo programma alla volta può usare la porta del caricatore.
- NegPy usa per primo uno Scanlight. Per eseguire la scansione con il caricatore, scollega lo Scanlight.
- Il caricatore non ha un LED bianco né una luce a infrarossi. NegPy offre comunque le preimpostazioni a luce bianca
  e per esse accende insieme rosso, verde e blu allo stesso livello.
- **Pellicola** stabilisce cosa fa il caricatore quando fai clic su Scansiona:
  - **Disattivata**: la pellicola resta dov'è.
  - **Fotogramma successivo**: dopo ogni scansione, il caricatore porta la pellicola al fotogramma successivo.
  - **Intera striscia**: un clic scansiona tutti i fotogrammi della striscia. **Ferma** termina la striscia.
    NegPy conserva i fotogrammi già scansionati e scarta il fotogramma in corso.

  Riprendi non sposta mai la pellicola. Alla fine della striscia, il caricatore la espelle. L'ultimo
  fotogramma si ferma nella finestra, prima dell'uscita, quindi il caricatore lo spinge fuori. **Espelli** (accanto a
  Pellicola) fa uscire la striscia in qualsiasi momento.
- **Scatta Campo piatto**, con **Intera striscia**: dopo l'espulsione della striscia, NegPy fotografa il caricatore
  vuoto ai livelli dei LED della preimpostazione. Parte tre stop più rapido dell'otturatore di scansione e
  scatta di nuovo se il canale più luminoso è tosato o troppo scuro. NegPy salva lo scatto come profilo di Campo piatto
  del rullino, con il nome del rullino, e lo applica ai fotogrammi del rullino. Lo scatto
  resta nella cartella `flatfield/references`, fuori dal rullino. Una striscia che si ferma prima della
  fine non riceve alcun campo piatto. Il campo piatto non viene mai per primo: dopo che il caricatore ha fatto uscire la pellicola dalla
  finestra e l'ha riportata indietro, non avanza.
- Metti la striscia all'ingresso del caricatore. Il firmware 1.0.238 e successivi la allineano da soli al primo
  fotogramma. Con **Fotogramma successivo** o **Intera striscia**, Scansiona fa avanzare una striscia ancora ferma
  all'ingresso (`INSERT`), per esempio una che era nel caricatore quando NegPy ne ha aperto la porta, perché
  aprire la porta reimposta il caricatore.
- Mentre NegPy usa il caricatore:
  - Un tocco di S funziona come Scansiona: avvia una scansione, oppure la ferma. NegPy legge il pulsante (`BTN?`)
    e agisce quando lo rilasci. Con un fotogramma nella finestra, il caricatore esegue anche una breve esposizione
    propria, e la pellicola si muove solo quando è finita. Con una striscia all'ingresso, un tocco
    prima la allinea. Una pressione prolungata di S è lasciata al caricatore, che con il firmware 1.0.238 espelle la striscia.
  - Un pulsante freccia sposta la pellicola di poco a ogni tocco e la muove finché lo tieni premuto (`BTNMODE JOG`),
    perché Fotogramma successivo e Intera striscia spostano fotogrammi interi.
  - La sospensione automatica del caricatore è disattivata (`SLEEP 0`), perché un caricatore sospeso non risponde a nulla. Se il
    caricatore smette comunque di rispondere, NegPy ne riapre la porta al successivo utilizzo.
- Quando NegPy rilascia il caricatore, ripristina la modalità della luce, la modalità di controllo, la modalità dei
  pulsanti e il tempo di sospensione salvati del caricatore.

La luce pulsata propria del caricatore resta a Dock Connect.

### Calibrazione del sensore

Una scansione a esposizione singola sotto luce a banda stretta può avere tinte che nessun cursore corregge, come
gialli che tendono all'arancione. La causa è la **diafonia del sensore**: le bande passanti dei filtri colorati
della fotocamera si sovrappongono alle bande della sorgente, quindi ogni canale legge un po' degli altri LED. È una
proprietà fissa della coppia sensore e luce, indipendente dalla pellicola.

Tre modi di costruire un profilo danno correzioni quasi identiche (vedi
[Flussi di lavoro del profilo di diafonia del sensore](#flussi-di-lavoro-del-profilo-di-diafonia-del-sensore)):

- **Con una preimpostazione.** Una preimpostazione a Esposizione singola calibrata con **Crea profilo di diafonia del sensore** attivo
  salva un profilo misurato attraverso la base della pellicola (flusso A).
- **Acquisisci dalla fotocamera…** Con la fotocamera collegata, uno Scanlight connesso e nessuna pellicola nel
  supporto, NegPy accende a turno ogni LED, imposta l'otturatore e salva il profilo. Usa
  l'ISO e il diaframma della fotocamera e non richiede il Live View. Un unico profilo vale per tutte le pellicole
  e tutte le preimpostazioni (flusso B, passaggio 1).
- **Da file.** Tre acquisizioni RAW della sola luce (solo rossa, solo verde, solo blu),
  per una luce che NegPy non può controllare (flusso D, passaggio 1).

I profili ricavati dalla base della pellicola e dalla sola luce differiscono leggermente, e nessuno dei due è più preciso. Un
profilo per ogni coppia sensore e luce è sufficiente.

Il profilo separa ogni scansione con una matrice 3×3 nel dominio lineare, prima dell'inversione.
I profili sono file TOML nella cartella `NegPy/sensor`. Esegui di nuovo l'**Analisi del rullino** dopo aver
cambiato il profilo.

### Flussi di lavoro del profilo di diafonia del sensore

Costruisci un profilo nella calibrazione della preimpostazione che **+** apre nel pannello
**Preimpostazione**, oppure nel pannello **Calibrazione** della scheda Rullino. Una preimpostazione che ha creato il proprio profilo
lo assegna a ogni scansione. Un profilo del pannello Calibrazione lo assegni tu, una volta per rullino.

| La tua configurazione | Flusso | Profilo costruito in | Assegnato da |
|---|---|---|---|
| Fotocamera collegata e Scanlight, preimpostazione calibrata | [A](#a-la-preimpostazione-crea-il-profilo) | Calibrazione della preimpostazione | La preimpostazione |
| Fotocamera collegata e Scanlight, preimpostazione calibrata, un profilo per tutte le preimpostazioni | [B](#b-una-preimpostazione-calibrata-e-un-profilo-separato) | Pannello Calibrazione | Tu |
| Fotocamera collegata e Scanlight, preimpostazione manuale | [C](#c-una-preimpostazione-manuale) | Pannello Calibrazione | Tu |
| File scattati senza NegPy | [D](#d-file-non-scansionati-con-negpy) | Pannello Calibrazione | Tu |

Tutti e quattro valgono solo per i fotogrammi a Esposizione singola e a Miscela. Una scansione tricromica non richiede alcun profilo. Una
preimpostazione Miscela calibrata crea il proprio profilo, come nel flusso A.

#### A. La preimpostazione crea il profilo

Il caso predefinito, per le preimpostazioni a Esposizione singola e Miscela calibrate.

Una volta per pellicola:

1. Carica la pellicola. Nel pannello **Preimpostazione**, premi **+** accanto al menu
   delle preimpostazioni e scegli **Calibra sulla base della pellicola…**. Si apre la finestra di calibrazione con il proprio Live View.
2. Imposta l'ISO e il diaframma con cui esegui la scansione, fai clic sulla base trasparente della pellicola e dai alla
   preimpostazione il nome della pellicola.
3. Scegli **Esposizione singola** o **Miscela**, lascia attivo **Crea profilo di diafonia del sensore** e premi **Calibra e salva**.

Per ogni rullino di quella pellicola:

1. Scegli la preimpostazione nel menu.
2. Scansiona il rullino.

Note:

- Calibra di nuovo solo quando cambiano la fotocamera, l'obiettivo, la luce o la pellicola.
- La preimpostazione contiene il nome del profilo; il profilo è un file nella cartella `NegPy/sensor`.
  Se quel file viene eliminato, la preimpostazione esegue comunque la scansione e il rullino non riceve correzione.
- Ogni scansione rende il profilo della preimpostazione quello del rullino e attiva il RAW lineare, sostituendo un
  profilo che avevi scelto a mano per quel rullino.
- Una nuova calibrazione con il nome della preimpostazione scrive di nuovo il profilo, e lo fa anche qualsiasi
  altro profilo salvato con quel nome.
- Ogni preimpostazione ha il proprio profilo, misurato attraverso la base di quella pellicola. I fotogrammi aperti o
  scansionati in altro modo non lo ricevono.

#### B. Una preimpostazione calibrata e un profilo separato

Usalo quando un unico profilo deve valere per tutte le preimpostazioni e le pellicole dell'attrezzatura, quando
hai già un profilo per questo sensore e questa luce, oppure quando una calibrazione della preimpostazione segnala che
le sue esposizioni erano troppo deboli per misurarne uno.

1. Costruisci il profilo una volta per l'attrezzatura. Togli la pellicola dal supporto. Nel pannello
   **Calibrazione**, attiva **RAW lineare**, premi il pulsante di calibrazione sotto
   *Profilo di diafonia del sensore*, dai un nome al profilo, premi **Acquisisci dalla fotocamera…**
   e conferma. Per usare invece tre file della sola luce, sceglili nella stessa finestra come nel
   flusso D, passaggio 1.
2. Carica la pellicola. Calibra la preimpostazione come nel flusso A, con **Crea profilo di diafonia del sensore** disattivato.
3. Scansiona il primo fotogramma. Nel pannello **Calibrazione**, controlla che **RAW lineare** sia attivo,
   scegli il profilo in **Profilo** e premi **Rullino** nell'intestazione del pannello.
4. Scansiona il resto del rullino. I nuovi fotogrammi prendono il profilo del rullino.
5. Esegui di nuovo l'**Analisi del rullino** se alcuni fotogrammi sono stati scansionati prima del passaggio 3.

Note:

- La preimpostazione non imposta nulla sul rullino. Un rullino senza profilo scelto non riceve correzione.
- Un profilo della sola luce non ha base della pellicola, quindi la sua correzione è simile a quella di una preimpostazione, non
  identica.
- Costruisci di nuovo il profilo quando cambiano la fotocamera o la luce.
- Il profilo è un'impostazione di Calibrazione, quindi passa ai fotogrammi successivi che apri. Imposta
  **Profilo** su *Nessuna* per un rullino scansionato sotto un'altra luce.

#### C. Una preimpostazione manuale

Una preimpostazione manuale non misura nulla, quindi non può creare un profilo. Un rullino a esposizione singola scansionato
con essa non ha correzione finché non ne scegli una.

1. Costruisci il profilo come nel flusso B, passaggio 1, oppure usane uno che l'attrezzatura ha già.
2. Nel pannello **Preimpostazione**, premi **+** accanto a PREIMPOSTAZIONE, scegli **Imposta a mano**,
   imposta i livelli e l'esposizione, imposta **Modalità di acquisizione** su **Esposizione singola**
   e salva la preimpostazione.
3. Scansiona il primo fotogramma, poi assegna il profilo al rullino come nel flusso B, passaggio 3.

Valgono le note del flusso B. Il profilo non dipende dai livelli dei LED né
dall'otturatore, quindi un unico profilo vale per una preimpostazione manuale a qualsiasi livello.

#### D. File non scansionati con NegPy

Per i file RAW a esposizione singola scattati sulla scheda, con un altro programma di scatto remoto, o sotto una luce a banda stretta che NegPy non può controllare.

1. Costruisci il profilo per la fotocamera e la luce che hanno scattato i file. Se quell'attrezzatura è una
   fotocamera che NegPy può collegare e uno Scanlight, usa **Acquisisci dalla fotocamera…** come nel flusso B.
   Altrimenti, fotografa tre volte la sola luce senza pellicola nel supporto: solo rossa,
   solo verde, solo blu, in RAW, esposte appena sotto la tosatura. Nel pannello **Calibrazione**,
   attiva **RAW lineare**, premi il pulsante di calibrazione, scegli i tre file, dai un nome
   al profilo e premi **Calcola e salva**.
2. Apri il rullino. Nel pannello **Calibrazione**, attiva **RAW lineare**.
3. Scegli il profilo in **Profilo** e premi **Rullino** nell'intestazione del pannello.
4. Esegui di nuovo l'**Analisi del rullino**.

Note:

- I file devono essere RAW di fotocamera. Il profilo richiede il RAW lineare ed è disattivato senza di esso.
- I file di un'altra fotocamera o di un'altra luce richiedono un profilo proprio.
- Una luce che non può mostrare un colore alla volta non può misurare un profilo.
- Il profilo passa ai fotogrammi successivi che apri, come nel flusso B.

### Quando nessun profilo si applica

Le scansioni tricromiche non hanno diafonia del sensore, e NegPy salta la correzione per gli
elementi a terna. Nemmeno uno scanner per pellicola dedicato ne ha (il CCD monocromatico di un Coolscan legge un
LED alla volta). Il suo errore di colore residuo è la diafonia dei coloranti della pellicola, che la matrice di
**Diafonia** nel dominio delle densità corregge (vedi [CROSSTALK.md](CROSSTALK.md)).

## Risoluzione dei problemi

| Sintomo | Causa | Soluzione |
|---|---|---|
| Il punto della fotocamera indica **"in uso"** (suggerimento ambra) | Un altro programma tiene occupato il corpo. Su macOS è il demone di sistema delle fotocamere, risvegliato da qualsiasi app che sorveglia le fotocamere, spesso un client di sincronizzazione in background senza finestra (Google Drive come elemento di login, Dropbox). | Chiudi l'app in background, oppure rimuovila da **Impostazioni di Sistema → Generali → Elementi login**. Ricollegare il cavo libera la fotocamera finché quell'app non la richiede di nuovo. Anteprima, Foto o Acquisizione Immagine aperti fanno lo stesso. NegPy si ricollega quando la fotocamera è libera. |
| Nessuna fotocamera trovata, e nient'altro è in esecuzione | Il corpo non è in modalità PC Remote, oppure è collegato come archivio di massa/MTP. | Imposta la modalità di connessione USB della fotocamera su **PC Remote**. |
| `[-10] Timeout reading from or writing to the port`, e nessun altro programma la sta usando | Un programma si è arrestato mentre era collegato, e la *fotocamera* tiene ancora la vecchia sessione. NegPy riprova i timeout che si risolvono da soli, quindi questo non si è risolto. | Spegni e riaccendi la fotocamera, oppure scollega e ricollega il cavo. Nulla sul computer lo risolve. |
| Il Live View è nero | Il corpo è uscito dalla modalità PC Remote, oppure c'è il tappo sull'obiettivo. | Spegni e riaccendi la fotocamera. |
| La finestra di scansione si apre con **"nessun Live View"** invece di un'anteprima | La voce di libgphoto2 per questo corpo non prevede l'anteprima (Sony a6000), oppure è collegato in modalità MTP, dove nessun corpo la prevede. | Se il messaggio indica MTP, imposta la modalità USB su **PC Remote** e ricollega. Altrimenti la scansione funziona, ma inquadratura e messa a fuoco si impostano sulla fotocamera, e la calibrazione non è disponibile. |
| L'acquisizione dice che la fotocamera ha restituito un JPEG invece di un RAW | La qualità d'immagine è impostata su JPEG, oppure RAW+JPEG ha selezionato il file elaborato. | Imposta la qualità d'immagine su **solo RAW**, poi riprova. |
| Una scansione a Miscela dice che la fotocamera non è passata in posa B | Il modo di scatto è continuo (un corpo Sony rifiuta la posa B in qualsiasi modo a raffica), oppure la ghiera dei modi non è su M. | Imposta il modo di scatto su scatto singolo e la ghiera dei modi su M, poi esegui di nuovo la scansione. |
| Una scansione a Miscela dice che la fotocamera non ha inviato alcun file RAW | Il corpo non è in posa B, è impostato su JPEG, oppure il cavo di scatto non lo raggiunge. | Imposta la posa B e solo RAW. Con **Cavo**, controlla che il cavo sia nella presa jack da 3,5 mm dello Scanlight e che l'otturatore scatti dalla luce. |
| Una scansione a Miscela espone in modo leggermente diverso dalla scansione tricromica della stessa preimpostazione | NegPy ha attenuato o schiarito i LED per temporizzare i canali, e i LED non sono del tutto lineari. | Imposta un tempo di otturazione di 0,2 s o più per canale. Con **Cavo**, mantieni i tre canali entro l'impulso di 2,55 s. |
| Una scansione a Miscela perde l'inizio della luce rossa | Il corpo apre l'otturatore più di 0,35 s dopo la pressione della posa B. | Chiudi NegPy, lascia un fotogramma nella luce ed esegui `uv run python scripts/bulb_lead_probe.py`. Indica l'anticipo sicuro più breve per il tuo corpo. |
| Il selettore del diaframma è disattivato | L'obiettivo non ha un diaframma elettronico. | Imposta il diaframma sull'obiettivo. |
| Un'impostazione torna al vecchio valore | Le scritture delle proprietà sono asincrone. | NegPy controlla finché il valore non viene applicato e registra un avviso se non succede mai. In quel caso l'impostazione non è scrivibile nella modalità attuale del corpo. Prova **M**. |
| Lo Scanlight non viene rilevato | Porta USB-C sbagliata. | Solo una delle due porte trasporta dati; l'altra è solo alimentazione. Usa la porta dati. |
| Il Lento Dock non viene rilevato | Dock Connect tiene occupata la porta del caricatore, oppure è collegato uno Scanlight. | Chiudi Dock Connect. Scollega lo Scanlight. |
| Lo Scanlight non viene rilevato su Linux, e la porta dati è quella giusta | Il tuo utente non può aprire la porta seriale (la maggior parte delle distribuzioni la assegna al gruppo `dialout`). | `sudo usermod -aG dialout $USER`, poi esci dalla sessione e rientra. |

## Note e limiti

- **Solo USB.** libgphoto2 può raggiungere alcune fotocamere via rete, ma non i corpi Sony.
- **La Miscela non è stata provata su un corpo reale.** I comandi USB della posa B seguono i nomi di libgphoto2 per
  Canon, Nikon e Sony, e l'attrezzatura simulata copre entrambi i tipi di scatto.
- **Sono stati provati solo corpi Sony.** NegPy cerca ogni controllo e non presume alcun produttore.
  `iso` e `shutterspeed` hanno gli stessi nomi ovunque. Il diaframma è `f-number` su
  Sony e Panasonic, e `aperture` su Canon, Nikon, Fujifilm, Olympus e Sigma. L'estensione del RAW
  proviene dalla fotocamera, e lo scatto va in memoria, non su una scheda. Canon e
  Nikon usano la scheda per impostazione predefinita e non scattano senza. Sono graditi i resoconti con altre
  marche.
- **La lente di ingrandimento dipende dal produttore.** Sony mette il rapporto di ingrandimento e il punto di mira
  in un'unica proprietà, quindi un clic ingrandisce *e* punta. Canon (`eoszoom`) e Nikon
  (`liveviewimagezoomratio`) li separano, con uno spazio di coordinate sconosciuto, quindi un clic
  ingrandisce dove il corpo sta già guardando. Gli altri corpi non hanno la lente, e la funzione
  si disattiva. Un corpo che smette di trasmettere mentre è ingrandito (la Nikon D3300) torna al
  fotogramma intero, e la sua lente resta disattivata per la sessione. Su questi corpi, metti a fuoco con l'Indicatore
  di fuoco.
- **Provato su macOS.** Linux dovrebbe funzionare almeno altrettanto bene, ma non è verificato.
- **Velocità.** La maggior parte del tempo di una terna è la latenza di trasferimento per immagine del corpo. Gli scatti
  avvengono all'interno della sessione di Live View in corso, senza riconnessione per fotogramma.
- **Lento Dock.** Provato con un Dock 35 con firmware 1.0.224.
- **Riconoscimenti.** La sequenza R/G/B e l'approccio alla calibrazione dell'esposizione provengono da
  [rohanpandula/TriRGB](https://github.com/rohanpandula/TriRGB). La luce è
  [jackw01/scanlight](https://github.com/jackw01/scanlight). La fotocamera è pilotata tramite
  [python-gphoto2](https://github.com/jim-easterbrook/python-gphoto2). L'approccio a banda
  stretta segue il lavoro di Flückiger e colleghi sulla scansione tricromatica delle pellicole.
