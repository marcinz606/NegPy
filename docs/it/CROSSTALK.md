# Matrici di diafonia personalizzate

Il controllo **Diafonia** nel riquadro Calibrazione (scheda Rullino) separa i canali di colore. NegPy fornisce
una matrice integrata (**Generic C41**). Puoi aggiungere le tue, calibrate per la tua pellicola *e
la tua configurazione di scansione*, senza scrivere codice.

## Per quale pellicola è una matrice

Negativo a colori e Trasparenza usano coloranti diversi, quindi ogni profilo dichiara il proprio
processo:

```toml
process = "Transparency"    # oppure "Color Negative" (il valore predefinito se la chiave manca)
```

Il menu offre solo le matrici per la pellicola corrente, e la resa usa lo stesso valore come
condizione: un profilo non corrispondente diventa l'identità. I vecchi nomi delle modalità (`C41`, `B&W`, `E-6`)
si caricano ancora.

NegPy non fornisce alcuna matrice per Trasparenza. Sulle diapositive il menu Matrice parte vuoto, e il menu
e il cursore Intensità sono disattivati. Il pulsante dell'editor resta attivo: **+** nell'editor
crea una matrice per il processo corrente, e il controllo **Processo** dell'editor imposta il
processo su qualsiasi profilo salvi. La Diafonia è nascosta solo in Negativo B/N, che ha un'unica
emulsione e niente da separare.

> **La pellicola invertibile si legge in modo diverso.** Su un negativo, la separazione porta la resa *verso*
> la scena. Una trasparenza è l'immagine finita, quindi la separazione porta la resa *lontano* dalla
> diapositiva. In Trasparenza, la Diafonia è un controllo di separazione dei colori, non una correzione
> di fedeltà. Lascia Intensità a 0 per corrispondere alla diapositiva.

## Che cosa fa

I coloranti ciano, magenta e giallo di un negativo a colori disperdono ciascuno un po' di densità negli altri
canali, il che rende i colori impastati e poco separati. La correzione della diafonia moltiplica
il vettore della **densità del negativo** di ogni pixel per una matrice 3×3, *prima* della normalizzazione e della
curva di stampa. Gli assorbimenti secondari dei coloranti sono lineari nella densità dei coloranti del negativo, quindi le matrici
si ricavano in questo dominio.

La sorgente luminosa e i filtri colorati sovrapposti della fotocamera aggiungono lo stesso tipo di miscela
lineare, quindi un'unica 3×3 li corregge tutti e tre insieme. Una matrice è una proprietà di un'**intera configurazione
di scansione**, non di una pellicola, e la stessa pellicola può richiedere matrici diverse su attrezzature
diverse.

Due controlli correlati nello stesso pannello fanno un lavoro diverso. La matrice del **sensore** corregge
la fotocamera nell'acquisizione lineare. **Rifinitura della tinta** ruota la tinta per uno spettro luminoso insolito. Vedi
[USER_GUIDE.md](../../negpy/translations/USER_GUIDE.it.md) §10.4.

> ### Che cosa sono davvero le matrici incluse
>
> Ogni profilo incluso è **letto da una scheda tecnica pubblicata della densità spettrale dei coloranti, non
> misurato su una scansione**, da cui l'*(approx)* in ogni nome.
>
> - Modellano **solo i coloranti**. È la correzione completa solo quando l'acquisizione legge
>   ogni banda di colorante in modo pulito: uno **scanner a banda stretta** (il sensore monocromatico di un Coolscan legge un LED
>   alla volta, in modo del tutto pulito; la schiera trilineare di un Pakon ci va vicino, con una leggera dispersione
>   residua), un'acquisizione **tricromica**, oppure un'attrezzatura a Esposizione singola con
>   il **Profilo di diafonia del sensore** applicato. Una luce a banda larga e un sensore Bayer
>   aggiungono una miscela che una matrice dei soli coloranti non può modellare.
> - Le schede tecniche delle pellicole amatoriali di solito pubblicano un'unica curva *aggregata* di neutro a metà scala, non
>   curve C/M/Y separate. Per le pellicole Kodak, le curve dei coloranti della Vision3 sono adattate a
>   quella curva, il che presuppone che gli accoppianti di mascheratura della pellicola corrispondano a quelli della Vision3. Le altre
>   matrici amatoriali sono lette a occhio dal grafico.
>
> Su un'attrezzatura a banda stretta battono comunque l'identità, ma una matrice regolata da te è più
> affidabile. Le matrici incluse sono di sola lettura. Usa **Crea una copia modificabile** per partire da una di esse.

La matematica, per pixel, sul negativo grezzo decodificato:

```
density      = -log10(rgb_negative)
density_out  = M · density
```

`M` è la tua matrice 3×3. Il cursore **Diafonia** (0–1) la miscela con la matrice identità
e normalizza le righe del risultato, quindi `0` è disattivato e `1` è la matrice completa:

```
M_applied = I · (1 - strength) + M · strength
M_applied = M_applied / row_sums(M_applied)        # ogni riga normalizzata in modo che la somma sia 1
```

Ogni riga ha somma 1, quindi un grigio uniforme resta grigio. La matrice ridistribuisce solo le
*differenze* di colore tra i canali.

## Leggere la matrice

La matrice è ordinata per righe. **Le righe sono i canali di uscita**, **le colonne sono i canali di ingresso**:

|            | ingr. R | ingr. G | ingr. B |
| :--------- | :----- | :----- | :----- |
| **usc. R**  | 1.00   | -0.05  | -0.02  |
| **usc. G**  | -0.04  | 1.00   | -0.08  |
| **usc. B**  | -0.01  | -0.10  | 1.00   |

- La **diagonale** resta vicina a `1.0`, così ogni canale mantiene la propria densità.
- I termini **fuori diagonale** sono di solito piccoli e negativi. Sottraggono ciò che uno strato disperde
  in un altro.
- Mantieni le righe con somma vicina a `1.0`. Le righe vengono normalizzate, quindi scostamenti maggiori funzionano, ma sono
  più difficili da interpretare.

## Formato del file (TOML)

Metti i file `.toml` nella tua cartella utente:

```
<Documents>/NegPy/crosstalk/
```

Il programma legge sul posto la raccolta inclusa (`crosstalk/` nel repository), quindi quei profili
compaiono accanto ai tuoi e si aggiornano a ogni versione. Un nome incluso prevale su un file utente con
lo stesso nome. Ogni file è una matrice:

```toml
# my_film.toml
name = "Gold 200 + Spectracolor"   # nome visualizzato facoltativo; altrimenti si usa il nome del file
type = "tuned"                     # provenienza facoltativa; la raggruppa nel menu
matrix = [                         # 3x3, per righe (usc. R/G/B × ingr. R/G/B)
  [ 1.00, -0.05, -0.02],
  [-0.04,  1.00, -0.08],
  [-0.01, -0.10,  1.00],
]
```

- `matrix` è **obbligatoria**: esattamente 3 righe di 3 numeri.
- `name` è **facoltativo**. Senza di esso, il menu mostra il nome del file (senza `.toml`).
- `type` è **facoltativo**. Dice *da dove provengono i numeri*, e quindi quanto
  fidarsene. Il menu raggruppa i profili per tipo:

  | `type` | Gruppo nel menu | Significato |
  | :--- | :--- | :--- |
  | `measured` | Misurato | Adattata a scansioni reali di un riferimento noto. |
  | `tuned` | Tarato su un'attrezzatura | Regolata a occhio su fotogrammi reali. L'editor salva questo tipo. |
  | `specsheet-based` | Dalle schede tecniche (approssimativo) | Letta dalle curve pubblicate di densità dei coloranti. Ogni profilo incluso. |
  | qualsiasi altro valore o assente | Altro | Si carica e funziona, senza provenienza. |

- NegPy salta in silenzio i file malformati (forma sbagliata, valori non numerici, TOML non valido).
- Il nome `Generic C41` è riservato alla matrice integrata e viene ignorato se riutilizzato.

NegPy **incorpora la matrice scelta nella modifica** quando la selezioni, così le modifiche e le
preimpostazioni salvate restano riproducibili anche se in seguito sposti o elimini il file.

## Come usarla

1. Metti il tuo `.toml` in `<Documents>/NegPy/crosstalk/`.
2. Apri il riquadro **Calibrazione** nella scheda Rullino e trova il **menu della diafonia** sotto DIAFONIA. I nuovi
   file compaiono alla successiva sincronizzazione del pannello, per esempio quando cambi foto. Se il tuo
   non compare, riavvia.
3. Scegli il tuo profilo e alza **Intensità** sopra 0.
4. Scegli **Generic C41** per tornare alla matrice integrata.

La diafonia cambia ciò che leggono gli esposimetri della normalizzazione, quindi esegui di nuovo l'**Analisi del rullino** (e
salva di nuovo i limiti bloccati) dopo aver cambiato il profilo o l'intensità.

### Crearne una tua, in breve

Non servono dati spettrali né uno spettrofotometro per fare meglio di *Generic C41*:

1. Scegli un fotogramma di cui conosci i colori reali: fogliame, cielo, un cartoncino grigio, la pelle.
2. Parti da **Generic C41** e alza **Intensità** finché i colori non si separano, prima che diventino
   sgargianti. Su molte attrezzature basta questo.
3. È ancora sbagliato in un modo specifico? Apri l'editor della matrice, premi **Crea una copia modificabile** e
   cambia il termine di miscela della coppia che non va. Il verde che si disperde nel rosso è **Rosso ›
   Dal verde**. Cambia un termine alla volta, a piccoli passi. `0.02` si vede.
4. Dalle il nome della **combinazione**, per esempio `Gold 200 + Spectracolor`. Esegui di nuovo l'**Analisi
   del rullino** e controllala su un secondo fotogramma.

Se tutte le tinte virano nello stesso modo e i neutri vanno bene, il problema è lo spettro della luce.
**Rifinitura della tinta** è la soluzione più semplice, e una matrice la contrasterebbe. Colori impastati e poco separati sono
il segno della diafonia.

### Modificare le matrici nel programma

L'**icona dei cursori** accanto al menu apre l'**editor della matrice**:

- Sfoglia tutti i profili. Le matrici incluse e **Generic C41** mostrano un lucchetto e sono
  **di sola lettura**. I tuoi profili sono modificabili.
- Ogni canale (**Rosso**, **Verde**, **Blu**) ha due cursori, **Dal** ciascuno degli altri canali,
  da −1 a 1, e l'anteprima si aggiorna in tempo reale. La normalizzazione delle righe rende la diagonale
  ridondante, quindi sono modificabili solo i sei termini di miscela. Il cursore **Intensità dell'anteprima**
  controlla solo questa anteprima. Il cursore **Intensità** del riquadro applica la matrice.
- **Crea una copia modificabile** clona la matrice bloccata selezionata in un profilo modificabile.
- **Salva** scrive il profilo come `.toml` in `<Documents>/NegPy/crosstalk/`, così compare
  nel menu.
- **Applica e chiudi** conserva ciò che stavi vedendo in anteprima. **Annulla** ripristina.

### Misurare da una tabella colori

Una tabella fotografata sulla pellicola è l'unico riferimento che stabilisce davvero una matrice. Fotografa una SpyderCheckr 24 o
una ColorChecker 24 alla luce del giorno su un fotogramma del rullino, scansionala con la tua attrezzatura e aprila. Il
**pulsante con il campione di colore** accanto al menu apre **Misura dalla tabella** (solo Negativo a colori).

1. Trascina i quattro punti sui centri delle quattro tacche d'angolo della tabella. L'ordine dei punti e
   l'orientamento della tabella non contano: NegPy legge la griglia in ogni verso, anche di lato e specchiata,
   e tiene quello i cui colori corrispondono alla tabella.
2. Dai al profilo il nome della combinazione e premi **Misura e salva**.

L'adattamento parte dall'identità ed esegue la resa propria del fotogramma per ogni matrice di prova, così
normalizzazione e Rimozione della dominante misurano di nuovo ogni volta. Stampa sul supporto Digitale, così i coloranti di una carta
restano fuori dalla matrice dell'attrezzatura. Minimizza il CIEDE2000 rispetto ai valori
pubblicati della tabella (D50, adattati al D65 della resa), con la luminosità a metà peso perché il
tono è affidato alla curva di stampa, e una piccola attrazione verso l'identità. Un unico guadagno di esposizione fa prima corrispondere
i grigi medi. Il risultato viene salvato con `type = "measured"` e applicato con Intensità 1. Esegui di nuovo l'**Analisi
del rullino** sugli altri fotogrammi.

Dopo l'adattamento, ogni riquadro di campionamento mostra il ΔE00 di colore della sua tacca rispetto al valore della tabella: la differenza
di tinta e croma, senza la luminosità perché la imposta la curva di stampa. Verde è entro 2,
ambra entro 5, rosso oltre. Spostare un punto li cancella.

Una matrice misurata su una pellicola sotto una luce descrive solo quella coppia. La separazione del sensore avviene
prima della Diafonia, quindi l'adattamento misura solo ciò che lascia il profilo di diafonia del sensore del fotogramma; misura di nuovo
se cambi quel profilo.

## Contribuire con una matrice

Aggiungi il tuo `.toml` alla cartella [`crosstalk/`](../../crosstalk/) del repository e apri una pull request.
Le matrici incluse arrivano a tutti gli utenti con la versione successiva. Vedi
[`crosstalk/README.md`](../../crosstalk/README.md). Indica l'attrezzatura nel nome del profilo e nella
pull request: la pellicola, la sorgente luminosa e la fotocamera o lo scanner.
