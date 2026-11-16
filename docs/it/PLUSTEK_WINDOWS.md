# Configurazione USB su Windows (Plustek OpticFilm)

Il backend USB Plustek di NegPy usa il driver esterno [pyopticfilm](https://github.com/jboneng/pyopticfilm) (libusb tramite PyUSB), versione **1.3.10** o successiva. Il driver Plustek di serie per Windows non deve possedere il dispositivo.

## Requisiti

- Windows 10 o 11
- Scanner Plustek OpticFilm supportati:
  - **8200i SE** (`07B3:1825`, GL128)
  - **8100 V2** (`07B3:1824`, GL128)
  - **8300i SE** (`07B3:181F`, GL128)
  - **7600i v1** (`07B3:0C3B`, GL843; `bcdDevice` `0x0400`)
  - **7600i v2** (`07B3:0C3B`, GL845; `bcdDevice` `0x0605`)
- WinUSB (o libusbK) associato con [Zadig](https://zadig.akeo.ie/)
- NegPy con la dipendenza facoltativa `plustek` (`uv sync --group plustek` oppure `pip install negpy[plustek]`). Su Windows, pyopticfilm installa anche `libusb-package`

## 1. Verifica il dispositivo

Accendi lo scanner e collegalo, poi:

1. Apri **Gestione dispositivi**
2. Cerca Plustek tra i dispositivi di acquisizione immagini o USB
3. Proprietà → Dettagli → ID hardware deve contenere `VID_07B3&PID_1825`, `PID_1824`, `PID_181F` o `PID_0C3B` per il tuo modello

## 2. Associa WinUSB (Zadig)

1. Scarica [Zadig](https://zadig.akeo.ie/)
2. Seleziona **Options → List All Devices**
3. Seleziona Plustek Film Scanner (`07B3:1825`, `07B3:1824`, `07B3:181F` o `07B3:0C3B`)
4. Sostituisci il driver con **WinUSB**. Funziona anche libusbK
5. Conserva il programma di installazione del driver Plustek di serie. Ti servirà per ripristinare VueScan o il software del produttore in seguito

Finché WinUSB è associato, i programmi di scansione Plustek di serie per Windows non vedono il dispositivo.

## 3. Avvia NegPy ed esegui la scansione

Da una copia del codice sorgente:

```powershell
cd path\to\NegPy
uv sync --group plustek
make run
```

Funziona anche una versione pubblicata per Windows, che include pyopticfilm, PyUSB e libusb. Nella scheda Scansione, imposta Backend su **pyOpticfilm (Plustek)** e aggiorna l'elenco dei dispositivi. Lo SE compare quando WinUSB è associato.

## 4. Ripristina il driver del produttore

1. Scollega lo scanner
2. Gestione dispositivi → disinstalla il dispositivo WinUSB. Spunta "elimina il software del driver" se viene proposto
3. Reinstalla il pacchetto del driver Plustek o VueScan
4. Ricollega lo scanner

## Risoluzione dei problemi

| Sintomo | Causa probabile |
|---------|----------------|
| Elenco dei dispositivi vuoto | PID sbagliato, scanner scollegato, oppure ancora con il driver del produttore |
| `DriverBindingError` o accesso negato | WinUSB non è associato, oppure un'altra app tiene occupato il dispositivo |
| `UsbError` o errori di collegamento | Cavo o hub. Prova una porta diretta della scheda madre |
| L'interfaccia segnala che mancano USB / PyUSB | Installa il gruppo plustek: `uv sync --group plustek` |
| La prima scansione a una certa risoluzione richiede qualche secondo | Normale: AFE più una striscia di ombreggiatura nera e una bianca, come fa SilverFast. Il risultato viene conservato in cache per ogni risoluzione |
| Ombreggiatura ASIC pronta (preferibile) | Il registro mostra una media del bianco a colori di circa 50–57k, `median_gain` di circa 1,1–1,7x, poi `ASIC shading ready` e l'immagine con `shading=True` / DVDSET attivo. Dopo un cambio di driver, elimina `plustek_calib` ed esegui di nuovo la scansione per misurare di nuovo la calibrazione |
| Media del bianco a colori di circa 50–57k | Normale. A guadagno unitario DVDSET restituisce `raw − dark`, quindi una striscia luminosa legge vicino al fondo scala. I valori 11–13k nelle acquisizioni sono i *guadagni* che SilverFast ne ha calcolato (`0x2000` = 1,0), non livelli del bianco |
| Guadagno mediano fuori da 1,02–3,0x | Sotto l'intervallo la striscia è già all'obiettivo, quindi non c'è nulla da uniformare. Sopra, il percorso della luce è troppo scuro. Controlla la lampada, i guadagni AFE e che la testina sia sulla cromatura di riposo pulita |
| Carrello non in posizione di riposo dopo l'ombreggiatura o l'interruzione di una scansione | La striscia bianca arma `AGOHOME` durante la misura, e azzerare SCAN la parcheggia. Se il parcheggio è andato in timeout, usa SilverFast o spegni e riaccendi, poi riprova |
| Angoli più scuri rispetto a SilverFast con la finestra Completa | La finestra Completa include circa 0,8 mm di cromatura del supporto. L'ombreggiatura è solo per colonna, quindi un certo calo in Y su quei bordi è previsto rispetto a un'inquadratura SilverFast più stretta |
| Il positivo è molto scuro finché non ritagli (bordi bianchi sul negativo) | La cromatura del supporto è più luminosa della base della pellicola, quindi i limiti automatici di NegPy si agganciano a essa. Entrambi i percorsi limitano le alte luci del bordo all'interno della pellicola (`border highlight clamp…`). Il percorso ASIC ne ha bisogno, perché DVDSET per costruzione porta quella cromatura al fondo scala. Alza anche Processo → **Margine di analisi**, oppure ritaglia prima dell'automatico |
| Strisce verticali iridescenti a "codice a barre" | La tabella caricata era indicizzata in modo diverso dall'immagine. O la misura è finita nella posizione del guadagno (il guadagno deve essere `0xFFFF × 0x2000 / white`, proporzionale a `1/white`), oppure il blocco è stato impacchettato senza il suo riempimento. La tabella AHB è fatta di blocchi da 512 byte con 126 coppie `(dark, gain)` più due coppie di riempimento con `gain = 0`, quindi i record contigui scorrono di 8 byte per blocco. Elimina `plustek_calib` ed esegui di nuovo la scansione |
| Scena a rombo o inclinata (gli oggetti pendono) | (1) La X dell'immagine si è ridotta mentre l'USB scandiva ancora la riga intera. La tabella di ombreggiatura deve coprire ogni colonna acquisita (la finestra Completa a 1800 resta di circa 2592 px). (2) Una larghezza USB del ritaglio dispari a 1800, per esempio 2455. La larghezza di uscita deve restare pari (`optical_span_alignment` più un numero pari di pixel). Elimina `plustek_calib` (cache v9+) ed esegui di nuovo la scansione. Il registro deve mostrare un `pixels=` pari |
| Negativo molto scuro, o positivo slavato e chiaro, rispetto a SilverFast | Entrambi i percorsi fanno riferimento alla cromatura di *riposo*, che è più luminosa della luce nella posizione di scansione. A 1800 dpi la base della pellicola cade vicino al 42% del fondo scala, circa 1,2 stop sotto, e NegPy misura un negativo sottile. `expose_film_base` la solleva con un unico guadagno scalare legato al canale più luminoso (`… exposure makeup gain=…`). Il guadagno deve restare scalare, altrimenti neutralizza la maschera arancione di cui l'inversione ha bisogno |
| Forte dominante arancione o rosa sul positivo | Controlla `AFE search done gains=…` per un canale pari o vicino a `AFE_GAIN_MAX` (511). Al limite, il termine scuro di quel canale si tosa a 0 (`dark0=(0, …)`), quindi perde i neri e tinge l'intero fotogramma. La ricerca sostituisce il codice della sessione 04 di SF per ogni canale bloccato al limite. Un blocco persistente significa che l'obiettivo di guadagno AFE è irraggiungibile. L'obiettivo è un livello della *striscia AFE*, non un bianco di ombreggiatura, quindi non alzarlo per farlo corrispondere alla media di prova di SF di circa 50k |
| Forte dominante verde sul positivo | Il tetto della limitazione del bordo deve essere **per canale**. Un unico percentile congiunto appiattisce il margine a un grigio neutro a un livello che il canale più scuro non raggiunge mai nella pellicola (sessione 004: 27432 congiunto contro i 19306 propri del verde), quindi la Dmin automatica misura il verde sulla cromatura e lo solleva di 1,4x. Controlla `border highlight clamp peak_p99.7=(r,g,b)`. Ogni valore deve stare appena sopra il picco della pellicola di quel canale |
| `white clipped at the rail` | La striscia bianca è bloccata vicino a `0xFFFF`, quindi non ha una forma da uniformare. Abbassa i guadagni AFE. DVDSET resta disattivato e interviene l'allungamento sul computer |
| La scansione non riesce: media del bianco sotto 20000 | La striscia dopo il guadagno unitario è troppo scura per raggiungere l'obiettivo entro il limite di guadagno 4x. Spesso è una striscia AHB vecchia: scuro e AFE attendono che il buffer abbia dati **a riposo**, e un `0xa5` di motore occupato lì non è pronto. Controlla anche che la lampada sia spenta per lo scuro e che la testina sia sulla cromatura di riposo pulita. La cache v6+ ignora una calibrazione degenerata |
| `GL128 motor stayed enabled` mentre la scansione parte | Il motore non si è fermato dopo un avanzamento di posizionamento, quindi il driver non avvia il movimento successivo. Controlla che il supporto si muova liberamente, poi esegui di nuovo la scansione. Il registro riporta ogni avanzamento come `GL128 feed of N steps complete in …s, idle at …s` |
| La scansione non riesce: ombreggiatura ASIC a colori / campo di riposo libero | Il carrello non è sul sensore di riposo libero. Parcheggialo o spegni e riaccendi, poi riprova. La pellicola può restare caricata |
