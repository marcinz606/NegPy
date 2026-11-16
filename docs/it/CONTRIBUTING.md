# Contribuire a NegPy

## Prendere in carico una segnalazione

Chi contribuisce con accesso in sola lettura non può usare l'interfaccia di assegnazione di GitHub, quindi commenta la segnalazione:

- `/assign`: assegna la segnalazione a te
- `/unassign`: toglie te dalla segnalazione

Una reazione 👀 al tuo commento conferma che il comando è stato eseguito.

## Configurazione per lo sviluppo

NegPy richiede Python 3.13+, `uv` e una toolchain Rust.

### 1. Prerequisiti
Installa [uv](https://docs.astral.sh/uv/getting-started/installation/) e [rustup](https://rustup.rs/). `rustup` legge la versione della toolchain da `negpy_core/rust-toolchain.toml`.

La cartella `negpy_core/` è una libreria Rust con i kernel CPU di NegPy. Il primo `uv sync` la compila. Dopo aver modificato un file Rust, il successivo `uv run` o `make test` la ricompila. `negpy_core` non compila codice C o C++: `cargo-deny` (`negpy_core/deny.toml`) rifiuta una dipendenza che lo faccia. I suoi test si confrontano con opencv, che solo il gruppo `dev` installa; `make install` (`uv sync --all-groups`) lo porta.

Supporto per gli scanner (facoltativo):

- **SANE** (Linux/macOS): Coolscan e altri scanner per pellicola SANE. Installa la libreria di sistema, poi il gruppo `sane` (`uv sync --group sane` oppure `pip install negpy[sane]`):
  - **Linux**:
    ```bash
    sudo pacman -S sane  # arch
    sudo apt install libsane-dev  # debian/ubuntu
    ```
  - **macOS**:
    ```bash
    brew install sane-backends
    ```

- **Plustek USB** (Windows/macOS/Linux): il driver [pyopticfilm](https://github.com/jboneng/pyopticfilm) (**1.3.10** o successivo): `uv sync --group plustek` oppure `pip install negpy[plustek]`. Su Windows, pyopticfilm installa `libusb-package`, e lo scanner richiede WinUSB associato con Zadig al posto del driver del produttore o di SilverFast (`07b3:1825` 8200i SE, `07b3:1824` 8100 V2, `07b3:181f` 8300i SE, `07b3:0c3b` 7600i v1 e v2). Vedi [PLUSTEK_WINDOWS.md](PLUSTEK_WINDOWS.md).

- **Nikon Coolscan** (Windows/macOS/Linux): [nkscan](https://github.com/activexray/nkscan) 0.14 o successivo, distribuito come wheel: `uv sync --group nkscan` oppure `pip install negpy[nkscan]`. Su Linux, l'USB richiede una regola udev per il produttore `04b0`, e FireWire/SCSI richiede il modulo `sg`.


### 2. Ambiente Python

```bash
make install
```


### 3. Esecuzione in locale

```bash
make run
```

#### Una cartella utente separata per lo sviluppo

NegPy tiene i database, le cache, le preimpostazioni, i registri e `override.toml` in un'unica
cartella utente, `Documents/NegPy` per impostazione predefinita. Se usi NegPy anche per lavoro vero, dai alle
versioni di sviluppo una cartella propria: imposta `NEGPY_USER_DIR` su un percorso assoluto in un
file facoltativo `.env.local`, ignorato da git, che il `Makefile` legge:

```make
# .env.local
NEGPY_USER_DIR = $(HOME)/negpy-devhome
```

`make run` usa (e crea) poi quella cartella.

Scrivi `$(HOME)`, non `~` né `$HOME`: make lascia `~` così com'è e legge `$HOME` come `$H`
seguito da `OME`. make si ferma quando il valore non è assoluto. Su macOS la tua home è sotto
`/Users/tuonome`, non `/home/tuonome/...`.

Per azzerarla:

```bash
make clear-devhome
```

Questo elimina l'intera cartella dopo una conferma (`FORCE=1` la salta). Si rifiuta di
partire se `NEGPY_USER_DIR` non è impostata o punta alla cartella predefinita. Conserva una copia di qualsiasi
`override.toml` su cui fai affidamento.

Due cose restano fuori dalla cartella utente:

- I file collaterali `.negpy` accanto alle immagini sorgente. La successiva installazione che apre un'immagine
  ne trasferisce il file collaterale nel proprio database. L'esportazione dei file collaterali è disattivata per impostazione predefinita; se
  la attivi, prova su delle copie.
- Le esportazioni.

#### Esecuzione senza hardware di scansione

```bash
make run-sim
```

Avvia il programma con fotocamera, Scanlight e scanner per pellicola simulati
(`NEGPY_SIMULATE_HARDWARE=1`). Solo il modulo gphoto2, la porta seriale e un backend
di scanner sono finti.

- **Fotocamera**: espone fotogrammi da un negativo sintetico sotto il colore dello Scanlight e
  l'otturatore e l'ISO del corpo, salvati come DNG Bayer. Metti la ROI di calibrazione sulla fascia di base trasparente
  attorno all'immagine.
- **Scanlight**: un v4b che risponde all'interrogazione del firmware e riporta la temperatura dei LED.
- **Scanner per pellicola**: il backend "Simulated" ha un alimentatore (fotogrammi per indice, IR, Espelli), un
  dispositivo con prescansione (anteprima a finestra intera, poi ritaglio) e un dispositivo per rullini (fotogrammi trovati sulla
  striscia).

## Struttura del progetto

- `negpy/domain/`: modelli dei dati, tipi e interfacce di base.
- `negpy/features/`: logica di elaborazione delle immagini (Esposizione, Geometria, Lab, ecc.).
- `negpy/infrastructure/`: codice di sistema di basso livello (risorse GPU, caricatori di file).
- `negpy/kernel/`: servizi di sistema di base (Registro, Configurazione, cache).
- `negpy/services/`: orchestrazione di alto livello (motore di resa, servizio di esportazione).
- `negpy/desktop/`: interfaccia PyQt6 (Vista, Controller, Worker).
- `negpy_core/`: libreria Rust per i kernel CPU (`src/<area>.rs`), con i suoi wrapper PyO3 in `src/py/`.
- `tests/`: test unitari e di integrazione.

## Norme di scrittura del codice

Esegui sempre `make format` prima di un commit.

### 1. Stile e formattazione
- Ruff controlla e formatta.
- I suggerimenti di tipo sono obbligatori su ogni nuova definizione di funzione (`ty` è imposto). Non usare `cast` per aggirarlo.
- Docstring chiare e brevi su classi e metodi pubblici.
- Virgolette doppie per le stringhe, snake_case per variabili e funzioni, PascalCase per le classi.

### 2. Test
Le nuove funzioni includono test unitari (`pytest`) in `tests/`.

```bash
make test
```

`make test` salta i test contrassegnati `slow` (vedi `addopts` in `pyproject.toml`), inclusa la suite delle metriche di prestazione in `tests/metrics/`. Per eseguirla e scrivere un file JSON di risultati:

```bash
NEGPY_METRICS_OUT=metrics.json uv run pytest tests/metrics/ -m "slow" -q
```

I file di prova (Canon CR2, Nikon NEF, Sony ARW, Fuji RAF, Leica DNG) si scaricano da rawsamples.ch alla prima esecuzione (circa 20–30 MB ciascuno) e vengono conservati in `~/.cache/negpy-metrics/`. Un test viene saltato se il suo scaricamento non riesce. Per usare un file locale, imposta la variabile d'ambiente del formato:

```bash
NEGPY_PERF_RAW_CR2=/percorso/del/file.CR2 \
NEGPY_METRICS_OUT=metrics.json \
uv run pytest tests/metrics/ -m "slow" -q
```

Variabili disponibili: `NEGPY_PERF_RAW_CR2`, `NEGPY_PERF_RAW_NEF`, `NEGPY_PERF_RAW_ARW`, `NEGPY_PERF_RAW_RAF`, `NEGPY_PERF_RAW_DNG`.

### 3. Flusso di lavoro (il Makefile)
Ogni comando di sviluppo passa per `uv run`:
- `make install`: prepara e sincronizza l'ambiente.
- `make lint`: controlli di Ruff.
- `make type`: controlli dei tipi con `ty`.
- `make test`: test unitari.
- `make format`: formattazione automatica con `cargo fmt` e Ruff.
- `make rust`: `cargo fmt --check`, `cargo clippy` (gli avvisi fanno fallire) e `cargo test` per `negpy_core`.
- `make all`: formattazione, controlli Rust, lint, tipi e test in sequenza.
- `make clean`: rimuove cache e prodotti di compilazione.
- `make run-sim`: esegue con hardware di scansione simulato (vedi [sopra](#esecuzione-senza-hardware-di-scansione)).
- `make clear-devhome`: elimina la cartella utente di sviluppo (vedi [sopra](#una-cartella-utente-separata-per-lo-sviluppo)).


## Traduzioni

Racchiudi il testo visualizzato in `tr("English source", name=value)` e traduci l'intero modello prima di inserire i valori. Usa `mark` per le etichette di catalogo che fungono anche da chiavi di ricerca, poi chiama `tr` quando le visualizzi. Lascia invariati valori enumerativi, chiavi delle impostazioni, percorsi e testi dell'utente. La lingua è una preferenza del programma e ha effetto dopo un riavvio.

Registra le lingue in `LANGUAGES` (`negpy/kernel/system/i18n.py`), usando la locale corrispondente del catalogo Qt, e imposta le loro forme del plurale in `kernel/system/text.py`. `make translations` aggiorna ogni catalogo JSON registrato in `negpy/translations/`. Compila le nuove voci, poi esegui `make translations-check` per verificare copertura e segnaposto di tutti i cataloghi. Le guide dei pannelli localizzate usano gli stessi marcatori `<!-- panel:key -->` di `docs/USER_GUIDE.md`, più `src:<hash>` della sezione inglese che traducono; `tests/test_i18n.py` indica ogni pannello non aggiornato e il suo nuovo marcatore. I pannelli mancanti ricadono sull'inglese. Includi i nuovi file di catalogo e di guida tra le risorse del pacchetto.

Per aggiornare le traduzioni dopo i rilasci o le fusioni dal progetto originale, usa la
[skill update-translations](../../.claude/skills/update-translations/SKILL.md).

## Compilazione e pacchetti

Per compilare l'applicazione autonoma per il tuo sistema operativo con PyInstaller:

```bash
make build
```

Su macOS, `NEGPY_MACOS_ARCH` imposta l'architettura di destinazione del DMG. Per una versione Intel:

```bash
NEGPY_MACOS_ARCH=x86_64 make build
```
