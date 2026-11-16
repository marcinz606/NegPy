# Modelli dei nomi file

Le schede **Esporta** e **Scansione** danno il nome ai file con modelli Jinja2.

## Barra laterale Esporta

### Variabili disponibili

| Variabile | Descrizione | Esempio di risultato |
| :--- | :--- | :--- |
| `{{ original_name }}` | Nome del file sorgente senza estensione. | `DSC0123` |
| `{{ colorspace }}` | Spazio colore di esportazione. | `sRGB`, `Adobe RGB` |
| `{{ format }}` | Formato del file esportato (JPEG, TIFF, …). | `JPEG`, `TIFF` |
| `{{ paper_ratio }}` | Proporzioni selezionate. | `3:2`, `Original` |
| `{{ size }}` | Dimensioni di stampa in cm. Vuota con "Risoluzione originale". | `30cm` |
| `{{ dpi }}` | DPI di esportazione. Vuota con "Risoluzione originale". | `300dpi` |
| `{{ target_px }}` | Lato lungo in pixel. Vuota tranne che in modalità Pixel. | `2048px` |
| `{{ border }}` | "border" se la larghezza è > 0, altrimenti vuota. | `border` |
| `{{ date }}` | Data corrente, AAAAMMGG. | `20260125` |
| `{{ roll }}` | Nome del rullino di acquisizione Scanlight (Metadati → Rullino), oppure ricavato da un nome nella forma `{roll}_Frame{NNN}`. Non è il nome del rullino del riquadro Analisi del rullino. | `Roll001` |
| `{{ frame }}` | Numero del fotogramma di acquisizione (intero), oppure ricavato dal nome. Non impostato (`none`) se sconosciuto. Per aggiungere zeri iniziali usa `{{ frame\|pad(3) }}`. `"%03d" % frame` non funziona su un fotogramma non impostato, e l'intero modello ricade su `original_name`. | `12` |
| `{{ frame_padded }}` | Fotogramma con zeri iniziali (`012`), oppure vuoto se sconosciuto. Equivale a `{{ frame\|pad(3) }}`. | `012` |
| `{{ camera }}` | Marca + modello della fotocamera. | `Mamiya 7` |
| `{{ camera_make }}` / `{{ camera_model }}` | Marca / modello della fotocamera. | `Mamiya`, `7` |
| `{{ lens }}` | Modello dell'obiettivo, oppure la marca se il modello è vuoto. | `80mm f/4` |
| `{{ lens_make }}` / `{{ lens_model }}` | Marca / modello dell'obiettivo. | |
| `{{ focal_length }}` | Lunghezza focale in mm. | `80` |
| `{{ film }}` | Nome della pellicola. | `Ektacolor Pro 400` |
| `{{ film_iso }}` | ISO della pellicola. | `400` |
| `{{ film_manufacturer }}` | Produttore della pellicola. | `Kodak` |
| `{{ film_color_type }}` | Tipo di colore della pellicola. | `Color negative` |
| `{{ film_format }}` | Formato della pellicola (35mm, 120, …). Non è il `{{ format }}` dell'esportazione. | `35mm` |
| `{{ developer }}` | Rivelatore. | `D-76` |
| `{{ dilution }}` | Diluizione del rivelatore. | `1+1` |
| `{{ push_pull }}` | Sovra/sottosviluppo come intero (−3…+3, 0 = Normale). | `1` |
| `{{ development_time }}` | Tempo di sviluppo, mm-ss. | `9-30` |
| `{{ development_temperature }}` | Temperatura di sviluppo in °C. | `20` |
| `{{ scanning }}` | Nota sul metodo di scansione. | `DSLR copy-stand` |
| `{{ exposure }}` | Testo di esposizione impostato in Metadati. | `1/125s f/2.8` |
| `{{ capture_date }}` | Data di ripresa originale, AAAAMMGG. Una data parziale viene completata con il primo giorno. Vuota se non impostata. | `19980714` |
| `{{ capture_year }}` | Anno di ripresa originale. Vuota se non impostata. | `1998` |

I valori di attrezzatura e processo provengono dal pannello **Metadati**, oppure dai metadati salvati di ogni file in un lotto. NegPy ne toglie i caratteri non sicuri per un percorso.

### Esempi

| Modello | Risultato |
| :--- | :--- |
| `{{ original_name }}` | `DSC0123.jpg` |
| `{{ date }}_{{ original_name }}_{{ colorspace }}` | `20260125_DSC0123_Adobe_RGB.jpg` |
| `{{ original_name }}_{{ size }}_{{ dpi }}_{{ border }}` | `DSC0123_30cm_300dpi_border.jpg` |
| `PRINT_{{ original_name }}_{{ paper_ratio }}` | `PRINT_DSC0123_3:2.jpg` |
| `{{ roll }}_Frame{{ frame\|pad(3) }}_{{ film }}_{{ film_iso }}` | `Roll001_Frame012_Portra_400_400.jpg` |
| `{{ film }}_{{ camera }}_{{ original_name }}` | `Portra_400_Mamiya_7_DSC0123.jpg` |

## Barra laterale Scansione

### Variabili disponibili

| Variabile | Descrizione | Esempio di risultato |
| :--- | :--- | :--- |
| `{{ date }}` | Data corrente, AAAAMMGG. | `20260125` |
| `{{ seq }}` | Numero di sequenza (intero). | `1`, `2`, … |

Per aggiungere zeri iniziali usa l'operatore `%` di Python: `{{ "%03d" % seq }}`.

### Esempi

| Modello | Risultato |
| :--- | :--- |
| `{{ date }}_{{ "%03d" % seq }}` | `20260125_001.tif` |
| `roll_{{ date }}_{{ seq }}` | `roll_20260125_1.tif` |
| `plustek_{{ date }}_{{ "%04d" % seq }}` | `plustek_20260125_0001.tif` |

### Incremento automatico

La sequenza parte da `1` a ogni sessione di scansione e aumenta finché il nome file non è libero, così NegPy non sovrascrive mai un file.

## Pulizia del nome file

Entrambe le schede ripuliscono i separatori:

- Spazi, trattini e trattini bassi tra le variabili si riducono a un **unico trattino basso** (`_`).
- I separatori all'inizio e alla fine vengono rimossi.
- I separatori attorno a una variabile vuota (`{{ border }}` senza bordo impostato) vengono rimossi.
- `{{ original_name }}` (solo esportazione) viene inserito alla lettera, separatori compresi.

Modello di esempio: `{{ original_name }} - {{ border }} - final`

- Con bordo: `DSC0123_border_final.jpg`
- Senza bordo: `DSC0123_final.jpg`
