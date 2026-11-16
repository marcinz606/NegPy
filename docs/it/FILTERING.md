# Filtrare la Striscia pellicola

La casella di ricerca nella riga Rullino della barra degli strumenti filtra la Striscia pellicola per nome file, oppure con termini come `film:portra iso:>=400` (vedi la [Guida utente](../../negpy/translations/USER_GUIDE.it.md), §2). L'ordinamento, la modalità Cartella monitorata ed "Esporta rullino" usano solo l'insieme visibile (filtrato).

## Modalità semplice (predefinita)

NegPy confronta il testo con il nome di ogni file, ignorando maiuscole e minuscole.

| Testo | Corrisponde a |
|-------|---------|
| `IMG` | `IMG_0001.cr2`, `img_test.NEF` |
| `.cr2` | ogni file `.cr2` |
| `_42` | i file che contengono `_42` in qualsiasi punto del nome |

Fai clic sulla **×** nella casella per cancellare il filtro.

## Modalità espressioni regolari

Fai clic sull'interruttore **`.*`** dentro la casella di ricerca. NegPy compila il modello con `re.IGNORECASE` e lo confronta con `re.search`. Usa `^` e `$` per far corrispondere il nome intero.

| Modello | Corrisponde a |
|---------|---------|
| `^IMG_\d{4}` | `IMG_0001.cr2`, `IMG_0042.NEF` |
| `\.(cr2\|nef)$` | solo i file `.cr2` o `.nef` |
| `roll_\d+_scan` | file come `roll_3_scan.tif` |

Un'espressione regolare non valida (per esempio una `[` non chiusa) colora di rosso il bordo della casella e mantiene il filtro precedente.

## Comportamento

- NegPy toglie i file nascosti dall'insieme della selezione multipla. Il file attivo passa alla prima selezione visibile, oppure viene azzerato se nulla corrisponde. Un'azione sulla selezione, come Applica o Incolla impostazioni, non modifica mai un file nascosto.
- Esporta rullino scrive solo i file visibili.
- La modalità Cartella monitorata acquisisce comunque i nuovi file. Un nuovo file che non corrisponde al filtro resta nascosto finché non cancelli o allarghi il filtro.
- Il filtro si applica dopo l'ordinamento, quindi l'ordine visibile segue la tua scelta di Nome, Data o Scena e Crescente/Decrescente.
- Il filtro dura per una sessione.
