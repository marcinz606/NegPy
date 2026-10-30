# NegPy-Benutzerhandbuch

NegPy ist eine virtuelle Dunkelkammer, die Filmscans zu fertigen Bildern macht. Deine Originaldateien bleiben unverändert. NegPy speichert deine Bearbeitungen in einer lokalen Datenbank. Wähle die Sprache unter Einstellungen → Oberfläche → Sprache und starte NegPy neu. Die Option „Systemstandard“ folgt der Systemsprache und nutzt Englisch, wenn NegPy diese Sprache nicht kennt. Fehlt eine Übersetzung für einen Bereich, zeigt NegPy den englischen Text.

<!-- panel:library src:35989d38 -->
### Bibliothek

Der Tab **Bibliothek** (`Ctrl+L`) listet deine **Rollen**. Eine Rolle ist eine benannte Gruppe von Bildern, meist ein Film: eine **Ordnerrolle** oder eine **virtuelle Rolle**. Beim Import merkt sich NegPy den Ordner. NegPy verschiebt, benennt oder löscht deine Dateien nie.

*   **+**: **Ordner als Rolle importieren…** macht aus einem Ordner eine Rolle und öffnet sie. **Unterordner als Rollen importieren…** macht aus jedem Unterordner mit Bildern eine Rolle.
*   **Doppelklicke** auf eine Rolle (oder drücke Enter), um sie im Filmstreifen zu öffnen.
*   **⋮** oder Rechtsklick auf eine Rolle: **Öffnen**, **Alle entladen…** (die geöffnete Rolle), **Rollenanalyse**, **Umbenennen…**, **In Ordnerrolle speichern**, **Entfernte Bilder wiederherstellen…** und **Löschen…**. Entfernte Bilder wiederherstellen holt die Bilder zurück, die Aus Rolle entfernen aus einer Ordnerrolle genommen hat. Die Zeile unter dem Namen der Rolle zählt sie. Löschen behält die Dateien und ihre gemeinsamen Bearbeitungen. Es verliert die Einstellungen, die Basislinie, die Filmbasis, die Szenen, die Kartensperren und die nur in dieser Rolle gemachten Bearbeitungen der Rolle. Aktualisieren fügt eine gelöschte Ordnerrolle nicht wieder hinzu.
*   **Rollen filtern…** und **Sortieren** helfen dir beim Finden einer Rolle. **⋮ → Aktualisieren** findet neue Unterordner und zählt die Bilder neu. **⋮ → Suchfilter…** listet Ordnernamen, die übersprungen werden (zuerst `export`).

Eine Rolle, deren Ordner fehlt, zeigt **Ordner fehlt**. Passt ein Ordnername zu deiner Ausrüstung ("penf" für "Pen F"), öffnen sich die Rolleneinstellungen schon ausgefüllt.

Eine Ordnerrolle enthält nur die Dateien ihres Ordners. Eine Datei, die du aus einem anderen Ordner hinzufügst, bleibt nur für diese Sitzung.

Um die Bilder im Filmstreifen als Rolle zu speichern, nutze den Filmstreifen. **In Ordnerrolle speichern** macht aus Bildern, die alle aus einem Ordner stammen, die Rolle dieses Ordners. Seine anderen Dateien erscheinen als entfernt. **Als virtuelle Rolle speichern…** behält genau diese Bilder, aus beliebigen Ordnern. In Ordnerrolle speichern macht auch aus einer virtuellen Rolle eines Ordners die Rolle dieses Ordners. Die neue Rolle startet mit den Einstellungen, den Kartensperren und den nur in dieser Rolle gemachten Bearbeitungen der geöffneten Rolle. Ein Bild hat in jeder Rolle, die es enthält, dieselbe Bearbeitung. Rechtsklick → **In dieser Rolle unabhängig bearbeiten** gibt ihm in einer Rolle eine eigene Bearbeitung.

<!-- panel:frames src:b724d5df -->
### Filmstreifen

Der **Filmstreifen** unter der Arbeitsfläche zeigt die geöffneten Bilder in einer Reihe. Ziehe seinen oberen Rand, um die Bilder zu vergrößern.

Die Zeile **Rolle** der Werkzeugleiste enthält:

*   Den Namen der Rolle und die Anzahl der Bilder, der behaltenen und der verworfenen Bilder.
*   Das **Filterfeld**: Gib einen Teil eines Dateinamens ein oder Begriffe wie `film:portra iso:>=400` (siehe unten). Die Suchschaltfläche oder **Enter** durchsucht die ganze Bibliothek.
*   **Sortieren**: nach Name, Datum oder Szene.
*   **Bogenfilter**: alle Bilder, nur behaltene, verworfene ausblenden oder nur unmarkierte. Die Schaltfläche ist hervorgehoben, solange ein Filter Bilder ausblendet.
*   **Rolleneinstellungen** (das Etikett): Ausrüstung und Filmdetails für dieses Bild, die Auswahl oder die Rolle.
*   **Anwenden**: kopiert gewählte Einstellungen dieses Bildes auf die Auswahl oder die Rolle.
*   **⋮**: **Dateien hinzufügen…**, **Ordner hinzufügen…**, **In Ordnerrolle speichern**, **Als virtuelle Rolle speichern…**, **Vorschaubilder aktualisieren**, **Hot Folder** (lädt neue Dateien, die ein Scanner im Ordner des neuesten Bildes speichert), **Entladen…**, **Alle entladen…** (leert den Filmstreifen; die Rolle bleibt in der Bibliothek), **Rolle auf Standard zurücksetzen…**, **Entfernte Bilder wiederherstellen…** und **Hilfe zum Filmstreifen…**.

Die Raster-Schaltfläche öffnet den **Leuchttisch** (`Shift+G`): die ganze Rolle in Zeilen, zum Sortieren. Ein Klick wählt ein Bild, ein Doppelklick oder `Enter` öffnet es. Die Werkzeugleiste wirkt auf die gewählten Bilder: **Behalten**, **Verwerfen**, Drehen und Spiegeln, **Auto-Zuschnitt** und **Export**. ⋯ enthält den Rest des Rechtsklick-Menüs. Der Regler **Bilder pro Zeile**, `+` und `-` oder `Ctrl` mit dem Mausrad ändern die Bildgröße. `Esc` geht zurück.

#### Filtern

Ein einfaches Wort passt auf den Dateinamen. Begriffe der Form `field:value` passen auf Bilddaten, und ein vorangestelltes `-` schließt aus:

| Begriff | Findet |
|---|---|
| `film:portra` | Filmsorte, die "portra" enthält |
| `camera:"Nikon F3"` | Anführungszeichen für einen Wert mit Leerzeichen |
| `iso:>=400` | Zahlen nehmen auch `>`, `>=`, `<`, `<=` |
| `date:2024-03` · `shot:>=1998` | Dateidatum · Aufnahmedatum |
| `place:tokyo` · `lens:` · `roll:` | andere Metadatenfelder |
| `scene:beach` | Bilder in einer Szene |
| `keeper:` · `rejected:` · `edited:` | markierte oder bearbeitete Bilder |

Begriffe lassen sich kombinieren: `film:portra iso:>=400 -rejected:`. Metadaten-Begriffe passen nur auf Bilder, deren Metadaten du ausgefüllt hast. **Suche nach Bedeutung** (aktiviere sie unter Einstellungen → Leistung) findet Bilder über eine Beschreibung, zum Beispiel "ein Hund am Strand".

<!-- panel:analysis src:b296e118 -->
## 3. Analyse-Anzeige

Die Anzeige über den Tabs Rolle und Bild beschreibt das aktuelle Bild, während du bearbeitest.

#### Diagramm

Die Schaltflächen über dem Diagramm wählen eine Ansicht:

*   **Kurve**: die Kurve des Papiers, vom Negativ (links nach rechts) zum Abzug (unten nach oben). Steiler bedeutet mehr Kontrast. Das graue Histogramm zeigt das Negativ, das farbige den Abzug.
*   **Waveform**: Helligkeit des Abzugs über die Bildbreite.
*   **Parade**: Rot, Grün und Blau nebeneinander. Ein graues Motiv liegt in allen drei auf einer Höhe. **Parade-Überlagerung** legt die drei in einem Diagramm übereinander.
*   **Vektorskop**: die Farben des Abzugs rund um einen Mittelpunkt. Eine Gruppe neben der Mitte ist ein Farbstich.

**Log** zeigt kleine Bereiche, zum Beispiel beschnittene Enden.

#### Beschneidung

Dreiecke in den oberen Ecken warnen, wenn Schatten reines Schwarz (links) oder Lichter reines Weiß (rechts) erreichen. **Beschneidungs-Überlagerung** (die Halbkreis-Schaltfläche) markiert diese Stellen im Bild. **Zonenhervorhebung** (die Marker-Schaltfläche) zeigt nur die Zone unter dem Cursor im Bild.

#### Zonen und Zonenplatzierung

Der **Stufenkeil** druckt 21 Graustufen. Stufen, die in Schwarz oder Weiß verschmelzen, sind verlorene Tonwerte. Der **Zonenstreifen** zeigt, wie viel vom Bild in jede Zone fällt, von 0 (Schwarz) über V (mittleres Grau) bis X (Weiß). Doppelklicke darauf (oder drücke `Shift+Z`) für die **Zonen-Überlagerung**, die das Bild nach Zonen einfärbt.

Klicke auf eine Zone im Streifen, dann auf eine Stelle im Bild. NegPy stellt den Abzug so ein, dass diese Stelle in dieser Zone abgezogen wird. Ein Pin setzt die Abzugsdichte, ein zweiter auch die ISO-R-Gradation. **Zonen platzieren** (oder Enter) übernimmt das Ergebnis, `Esc` bricht ab.

#### Messwerte

Die Falte **Messwerte** enthält Zahlen zum Scan. Ein bernsteinfarbener Wert ist eine Warnung. Bewege den Mauszeiger über eine Zeile für Details.

*   **Negativ**: der Dichteumfang des Negativs (Δ). Der Mauszeiger zeigt flach, normal oder kontrastreich.
*   **Belichtung**: helles (positiv) oder dunkles (negativ) Bild, in Blenden.
*   **Beschneidung**: Anteil des Abzugs in reinem Schwarz, dann in reinem Weiß.
*   **Scan-Beschneidung**: Anteil jedes Kanals, Rot, Grün und Blau, am Maximum des Scanners. Keine Bearbeitung behebt das. Belichte den Scan niedriger.
*   **Reparatur**: Anteil des Bildes, den die Staubentfernung neu geschrieben hat.
*   **Gamut**: Anteil, den der Proof-Drucker nicht drucken kann, solange du einen Softproof nutzt.

<!-- panel:autocrop src:3047f00d -->
### 4.1 Zuschnitt

Wähle den Teil des Bildes, der bleibt, von Hand oder mit dem Auto-Zuschnitt.

*   **Zeichnen**: Ziehe einen Zuschnitt auf das Bild. Ziehe an einer Ecke, um die Größe zu ändern, im Inneren, um ihn zu verschieben, oder außerhalb des Rahmens, um ihn zu drehen. `Enter` oder **Fertig** behält den Zuschnitt, `Esc` stellt den vorherigen Zuschnitt wieder her. Das **×** entfernt einen gezeichneten Zuschnitt. Beim Zeichnen wird der Auto-Zuschnitt ausgeschaltet.
*   **Auto-Zuschnitt** (auch in der Leiste des Tabs Geometrie): findet die Ränder des Bildes und schneidet darauf zu.
*   **Verhältnis**: die Form des Zuschnitts, zum Beispiel `3:2` oder `1:1`, passend zum Bild gedreht. **Frei** nutzt die Form, die NegPy findet, und du kannst jeweils eine Seite ziehen. **Erkennen** wählt die nächste Standardform.
*   **Hilfslinien**: Kompositionslinien, zum Beispiel Drittel, während du zeichnest. Die Dreh-Schaltfläche dreht sie.
*   **Modus** (Falte Auto-Zuschnitt-Einstellungen): **Nur Bild** schneidet auf das Bild zu. **Filmrand** behält den ganzen Film mit Rand und Perforation.
*   **Zuschnitt-Versatz** und **Randbeschnitt** (Falte unter Modus): verschieben den Zuschnitt nach innen (positiv) oder außen (negativ) gegenüber dem gefundenen Rand und legen fest, wie weit er in den Filmrand schneidet (Nur Bild). 0% hält am Filmrand an, 100% am Bildrand. Über 100% schneidet ins Bild, um einen weißen Rand zu entfernen.
*   **Erst ausrichten**: richtet das Bild an seinen Filmkanten aus, bevor der Auto-Zuschnitt es zuschneidet.
*   **Alle Bilder zuschneiden** (Nur Bild): schneidet alle Bilder zusammen zu. Klare Bilder helfen unklaren, und gezeichnete Zuschnitte bleiben.

Der Auto-Zuschnitt braucht die Scannerlichtquelle oder den Halter als hellsten Teil des Scans. Er lässt ein Bild unverändert, wenn das Bild so hell ist wie die Lichtquelle oder das nächste Bild einen großen Teil einer Seite füllt. Schneide ein solches Bild von Hand zu oder nutze **Filmrand** und schneide nach innen.

<!-- panel:geometry src:913e25e7 -->
### 4.2 Geraderichten

Drehe das Bild und richte es gerade aus.

*   **Drehen** und **Spiegeln**: drehen das Bild um eine Vierteldrehung oder spiegeln es. Die Werkzeugleiste hat dieselben Schaltflächen.
*   **Ausrichten**: Zeichne eine Linie entlang des Horizonts oder einer senkrechten Kante. Das Bild dreht sich, bis die Linie gerade ist.
*   **An Film ausrichten** (auch in der Leiste des Tabs Geometrie): richtet das Bild an den Kanten des Films aus. Der Bildinhalt zählt dabei nicht. Ein schief fotografiertes Bild bleibt also schief.
*   **Feindrehung**: dreht das Bild in kleinen Schritten. Positiv ist im Uhrzeigersinn.
*   **Neigung** und **Schwenkung** (Falte): richten ein Gebäude aus, das von unten fotografiert wurde (Neigung), oder eine Wand, die von der Seite fotografiert wurde (Schwenkung). **Linien zeichnen** setzt beide aus Linien entlang der vier Filmkanten.
*   **Leere Ecken zuschneiden**: schneidet die leeren Ecken ab, die Drehung, Neigung und Schwenkung hinterlassen.

Schneide vor einer starken Neigung oder Schwenkung zu. Sonst kann der Filmrand ins Bild kommen und den Abzug abdunkeln. Objektivverzeichnung korrigierst du unter **Optik** ([§10.3](#103-optics)).

<!-- panel:color src:2abd7418 -->
### 5.1 Filterung: Farbabgleich

Korrigiert die Farbe des Abzugs, wie die Farbfilter eines Vergrößerers in der Dunkelkammer. **Global / Schatten / Lichter** wendet die Farbregler auf das ganze Bild an oder vor allem auf seine dunklen oder hellen Teile.

*   **WB wählen**: Klicke im Bild auf etwas, das grau sein soll. NegPy korrigiert die Farbe so, dass es grau wird.
*   **Temperatur**: macht das Bild wärmer oder kühler.
*   **Cyan / Magenta / Gelb**: verschieben die Farbe zwischen Cyan und Rot, Magenta und Grün oder Gelb und Blau.
*   **Ring-around** (`Shift+F`): zeigt das Bild als Raster leicht unterschiedlicher Farben. Klicke auf die Version, die richtig aussieht. `Esc` schließt es.
*   **Farbstichentfernung**: entfernt einen Farbstich, der in den dunklen und hellen Teilen des Bildes unterschiedlich ist. Der Start liegt bei 0,5 (halb) bei Farbnegativen und bei 0 bei Dias. Blätter, Himmel und Wasser gelten nicht als grau und behalten ihre Farbe.
*   **Filter**: ein Farbfilter, wie er bei Schwarzweißfilm vor dem Kameraobjektiv sitzt. Er erscheint nur bei einem Farbnegativ, das in S/W abgezogen wird, und ersetzt die Regler darüber. Jeder Filter dunkelt die Farben ab, die seiner eigenen gegenüberliegen. Gelb #8, Tiefgelb #15, Orange #21 und Rot #25 machen einen blauen Himmel Schritt für Schritt dunkler. Rot macht außerdem Blätter dunkel und Haut blass. Gelbgrün #11 und Grün #58 machen Blätter heller und Haut dunkler. Keiner gibt den Look eines normalen Schwarzweißfilms.

<!-- panel:tone src:c1700dd9 -->
### 5.2 Tonwerte: Helligkeit und Kontrast

Wie hell oder dunkel der Abzug ist und wie viel Kontrast er hat.

*   **Abzugsdichte**: wie hell oder dunkel der Abzug ist (die Belichtungszeit in der Dunkelkammer). Niedriger ist heller.
*   **ISO-R-Gradation**: Kontrast (die Papiergradation in der Dunkelkammer). Eine niedrigere Zahl gibt mehr Kontrast. Etwa 110 ist normal.
*   **Auto** (der Zauberstab, auch in der Leiste des Tabs Belichtung): **Auto-Dichte** und **Auto-Gradation** stellen Helligkeit und Kontrast für jedes Bild ein, damit eine Rolle gleichmäßig abgezogen wird. Ein Regler zeigt dann den automatischen Wert: Bewege ihn zum Anpassen, doppelklicke darauf für den Rückweg. Ein Rechtsklick auf den Zauberstab schaltet beide ein oder aus. Das Auto-Menü enthält außerdem:
    *   **Zielwerte festlegen…**: die Helligkeit und den Kontrast, die Auto anstrebt.
    *   **Auto-Dichte/Gradation fixieren**: behält die automatischen Werte dieses Bildes und schaltet Auto dafür aus.
    *   **Schwarz halten**: Wenn du die Schatten aufhellst, bleiben die tiefsten Schwarztöne schwarz.
*   **Probestreifen** (`Shift+T`): ein Raster von Versionen mit unterschiedlicher Helligkeit und unterschiedlichem Kontrast. Klicke auf die, die dir gefällt. `Esc` schließt es.
*   **Schattendichte** / **Lichterdichte** (Falte unter Abzugsdichte): dunkeln nur die Schatten oder nur die Lichter ab oder hellen sie auf.
*   **Schattengradation** / **Lichtergradation** (Falte unter ISO-R-Gradation): Kontrast nur in den Schatten oder nur in den Lichtern.
*   **Druckhilfen** (Falte):
    *   **Vorbelichtung**: bringt Details in den hellsten Bereichen zurück.
    *   **Diffusion**: ein weiches Glühen. Dunkle Bereiche breiten sich in helle aus.
    *   **Kontrastmaske**: öffnet große dunkle Bereiche, hält große helle zurück und erhält feine Details. In der Dunkelkammer entsteht dieser Effekt mit einer weichen, unscharfen Maske auf dem Negativ. Beim digitalen Abzug erreichen Fujifilm-Frontier-Scanner mit HyperTone einen ähnlichen Effekt. Bereiche weit vom mittleren Grau ändern sich am meisten, daher ändert sich ein kontrastarmes Bild wenig.
*   **Global / R / G / B**: R, G oder B wendet die Gradationen nur auf eine Farbe an, für einen Farbstich, der in Schatten und Lichtern unterschiedlich ist.

**Weißpunkt** und **Schwarzpunkt** findest du auf der Karte **Messung** im Tab Rolle ([§10.5](#105-metering-negative--positive)).

<!-- panel:paper src:00cfca5d -->
### 5.3 Papierverhalten: das Papier

Das Papier, auf das das Bild abgezogen wird, und wie es Licht in Tonwerte von Schwarz bis Weiß umsetzt.

*   **Papier**: das Abzugspapier. Ein echtes Papier legt Kontrastkurve und Farben nach den Daten seines Herstellers fest. **Digital (Adobe RGB)**, der Standard, ist ein ideales Papier. **Digital (Pakon)** hat die Farben eines Pakon-Laborscanners. **Digital (Frontier)** hat die Farben und die Kontrastkurve eines Fujifilm-Frontier-Laborscans.
*   **Papier-Probestreifen** (die gestreifte Schaltfläche): zieht das Bild auf jedem Papier ab. Klicke auf ein Papier für die große Ansicht, wechsle mit den Pfeiltasten und drücke **Behalten**, um es zu nutzen.
*   **Farbstofftrennung**: Farbstärke im Abzug. 1 ist unverändert. **Trennungsdämpfung** legt mehr von der Änderung auf matte Farben und weniger auf kräftige. **Chroma** im Tab Look ändert stattdessen alle Farben gleichmäßig.
*   **Papierweiß**: zeigt das Weiß von echtem Papier, etwas dunkler als reines Weiß.
*   **Papierschwarz**: zeigt das Schwarz von echtem Papier, etwas heller als reines Schwarz.
*   **Kurvenform** (Falte):
    *   **Snap**: Kontrast der Mitteltöne. Schwarz und Weiß bewegen sich nicht.
    *   **Fuß**: formt die Schatten. Positiv hebt sie an, negativ macht sie tiefer. **Fußbreite** legt fest, wie weit er in die Mitteltöne reicht.
    *   **Schulter**: formt die Lichter. Positiv macht sie weicher, negativ heller, wobei Details verloren gehen können. **Schulterbreite** legt fest, wie weit sie reicht.
*   **Global / R / G / B**: R, G oder B wendet die Kurvenregler auf eine Farbschicht an.

<!-- panel:local src:7d85677b -->
### 5.4 Abwedeln & Nachbelichten: Bereiche aufhellen oder abdunkeln

Zeichne eine Form. Nur dieser Bereich wird dann heller (Abwedeln) oder dunkler (Nachbelichten). Ein Dia nutzt diese Karte nicht, aber seine Masken bleiben erhalten.

*   **Zeichnen**: Klicke, um Punkte zu setzen. Schließe die Form mit Doppelklick, `Enter` oder einem Klick nahe am Startpunkt. Ziehe einen Punkt, um ihn zu verschieben. Klicke auf ein "+" an einer Kante, um einen Punkt hinzuzufügen. Klicke mit der rechten Maustaste auf einen Punkt, um ihn zu löschen.
*   **Oval**: Ziehe ein Oval auf. Mit seinen Griffen verschiebst, streckst und drehst du es.
*   **Abdeckkarte**: eine Nachbelichtung, die ausläuft, wie eine Karte, die über das Papier bewegt wird. Ziehe von der Kante mit voller Stärke (durchgezogene Linie) bis dorthin, wo sie endet (gestrichelte Linie).
*   **Maskenliste**: Klicke auf eine Maske, um sie auszuwählen. Mit den Symbolen schaltest du sie ein oder aus, kehrst sie um (wirkt überall außer innerhalb), zeigst ihren Umriss oder löschst sie. Die Regler darunter ändern die ausgewählte Maske.
*   **Nachbelichten**: Positiv dunkelt den Bereich ab, negativ hellt ihn auf. Die Einheit sind Blenden: +1 verdoppelt das Licht.
*   **Kantenweichheit**: wie weich die Kante ist.
*   **Gradation**: Kontrast innerhalb der Maske. Negativ gibt mehr Kontrast.
*   **Vorbelichtung**: holt Zeichnung in hellen Bereichen innerhalb der Maske zurück, wie ein Preflash.
*   **Tonwertgrenze**: **Lichter** oder **Schatten** lässt die Maske nur auf Töne wirken, die heller oder dunkler sind als der **Tonwertbereich** (0 ist Schwarz, V mittleres Grau, X Weiß). Eine Himmelsnachbelichtung endet dann an der Horizontlinie. **Tonwertübergang** legt fest, wie allmählich sie einsetzt.

**Abzugsnotizen** (`Shift+N`) markieren die Masken mit ihren Nummern und Werten im Bild, wie Notizen eines Druckers im Labor.

<!-- panel:lab src:e8e4a89c -->
### 6.1 Lab: Farbe und Detail

Die Anpassungen, die ein Lab-Scanner nach dem Abzug vornimmt.

*   **Chroma**: Farbstärke. 1 ist unverändert, 0 ist Schwarzweiß.
*   **Hautschutz**: verhindert, dass Hauttöne zu farbstark werden.
*   **Chroma-Entrauschen**: entfernt farbige Punkte. Das Filmkorn bleibt. Bei Farbfilm ist es auf 2,5 eingeschaltet.
*   **Schärfen**: schärft Details ohne farbige Kanten. Die Falte enthält **Methode** (Unsharp Mask macht Kanten klarer, Dekonvolution macht die Unschärfe des Scanners rückgängig), **Radius** (die Größe des Details in exportierten Pixeln; beurteile ihn in der **Exportgröße** oder mit der **Kornschärfe-Lupe**) und **Maskierung** (schärft nur Kanten, damit Himmel und Haut glatt bleiben).
*   **Auto-Maskierung** (der Zauberstab neben Methode, auch in der Kopfzeile des Tabs): stellt die Maskierung aus dem Korn jedes Bildes ein. Die Maskierung zeigt den verwendeten Wert, und ein Ziehen verschiebt ihn von dort nach oben oder unten. Ein Export misst das Korn in seiner eigenen Größe neu.
*   **CLAHE**: lokaler Kontrast, um Details hervorzuheben. Hohe Werte können unnatürlich wirken.
*   **Glühen** (Falte Effekte): ein weiches Glühen um helle Bereiche, wie der Weichzeichner eines Minilab-Scanners.
*   **Lichthof** (Falte Effekte): ein rotes Glühen um helle Bereiche, wie es manche Filme zeigen.

<!-- panel:altproc src:1b810792 -->
### 6.2 Alternative Verfahren

Historische Abzugsverfahren für S/W-Negative. Wähle **Ohne**, **Lith**, **Cyanotypie** oder **Sabattier-Effekt**. Es erscheinen nur die Regler dieses Verfahrens.

#### Lith

Cremige, warme Lichter und harte, tiefe Schwärzen. Das Papier unter Papierverhalten bestimmt die Farbe. Auf einem Lith-Abzug wirken nur die Toner Selen und Gold.

*   **Belichtung**: mehr gibt wärmere, farbigere Lichter und weichere Töne.
*   **Abbruchpunkt**: wann der Abzug den Entwickler verlässt. Höher gibt tiefere, kältere Schwärzen.
*   **Übergangshärte**: wie plötzlich die Schatten schwarz werden.

#### Cyanotypie

Ein blauer Abzug mit grünen Lichtern. Toner wirken hier nicht; nutze Bleichen und Tannin.

*   **Sensibilisator**: **Klassisch** gibt ein helles Blau mit grünen Lichtern. **Neu** gibt tiefere, sauberere Blautöne.
*   **Belichtung**: mehr macht den Abzug blauer.
*   **Belichtungsumfang**: Kontrast. Ein kürzerer Umfang gibt mehr Kontrast.
*   **Bleichen**: entfernt Blau, zuerst in den Lichtern.
*   **Tannin**: färbt gebleichte Bereiche braun. Bleiche zuerst für ein volles Braun.

#### Sabattier-Effekt

Ein Lichtblitz während der Entwicklung. Die hellen Töne kehren sich zu Grau um, und entlang der Kanten zwischen Dunkel und Hell erscheinen dünne helle Linien (Mackie-Linien).

*   **Stärke**: die Dauer des Blitzes. Über 1 kehren sich die hellen Töne um.
*   **Zweitbelichtung**: wo die Umkehrung beginnt. Dunklere Töne bleiben, wie sie sind.
*   **Bewegung**: weniger Bewegung gibt breitere Mackie-Linien.

<!-- panel:toning src:8fc7301e -->
### 6.3 Tonung

Toner sind chemische Bäder, die die Farbe eines S/W-Abzugs ändern. Die Splittonung gibt jedem Abzug Farbe.

**Chemische Tonung** (S/W-Negative) wirkt in der gezeigten Reihenfolge:

*   **Selen**: tiefere, kühlere Schwärzen.
*   **Sepia**: warmes Braun, von den Lichtern abwärts.
*   **Gold**: Blauschwarz. Über Sepia gibt es orangerote Lichter.
*   **Eisenblau**: blaue Schatten und marineblaue Schwärzen.
*   **Kupfer**: Rosa bis Ziegelrot. Die Schwärzen werden schwächer.
*   **Vanadium**: grüne Mitteltöne und Lichter. Die Schwärzen bleiben schwarz.

Auf einem Lith-Abzug wirken nur Selen und Gold. Auf einer Cyanotypie wirkt kein Toner.

**Splittonung** (alle Abzüge): **Schattenstärke** und **Lichterstärke** geben den Schatten und den Lichtern Farbe. **Schattenfarbton** und **Lichterfarbton** wählen die Farbe und erscheinen, sobald die Stärke über 0 liegt.

<!-- panel:leak src:d58b56d6 -->
### 7.1 Lichteinfall

Mindert den Schleier, den ein Lichteinfall einem Negativ oder Dia hinzufügt, vor jeder Staubreparatur und nur unter den Lichteinfällen, die du setzt. Ausgebrannter Film bleibt: Repariere oder klone ihn. Nicht verfügbar für Positiv-Bilder und Komposite.

*   **Lichteinfall mindern** schaltet die Korrektur ein. **Auto** (Zauberstab) setzt eine Form auf jeden Lichteinfall, den der Film belegt: eine Kante, wo belichteter Filmrand ins Bild weiterläuft, einen Streifen entlang eines geraden Lichteinfalls. Ein Schleier, der an der Bildgrenze endet, ist Randschleier oder Randaufdruck, und Auto lässt ihn.
*   **Von der Rolle** (Filmsymbol): Sind andere Bilder ausgewählt, überträgt es die Lichteinfälle dieses Bildes auf jedes davon, und jedes Bild liest Stärke und Farbe selbst. Ohne Auswahl wählt es die Bilder aus, auf denen Auto einen Lichteinfall findet.
*   **Film-Dmax** (Aus, oder 0,05 bis 4,0 D): die maximale Dichte des Films, für einen Film, dessen dichtes Ende gesättigt ist.
*   **Schichten wiederherstellen**: baut eine Farbschicht, die der Lichteinfall bis zur höchsten Dichte getrieben hat, aus den Schichten wieder auf, die das Bild noch enthalten.
*   **Korn** und **Kontrast** wirken nur, wo Lichteinfall entfernt wurde: Korn beruhigt gestrecktes Korn, Kontrast stellt abgeflachten Kontrast wieder her.
*   **Streifen**: Ziehe entlang der stärksten Linie des Lichteinfalls und ziehe dann jede gestrichelte Kante dorthin, wo er ausläuft.
*   **Kante**: Ziehe entlang der Seite, von der der Lichteinfall kommt. Die Linie rastet dort ein. Nimmt der Beschnitt den belichteten Spalt zwischen den Bildern auf, entfernt die Kante ihn entlang der ganzen Seite.
*   **Fleck**: Ziehe ein Oval über einen runden Lichteinfall.
*   **Malen**: Male über einen Lichteinfall ohne Linie oder Mitte, breiter als der Lichteinfall.
*   **Ausgewählter Lichteinfall**: **Stärke** (0 bis 400%) legt fest, wie viel des Lichteinfalls entfernt wird, **Auslauf**, wie ein Streifen, Fleck oder gemalter Bereich zum Rand hin ausläuft, **Farbe**, woher seine Farbe kommt (**Auto**, **Rückseite** für Licht durch den Träger, **Vorderseite** für Licht auf der Schichtseite), und **Farbabgleich** nimmt mehr oder weniger von seiner Hauptfarbe heraus. Farbe und Farbabgleich sind bei S/W-Film ausgeblendet.
*   **Ausgebranntes zeigen** (Flamme) markiert Film, den der Lichteinfall in jeder Schicht ausgebrannt hat.

<!-- panel:retouch src:8d07a6c2 -->
### 7.2 Retusche: Staub, Haare, Kratzer

Entfernt Staub, Haare und Kratzer. Jede Reparatur wird aus dem sauberen Film darum herum aufgebaut.

*   **Überlagerung**: **Markiert** zeigt, was die Staubentfernung gefunden hat, **IR** den Infrarotkanal des Scans.
*   **Optische Entfernung** (auch in der Leiste des Tabs Finish): findet Staub und Haare selbst. **Flecken-Schwelle** und **Haar-Schwelle**: niedriger findet mehr, kann aber echte Details entfernen; 1 ist aus. **Größe**: der größte Staub, der entfernt wird. Um einen Bereich zu schützen, ziehe mit der rechten Maustaste darüber oder klicke mit der rechten Maustaste auf eine Markierung. Die Cursor-Schaltfläche überspringt das Rechtsklick-Menü.
*   **IR-Entfernung** (Scans mit Infrarotkanal, auch in der Leiste des Tabs Finish): findet Staub im Infrarot, wo ihn das Bild nicht zeigt. **IR-Schwelle**: niedriger findet mehr. S/W und Kodachrome werden übersprungen, weil ihr Bild auch im Infrarot sichtbar ist. Der Kanal stammt aus 4-Kanal-TIFF- und DNG-Scans oder aus einer `_IR.tif`-Datei neben dem Scan.
*   **Heilen**: klicke oder male über Staub. Es werden nur auffällige Markierungen repariert, also male großzügig.
*   **Kratzer**: klicke Punkte entlang eines Kratzers oder Haars und drücke dann `Enter`.
*   **Linie**: klicke einmal auf einen langen, geraden Kratzer. NegPy findet und repariert dann die ganze Linie. **Linienempfindlichkeit**: niedriger findet schwächere Linien.
*   **Pinselgröße**: auch mit `Alt`+Scrollen auf dem Bild.
*   **Klonen**: kopiert Film aus einem anderen Bereich über einen Defekt, den Heilen nicht beheben kann. Der erste Klick (oder `Alt`-Klick) wählt die Quelle, danach malst du. **Quelle setzen** wählt eine neue. **Ton anpassen** übernimmt Helligkeit und Farbe um den Defekt. **Stärke** legt fest, wie stark er deckt, **Kantenweichheit**, wie weich seine Kante ist.
*   **Letzte rückgängig** und **Alle entfernen** entfernen manuelle Reparaturen. Klicke mit der rechten Maustaste auf eine im Bild, um sie zu löschen.

<!-- panel:finish src:0e98fdde -->
### 7.3 Finish: Vignette, Bildbühne, Rand

Wie der Abzug präsentiert wird. Die übrigen Regler jedes Effekts erscheinen, sobald sein erster Regler nicht 0 ist.

*   **Vignette**: **Nachbelichten** dunkelt die Ränder ab (positiv) oder hellt sie auf (negativ). **Größe** legt fest, wie weit sie reicht. **Rundung** reicht von rund (0) bis zur Form des Abzugs (1).
*   **Ausgefeilte Bildbühne**: **Breite** zeichnet einen rauen schwarzen Rahmen um das Bild, wie eine im Labor breiter gefeilte Negativbühne. **Rauheit**, **Streulicht** (ein graues Glühen außerhalb des Rahmens) und **Ecken** formen ihn.
*   **Rand**: **Breite** legt ein Passepartout um den Abzug. **Unteres Gewicht** macht den unteren Rand breiter, wie bei einem Bilderrahmen. **Farbe**: **Papierweiß** passt zum Papier, **Benutzerdefiniert** nutzt eine Farbe, die du wählst.

<!-- panel:presets src:8338af2d -->
### Vorgaben

Gespeicherte Einstellungen, die du auf andere Bilder anwenden kannst.

*   **Anwenden** (oder Doppelklick auf eine Vorgabe): wendet die ausgewählte Vorgabe auf dieses Bild an.
*   **Speichern…**: wähle, welche der aktuellen Einstellungen du als neue Vorgabe behältst. Eine Vorgabe kann die Einstellungen für Zuschnitt und Optik mit dem Flat-Field-Profil enthalten, aber nie Zuschnitt, Drehung oder Spiegelung eines einzelnen Bildes.
*   **Stift** und **Papierkorb**: bearbeiten oder löschen die ausgewählte Vorgabe.

<!-- panel:work_prints src:6d4f4720 -->
### Arbeitsabzüge

Ein **Arbeitsabzug** ist eine Version dieses Bildes, die du unter einem Namen behältst, wie die Probeabzüge auf dem Weg zum fertigen Abzug.

*   **Arbeitsabzug speichern…** (`Ctrl+Shift+S`) behält die aktuelle Bearbeitung.
*   **Klicke** auf einen Arbeitsabzug, um ihn zu verwenden. `Ctrl+Z` geht zurück.
*   **⋮** oder Rechtsklick: **Mit dieser Version vergleichen**, **Diese Version exportieren…**, **Umbenennen…** oder **Löschen…**. **Mit dieser Version vergleichen** öffnet **Vorher / Nachher** mit dem Arbeitsabzug links.
*   **Metadaten laden** (zuerst aus): stellt auch die mit dem Arbeitsabzug gespeicherten Metadaten wieder her.

<!-- panel:edit_history src:59f9ee38 -->
### Bearbeitungsverlauf

Jede Änderung, die du gemacht hast, die neueste oben (die letzten 100). **Jetzt** markiert den aktuellen Stand. Nach einem Rückgängig erscheinen spätere Schritte als **Rückgängig gemacht**, bis zu deiner nächsten Bearbeitung. Klicke auf einen Schritt, um dorthin zurückzugehen. **⋮** oder Rechtsklick → **Diese Version exportieren…** exportiert einen früheren Stand. **Mit diesem Schritt vergleichen** öffnet **Vorher / Nachher** mit diesem Schritt links, auf deinem aktuellen Zuschnitt. Drücke `\`, um den Vergleich zu schließen.

<!-- panel:film src:84c36b77 -->
### 10.1 Einrichtung

Was der Film ist und wie du ihn gescannt hast. Diese Angaben bestimmen, welche Karten und Regler erscheinen.

*   **Farbe / S/W / Dia**: Farbnegativ (C-41), ein Schwarzweißabzug oder Dia (E-6 und ähnlich). S/W druckt ein S/W-Negativ oder ein Farbnegativ in Schwarzweiß. Bei einem Farbnegativ bietet die Karte Filterung dann einen Kontrast-Filter.
*   **Positiv** (nur Dia): für eine Datei, die schon ein Positiv ist, etwa ein gescannter Abzug oder der Export einer anderen App. NegPy überspringt dann Inversion und Messung.
*   **Kamera / Filmscanner**: womit du gescannt hast. NegPy liest das aus jeder Datei (eine Kamera-RAW-Datei ist eine Kamera), bis du eine Wahl triffst. Deine Wahl gilt für jedes Bild und legt den Scanner im Scan-Reiter fest.
*   **Weißlicht / Schmalbandlicht** (nicht für ein S/W-Negativ): das Licht hinter dem Negativ. Weißlicht ist ein Leuchtkasten oder LED-Panel. Schmalband ist ein Licht mit reinem Rot, Grün und Blau, etwa ein Scanlight.

Die Scan-Angaben sind eine Einstellung für die ganze App. Die Liste darunter zeigt, wie eine RAW-Datei dekodiert wird und welches Papier verwendet wird.

Eine Kamera zeigt die Karten Raw-Dekodierung und Optik. Eine Kamera mit Schmalbandlicht zeigt auch Trichrom und Kalibrierung. Eine Änderung aktualisiert jedes geöffnete Bild (Ctrl+Z macht sie pro Bild rückgängig), aber ein Papier, das du selbst gewählt hast, bleibt. Ein Kamera-Scan legt Dekodierung und Papier für jedes Bild aus seinem eigenen Licht fest.

<!-- panel:assembly src:4fce6e79 -->
### 10.7 Bildzusammenstellung

Wie aus Dateien Bilder werden. Jede Rolle behält hier ihre eigenen Einstellungen.

#### Trichrom

Für eine Kamera mit Schmalbandlicht, die von jedem Bild drei Aufnahmen macht: Rot, Grün und Blau.

*   **Trichrom-Modus**: fügt die drei Aufnahmen zu einem Farbbild zusammen. Aufnahmen werden nach Aufnahmezeit (oder Dateiname) gruppiert, also mache die drei Aufnahmen jedes Bildes direkt nacheinander. Aufnahmen, die sich nicht gruppieren lassen, bleiben einzeln.
*   **Tripel bearbeiten…**: wähle die drei Aufnahmen des aktuellen Bildes von Hand. **Kanäle ausrichten** richtet sie aus, um Farbsäume zu entfernen.
*   **Bild als TIFF-Negativ zusammenführen…** (Rechtsklick): speichert ein Tripel oder einen Stitch als ein TIFF neben seiner ersten Quelldatei, mit seiner Bearbeitung. Eine Option verschiebt die Quelldateien in den Papierkorb.

#### Halbformat

**Halbformat-Modus** teilt jeden Scan für Halbformatkameras in zwei Bilder. Jede Hälfte hat ihre eigene Bearbeitung. Jede Rolle behält ihre eigene Einstellung. Ohne geöffnete Rolle gilt sie für Bilder ohne Rolle bis **Alle entladen**, und **Deine letzte Sitzung wieder öffnen** behält sie. Sie ist ausgegraut, solange ein Bild aus einer Rolle geladen ist, weil dieses Bild seiner eigenen Rolle folgt.

*   **Anpassen…**: ziehe den weißen Kasten für den Zuschnitt und die orange Linie für die Teilung. **Teilungsrichtung** und **Schnittbreite** legen die Lücke fest, **Automatisch erkennen** findet sie neu, und **Anwenden** speichert für dieses Bild, die Auswahl oder alle Bilder.
*   **Alle erkennen**: findet die Teilungen auf jedem Scan neu.
*   **Teilung aufheben** (bei einem geteilten Bild): macht es wieder zu einem Bild und löscht die Bearbeitungen beider Hälften.

Klicke mit der rechten Maustaste auf eine Hälfte für **Teilung für dieses Bild anpassen…** und **Teilung auf Rollenstandard zurücksetzen**. Wenn du Halbformat ausschaltest, bleiben bearbeitete Hälften als **Diptychon** nebeneinander. Schalte es wieder ein, um sie zu bearbeiten.

<!-- panel:sensor src:2c94daa5 -->
### 10.3 Kalibrierung: Farben deines Scan-Aufbaus

Korrigiert Farbfehler, die von Lichtquelle, Kamera und Film kommen.

**Sensor-Farbvermischungsprofil** (eine Einzelbelichtung bei schmalbandigem Licht, Lineares RAW): entfernt die Farbvermischung zwischen den Kanälen des Sensors. Führe nach einer Änderung die **Rollenanalyse** erneut aus.

*   **Profil**: das Sensor-Farbvermischungsprofil. Eigene `.toml`-Profile gehören in `<Documents>/NegPy/sensor/`.
*   **Methode**: **Zweiskalig** (Standard) entfernt den Mischanteil und verhindert Sprenkel bei sehr starken Farben. **Linear** ist exakt, aber starke Farben können sprenkeln. **Dichte** sprenkelt nicht, starke Farben sind aber etwas matter.
*   **Kalibrieren**: erstellt ein Profil aus drei Aufnahmen der freien Lichtquelle (Rot, Grün und Blau), aus Dateien oder von einer angeschlossenen Kamera (siehe CAMERA_SCANNING.md).

**Farbvermischung** (Farbfilm): Filmfarbstoffe, deine Lichtquelle und dein Sensor mischen die Farben leicht. Eine Matrix trennt sie wieder. Führe nach einer Änderung die **Rollenanalyse** erneut aus.

*   **Matrix**: die Matrizen für diesen Filmtyp. Matrizen mit *(approx)* stammen aus Filmdatenblättern und sind ein Ausgangspunkt. Eigene `.toml`-Dateien gehören in `<Documents>/NegPy/crosstalk/` (siehe CROSSTALK.md). Die Regler-Schaltfläche öffnet den Editor. **+** erstellt eine neue Matrix. NegPy enthält keine Dia-Matrix.
*   **Stärke**: wie stark die Matrix wirkt.
*   Die Farbfeld-Schaltfläche öffnet **Aus Testtafel messen** (Farbnegative): Sie misst eine Matrix aus einer Aufnahme eines ColorChecker 24 oder SpyderCheckr 24 auf dem Film. Ziehe die vier Punkte auf die Eckfelder.

**Farbton-Trimmung** (Kamera mit schmalbandigem Licht): dreht alle Farben auf dem Farbkreis, für ein Licht, das Gelb orange oder Grün oliv färbt. Lass den Wert bei weißem Licht auf 0. Sie wirkt auch bei Dias, wo das Sensor-Farbvermischungsprofil nicht wirkt.

<!-- panel:process src:61c1c291 -->
### 10.5 Messung: Negativ → Positiv

Um ein Negativ in ein Positiv umzuwandeln, misst NegPy die dunkelsten und hellsten Stellen (Schwarz- und Weißpunkt) und die Farbe des Films. Diese Regler bestimmen, was NegPy misst.

*   **Bild neu analysieren**: misst dieses Bild oder jedes ausgewählte Bild erneut anhand seines aktuellen Zuschnitts. Das Bild nutzt dann eigene Grenzen, nicht die Basislinie der Rolle oder Szene.
*   **Region zeichnen**: Zeichne einen Bereich, dann misst NegPy nur diesen Bereich. Doppelklicke hinein oder drücke `Enter` zum Bestätigen. `Esc` stellt die vorherige Region wieder her. **×** löscht sie.
*   **Filmbasis**: Klicke auf den klaren Film zwischen den Bildern, um die Filmbasis für die Analyseanzeige festzulegen. Es ist die Basis der Rolle, ohne geöffnete Rolle die des Bildes selbst. **×** löscht sie.
*   **Analysepuffer**: ignoriert beim Messen einen Rand um das Bild, damit Filmkanten und Halter nicht zählen.
*   **Beschneidung**: wie viel des Abzugs reines Schwarz oder reines Weiß ist.
*   **Grenzen sperren**: behält Schwarz- und Weißpunkt dieses Bildes, auch wenn sich der Zuschnitt oder die Rollenanalyse ändert.
*   **Luma-Bereichsbeschnitt** (Klappbereich): wie eng Schwarz und Weiß gesetzt werden. Positive Werte geben mehr Kontrast, für dichte oder verschleierte Negative. Negative Werte erhalten mehr Details in Schatten und Lichtern.
*   **Farbbeschnitt** (Klappbereich): wie eng die Farbbalance gesetzt wird.
*   **Weißpunkt** / **Schwarzpunkt** (Klappbereich): verschieben den gemessenen Weiß- und Schwarzpunkt. Ein positiver Weißpunkt hellt den Abzug auf. Ein positiver Schwarzpunkt hebt die Schwarztöne an. **R / G / B** verschiebt nur eine Farbe.

Ein **Dia** braucht keine Umkehrung: Die Tonwertregler formen das Bild direkt. Auto-Dichte und Auto-Gradation sind aus. Schalte sie für ein verblasstes Dia ein. In Lightroom-Begriffen ist Belichtung die Abzugsdichte (niedriger ist heller), Kontrast die ISO-R-Gradation, und Schatten und Lichter sind die Dichte von Schatten und Lichtern.

<!-- panel:baseline src:371ff87f -->
### 10.6 Rollenanalyse

Misst die ganze Rolle einmal und teilt das Ergebnis, damit die Bilder eines Films gleich abgezogen werden. NegPy misst die Rolle und jede Szene selbst, sobald Bilder dazukommen. Kein Bild nutzt eine Basislinie, bis du einen Mittelwert einschaltest. Ohne geöffnete Rolle ist die Karte ausgegraut: Öffne oder speichere zuerst eine Rolle.

*   **Neu analysieren**: misst jedes geladene Bild (außer Szenen- und gesperrte Bilder) speichert den Durchschnitt als Basislinie der Rolle und schaltet die Mittelwerte für diese Bilder ein. Bilder, die stark von den anderen abweichen, behalten ihre eigene. Zuerst werden die Bilder zugeschnitten, die auf den Auto-Zuschnitt warten.
*   **Dieses Bild verwenden**: macht die Messung dieses Bildes zur Basislinie der Rolle.
*   **Mittelwert verwenden**: Ein Rechtsklick auf einen der drei schaltet alle Mittelwerte ein oder aus. **Luma** nutzt Schwarz- und Weißpunkt der Rolle, **Farbe** die Farbbalance der Rolle und **Farbstich** (Farbnegative) die Graubalance der Rolle für die Farbstichentfernung. Bei aktivem Farbstich behält ein Bild unter anderem Licht etwas von diesem Licht, wie bei einem Frontier-Abzug. Ein Magenta-Stich bleibt, ein warmer oder grüner Stich bleibt teilweise, und ein blauer Stich verschwindet. Mit Luma und Farbe wird die Rolle gleichmäßig abgezogen. Sind beide aus, wird jedes Bild einzeln gemessen.
*   **Basislinie**: zeigt, woher die Basislinie stammt (Rolle, Szene oder Bild). Sie warnt, wenn keine vorhanden ist, wenn sie veraltet ist oder wenn die Farbe dieses Bildes stark vom Durchschnitt abweicht.
*   **Rollen**: wähle die gespeicherte Basislinie einer anderen Rolle. Eine Rolle mit Häkchen hat eine.
*   **Szenen**: Gruppen von Bildern mit eigener Basislinie, zum Beispiel ein Motiv unter einem Licht ([Szenen](#scenes)). **Analysieren** misst die Szene, **Auswählen** wählt ihre Bilder aus und **Löschen** entfernt die Gruppe.
*   **Szenen erkennen**: findet die Gruppen von Bildern unter anderem Licht als die Rolle, in beliebiger Bildreihenfolge, und zeigt sie auf dem Leuchttisch. Jede Gruppe wartet unter **Gefunden**. Um eine Gruppe zu ändern, ziehe Bilder auf eine andere Gruppe oder auf ein Bild ohne Szene, um sie auszuschließen. Rechtsklick › **Szene** macht dasselbe. Gib ihr einen Namen und **Übernehme** sie als Szene mit eigener Basislinie, oder **Verwirf** sie.

<!-- panel:demosaic src:99177ab3 -->
### 10.6 Raw-Dekodierung

Wie eine Kamera-RAW-Datei in Pixel umgewandelt wird. Wird bei einer Kamera angezeigt.

*   **Kamera-WB / Lineares RAW**: Kamera-WB nutzt den Weißabgleich der Kamera. Lineares RAW dekodiert neutral, für schmalbandiges Licht. Die Einrichtung stellt es für dich ein.
*   **Vorschau** / **Export**: die Demosaicing-Methode, die die Farben ergänzt, die ein Sensorpixel nicht sieht. **Auto** ist am Bildschirm schnell und nutzt für den Export **AHD**. **VNG** ist glatt und **PPG** schnell. **DCB** und **DHT** erhalten mehr feine Details, können aber Korn in Muster verwandeln. **AAHD** macht Kanten weicher. Es werden nur die Methoden deines LibRaw-Builds aufgeführt.
*   **Wiederherstellung** (Dias): rekonstruiert ausgebrannte Lichter in einer Kamera-RAW-Datei. **Mischen** passt zu Sonne und Himmel. **Rekonstruieren** ist stärker, kann aber die Farbe falsch treffen.

<!-- panel:optics src:6fce13e8 -->
### 10.7 Optik

Korrekturen für Objektiv und Licht, gleich für jedes Bild, das du mit diesem Aufbau scannst. Wird bei einer Kamera angezeigt.

#### Objektivkorrektur

*   **Eingebettet**: nutzt die eigene Objektivkorrektur der Datei (DNG, Sony ARW, Panasonic RW2). **Verzeichnung** richtet gekrümmte Linien gerade und **CA** entfernt Farbsäume. Stelle sie vor dem Zuschnitt oder der Retusche ein.
*   **Verzeichnungskorrektur**: richtet Linien gerade, die nach außen (positiv) oder nach innen (negativ) wölben. Nutze die Filmkante als gerade Führung.

#### Flat-Field-Korrektur: gleichmäßiges Licht

Entfernt ungleichmäßiges Licht, zum Beispiel dunklere Ecken, mit einer Aufnahme der freien Lichtquelle.

*   **Profil**: **+** speichert eine Aufnahme der freien Lichtquelle als benanntes Profil. Die Papierkorb-Schaltfläche löscht es, und jedes Bild, das es nutzt, verliert seine Korrektur.
*   **Flat Field anwenden**: wendet das gewählte Profil auf diese Rolle an.
*   **Flat Field prüfen**: zeigt, wie gut das Profil seine eigene Aufnahme ausgleicht. Gleichmäßiges Grau ist gut. Ein Streifen oder Fleck zeigt, wo es abweicht.

Ein Bild behält sein eigenes Profil, solange es nicht seiner Rolle folgt. So kann jedes Licht und jedes Filmformat eines haben. Ein neues Bild oder ein Bild mit fehlendem Profil nimmt das zuletzt gewählte Profil.

<!-- panel:metadata_gear src:208e1498 -->
### Analoge Ausrüstung

*   **Kamera / Objektiv / Filmsorte**: wähle aus deiner eigenen Ausrüstung (Reiter **Meine Ausrüstung**). **Andere…** öffnet den vollständigen eingebauten Katalog.
*   **Leeren**: leert alle drei.
*   **Aus Ordnernamen ableiten**: füllt leere Felder (Kamera, Film, ISO, Datum) aus dem Ordnernamen.

<!-- panel:metadata_capture src:a84922a9 -->
### Aufnahme

*   **Datum**: wann du das Bild aufgenommen hast, als `1998`, `1998-07`, `1998-07-14` oder `1998-07-14 16:30`.
*   **Ort**: die Pin-Schaltfläche öffnet eine Karte. Du kannst auch Koordinaten oder einen Kartenlink einfügen. Die Karte nutzt OpenStreetMap.

<!-- panel:metadata_process src:9dd71862 -->
### Entwicklung

Wie der Film entwickelt wurde.

*   **Gespeicherte Entwicklung**: ein Rezept aus Meine Ausrüstung, das die Felder darunter füllt. Wenn du in ein Feld tippst, wird es getrennt.
*   **Filmformat**: die Filmgröße, zum Beispiel `35mm`, `120` oder `4×5`.
*   **Entwickler** und **Verdünnung**: zum Beispiel `D-76` und `1+1`.
*   **Push / Pull**: von `Push +3` bis `Pull -3`.
*   **Zeit** und **Temperatur (°C)**: die Zeit als `9:30` oder in Minuten.
*   **Leeren**: leert die Entwicklungsfelder.

<!-- panel:metadata_scanning src:eab4bcdf -->
### Scannen

*   **Gespeicherter Aufbau**: ein Scan-Aufbau aus Meine Ausrüstung.
*   **Scannen**: wie du gescannt hast, oder Notizen.
*   **Rolle / Bild**: der Rollenname und die Bildnummer. Dateinamen können sie nutzen (`{{ roll }}`, `{{ frame }}`).
*   **Nummerieren…**: nummeriert die gewählten Bilder oder die ganze Rolle in Filmreihenfolge. Lege Start, Schrittweite und Richtung fest. **Verworfene Bilder überspringen** lässt verworfene Bilder aus.
*   **Leeren**: leert den Aufbau und die Notiz.

<!-- panel:metadata_exposure src:9c0029fe -->
### Belichtung

Verschlusszeit, Blende, ISO und Belichtungskorrektur der Kamera (EV) der Originalaufnahme, falls du sie kennst. Klicke auf das Schloss, um freien Text wie `1/125s f/2.8 ISO 400 -1/3ev` einzugeben. Eine Korrektur steht in Blendenstufen vor `ev`, und Summen wie `1+2/3ev` werden addiert.

<!-- panel:metadata_preview src:3cf26bec -->
### Metadaten-Vorschau

Zeigt, was in die exportierte Datei geschrieben wird. Die Schild-Schaltfläche **Originalmetadaten schützen** exportiert stattdessen die unveränderten Metadaten der Quelldatei. **Beschreibung…** wählt die Felder, aus denen die Bildbeschreibung besteht (zunächst Kamera, Objektiv, Film und ISO).

Kamera, Objektiv und Film kommen in die Standard-EXIF-Felder, dein Scanner in die eigenen XMP-Felder von NegPy. Eine in der Kamera vergebene Sternebewertung wird übernommen.

<!-- panel:gear_items src:1c45c908 -->
### Meine Ausrüstung

Deine Ausrüstungsliste, die jede Ausrüstungsauswahl nutzt. **Kategorie** wählt **Kameras**, **Objektive**, **Filmsorten**, **Entwicklungen** (Entwicklungsrezepte) oder **Scan-Aufbauten**. Sie wird in `~/NegPy/gear/` gespeichert.

*   **+**: kopiert einen Eintrag aus dem eingebauten Katalog, oder **Eigenes hinzufügen**, um einen einzutippen.
*   **Katalog**: zeigt zusätzlich die eingebauten Modelle.
*   **Kopieren / Papierkorb**: dupliziert oder löscht den gewählten Eintrag. Eingebaute Einträge lassen sich nicht löschen.

<!-- panel:gear_presets src:17a221ff -->
### Vorgaben

Gespeicherte Metadatensätze, zum Beispiel eine Kamera mit Objektiv und Film. Sie werden in `~/NegPy/presets/metadata/` gespeichert.

*   **Auf dieses Bild anwenden**: schreibt die Felder der Vorgabe auf dieses Bild. Andere Felder bleiben.
*   **+**: speichert die Metadaten des aktuellen Bildes unter einem Namen.
*   **⋮** oder Rechtsklick auf eine Vorgabe: anwenden, duplizieren, bearbeiten oder löschen.
*   **Details**: bearbeite die Felder der Vorgabe. **Notizen** ist freier Text.

Eine Filmsorte legt das Format fest. Setze ein Format wie `6×7` daher nach der Filmsorte.

<!-- panel:export_presets src:b7511145 -->
### Vorgaben

Gespeicherte Export-Rezepte: Format, Größe, Farbe, Ordner und Dateiname. Schalte die gewünschten ein, dann schreibt **Vorgaben exportieren** jedes Bild einmal pro Vorgabe. **Verwalten** bearbeitet sie. Vorgaben funktionieren nicht mit linearer Ausgabe.

<!-- panel:export_alongside src:ef285087 -->
### Begleitexporte

Zusätzliche Dateien neben dem Abzug. Jede hat links einen Schalter und rechts eine einmalige Aktion.

*   **Abzugsnotizen**: eine beschriftete Kopie des Bildes mit Abwedeln, Nachbelichten und Abzugseinstellungen, wie sie ein Dunkelkammer-Printer führt. **Vorschau** zeigt sie (`Shift+N`). **Notizen exportieren** speichert sie.
*   **Auto-Dichte / Gradation**: **Beim Export festschreiben** fixiert die automatische Helligkeit und den Kontrast jedes exportierten Bildes, damit sie sich später nicht ändern. **In Rolle festschreiben** tut das jetzt für jedes sichtbare Bild.
*   **Bearbeitungs-Sidecar-Dateien**: **Beim Export speichern** schreibt neben jede Quelle eine `.negpy`-Datei mit der Bearbeitung. **Sidecar-Dateien exportieren** schreibt sie für alle sichtbaren Bilder. Rechtsklick auf ein Bild → **Bearbeitung aus Sidecar-Datei laden…** liest eine zurück.

<!-- panel:contact_sheet src:e27d234d -->
### Kontaktbogen

Auf dem Tab **Neben dem Export**. Ein Kontaktbogen zeigt die ganze Rolle als Filmstreifen auf einem Blatt Papier, wie ein Kontaktabzug in der Dunkelkammer.

*   **Pfad**: der Ordner für die Bögen. Leer verwendet den Exportordner.
*   **Kontaktbogen…**: öffnet den Bogen. Ziehe an einer Kante des Papiers, um die Größe zu ändern.
*   **Filmformat**: 35mm, 35mm Half Frame oder 120 (bei 120 mit **Bildgröße**).
*   **Papier**, **Breite**, **Höhe** und **DPI**: Größe und Auflösung des Bogens. Das Standardpapier fasst eine Rolle mit 36 Aufnahmen.
*   **Reihenfolge**: nach Dateiname, nach Datum oder nach Szene, mit einem neuen Streifen oder einer neuen Zeile für jede Szene.
*   **Abzug**: **Wie bearbeitet** oder **Reiner Probeabzug**: die ganze Rolle mit einer Belichtung und normalem Kontrast. So siehst du, welche Negative dünn oder dicht sind.
*   **Bilder wählen**: Klicke auf ein Bild, um es auszulassen oder wieder aufzunehmen. Verworfene Bilder sind zu Beginn ausgelassen.
*   **Randaufdruck**: die Markierungen des Filmherstellers am Rand, zum Beispiel Filmname und Bildnummern.
*   **Rollenbeschriftung**: Rollenname, Film, Kamera und Datum über den Streifen. Gib in **Beschriftungstext** deinen eigenen Text ein, um ihn stattdessen zu drucken.
*   **Weißes Papier**: druckt auf Weiß und spart so Tinte auf einem Heimdrucker.
*   **Filmränder**: Aus druckt die Bilder aufrecht, jedes in seiner eigenen Form, in Zeilen wie auf dem Leuchttisch. Der Bogen hat dann keine Perforation und keinen **Randaufdruck**.

Die Bögen werden als `contact_sheet.jpg` gespeichert (`contact_sheet_1of2.jpg` und folgende bei einer langen Rolle).

<!-- panel:soft_proof src:193ff940 -->
### Softproof

Der Tab **Proof** zeigt am Bildschirm, wie das Bild auf einem bestimmten Drucker und Papier aussieht. Er ändert nur die Vorschau, nie den Export. Papier zeigt keine so leuchtenden Farben wie ein Bildschirm, darum wirkt ein korrekter Proof matter.

*   **Proof am Bildschirm** (`Shift+P`): schaltet den Proof ein oder aus.
*   **Vorgabe**: ein gespeichertes Setup aus Drucker und Papier.
*   **Profil**: das Drucker- oder Papierprofil. Es folgt dem Exportprofil, bis du eines wählst.
*   **Umsetzung**: **Relativ farbmetrisch** behält druckbare Farben genau und beschneidet den Rest. **Wahrnehmung** staucht alle Farben und passt zu sehr bunten Bildern. **Sättigung** ist für Diagramme und Grafiken.
*   **Schwarzpunktkompensation**: bildet den dunkelsten Ton auf das Schwarz des Papiers ab.
*   **Papierweiß simulieren** und **Tintenschwarz simulieren**: zeigen das echte Weiß und Schwarz des Papiers.
*   **Gamutwarnung**: zeigt Farben, die der Drucker nicht drucken kann, in Grau.
*   **Anzeige**: das Profil deines Monitors, automatisch erkannt.

<!-- panel:scan_source src:939ed7bb -->
### Scanner

*   **Scanner**: **Filmscanner** für einen eigenen Filmscanner (die Karten Gerät, Film & Qualität und Bildausschnitt), oder **Kamera** für eine Kamera auf einem Reprostativ (die Karten Kamera und Vorgabe).

<!-- panel:scan_device src:942d8a35 -->
### Gerät

*   **Backend**: **SANE** (Linux und macOS), **Nikon Coolscan (nkscan)** oder **pyOpticfilm (Plustek)**. Die Regler richten sich nach den Angaben des Scanners.
*   **Gerät**: der Scanner. Mit den Pfeilen aktualisierst du die Liste. Das Auswurfmenü hat **Jetzt auswerfen** und bei einem Streifeneinzug **Auswerfen, wenn fertig**. Das gibt den Streifen nach einem Stapel zurück.
*   **Debug-Protokoll** (Coolscan): schreibt `nkscan.log` in den NegPy-Ordner. Für einen Fehlerbericht wähle **Trace**, wiederhole das Problem und hänge die Datei an.

NegPy hält macOS und Windows während des Scans wach. Schalte unter Linux vor einer langen Rolle den automatischen Ruhezustand aus.

**Plustek**: Der OpticFilm 8200i SE, 8100 V2, 8300i SE und 7600i v1 funktionieren. Probiere bei anderen Modellen SANE. Richte unter Windows zuerst den USB-Treiber ein ([PLUSTEK_WINDOWS.md](PLUSTEK_WINDOWS.md)). **Coolscan**: braucht kein SANE. Unter Linux braucht USB eine udev-Regel für den Hersteller `04b0`.

<!-- panel:scan_quality src:c3afa46a -->
### Film & Qualität

*   **Film**: Farbnegativ, S/W-Negativ, Dia oder Kodachrome. Er bestimmt die Bilderkennung, die Messung und ob Infrarot angeboten wird.
*   **Filmformat**: die Bildlänge (135, 66, 645 und andere). **Auto** funktioniert, wenn der Halter sie festlegt.
*   **Auflösung** und **Bittiefe**: dpi und Bits pro Kanal des Scans.
*   **Scanmodus** (Plustek): **Mehrfachdurchgang** mittelt mehrere Scans für weniger Rauschen. **Adaptive Mehrfachbelichtung** verbindet eine kurze und eine lange Belichtung für mehr Umfang. Plustek-Scanner bieten sie bei 7200 dpi nicht an.
*   **Samples** (Coolscan): liest jede Zeile mehrfach, für weniger Rauschen in den Schatten. Langsamer.
*   **Infrarot**: **IR** scannt einen Infrarotkanal für die Staubentfernung auf der Karte Retusche. **ICE** (Coolscan) entfernt den Staub stattdessen im Scanner. Ein Scanner ohne ICE zeigt nur einen **IR**-Schalter. Nicht für S/W oder Kodachrome.
*   **Superfine** (Coolscan): scannt eine Zeile pro Durchgang. Langsamer.
*   **Autofokus**, **Auto-Belichtung** und **Belichtung**: Fokus und Belichtung am Scanner, vor dem Scan.

<!-- panel:scan_framing src:945dad86 -->
### Bildausschnitt

*   **Bilder**: `1-6`, `1,2,5` oder leer für alle. Die Streifenvorschau füllt das Feld aus.
*   **Streifenvorschau…** (Streifeneinzug): zeigt alle Bilder, um sie zu platzieren und zu wählen. **Vorschau…** (manueller Halter): legt ein Zuschnittsfenster fest. **Leeren** setzt es zurück. **Bilder erkennen** findet die Bilder nach einer Bewegung des Films erneut.
*   **Bild messen…** (Coolscan): misst die Belichtung an einem Bild und nutzt sie für jeden späteren Scan, damit alle Bilder einer Rolle zusammenpassen. **Entsperren** beendet das. Miss für jede Rolle neu.
*   **Prescan…** (Plustek): eine schnelle Vorschau des ganzen Bereichs. Ziehe einen Zuschnitt auf, dann liest der nächste Scan nur diesen Bereich.

<!-- panel:scan_camera src:813b8278 -->
### Kamera

Scannt mit einer Kamera auf einem Reprostativ, per USB im Modus **PC Remote** verbunden (macOS und Linux; sie braucht `python-gphoto2`). Mit einem NegPy-**Scanlight** steuert NegPy auch das Licht. CAMERA_SCANNING.md beschreibt die Einrichtung.

*   **Kamera** / **Licht**: der Verbindungsstatus. Das Licht wird bernsteinfarben, wenn es warm läuft. Unter Linux muss dein Benutzer für das Scanlight in der Gruppe `dialout` sein.
*   **Live-Ansicht**: ein Livebild der Kamera. Klicke, um die Fokuslupe zu setzen, und fokussiere, bis die **Fokusanzeige** **am Maximum** zeigt. Stelle ISO, Verschlusszeit und Blende hier oder an der Kamera ein. Beim Schließen kehrt das Scanlight in den vorherigen Zustand zurück.
*   **Histogramm** und **Zebras**: das Histogramm des Bildes und Streifen auf Bereichen kurz vor der Beschneidung, in den Beschneidungsfarben, die Farbsehen festlegt.
*   **Messen**: stellt die Verschlusszeit so ein, dass der hellste Bereich, der nicht beschnitten werden darf (der klare Film eines Negativs, die Lichter eines Dias), knapp unter der Beschneidung liegt. Miss vor dem ersten Scan. Ohne kalibrierte Vorgabe nutzt ein Scan die Belichtung, die an der Kamera eingestellt ist, und NegPy fragt nach, bevor es ungemessen scannt.
*   **Scan** nimmt das nächste Bild auf und importiert es. **Neu aufnehmen** (der Wiederholungspfeil) nimmt das letzte Bild erneut auf. Wenn ein Scan beschneidet, zeigt die Statuszeile, wie viel vom Bild beschnitten wurde. Im Fenster der Live-Ansicht scannt `Space` oder `S` und `R` nimmt neu auf. Mit einem Scanlight wählen **R**, **G**, **B** und **IR**, welche Belichtungen Neu aufnehmen wiederholt.
*   **Aufnahmeverzögerung**: wartet vor der ersten Belichtung, damit sich ein wackliger Tisch beruhigt.
*   **Licht eingeschaltet lassen** (die Glühbirne): hält das Scanlight nach jedem Scan an, damit du das nächste Bild einrichten kannst.

<!-- panel:scan_light src:e58660c1 -->
### Vorgabe

Sichtbar, solange ein Scanlight verbunden ist. Eine Vorgabe enthält die Lichtwerte und die Kamerabelichtung für eine Filmart.

*   **Vorgabe**: Licht und Belichtung für jeden Scan. Mit **+** fügst du eine auf drei Arten hinzu:
    *   **Integrierte Vorgaben**: sofort nutzbar, ohne Kalibrierung: **Weißlicht**, **Weißlicht + IR**, **Trichrom** und **Trichrom + IR**. Sie nutzen die Belichtung der Kamera, die der Belichtungsmesser einstellt. Weißlicht braucht ein Scanlight mit weißer LED (v4, Big Scanlight).
    *   **Auf der Filmbasis kalibrieren…**: Setze das Rechteck auf klaren Film und starte die Kalibrierung. NegPy ermittelt die Lichtwerte und die Verschlusszeit.
    *   **Von Hand einstellen**: Stelle die Regler selbst ein und speichere sie als Vorgabe. Ohne verbundene Kamera speichert sie keine Belichtung, und ein Scan nutzt die eigene der Kamera.
*   **Rot**, **Grün**, **Blau**, **Weiß**: die Lichtwerte (0 bis 255). **Licht an** schaltet das Licht ein, um den Film einzurichten, oder aus.
*   **Aufnahmemodus**: **Trichrom** macht eine Belichtung pro Farbe und fügt sie zusammen. **Einzelbelichtung** macht eine Belichtung mit allen drei Farben an.
*   **Sensor-Farbvermischungsprofil erstellen** (Einzelbelichtung): speichert zusätzlich ein Sensor-Farbvermischungsprofil für diese Vorgabe (siehe Kalibrierung, §10.4).
*   **ISO**, **Verschlusszeit**, **Blende**: die Belichtung der Vorgabe. Eine Änderung gilt, bis du die Vorgabe erneut wählst. Die Messtaste misst.
*   **IR-Verschlusszeit** und **Infrarot** (Lichter mit Infrarot-LEDs): fügt jedem Scan eine Infrarotbelichtung für die Staubentfernung hinzu, gespeichert als `_IR`-Datei. Es braucht eine Kamera, die Infrarot sieht (Vollspektrum oder umgebaut), und manuellen Fokus. Nur für Farbfilm.
*   **Kanalverzögerung**: eine Pause zwischen den Belichtungen für Rot, Grün und Blau, für Kameras, die sie brauchen.

<!-- panel:scan_output src:3015443a -->
### Ausgabe

Gilt für beide Scanner.

*   **Ordner**: wo die Scans gespeichert werden.
*   **Scan als Rolle**: der Ordner öffnet sich als Rolle, und die Bilder erscheinen beim Speichern.
*   **Ordner als Rolle**: Scans landen im Ordner selbst. Aus schreibt in einen Unterordner **Rolle**, und ein neuer Rollenname beginnt eine neue Rolle. **+** neben Rolle springt zum nächsten Namen (`Roll001` zu `Roll002`).
*   **Format** und **Dateiname** (Filmscanner): TIFF oder TIFF (mono) für S/W, und das Muster für den Dateinamen (`{{ date }}`, `{{ seq }}`).

<!-- panel:scan_strip src:135ca0e1 -->
### Streifenvorschau

Zeigt den ganzen Streifen, um die Bilder vor dem Scan zu platzieren und zu wählen. **Bildausschnitt anwenden** behält den Bildausschnitt, und **Scan** scannt. **Vorschau stoppen** beendet eine Vorschau vorzeitig.

*   **Zuschnitt**: Ziehe auf einem Bild. **Zuschnitte leeren** entfernt alle Zuschnitte.
*   **Versatz**: verschiebt jedes Bild entlang des Films, damit es zu den Bildern passt. Der weiße Rahmen zeigt das Bild, das NegPy gefunden hat.
*   **Drift**: ein Versatz, der von Bild zu Bild wächst.
*   Der Regler unter jeder Kachel verschiebt nur dieses Bild.
*   **Größe** legt die Kachelgröße fest. **Alle**, **Keine** und der Schalter jedes Bildes wählen Bilder. **Vorschau-Bild** scannt ein Bild erneut.
