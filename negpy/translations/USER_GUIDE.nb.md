# NegPy brukerveiledning

NegPy gjør skanninger av filmnegativer og dias til ferdige bilder. Originalfilene dine endres aldri. NegPy lagrer redigeringene dine i en lokal database. Velg språk under Innstillinger → Grensesnitt → Språk, og start NegPy på nytt. Systemstandard følger systemspråket og bruker engelsk når NegPy ikke har det språket. Når et panel mangler oversettelse, viser NegPy den engelske teksten.

<!-- panel:library src:35989d38 -->
### Bibliotek

Fanen **Bibliotek** (`Ctrl+L`) viser **rullene** dine. En rull er en navngitt gruppe bilder, vanligvis én film: en **mapperull** eller en **virtuell rull**. Importen husker mappen. NegPy flytter, gir nytt navn til eller sletter aldri filene dine.

*   **+**: **Importer mappe som rull…** gjør en mappe til én rull og åpner den. **Importer undermapper som ruller…** gjør hver undermappe med bilder til en rull.
*   **Dobbeltklikk** en rull (eller trykk Enter) for å åpne den i filmstripen.
*   **⋮** eller høyreklikk en rull: **Åpne**, **Last ut alle…** (den åpne rullen), **Rullanalyse**, **Gi nytt navn…**, **Lagre i mapperull**, **Gjenopprett fjernede bilder…** og **Slett…**. Gjenopprett fjernede bilder henter tilbake bildene som Fjern fra rull tok ut av en mapperull. Linjen under navnet på rullen teller dem. Slett beholder filene og de delte redigeringene deres. Rullens innstillinger, grunnlag, filmbase, scener, kortlåser og redigeringene som bare ble gjort i den rullen, går tapt. Oppdater legger ikke til en slettet mapperull igjen.
*   **Filtrer ruller…** og **Sorter** hjelper deg å finne en rull. **⋮ → Oppdater** finner nye undermapper og teller bildene på nytt. **⋮ → Søkefiltre…** viser mappenavn som hoppes over (først `export`).

En rull der mappen er borte, viser **mappen mangler**. Et mappenavn som samsvarer med utstyret ditt ("penf" for "Pen F"), åpner Rullinnstillinger ferdig utfylt.

En mapperull inneholder bare filene i sin egen mappe. En fil du legger til fra en annen mappe, blir værende bare i denne økten.

Bruk filmstripen for å lagre bildene i filmstripen som en rull. **Lagre i mapperull** gjør bilder som alle kommer fra én mappe, til rullen for den mappen. De andre filene i mappen vises som fjernet. **Lagre som virtuell rull…** beholder nøyaktig disse bildene, fra hvilke som helst mapper. Lagre i mapperull gjør også en virtuell rull fra én mappe til rullen for den mappen. Den nye rullen starter med den åpne rullens innstillinger, kortlåser og redigeringene som bare ble gjort i den rullen. Et bilde har én redigering i hver rull som inneholder det. Høyreklikk → **Rediger uavhengig i denne rullen** gir det en egen redigering i én rull.

<!-- panel:frames src:c1f54a29 -->
### Filmstripe

**Filmstripen** under lerretet viser de åpne bildene på én rad. Dra den øvre kanten for å gjøre bildene større.

Raden **Rull** på verktøylinjen inneholder:

*   Navnet på rullen og antallet bilder, beholdte og forkastede.
*   **Filterfeltet**: skriv en del av et filnavn, eller uttrykk som `film:portra iso:>=400` (se nedenfor). Søkeknappen, eller **Enter**, søker i hele biblioteket.
*   **Sorter**: etter navn, dato, scene, eller etter rull og bildenummer.
*   **Arkfilter**: alle bilder, bare beholdte, skjul forkastede, eller bare umerkede. Knappen er uthevet mens et filter skjuler bilder.
*   **Rullinnstillinger** (merkelappen): detaljer om utstyr og film for dette bildet, utvalget eller rullen.
*   **Bruk**: kopierer valgte innstillinger fra dette bildet til utvalget eller rullen.
*   **⋮**: **Legg til filer…**, **Legg til mappe…**, **Lagre i mapperull**, **Lagre som virtuell rull…**, **Oppdater miniatyrbilder**, **Aktiv mappe** (laster inn nye filer som en skanner lagrer i mappen til det nyeste bildet), **Last ut…**, **Last ut alle…** (tømmer filmstripen; rullen blir værende i biblioteket), **Tilbakestill bilder til standard…**, **Gjenopprett fjernede bilder…** og **Veiledning for filmstripen…**.

Rutenettknappen åpner **lysbordet** (`Shift+G`): hele rullen i rader, for sortering. Et klikk velger et bilde, og et dobbeltklikk eller `Enter` åpner det. Verktøylinjen virker på de valgte bildene: **Behold**, **Forkast**, roter og speilvend, **Autobeskjær** og **Eksporter**, og ⋯ inneholder resten av høyreklikkmenyen. Glidebryteren **Bilder per rad**, `+` og `-`, eller `Ctrl` med rullehjulet endrer bildestørrelsen. `Esc` går tilbake, og det gjør også et lerretsverktøy eller en titt.

#### Filtrering

Et vanlig ord samsvarer med filnavnet. `field:value`-uttrykk samsvarer med bildedata, og en innledende `-` utelukker:

| Uttrykk | Finner |
|---|---|
| `film:portra` | filmtype som inneholder "portra" |
| `camera:"Nikon F3"` | anførselstegn for en verdi med mellomrom |
| `iso:>=400` | tall tar også `>`, `>=`, `<`, `<=` |
| `date:2024-03` · `shot:>=1998` | fildato · opptaksdato |
| `place:tokyo` · `lens:` · `roll:` | andre metadatafelt |
| `scene:beach` | bilder i en scene |
| `keeper:` · `rejected:` · `edited:` | merkede eller redigerte bilder |

Uttrykk kan kombineres: `film:portra iso:>=400 -rejected:`. Metadatauttrykk samsvarer bare med bilder der du har fylt ut metadataene. **Søk etter betydning** (slå det på i Innstillinger → Ytelse) finner bilder etter en beskrivelse, for eksempel "en hund på en strand".

<!-- panel:analysis src:cf165609 -->
## 3. Analyseavlesning

Denne avlesningen beskriver det gjeldende bildet mens du redigerer.

#### Diagram

Velg en diagramvisning:

*   **Kurve**: papirets kurve, fra negativet (bortover) til kopien (oppover). Brattere betyr mer kontrast. Det grå histogrammet er negativet, det fargede er kopien.
*   **Bølgeform**: lysheten i kopien over bildet.
*   **Parade**: rød, grønn og blå side om side; et grått motiv ligger i samme høyde i alle tre. **Paradeoverlegg** legger de tre i ett plott.
*   **Vektorskop**: fargene i kopien rundt et midtpunkt. En klynge utenfor midten er et fargestikk.

**Log** viser små områder, som klipping.

#### Klipping

Trekanter i de øvre hjørnene advarer når skygger blir helt svarte (venstre) eller høylys helt hvite (høyre). **Klippeoverlegg** (knappen med halvsirkelen) markerer disse delene på bildet. **Soneutheving** (markørknappen) viser bare sonen under pekeren på bildet.

#### Soner og soneplassering

**Gråkilen** kopierer 21 grå trinn; trinn som smelter sammen i svart eller hvitt, er tapte toner. **Sonestripen** viser hvor mye av bildet som havner i hver sone, fra 0 (svart) gjennom V (mellomgrå) til X (hvit). Dobbeltklikk den (eller trykk `Shift+Z`) for **soneoverlegget**, som farger bildet etter sone.

Klikk en sone på stripen, og så et punkt på bildet, så angir NegPy kopien slik at punktet kopieres i den sonen. Én nål angir Kopitetthet; en andre angir også ISO-R-gradering. **Plasser soner** (eller Enter) bruker resultatet; `Esc` avbryter.

#### Avlesninger

Folden **Avlesninger** inneholder tall om skanningen. En gul verdi er en advarsel; hold pekeren over en rad for detaljer.

**Filmbase** (trådkorset): klikk den klare filmen mellom bildene for å angi basen avlesningene måler fra.

*   **Negativ**: tetthetsomfanget til negativet (Δ). Hold pekeren over for flatt, normalt eller kontrastrikt.
*   **Eksponering**: lyst (positiv) eller mørkt (negativ) bilde, i blendertrinn.
*   **Klipping**: andel av kopien som er helt svart, og så helt hvit.
*   **Skanneklipping**: andel av hver kanal ved skannerens maksimum. Ingen redigering retter det; eksponer skanningen lavere.
*   **Reparer**: andel av bildet som støvfjerningen skrev om.
*   **Fargeomfang**: andel som prøveskriveren ikke kan gjengi, under prøvetrykk på skjerm.

<!-- panel:autocrop src:3047f00d -->
### 4.1 Beskjæring

Velg den delen av bildet du vil beholde, selv eller med autobeskjæring.

*   **Tegn**: dra en beskjæring på bildet. Dra et hjørne for å endre størrelsen, innsiden for å flytte den, eller utenfor boksen for å rotere. `Enter` eller **Ferdig** beholder beskjæringen, og `Esc` setter tilbake beskjæringen du hadde før. **×** fjerner en beskjæring du har tegnet. Tegning slår av autobeskjæring.
*   **Autobeskjær** (også på linjen i fanen Geometri): finner kantene i bildet og beskjærer etter dem.
*   **Forhold**: formen på beskjæringen, for eksempel `3:2` eller `1:1`, dreid for å passe bildet. **Fri** bruker formen NegPy finner, og du kan dra én side om gangen. **Oppdag** velger den nærmeste standardformen.
*   **Veiledning**: komposisjonslinjer, for eksempel tredjedeler, mens du tegner. Roterknappen dreier dem.
*   **Modus** (folden Innstillinger for autobeskjæring): **Bare bildet** beskjærer til bildet. **Filmkant** beholder hele filmen med kanten og hullene.
*   **Beskjæringsforskyvning** og **Justering av filmkant** (folden under Modus): flytter beskjæringen innenfor (positiv) eller utenfor (negativ) den funne kanten, og angir hvor langt den skjærer inn i filmkanten (Bare bildet). 0 % stopper ved filmkanten og 100 % ved bildekanten; over 100 % skjærer inn i bildet for å fjerne en hvit kant.
*   **Rett opp først**: retter bildet etter filmkantene før autobeskjæring beskjærer det.
*   **Beskjær alle bilder** (Bare bildet): beskjærer alle bilder sammen. Tydelige bilder hjelper utydelige, og beskjæringer du har tegnet, blir stående.

Autobeskjæring krever at skannerlyset eller holderen er den lyseste delen av skanningen. Den lar et bilde være i fred når bildet er like lyst som lyset, eller når neste bilde fyller en stor del av én side. Beskjær et slikt bilde selv, eller bruk **Filmkant** og juster innover.

<!-- panel:geometry src:913e25e7 -->
### 4.2 Rett opp

Drei bildet og sett det i vater.

*   **Roter** og **Speilvend**: drei bildet en kvart omdreining, eller speilvend det. Verktøylinjen har de samme knappene.
*   **Vater**: tegn en linje langs horisonten eller en loddrett kant, så roteres bildet til det er i vater.
*   **Rett mot film** (også på linjen i fanen Geometri): retter bildet inn etter kantene på filmen. Den ser bort fra innholdet i bildet, så et bilde tatt på skrå blir stående på skrå.
*   **Finrotasjon**: dreier bildet litt. Positiv er med klokken.
*   **Tilt** og **Sving** (fold): rett opp en bygning fotografert nedenfra (Tilt) eller en vegg fotografert fra siden (Sving). **Tegn linjer** angir begge ut fra linjer langs de fire filmkantene.
*   **Beskjær tomme hjørner**: beskjærer bort de tomme hjørnene som rotasjon, Tilt og Sving etterlater.

Beskjær før en stor Tilt eller Sving, ellers kan filmkanten komme til syne og gjøre kopien mørkere. Objektivforvrengning rettes i **Optikk** ([§10.3](#103-optics)).

<!-- panel:color src:2abd7418 -->
### 5.1 Filtrering: fargebalanse

Korrigerer fargen i kopien, slik fargefiltrene i et forstørrelsesapparat i mørkerommet gjør. **Global / Skygger / Høylys** bruker fargekontrollene på hele bildet, eller mest på de mørke eller lyse delene.

*   **Velg WB**: klikk på noe i bildet som skal være grått, så korrigerer NegPy fargen slik at det blir grått.
*   **Temperatur**: gjør bildet varmere eller kaldere.
*   **Cyan / Magenta / Gul**: forskyver fargen mellom cyan og rødt, magenta og grønt, eller gult og blått.
*   **Fargering** (`Shift+F`): viser bildet som et rutenett av litt ulike farger. Klikk den som ser riktig ut. `Esc` lukker den.
*   **Fjerning av fargestikk**: fjerner et fargestikk som er ulikt i de mørke og lyse delene av bildet. Den starter på 0,5 (halvparten) for fargenegativer og på 0 for dias. Blader, himmel og vann regnes ikke som grått, så de beholder fargen.
*   **Filter**: et farget filter, som på kameraobjektivet for svart-hvitt-film. Det vises bare for et fargenegativ som kopieres i S/H, og erstatter kontrollene ovenfor. Hvert filter gjør fargene motsatt av sin egen mørkere. Gul #8, Dyp gul #15, Oransje #21 og Rød #25 gjør en blå himmel mørkere, trinn for trinn. Rød gjør også blader mørke og hud blek. Gulgrønn #11 og Grønn #58 gjør blader lysere og hud mørkere. Ingen gir utseendet til en vanlig svart-hvitt-film.

<!-- panel:tone src:c1700dd9 -->
### 5.2 Tone: lyshet og kontrast

Hvor lys eller mørk kopien er, og hvor mye kontrast den har.

*   **Kopitetthet**: hvor lys eller mørk kopien er (eksponeringstiden i mørkerommet). Lavere er lysere.
*   **ISO-R-gradering**: kontrast (papirgraderingen i mørkerommet). Et lavere tall gir mer kontrast; omtrent 110 er normalt.
*   **Auto** (tryllestaven, også på linjen i fanen Eksponering): **Autotetthet** og **Autogradering** angir lyshet og kontrast for hvert bilde, så en rull kopieres jevnt. En glidebryter viser da den automatiske verdien: flytt den for å justere, og dobbeltklikk den for å gå tilbake. Høyreklikk tryllestaven for å slå begge på eller av. Auto-menyen har også:
    *   **Angi mål…**: lysheten og kontrasten Auto sikter mot.
    *   **Bak inn autotetthet/-gradering**: beholder de automatiske verdiene for dette bildet og slår av Auto for det.
    *   **Hold svart**: når du gjør skyggene lysere, forblir de dypeste svarttonene svarte.
*   **Prøvestripe** (`Shift+T`): et rutenett av versjoner med ulik lyshet og kontrast. Klikk den du liker. `Esc` lukker den.
*   **Skyggetetthet** / **Høylystetthet** (folden under Kopitetthet): gjør bare skyggene eller høylysene mørkere eller lysere.
*   **Skyggegradering** / **Høylysgradering** (folden under ISO-R-gradering): kontrast bare i skyggene eller høylysene.
*   **Kopieringshjelpemidler** (fold):
    *   **Forbelysning**: henter tilbake detaljer i de lyseste områdene.
    *   **Diffusjon**: en myk glød; mørke områder sprer seg inn i lyse.
    *   **Kontrastmaske**: åpner store mørke områder og holder tilbake store lyse, og beholder fine detaljer. I mørkerommet får en kopist dette med en myk, uskarp maske lagt på negativet. I digital kopiering får Fujifilm Frontier-skannere en lignende virkning med HyperTone. Områder langt fra mellomgrå endres mest, så et bilde med lav kontrast endres lite.
*   **Global / R / G / B**: R, G eller B bruker graderingene på bare én farge, for et fargestikk som er ulikt i skygger og høylys.

**Hvitpunkt** og **Svartpunkt** er på kortet **Lysmåling** i fanen Rull ([§10.5](#105-metering-negative--positive)).

<!-- panel:paper src:00cfca5d -->
### 5.3 Papirrespons: papiret

Papiret bildet kopieres på, og hvordan det gjør lys om til toner fra svart til hvitt.

*   **Kopipapir**: papiret. Et ekte papir angir kontrastkurven og fargene ut fra produsentens data. **Digital (Adobe RGB)**, standarden, er et ideelt papir. **Digital (Pakon)** har fargene til en Pakon-laboratorieskanner. **Digital (Frontier)** har fargene og kontrastkurven til en laboratorieskanning fra Fujifilm Frontier.
*   **Prøvestripe for papir** (den stripete knappen): kopierer bildet på hvert papir. Klikk et papir for å se det stort, bla med piltastene, og trykk **Behold** for å bruke det.
*   **Fargestoffseparasjon**: fargestyrke i kopien. 1 er uendret. **Separasjonsdemping** legger mer av endringen på matte farger og mindre på sterke. **Kroma** i fanen Utseende endrer i stedet alle farger likt.
*   **Papirhvitt**: viser hvitfargen til ekte papir, litt mørkere enn rent hvitt.
*   **Papirsvart**: viser svarttonen til ekte papir, litt lysere enn rent svart.
*   **Kurveform** (fold):
    *   **Snap**: mellomtonekontrast. Svart og hvitt flytter seg ikke.
    *   **Tå**: former skyggene. Positiv løfter dem, negativ gjør dem dypere. **Tåbredde** angir hvor langt inn i mellomtonene den når.
    *   **Skulder**: former høylysene. Positiv mykner dem, negativ gjør dem lysere og kan miste detaljer. **Skulderbredde** angir hvor langt den når.
*   **Global / R / G / B**: R, G eller B bruker kurvekontrollene på ett fargelag.

<!-- panel:local src:7d85677b -->
### 5.4 Skygging og etterbelysning: gjør områder lysere eller mørkere

Tegn en form, og gjør så bare det området lysere (skygging) eller mørkere (etterbelysning). Et dias bruker ikke dette kortet, men maskene blir stående.

*   **Tegn**: klikk for å plassere punkter. Dobbeltklikk, trykk Enter eller klikk nær starten for å lukke formen. Dra et punkt for å flytte det, klikk en "+" på en kant for å legge til et, og høyreklikk et punkt for å slette det.
*   **Oval**: dra ut en oval. Håndtakene flytter, strekker og dreier den.
*   **Kort**: en etterbelysning som toner ut, som et kort som flyttes over papiret. Dra fra kanten med full styrke (heltrukket linje) til der den slutter (stiplet linje).
*   **Maskeliste**: klikk en maske for å velge den. Ikonene slår den på eller av, inverterer den (virker overalt unntatt innenfor den), viser omrisset eller sletter den. Glidebryterne nedenfor endrer den valgte masken.
*   **Etterbelysning**: positiv gjør området mørkere, negativ gjør det lysere. Enheten er blendertrinn: +1 dobler lyset.
*   **Mykning**: hvor myk kanten er.
*   **Gradering**: kontrast innenfor masken. Negativ gir mer kontrast.
*   **Forbelysning**: henter tilbake detaljer i lyse områder innenfor masken, som Forbelysning på kortet Tone.
*   **Tonegrense**: **Høylys** eller **Skygger** får masken til å virke bare på toner lysere eller mørkere enn **Tonesone** (0 er svart, V mellomgrå, X hvit). En etterbelysning av himmelen stopper da ved horisonten. **Tonemykhet** angir hvor gradvis den starter.

**Kopieringsnotater** (`Shift+N`) markerer maskene på bildet med numre og verdier, slik notatene til en kopist gjør.

<!-- panel:lab src:e8e4a89c -->
### 6.1 Lab: farge og detaljer

Justeringene en laboratorieskanner gjør etter kopien.

*   **Kroma**: fargestyrke. 1 er uendret, 0 er svart-hvitt.
*   **Hudbeskyttelse**: hindrer at hudtoner får for sterk farge.
*   **Kromastøyreduksjon**: fjerner fargede prikker. Filmkornet blir stående. Den er på 2,5 for fargefilm.
*   **Skarphet**: gjør detaljer skarpere uten fargede kanter. Folden inneholder **Metode** (Uskarp maske gjør kantene skarpere; Dekonvolusjon opphever skannerens uskarphet), **Radius** (størrelsen på detaljene, i eksporterte piksler; vurder den i **Eksportstørrelse** eller med **Kornfokuserer**) og **Maskering** (gjør bare kanter skarpere, så himmel og hud holder seg jevne).
*   **Automaskering** (tryllestaven ved siden av Metode, også i fanehodet): angir Maskering ut fra kornet i hvert bilde. Maskering viser verdien som brukes, og en dragning flytter den opp eller ned derfra. En eksport måler kornet på nytt i sin egen størrelse.
*   **CLAHE**: lokal kontrast, for å få frem detaljer. Sterke verdier kan se unaturlige ut.
*   **Glød** (folden Effekter): en myk glød rundt lyse områder, som den myke fokusen i en minilab-skanner.
*   **Halasjon** (folden Effekter): en rød glød rundt lyse områder, slik noen filmer gir.

<!-- panel:altproc src:1b810792 -->
### 6.2 Alternative prosesser

Historiske kopieringsprosesser for S/H-negativer. Velg **Ingen**, **Lith**, **Cyanotypi** eller **Sabattier**, så vises bare kontrollene for den prosessen.

#### Lith

Kremaktige, varme høylys og harde, dype svarttoner. Kopipapiret under Papirrespons angir fargen. Bare selen- og gulltonerne virker på en lith-kopi.

*   **Eksponering**: mer gir varmere, mer fargerike høylys og mykere toner.
*   **Snappunkt**: når kopien tas ut av fremkalleren. Høyere gir dypere, kaldere svarttoner.
*   **Brådhet**: hvor brått skyggene blir svarte.

#### Cyanotypi

En blå kopi med grønne høylys. Tonere virker ikke på den; bruk Bleking og Tannin.

*   **Sensibilisator**: **Klassisk** gir en lys blåfarge med grønne høylys. **Ny** gir dypere, renere blåtoner.
*   **Eksponering**: mer gjør kopien blåere.
*   **Eksponeringsskala**: kontrast. En kortere skala gir mer kontrast.
*   **Bleking**: fjerner blått, høylysene først.
*   **Tannin**: gjør blekede områder brune. Bruk Bleking først for en helt brun tone.

#### Sabattier

Et lysglimt under fremkallingen. De lyse tonene snus mot grått, og tynne lyse linjer (Mackie-linjer) dukker opp langs kantene mellom mørkt og lyst.

*   **Styrke**: lengden på lysglimtet. Over 1 snus de lyse tonene.
*   **Ny eksponering**: hvor reverseringen starter. Mørkere toner blir som de er.
*   **Agitasjon**: mindre agitasjon gir bredere Mackie-linjer.

<!-- panel:toning src:8fc7301e -->
### 6.3 Toning

Tonere er kjemiske bad som endrer fargen på en S/H-kopi. Delt toning legger farge til enhver kopi.

**Kjemisk toning** (S/H-negativer) virker i rekkefølgen som vises:

*   **Selen**: dypere, kaldere svarttoner.
*   **Sepia**: varm brun, fra høylysene og nedover.
*   **Gull**: blåsvart. Over sepia gir den oransjerøde høylys.
*   **Jernblå**: blå skygger og marineblå svarttoner.
*   **Kobber**: rosa til teglrødt. Svarttonene blir svakere.
*   **Vanadium**: grønne mellomtoner og høylys. Svarttonene forblir svarte.

På en lith-kopi virker bare Selen og Gull. På en cyanotypi virker ingen toner.

**Delt toning** (alle kopier): **Skyggestyrke** og **Høylysstyrke** legger farge til skyggene og høylysene. **Skyggetone** og **Høylystone** velger fargen, og vises når styrken er over 0.

<!-- panel:retouch src:8d07a6c2 -->
### 7.1 Retusj: støv, hår, riper

Fjerner støv, hår og riper. Hver reparasjon bygges opp fra den rene filmen rundt den.

*   **Overlegg**: **Merket** viser hva støvfjerningen fant, **IR** skanningens infrarøde kanal.
*   **Optisk fjerning** (også på linjen i fanen Etterarbeid): finner støv og hår selv. **Flekkterskel** og **Hårterskel**: lavere finner flere, men kan fjerne ekte detaljer; 1 er av. **Størrelse**: det største støvet som fjernes. For å beskytte et område høyredrar du over det eller høyreklikker et merke; pekerknappen hopper over høyreklikkmenyen.
*   **IR-fjerning** (skanninger med en infrarød kanal, også på linjen i fanen Etterarbeid): finner støv i infrarødt, der bildet ikke vises. **IR-terskel**: lavere finner flere. S/H og Kodachrome hoppes over: bildet vises også i infrarødt. Kanalen kommer fra TIFF- og DNG-skanninger med fire kanaler eller en `_IR.tif`-fil ved siden av skanningen.
*   **Lege**: klikk eller mal over støv. Bare merker som skiller seg ut, repareres, så mal romslig.
*   **Ripe**: klikk punkter langs en ripe eller et hår, og trykk så Enter.
*   **Linje**: klikk én gang på en lang, rett ripe, så finner og reparerer NegPy hele linjen. **Linjefølsomhet**: lavere finner svakere linjer.
*   **Penselstørrelse**: også `Alt`+rulling på bildet.
*   **Klon**: kopierer film fra et annet område over en feil som Lege ikke kan rette. Det første klikket (eller `Alt`-klikk) velger kilden; mal så. **Angi kilde** velger en ny. **Tilpass tone** henter lysheten og fargen rundt feilen. **Styrke** angir hvor mye den dekker, **Mykning** hvor myk kanten er.
*   **Angre siste** og **Fjern alle** fjerner manuelle reparasjoner; høyreklikk en på bildet for å slette den.

<!-- panel:finish src:0e98fdde -->
### 7.2 Etterbehandling: vignett, negativholder, kant

Hvordan kopien presenteres. De andre glidebryterne for hver effekt vises når den første glidebryteren ikke er 0.

*   **Vignett**: **Etterbelysning** gjør kantene mørkere (positiv) eller lysere (negativ). **Størrelse** angir hvor langt den når. **Rundhet** går fra rund (0) til formen på kopien (1).
*   **Filet negativholder**: **Bredde** tegner en grov svart ramme rundt bildet, slik en negativholder i mørkerommet som er filt bredere, gjør. **Ruhet**, **Lysspredning** (en grå glød utenfor rammen) og **Hjørner** former den.
*   **Kant**: **Bredde** legger et passepartout rundt kopien. **Vekt nederst** gjør den nedre kanten bredere, som på et passepartout til innramming. **Farge**: **Papirhvitt** tilpasser seg papiret, **Egendefinert** bruker en farge du velger.

<!-- panel:presets src:32c2813a -->
### Forhåndsinnstillinger

Lagrede innstillinger du kan bruke på andre bilder.

*   **Bruk** (eller dobbeltklikk en forhåndsinnstilling): bruker den valgte forhåndsinnstillingen på dette bildet, de valgte bildene eller hele rullen.
*   **Lagre…**: velg hvilke av de gjeldende innstillingene som skal beholdes som en ny forhåndsinnstilling. En forhåndsinnstilling kan inneholde innstillingene for Beskjæring og Optikk med flatfeltprofilen, men aldri beskjæringen, rotasjonen eller speilvendingen av ett bilde.
*   **Penn** og **Papirkurv**: rediger eller slett den valgte forhåndsinnstillingen.

<!-- panel:work_prints src:6d4f4720 -->
### Arbeidskopier

En **arbeidskopi** er en versjon av dette bildet som du beholder under et navn, som prøvekopiene på veien mot den ferdige.

*   **Lagre arbeidskopi…** (`Ctrl+Shift+S`) beholder den gjeldende redigeringen.
*   **Klikk** en arbeidskopi for å bruke den. `Ctrl+Z` går tilbake.
*   **⋮** eller høyreklikk: **Sammenlign med denne versjonen**, **Eksporter denne versjonen…**, **Gi nytt navn…** eller **Slett…**. **Sammenlign med denne versjonen** åpner **Før / etter** med arbeidskopien til venstre.
*   **Last inn metadata** (av i starten): gjenoppretter også metadataene som er lagret med arbeidskopien.

<!-- panel:edit_history src:59f9ee38 -->
### Redigeringshistorikk

Hver endring du har gjort, den nyeste øverst (de siste 100). **Nå** markerer den gjeldende tilstanden; etter en angring vises senere trinn som **Angret** til neste redigering. Klikk et trinn for å gå tilbake til det. **⋮** eller høyreklikk → **Eksporter denne versjonen…** eksporterer en tidligere tilstand. **Sammenlign med dette trinnet** åpner **Før / etter** med det trinnet til venstre, på den gjeldende beskjæringen din. Trykk `\` for å lukke den.

<!-- panel:film src:b72d2369 -->
### 10.1 Oppsett

Hva filmen er, og hvordan du skannet den. Disse valgene bestemmer hvilke kort og kontroller som vises.

*   **Farge / S/H / Dias**: fargenegativ (C-41), en svart-hvitt-kopi, eller dias (E-6 og lignende). S/H kopierer et S/H-negativ eller et fargenegativ i svart-hvitt. For et fargenegativ gir kortet Filtrering da et kontrastfilter.
*   **Positiv** (bare Dias): for en fil som allerede er et positiv, for eksempel en skannet kopi eller en eksport fra en annen app. NegPy hopper da over invertering og lysmåling. Et dias som ikke er en kamera-RAW, for eksempel en TIFF, JPEG, JXL eller en skanning fra Pakon, Noritsu, Coolscan eller Flextight, starter som Positiv. Slå det av for å gi en falmet skanning diasets tonekurve. En JXL lastes inn som lineær.
*   **Koding** (positiv TIFF): hvordan verdiene koder lys. **Auto** leser filens ICC-profil, og ellers lineær over 8 bit og sRGB ved 8 bit. **Lineær** er lineære utdata fra en skanner, for eksempel Nikon Scan med fargestyring av. **2.2** er gamma 2,2 i Adobe RGB-primærfarger. Auto hopper over skannerens egen kalibreringsprofil og leser filen som umerket. Velg Lineær eller 2.2 for å passe skannerprogramvaren.
*   **Kamera / Filmskanner**: hva du skannet med. NegPy leser det fra hver fil (en kamera-RAW er et kamera) til du velger noe. Valget ditt gjelder for hvert bilde og angir skanneren i fanen Skann.
*   **Hvitt lys / Smalbåndslys** (ikke for et S/H-negativ): lyset bak negativet. Hvitt lys er en lysboks eller LED-plate. Smalbånd er et lys med rent rødt, grønt og blått, for eksempel en Scanlight.

Skannevalgene er én innstilling for hele appen. Listen under dem viser hvordan en RAW-fil dekodes og hvilket kopipapir som brukes.

Et kamera viser kortene RAW-dekoding og Optikk. Et kamera med smalbåndslys viser også Trichrome og Kalibrering. En endring oppdaterer hvert åpent bilde (Ctrl+Z angrer den per bilde), men et kopipapir du har valgt selv, blir stående. En kameraskanning angir dekodingen og kopipapiret for hvert bilde ut fra sitt eget lys.

<!-- panel:demosaic src:577713a8 -->
### 10.2 RAW-dekoding

Hvordan en kamera-RAW-fil gjøres om til piksler. Vises for et kamera.

*   **Kamera-WB / Lineær RAW**: Kamera-WB bruker kameraets hvitbalanse. Lineær RAW dekoder nøytralt, for smalbåndslys. Oppsett angir det for deg.
*   **Forhåndsvisning** / **Eksport**: demosaikkmetoden, som fyller inn fargene hver sensorpiksel ikke kan se. **Auto** er raskt på skjermen og bruker **AHD** ved eksport. **VNG** er jevn og **PPG** rask. **DCB** og **DHT** beholder flere fine detaljer, men kan gjøre korn om til mønstre. **AAHD** mykner kanter. Bare metodene i din LibRaw-versjon vises.
*   **Gjenoppretting** (dias): gjenoppbygger utbrente høylys i en kamera-RAW. **Blanding** passer sol og himmel. **Rekonstruer** er sterkere, men kan gi feil farge. Et nytt bilde fra filmskanneren starter med Gjenoppretting av.

<!-- panel:optics src:33b2d330 -->
### 10.3 Optikk

Korreksjoner for objektivet og lyset, de samme for hvert bilde du skanner med dette oppsettet. Vises for et kamera.

#### Objektivkorreksjon

*   **Innebygd**: bruker filens egen objektivkorreksjon (DNG, Sony ARW, Panasonic RW2). **Forvrengning** retter opp buede linjer og **CA** fjerner fargekanter. Angi det før du beskjærer eller retusjerer.
*   **Forvrengningskorreksjon**: retter opp linjer som buer utover (positiv) eller innover (negativ). Bruk filmkanten som en rett linje å gå etter.

#### Flatfeltkorreksjon: jevn ut lyset

Fjerner ujevnt lys, for eksempel mørkere hjørner, med et bilde av den bare lyskilden.

*   **Profil**: **+** lagrer et bilde av det bare lyset som en navngitt profil. Papirkurvknappen sletter den, og hvert bilde som bruker den, mister korreksjonen sin.
*   **Bruk flatfelt**: bruker den valgte profilen på denne rullen.
*   **Kontroller flatfelt**: viser hvor godt profilen jevner ut sitt eget bilde. Jevnt grått er bra; et bånd eller en flekk viser hvor den avviker.

Et bilde beholder sin egen profil med mindre det følger rullen, så hvert lys og filmformat kan ha én. Et nytt kamerabilde, eller et kamerabilde der profilen mangler, får den siste profilen du valgte på et kamerabilde. Et nytt bilde fra filmskanneren starter uten Optikk-korreksjoner. Med Filmskanner valgt vises dette kortet bare på et bilde som har en korreksjon.

<!-- panel:sensor src:3a992073 -->
### 10.4 Kalibrering: farger fra skanneoppsettet ditt

Korrigerer fargefeil som kommer fra lyset, kameraet og filmen din.

**Sensorkrysstaleprofil** (én eksponering under smalbåndslys, lineær RAW): fjerner fargen som lekker mellom sensorens kanaler. Kjør **Rullanalyse** på nytt etter en endring.

*   **Profil**: sensorkrysstaleprofilen. Dine egne `.toml`-profiler legges i `<Documents>/NegPy/sensor/`.
*   **Metode**: **To skalaer** (standard) fjerner lekkasjen og hindrer at svært sterke farger blir prikkete. **Lineær** er nøyaktig, men sterke farger kan bli prikkete. **Tetthet** gir ingen prikker, men sterke farger blir litt mattere.
*   **Kalibrer**: lager en profil fra tre bilder av det bare lyset (rødt, grønt og blått), fra filer eller et tilkoblet kamera (se CAMERA_SCANNING.md).

**Krysstale** (fargefilm): fargestoffene i filmen, lyset ditt og sensoren din blander fargene litt, og en matrise skiller dem igjen. Kjør **Rullanalyse** på nytt etter en endring.

*   **Matrise**: matrisene for denne filmtypen. De som er merket *(omtrentlig)*, kommer fra datablader for film og er et utgangspunkt. Dine egne `.toml`-filer legges i `<Documents>/NegPy/crosstalk/` (se CROSSTALK.md). Glidebryterknappen åpner redigeringsverktøyet; **+** lager en ny matrise. NegPy har ingen medfølgende matrise for dias.
*   **Styrke**: hvor mye av matrisen som brukes.
*   Fargeprøveknappen åpner **Mål fra kart** (fargenegativer): den måler en matrise fra et ColorChecker 24 eller SpyderCheckr 24 fotografert på filmen. Dra de fire prikkene til hjørnefeltene.

**Justering av fargetone** (kamera med smalbåndslys): dreier alle farger rundt fargehjulet, for et lys som gjør gult oransje eller grønt olivenfarget. Behold 0 for hvitt lys. Den virker også på dias, der sensorkrysstaleprofilen ikke gjør det.

#### 10.4.1 Fargegjenoppretting: falmede dias

Gir et falmet dias fargelagene tilbake.

*   **Rød overlevelse**: hvor mye rødt fargestoff som er igjen. Senk den til kontrasten kommer tilbake.
*   **Grønn overlevelse** / **Blå overlevelse**: det samme for grønt og blått, mot rødt.
*   **Fjern flekk**: gjør gulnede klare områder hvite, og kjører Beregn. Prøv det når Beregn gjør ikonet sitt gult.
*   **Beregn**: angir glidebryterne ut fra de grå tonene i dette bildet. **Mellomtone** passer et sterkt falmet dias. **Gråområde** passer en mild falming med dype skygger.

<!-- panel:process src:6572d2a7 -->
### 10.5 Lysmåling: negativ → positiv

For å gjøre et negativ om til et positiv måler NegPy de mørkeste og lyseste delene (svart- og hvitpunktet) og fargen på filmen. Disse kontrollene angir hva den måler.

*   **Mål alene**: tar dette bildet, eller hvert valgte bilde, ut av gjennomsnittet for rullen eller scenen og måler det for seg, fra den gjeldende beskjæringen.
*   **Tegn område**: tegn et område, så måler NegPy bare det området. Dobbeltklikk innenfor eller trykk `Enter` for å bekrefte, og `Esc` setter tilbake området du hadde før. **×** fjerner det.
*   **Analysebuffer**: ser bort fra en kant rundt bildet ved måling, så filmkanter og holdere ikke teller med.
*   **Klipping**: hvor mye av kopien som er helt svart eller helt hvit.
*   **Lås grenser**: beholder svart- og hvitpunktet for dette bildet, selv når beskjæringen eller rullanalysen endres.
*   **Luma-områdeklipping** (fold): hvor stramt svart og hvitt settes. Positiv gir mer kontrast, for tette eller tåkete negativer. Negativ beholder flere detaljer i skygger og høylys.
*   **Fargeklipping** (fold): hvor stramt fargebalansen settes.
*   **Dmin** / **Dmax** (folden Velg selv): for et bilde som NegPy måler dårlig. Klikk **Dmin**, og så filmbasen eller den dypeste skyggen, for å angi svartpunktet. Klikk **Dmax**, og så det lyseste høylyset, for å angi hvitpunktet. Hver **×** fjerner sitt eget punkt. Begge er av mens bildet kopieres etter rullens gjennomsnitt.
*   **Hvitpunkt** / **Svartpunkt** (fold): flytter det målte hvit- og svartpunktet. Et positivt hvitpunkt gjør kopien lysere; et positivt svartpunkt løfter svarttonene. **R / G / B** flytter bare én farge.

Et **dias** trenger ingen invertering: tonekontrollene former bildet direkte. Autotetthet og autogradering starter av; slå dem på for et falmet dias. Med Lightroom-begreper er Eksponering lik Kopitetthet (lavere er lysere), Kontrast lik ISO-R-gradering, og Skygger og Høylys lik Skyggetetthet og Høylystetthet.

<!-- panel:baseline src:7fc2c469 -->
### 10.6 Rullanalyse

Måler hele rullen én gang og deler resultatet, så bildene fra én film kopieres likt. NegPy måler rullen og hver scene etter hvert som bildene kommer inn, og på nytt etter en endring av beskjæring eller lysmåling. Ingen bilder bruker et grunnlag før du slår på et gjennomsnitt. Når ingen rull er åpen, er kortet nedtonet: åpne eller lagre en rull først.

*   **Analyser på nytt**: måler hvert synlig bilde utenfor en scene, lagrer gjennomsnittet som rullens grunnlag og slår på gjennomsnittene for rullen og de nye bildene. Bilder som er langt fra resten, beholder sine egne.
*   **Bruk dette bildet**: gjør dette bildets egen måling til rullens grunnlag; Fargestikk slås av.
*   **Bruk gjennomsnitt**: høyreklikk en av de tre for å slå hvert gjennomsnitt på eller av. **Luma** bruker rullens svart- og hvitpunkt, **Farge** rullens fargebalanse, og **Fargestikk** (fargenegativer) rullens gråbalanse for fjerning av fargestikk. Med Fargestikk på beholder et bilde tatt under et annet lys noe av det lyset, som på en Frontier-kopi. Et magentastikk blir værende, et varmt eller grønt stikk blir delvis værende, og et blått stikk forsvinner. Med Luma og Farge på kopieres rullen jevnt; med begge av måles hvert bilde for seg.
*   **Grunnlag**: viser hvor grunnlaget kom fra (rull, scene eller bilde). Den advarer når det ikke finnes noe, når det er utdatert, eller når fargen i dette bildet er langt fra gjennomsnittet.
*   **Ruller**: velg en annen rulls lagrede grunnlag. En rull med hake har et.
*   **Scener**: grupper av bilder med sitt eget grunnlag, for eksempel ett motiv under ett lys ([Scener](#scenes)). **Analyser** måler scenen, **Velg** velger bildene i den og **Slett** fjerner gruppen.
*   **Oppdag scener**: finner gruppene av bilder under et annet lys enn rullen, i hvilken som helst bilderekkefølge, og viser dem på lysbordet. Hver gruppe venter under **Funnet**. For å endre en gruppe drar du bilder til en annen gruppe, eller til et bilde uten scene for å utelate dem; høyreklikk › **Scene** gjør det samme. Gi gruppen et navn, og **Godta** den som en scene med eget grunnlag, eller **Forkast** den.

<!-- panel:assembly src:96b2e550 -->
### 10.7 Bildesammensetning

Hvordan filer blir til bilder. Hver rull beholder sine egne innstillinger her.

#### Trichrome

For et kamera med smalbåndslys som tar tre bilder av hvert bilde: rødt, grønt og blått.

*   **Trichrome-modus**: slår de tre opptakene sammen til ett fargebilde. Opptakene grupperes etter opptakstid (eller filnavn), så ta de tre opptakene av hvert bilde rett etter hverandre. Opptak som ikke grupperes, blir værende hver for seg.
*   **Slå sammen eksponeringer (Trichrome)** (gruppeikonet, eller høyreklikk): velg de tre opptakene av et bilde for å slå dem sammen. NegPy finner hvilket opptak som er rødt, grønt og blått.
*   **Rediger Trichrome…**: velg de tre opptakene av det gjeldende bildet selv. **Juster kanaler** retter dem inn mot hverandre for å fjerne fargekanter. **Del opp Trichrome** (ikonet for å dele opp gruppen, eller høyreklikk) skiller dem igjen.
*   **Slå sammen bilde til TIFF-negativ…** (høyreklikk): lagrer en triplett eller en sammensying som én TIFF ved siden av den første kildefilen, med redigeringen. Et alternativ flytter kildefilene til papirkurven.

#### Halvformat

**Halvformatmodus** deler hver skanning i to bilder, for halvformatkameraer. Hver halvdel har sin egen redigering. Hver rull beholder sin egen innstilling. Når ingen rull er åpen, gjelder den for bilder som ikke er i noen rull, til **Last ut alle**, og **Åpne forrige økt på nytt** beholder den. Den er nedtonet mens et bilde fra en rull er lastet inn, fordi det bildet følger sin egen rull.

*   **Juster…**: dra den hvite boksen for å beskjære og den oransje linjen for å angi delingen. **Delingsretning** og **Snittykkelse** angir mellomrommet, **Oppdag automatisk** finner dem igjen, og **Bruk** lagrer til dette bildet, utvalget eller alle bilder.
*   **Oppdag alle**: finner delingene i hver skanning på nytt.
*   **Del opp** (på et delt bilde): gjør det til ett bilde igjen og sletter redigeringene av begge halvdelene.

Høyreklikk en halvdel for **Juster delingen for dette bildet…** og **Tilbakestill delingen til rullens standard**. Når du slår av halvformat, blir redigerte halvdeler stående side om side som et **diptyk**; slå det på igjen for å redigere dem.

<!-- panel:metadata_gear src:208e1498 -->
### Analogt utstyr

*   **Kamera / Objektiv / Filmtype**: velg fra ditt eget utstyr (fanen **Mitt utstyr**). **Annet…** åpner hele den innebygde katalogen.
*   **Tøm**: tømmer alle tre.
*   **Utled fra mappenavnet**: fyller ut tomme felt (kamera, film, ISO, dato) ut fra mappenavnet.

<!-- panel:metadata_capture src:a84922a9 -->
### Opptak

*   **Dato**: når du tok bildet, som `1998`, `1998-07`, `1998-07-14` eller `1998-07-14 16:30`.
*   **Sted**: nålknappen åpner et kart. Du kan også lime inn koordinater eller en kartlenke. Kartet bruker OpenStreetMap.

<!-- panel:metadata_process src:9dd71862 -->
### Prosess

Hvordan filmen ble fremkalt.

*   **Lagret prosess**: en oppskrift fra Mitt utstyr som fyller ut feltene nedenfor. Når du skriver over et felt, kobles det fra.
*   **Filmformat**: filmstørrelsen, for eksempel `35mm`, `120` eller `4×5`.
*   **Fremkaller** og **Fortynning**: for eksempel `D-76` og `1+1`.
*   **Push / Pull**: fra `Push +3` til `Pull -3`.
*   **Tid** og **Temperatur (°C)**: tid som `9:30` eller i minutter.
*   **Tøm**: tømmer prosessfeltene.

<!-- panel:metadata_scanning src:eab4bcdf -->
### Skanning

*   **Lagret oppsett**: et skanneoppsett fra Mitt utstyr.
*   **Skanning**: hvordan du skannet, eller notater.
*   **Rull / Bilde**: rullnavnet og bildenummeret. Filnavn kan bruke dem (`{{ roll }}`, `{{ frame }}`).
*   **Sekvens…**: nummererer de valgte bildene, eller hele rullen, i filmrekkefølge. Angi start, trinn og retning; **Hopp over forkastede bilder** utelater forkastede.
*   **Tøm**: tømmer oppsettet og notatet.

<!-- panel:metadata_exposure src:9c0029fe -->
### Eksponering

Lukkertid, blender, ISO og kameraets eksponeringskompensasjon (EV) for det opprinnelige bildet, hvis du vet dem. Klikk låsen for å skrive fritekst som `1/125s f/2.8 ISO 400 -1/3ev`. En kompensasjon er blendertrinn foran `ev`, og summer som `1+2/3ev` legges sammen.

<!-- panel:metadata_preview src:3cf26bec -->
### Forhåndsvisning av metadata

Viser hva som legges i den eksporterte filen. Skjoldknappen, **Beskytt opprinnelige metadata**, eksporterer i stedet kildefilens egne metadata uendret. **Beskrivelse…** velger feltene som utgjør bildebeskrivelsen (kamera, objektiv, film og ISO i starten).

Kamera, objektiv og film legges i de vanlige EXIF-feltene, og skanneren din i NegPys egne XMP-felt. En stjernevurdering angitt i kameraet blir med.

<!-- panel:gear_items src:1c45c908 -->
### Mitt utstyr

Utstyrslisten din, som brukes av hver utstyrsvelger. **Kategori** velger **Kameraer**, **Objektiver**, **Filmtyper**, **Prosesser** (fremkallingsoppskrifter) eller **Skanneoppsett**. Den lagres i `~/NegPy/gear/`.

*   **+**: kopier et element fra den innebygde katalogen, eller **Legg til egendefinert** for å skrive inn et selv.
*   **Katalog**: vis også de innebygde modellene.
*   **Kopier / Papirkurv**: dupliser eller slett det valgte elementet. Innebygde elementer kan ikke slettes.

<!-- panel:gear_presets src:17a221ff -->
### Forhåndsinnstillinger

Lagrede sett med metadata, for eksempel ett kamera, ett objektiv og én film. De lagres i `~/NegPy/presets/metadata/`.

*   **Bruk på dette bildet**: skriver feltene fra forhåndsinnstillingen til dette bildet; andre felt blir stående.
*   **+**: lagrer metadataene for det gjeldende bildet under et navn.
*   **⋮** eller høyreklikk en forhåndsinnstilling: bruk, dupliser, rediger eller slett den.
*   **Detaljer**: rediger feltene i forhåndsinnstillingen. **Notater** er fritekst.

En filmtype angir formatet, så angi et format som `6×7` etter filmtypen.

<!-- panel:export_presets src:b7511145 -->
### Forhåndsinnstillinger

Lagrede eksportoppskrifter: format, størrelse, farge, mappe og filnavn. Slå på dem du vil ha, så skriver **Eksportforhåndsinnstillinger** hvert bilde én gang per forhåndsinnstilling. **Administrer** redigerer dem. Forhåndsinnstillinger virker ikke med lineært resultat.

<!-- panel:export_alongside src:ef285087 -->
### Ved siden av eksportene

Ekstra filer ved siden av kopien. Hver har en bryter til venstre og en engangshandling til høyre.

*   **Kopieringsnotater**: en merket kopi av bildet med skygging og etterbelysning og kopiinnstillinger, slik en mørkeromskopist tar vare på. **Forhåndsvisning** viser den (`Shift+N`); **Eksporter notater** lagrer den.
*   **Autotetthet / -gradering**: **Bak inn ved eksport** fastsetter den automatiske lysheten og kontrasten for hvert eksporterte bilde, så de ikke endres senere. **Bak inn i rullen** gjør det for hvert synlig bilde nå.
*   **Redigeringssidefiler**: **Lagre ved eksport** skriver en `.negpy`-fil med redigeringen ved siden av hver kilde. **Eksporter sidefiler** skriver dem for alle synlige bilder. Høyreklikk et bilde → **Last inn redigering fra sidefil…** leser en tilbake.

<!-- panel:contact_sheet src:ddc8d6ef -->
### Kontaktark

På fanen **Ved siden av eksportene**. Et kontaktark viser hele rullen som filmstriper på ett papirark, slik en kontaktkopi i mørkerommet gjør.

*   **Bane**: mappen for arkene. Tom bruker eksportmappen.
*   **Kontaktark…**: åpner arket. Dra en kant av papiret for å endre størrelsen.
*   **Filmformat**: 35 mm, 35 mm halvformat eller 120 (med **Bildestørrelse** for 120).
*   **Papir**, **Bredde**, **Høyde** og **DPI**: størrelsen og oppløsningen til arket. Standardpapiret har plass til en rull med 36 bilder.
*   **Rekkefølge**: etter filnavn, etter dato, etter rull og bildenummer, eller etter scene med en ny stripe eller rad for hver scene.
*   **Kopi**: **Som redigert**, eller **Rett prøve**: hele rullen med én eksponering og normal kontrast, så du ser hvilke negativer som er tynne eller tette.
*   **Velg bilder**: klikk et bilde for å utelate det eller sette det tilbake. Forkastede er utelatt i starten.
*   **Kantpåskrift**: produsentens merking langs kantene, for eksempel filmnavnet, bildenumrene og de fargede linjene en filmtype trykker ved perforeringen.
*   **Rulletikett**: rullnavnet, filmen, kameraet og datoen over stripene. Skriv din egen tekst i **Etikettekst** for å skrive den i stedet.
*   **Hvitt papir**: kopierer på hvitt, noe som sparer blekk på en hjemmeskriver.
*   **Filmkanter**: av kopierer bildene stående, hvert i sin egen form, i rader som på lysbordet. Arket har da ingen perforering og ingen **Kantpåskrift**.

Arkene lagres som `contact_sheet.jpg` (`contact_sheet_1of2.jpg` og videre for en lang rull).

<!-- panel:soft_proof src:0254272b -->
### Prøvetrykk på skjerm

Fanen **Prøve** viser på skjermen hvordan bildet vil se ut når det skrives ut på en gitt skriver og et gitt papir. Den endrer bare forhåndsvisningen, aldri eksporten. Papir kan ikke vise farger like klare som en skjerm, så et riktig prøvetrykk ser mattere ut.

*   **Prøvetrykk på skjerm** (`Shift+P`): slår prøvetrykket på eller av. Mens det er på, viser bildeinfoen over bildet **Prøve** og profilen.
*   **Forhåndsinnstilling**: et lagret oppsett for skriver og papir.
*   **Profil**: skriver- eller papirprofilen. Den følger eksportprofilen til du velger en.
*   **Hensikt**: **Relativ kolorimetrisk** holder farger som kan gjengis, nøyaktige og klipper resten. **Perseptuell** komprimerer alle farger, noe som passer svært fargerike bilder. **Metning** er for diagrammer og grafikk.
*   **Svartpunktkompensasjon**: tilordner den mørkeste tonen til papirets svart.
*   **Simuler papirhvitt** og **Simuler blekksvart**: viser papirets virkelige hvitt og svart.
*   **Advarsel om fargeomfang**: viser farger skriveren ikke kan gjengi, som grått.
*   **Skjerm**: profilen til skjermen din, funnet automatisk.

<!-- panel:scan_source src:939ed7bb -->
### Skanner

*   **Skanner**: **Filmskanner** for en egen filmskanner (kortene Enhet, Film og kvalitet og Innramming), eller **Kamera** for et kamera på et reprostativ (kortene Kamera og Forhåndsinnstilling).

<!-- panel:scan_device src:a18b5f39 -->
### Enhet

*   **Motor**: **SANE** (Linux og macOS), **Nikon Coolscan (nkscan)** eller **pyOpticfilm (Plustek)**. Kontrollene følger det skanneren rapporterer.
*   **Enhet**: skanneren; pilene oppdaterer listen. Utmatingsmenyen har **Mat ut nå** og, på en stripemater, **Mat ut når ferdig**, som returnerer stripen etter en satsvis kjøring.
*   **Feilsøkingslogg** (Coolscan): skriver `nkscan.log` i NegPy-mappen. For en feilrapport angir du **Trace**, gjentar problemet og legger ved filen.

NegPy holder macOS og Windows våkne mens den skanner. På Linux slår du av automatisk hvilemodus før en lang rull.

**Plustek**: OpticFilm 8200i SE, 8100 V2, 8300i SE, 7600i v1 og 7600i v2 virker; for andre modeller kan du prøve SANE. På Windows setter du først opp USB-driveren ([PLUSTEK_WINDOWS.md](PLUSTEK_WINDOWS.md)). **Coolscan**: trenger ikke SANE. På Linux krever USB en udev-regel for leverandøren `04b0`.

<!-- panel:scan_quality src:0e35bd9a -->
### Film og kvalitet

*   **Film**: fargenegativ, S/H-negativ, dias eller Kodachrome. Den styrer bildeoppdaging, lysmåling, og om infrarødt tilbys.
*   **Filmformat**: bildelengden (135, 66, 645 og andre). **Auto** virker der holderen bestemmer den.
*   **Oppløsning** og **Bitdybde**: skanningens dpi og bit per kanal.
*   **Skannemodus** (Plustek): **Flerpassering** tar gjennomsnittet av flere skanninger for mindre støy; **Adaptiv multieksponering** kombinerer en kort og en lang eksponering for større omfang, med den lange eksponeringen angitt ut fra hvert bilde; **Fast multieksponering** bruker skannerens ene lange eksponering på hvert bilde, slik SilverFast gjør (8200i SE, 8100 V2 og 8300i SE). 7600i v2 tilbyr adaptiv multieksponering, men ikke flerpassering.
*   **Prøver** (Coolscan): leser hver linje flere ganger, for mindre støy i skyggene. Tregere.
*   **Infrarød**: **IR** skanner en infrarød kanal, for støvfjerning på kortet Retusj. **ICE** (Coolscan) fjerner i stedet støvet i skanneren. En skanner uten ICE viser én **IR**-bryter. Ikke for S/H eller Kodachrome.
*   **Superfin** (Coolscan): skanner én linje per passering. Tregere.
*   **Autofokus**, **Autoeksponering** og **Eksponering**: fokus og eksponering på skanneren, før skanningen.

<!-- panel:scan_framing src:945dad86 -->
### Innramming

*   **Bilder**: `1-6`, `1,2,5`, eller tomt for alle. Stripeforhåndsvisningen fyller det ut.
*   **Forhåndsvis stripe…** (stripemater): forhåndsviser hvert bilde, for å plassere og velge bildene. **Forhåndsvis…** (manuell holder): angir ett beskjæringsvindu. **Tøm** tilbakestiller det. **Oppdag bilder** finner bildene på nytt etter at filmen har flyttet seg.
*   **Mål bilde…** (Coolscan): måler eksponeringen på ett bilde og bruker den for hver senere skanning, så alle bildene i en rull samsvarer. **Lås opp** avslutter det. Mål på nytt for hver rull.
*   **Forhåndsskann…** (Plustek): en rask forhåndsvisning av hele området. Dra en beskjæring, så leser neste skanning bare det området.

<!-- panel:scan_camera src:0145e5d1 -->
### Kamera

Skanner med et kamera på et reprostativ, koblet til med USB i **PC Remote**-modus (macOS og Linux; det krever `python-gphoto2`). Med en NegPy **Scanlight** styrer NegPy også lyset. CAMERA_SCANNING.md forklarer oppsettet.

*   **Kamera** / **Lys**: tilkoblingsstatusen. Lyset blir gult når det går varmt. På Linux krever Scanlight at brukeren din er i gruppen `dialout`.
*   **Direktevisning**: et levende bilde fra kameraet. Klikk for å plassere fokusluppen, og fokuser til **Fokusmåler** viser **på topp**. Angi ISO, lukkertid og blender her eller på kameraet. Når den lukkes, går Scanlight tilbake til slik den var før.
*   **Histogram** og **Sebraer**: histogrammet for bildet, og striper på deler som er i ferd med å klippes, i klippefargene som Fargesyn angir.
*   **Måler**: angir lukkertiden slik at den lyseste delen som ikke må klippes (den klare filmen i et negativ, høylysene i et dias), havner like under klipping. Mål før første skanning. Uten en kalibrert forhåndsinnstilling bruker en skanning eksponeringen som er angitt på kameraet, og NegPy spør før den skanner uten lysmåling.
*   **Skann** tar neste bilde og importerer det; **Ta på nytt** (gjør om-pilen) tar det siste bildet igjen. Når en skanning klipper, forteller statuslinjen hvor mye av bildet som ble klippet. I vinduet Direktevisning skanner `Space` eller `S`, og `R` tar på nytt. Med en Scanlight velger **R**, **G**, **B** og **IR** hvilke eksponeringer Ta på nytt gjentar.
*   **Opptaksforsinkelse**: venter før den første eksponeringen, så et ustøtt bord rekker å falle til ro.
*   **La lyset være på** (pæren): lar Scanlight lyse etter hver skanning, så du kan ramme inn neste bilde.
*   **Film** (med en Lento Dock): hva holderen gjør når du klikker Skann. **Neste bilde** flytter filmen ett bilde etter hver skanning. **Hele stripen** skanner hvert bilde og mater så ut stripen; Stopp avslutter det. Ta på nytt flytter aldri filmen. **Mat ut** flytter stripen ut. S-knappen på holderen virker som Skann.
*   **Ta flatfelt**: når en hel stripe er matet ut, fotograferes den tomme holderen og lagres som rullens flatfeltprofil.

<!-- panel:scan_light src:417cc437 -->
### Forhåndsinnstilling

Vises mens en Scanlight er tilkoblet. En forhåndsinnstilling inneholder lysnivåene og kameraeksponeringen for én type film.

*   **Forhåndsinnstilling**: lyset og eksponeringen for hver skanning. **+** legger til en på tre måter:
    *   **Innebygde forhåndsinnstillinger**: klare til bruk, uten kalibrering: **Hvitt lys**, **Hvitt lys + IR**, **Trichrome** og **Trichrome + IR**. De bruker kameraets eksponering, som måleren angir. Hvitt lys krever en Scanlight med hvit LED (v4, Big Scanlight).
    *   **Kalibrer på filmbasen…**: plasser rektangelet på klar film og kjør den. NegPy finner lysnivåene og lukkertiden.
    *   **Angi selv**: angi glidebryterne selv, og lagre dem så som en forhåndsinnstilling. Uten tilkoblet kamera lagrer den ingen eksponering, og en skanning bruker kameraets egen.
*   **Rød**, **Grønn**, **Blå**, **Hvit**: lysnivåene (0 til 255). **Lys på** slår på lyset for å ramme inn filmen, eller av.
*   **Opptaksmodus**: **Trichrome** tar én eksponering per farge og slår dem sammen; **Én eksponering** tar én eksponering med alle tre fargene på. **Blanding** holder kameraet åpent i Bulb og tenner rødt, grønt og blått etter tur, for én fil. En Trichrome- eller Blanding-forhåndsinnstilling kan bytte mellom de to.
*   **Utløser** (Blanding): hva som holder lukkeren åpen. **USB** bruker kameraets egen bulb-styring. **Kabel** bruker en utløserkabel i Scanlights 3,5 mm-kontakt, for eksponeringer opptil 2,55 s.
*   **Sensorprofil** (Blanding-forhåndsinnstilling uten egen profil): sensorkrysstaleprofilen for skanningene. Skanningen venter på en. Knappen med glassene lager en.
*   **Lag sensorkrysstaleprofil** (Én eksponering eller Blanding): lagrer også en sensorkrysstaleprofil som skanningene med denne forhåndsinnstillingen bruker (se Kalibrering, §10.4).
*   **ISO**, **Lukker**, **Blender**: eksponeringen for forhåndsinnstillingen. En endring gjelder til du velger forhåndsinnstillingen igjen. Måleknappen måler.
*   **IR-lukker** og **Infrarød** (lys med infrarøde LED-er): legger til en infrarød eksponering i hver skanning for støvfjerning, lagret som en `_IR`-fil. Det krever et kamera som ser infrarødt (fullspekter eller konvertert) og manuell fokus. Bare fargefilm.
*   **Kanalforsinkelse**: en pause mellom den røde, grønne og blå eksponeringen, for kameraer som trenger det.

<!-- panel:scan_output src:3015443a -->
### Utdata

Felles for begge skannerne.

*   **Mappe**: hvor skanninger lagres.
*   **Skann som rull**: mappen åpnes som en rull, og bildene lastes inn etter hvert som de lagres.
*   **Mappe som rull**: skanninger legges i selve mappen. Av skriver inn i en undermappe **Rull**, og et nytt rullnavn starter en ny rull. **+** ved siden av Rull går til neste navn (`Roll001` til `Roll002`).
*   **Format** og **Filnavn** (filmskanner): TIFF, eller TIFF (mono) for S/H, og filnavnmønsteret (`{{ date }}`, `{{ seq }}`).

<!-- panel:scan_strip src:135ca0e1 -->
### Stripeforhåndsvisning

Viser hele stripen, for å plassere og velge bildene før skanningen. **Bruk innramming** beholder innrammingen og **Skann** skanner. **Stopp forhåndsvisning** avslutter en forhåndsvisning tidlig.

*   **Beskjær**: dra på et bilde. **Fjern beskjæringer** fjerner alle beskjæringer.
*   **Forskyvning**: flytter hvert bilde langs filmen for å rette det inn etter bildene. Den hvite boksen viser bildet NegPy fant.
*   **Avdrift**: en forskyvning som vokser fra bilde til bilde.
*   Glidebryteren under hver flis flytter bare det bildet.
*   **Størrelse** angir flisstørrelsen. **Alle**, **Ingen** og bryteren for hvert bilde velger bilder. **Forhåndsvis bilde** skanner ett bilde på nytt.
