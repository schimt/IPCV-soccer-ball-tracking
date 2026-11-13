# Testklip 2: start og genoptag annotation

**Flyttet 27. oktober 2026:** Projektet ligger nu i repoets rodmappe (`.`). Absolutte stier i `configs/two_clip_experiment.json` er gjort relative (hashes uændrede); se `results/two_clip_experiment/path_migration.md`. Næste skridt er stadig efterkontrollen: `powershell -NoProfile -ExecutionPolicy Bypass -File .\Review-Test2Quality.ps1`.

**Opdateret efter genoptagelse:** Alle 46 prioriterede frames er bekræftet. Den almindelige efterkontrol har stadig resterende frames; `Review-Test2Quality.ps1` viser det aktuelle antal. En fejl, som genudtrak stikprøven efter statusrettelser, er rettet: den oprindelige stikprøve bevares nu. Kun nye synlighedsskift kan føje frames til. Alle dine tidligere bekræftelser og rettelser er bevaret. Ændringen vedrører kun efterkontrollens kø, ikke metoder eller evalueringskriterier.

**Efterkontrol 19. oktober:** Første gennemgang er afsluttet (1.500/1.500). Se `results/two_clip_experiment/annotation_quality/README.md` for konkrete mistænkelige labels og backup. Start med `powershell -NoProfile -ExecutionPolicy Bypass -File .\Review-Test2Quality.ps1 -Priority` (46 frames), og kør derefter samme kommando uden `-Priority` for de resterende frames i den samlede liste på 294. Disse nye kommandoer genoptager også efterkontrollen ved at springe allerede efterkontrollerede, uændrede frames over. Førstegangsgennemgangen nedenfor er historik; annotationerne er ikke nulstillet.

Videoen er kopieret uden ændringer fra `new_video/denmark_wales_clip_1min.mp4` til `data/test_2/denmark_wales_clip_1min.mp4`. Den separate fil `data/test_2/annotations.csv` indeholder frames 0-1499 med stride 1. Klip 1 og dets annotationer er bevaret.

Åbn PowerShell og kør:

```powershell
# Kør fra repoets rodmappe
powershell -NoProfile -ExecutionPolicy Bypass -File .\Annotate-Test2.ps1
```

Alternativt direkte, uden PowerShell-script:

```powershell
.\.venv\Scripts\python.exe -m balltrack annotate --video data/test_2/denmark_wales_clip_1min.mp4 --output data/test_2/annotations.csv --stride 1 --pending-only
```

Træk med venstre museknap for at tegne eller rette boksen. Højreklik zoomer ind/ud. `r` godkender boksen og går videre. `o` = okkluderet, `u` = usikker og `f` = uden for billedet; disse taster gemmer status og går videre. `q` gemmer og lukker. Samme kommando genoptager første ikke-gennemgåede frame. Orange boks og `PENDING` betyder ikke-godkendt. `x` nulstiller; `a`/`d` navigerer, `n` finder næste manglende gennemgang. Gæt aldrig skjulte positioner. Ved tvivl bruges `u`. Der er ingen automatisk foreslåede bokse i den nye fil.

Pending-only springer allerede godkendte frames over. Hvis en godkendt frame skal rettes, kør den direkte kommando uden `--pending-only` og navigér med `a`/`d`; alle andre annotationer bevares. Luk den gamle annotationssession før en ny åbnes.

## Efter de 1500 frames

Kør først denne kommando. Den afviser ufuldstændige eller ikke-manuelt godkendte annotationer og opretter en separat kvalitetsliste uden modelpredictions:

```powershell
.\.venv\Scripts\python.exe -m balltrack.two_clip quality-plan
powershell -NoProfile -ExecutionPolicy Bypass -File .\Annotate-Test2.ps1 -QualityReview
```

Listen omfatter første/sidste frame, begge sider af alle synlighedsskift og tilfældige 10 % af synlige frames (seed 42). Gennemgå listen på den rå video: `r` bekræfter, eller ret boks/status. Listen bruger det eksisterende annotationsværktøj; den nulstiller ingen godkendelser. Ved genstart af denne ekstra liste vises første listeframe igen, fordi første gennemgang allerede er komplet; navigér med `d` til den ønskede listeframe. Auditloggen bevarer allerede udført kvalitetskontrol.

```powershell
.\.venv\Scripts\python.exe -m balltrack.two_clip check
```

Hvis rettelser giver nye synlighedsskift, tilføjer kontrollen dem til listen. Åbn `-QualityReview` igen og bekræft de tilføjede frames; kør derefter `check` igen. Først når kontrollen består, er referencegrundlaget klar.

## Evaluering efter afsluttet manuel kontrol

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\Evaluate-TwoClips.ps1
```

Kommandoen kontrollerer hashes, alle labels og kvalitetsloggen før inference. Den behandler hele klip 2 med de fire fastlåste versioner uden manuelle rettelser. Den genbruger hele klip 1's bevarede predictions og historiske FPS. Der køres ingen træning eller tuning. Kør uden andre tunge GPU/CPU-opgaver. Et allerede påbegyndt outputforløb overskrives ikke; ved afbrydelse bevares filerne til inspektion.

Resultater: `results/two_clip_experiment/completed/comparison.csv` med begge klip separat og `comparison_all_modes.csv` med de sekundære observed-only-mål. Predictions og resultatvideoer: `runs/two_clip_experiment/test_2/`. Separate GT-videoer og frameoversigter: `results/two_clip_experiment/completed/test_1/` og `test_2/`. De indeholder aldrig opfundet GT for skjulte/usikre positioner.

Efter evaluering opdateres tabeller/rapporttekst med `.venv/Scripts/python.exe -m balltrack.two_clip_report`. Den visuelle fejlanalyse og PDF-kontrollen færdiggøres derefter med de faktiske outputs. Indtil da er klip 2 tydeligt markeret som afventende i rapporten. De fulde fastlåste regler findes i `docs/two_clip_protocol.md`.
