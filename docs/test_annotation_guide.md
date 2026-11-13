# Fortsæt gennemgangen af alle 1.500 frames

Denne guide dokumenterer **testklip 1** og det tidligere forløb. For det nye Danmark–Wales-klip bruges [test_2_annotation_guide.md](test_2_annotation_guide.md) og `Annotate-Test2.ps1`. Den tidligere begrænsning til én testvideo gælder ikke den brugerbestilte udvidelse af 19. oktober 2026.

**Status 6. oktober 2026:** Alle 1.500 frames og den efterfølgende målrettede gennemgang af 31 kvalitetsflag er afsluttet. Brugeren ændrede 22 labels og bekræftede 9; frame 1420 er rettet til synlig bold. Nye resultater ligger i `results/test_reviewed/` med korrigeret reference og før/efter-log. De oprindelige resultater og backup er bevaret. `--pending-only` har ingen resterende frames at vise. Modeller, predictions, parametre og vægte er uændrede. Afsnittet nedenfor beskriver den tidligere udvidelse og værktøjets betjening.

Kilden er brugerens valgte https://www.youtube.com/watch?v=t0GESlaVNdE, lokalt `data/test/source_video.mp4`. Projektets eneste sluttest er `data/test/clip.mp4`: intervallet **[00:00.000, 01:00.000)** med alle 1.500 frames ved 25 fps, 1350 × 1080 pixels. Videoen og dens hashes er uændrede. Der vælges ingen anden testvideo, og SoccerNet bruges fortsat kun til træning og separat validering.

Gennemgangen er udvidet til **0, 1, 2, …, 1499**: alle 1.500 frames med 0,04 sekunders mellemrum, altså 25 Hz. Dine 300 godkendte annotationer fra stride 5 er bevaret uændret. De resterende **1.200 frames** skal gennemgås manuelt. Værktøjet viser kun ikke-gennemgåede frames, først frame 1, og springer alle godkendte frames over. Begge metoder behandler og evalueres efter afsluttet review på det samme fulde klip.

## Start eller genoptag

Kør præcis dette i PowerShell:

```powershell
# Run from the repository root
& '.\.venv\Scripts\python.exe' -m balltrack annotate --video data/test/clip.mp4 --output data/test/annotations.csv --stride 1 --pending-only
```

`Annotate-Test.ps1` udfører samme start. Brug ét annotationsvindue ad gangen. Værktøjet genoptager ved den første ikke-gennemgåede frame. Kommandoen nulstiller intet. Godkendte frames vises ikke med `--pending-only`, heller ikke ved frem/tilbage eller afspilning. Efter sidste godkendelse gemmes og lukkes automatisk. Hvis alt er gennemgået ved næste opstart, vises blot en besked. Hvis en allerede godkendt annotation senere skal rettes, kan samme kommando køres uden `--pending-only` for adgang til alle frames.

| Handling | Betjening |
|---|---|
| Tegn eller ret en boks som forslag | Træk med venstre museknap |
| Zoom før tegning / vis hele billedet | Højreklik / højreklik igen |
| Gem og godkend boks eller status; næste ikke-gennemgåede frame | `r` |
| Gem okkluderet; næste ikke-gennemgåede frame | `o` |
| Gem usikker; næste ikke-gennemgåede frame | `u` |
| Gem uden for billedet; næste ikke-gennemgåede frame | `f` |
| Forrige / næste ikke-gennemgåede frame uden godkendelse | `a` / `d` eller mellemrum |
| Næste ikke-gennemgåede frame | `n` |
| Afspil/pause ikke-gennemgåede frames uden godkendelse | `p` |
| Nulstil aktuel boks/status til umærket | `x` |
| Gem og luk | `q` |

En tegnet boks gemmes straks, men kræver **`r`** for godkendelse. Det gælder også, hvis du lukker og genoptager senere. `r` kræver en eksisterende boks eller status; umærkede frames springes ikke automatisk over. `o/u/f` godkender din synlighedsbeslutning med det samme og fjerner eventuelle koordinater. Når der ikke er flere ikke-gennemgåede frames fremme, fortsætter godkendelse ved eventuelle oversprungne frames tidligere i klippet. Orange bokse er forslag. Gennemgå hvert billede visuelt; hold ikke godkendelsestasten nede.

## Backup og bevarelse af færdigt arbejde

Hvis CSV eller metadata ændres, mens et annotationsvindue er åbent, stopper værktøjet nu uden at overskrive de ændrede filer. Vinduets egen kopi gemmes i `data/test/backups/*_session_conflict/`. Luk gamle vinduer og genstart; brug altid `clip.mp4`. Lukning uden ændringer omskriver ikke længere CSV'en. Ved reparationen af en CSV, der var skrevet tilbage til 300 rækker, blev en ekstra backup gemt i `data/test/backups/20261005T123228742890Z_schedule_repair/`; alle 300 godkendelser blev bevaret.

Før udvidelsen blev den færdige `annotations.csv` med 300 rækker og 300 manuelle godkendelser kopieret byte for byte til:

`data/test/backups/20261005T122128530466Z_before_stride1/annotations.csv`

Mappen indeholder også metadata, auditlog, fremdrift, konfiguration og klipregistrering samt `backup_record.json` med hashes. Den aktive CSV har nu alle 1.500 rækker. De 300 færdige annotationer er bevaret med geometri, synlighed, kilde og godkendelse. Forslag til de andre 1.200 frames er hentet fra den tidligere fulde backup `20261005T084407037554Z_before_stride5`; alle tilføjede rækker har `reviewed=0`, også hvis den gamle backup havde godkendelser. Udvidelsen er registreret i auditloggen. Ingen model er kørt for at generere labels.

Assistentforslag og lineær interpolation er kun hjælp til gennemgangen. Efter din konkrete kontrol gemmer `r` en manuel godkendelse med `source=manual_review`; auditloggen bevarer den tidligere kilde, geometri og status. `o/u/f` bruger `source=manual`. Slutevalueringen kræver `reviewed=1` og en af disse to manuelle kilder. Rå `interpolated_linear`- eller assistentforslag kan ikke tælle som facit, heller ikke hvis review-flaget alene sættes til 1.

Valgfri interpolation med `b` ved en synlig startboks og `i` ved en synlig slutboks kan kun udfylde umærkede mellemframes. Den må ikke overskrive dine godkendelser. Alle genererede bokse kræver individuel kontrol med `r`; hver af de 1.200 tilføjede frames skal kontrolleres på sit faktiske billede.

## Synlighed og kontrol

Tegn en tæt boks omkring den synlige bold eller identificerbar bevægelsessløring. En delvist skjult, men lokaliserbar bold kan markeres synlig efter samme konsekvente regel. Brug `o` når bolden er skjult, `f` når den er uden for billedet, og `u` når tilstedeværelse eller placering ikke kan afgøres. Gæt ikke skjulte koordinater.

Klippet indeholder spil, frispark, et skud, nærbilleder/publikum, jubel og starten af en replay. Behold alle billeder i videoen. Begyndelsens bold mod publikum, hænder/ben omkring 14–17 sekunder, skuddet omkring 32–35 sekunder og replay fra cirka 56 sekunder kræver særlig opmærksomhed. Brug zoom; de to metoders predictions vises ikke under annotation. Kontrollér udvalgte synlighedsovergange og genkontrollér et tilfældigt udsnit på 10% af de synlige samples før afslutning.

```powershell
& '.\.venv\Scripts\python.exe' -m balltrack check-annotations --annotations data/test/annotations.csv --require-complete
```

Kontrollen kræver alle 1.500 manuelle gennemgange, korrekte bokse/timestamps, den registrerede video og stride 1. `Evaluate-Test.ps1` udfører kontrollen før tracking og sammenligner begge metoder på præcis disse annotationer. Usikre frames tæller som gennemgåede, men udelades fra metrikkerne og optælles separat. Metodeparametre og modelvægte holdes uændrede; deres hashes er gemt i klipregistreringen.

## Hvad resultaterne kan vise

Begge metoder evalueres kvantitativt på præcis samme 1.500 manuelt gennemgåede annotationer. Prediktionsfilerne indeholder alle 1.500 frames, og hastighedsmålinger og resultatvideoer dækker hele minuttet. Ikke-gennemgåede eller rå interpolerede bokse må aldrig tælle som facit.

Stride 1 gør fejl og synlighedsskift målbare på hver optaget frame. En tabt periode er antal fejlede synlige frames gange 0,04 sekunder. Hændelser mellem videoens frames er stadig uobserverede. Naboframes i ét redigeret TV-klip er korrelerede; replay, stillestående bold og annoteringsusikkerhed begrænser repræsentativiteten. Sluttestbilleder og sluttestfejl må ikke bruges til metode- eller parametervalg.
