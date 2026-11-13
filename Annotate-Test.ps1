$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
& '.\.venv\Scripts\python.exe' -m balltrack annotate --video data/test/clip.mp4 --output data/test/annotations.csv --stride 1 --pending-only
if ($LASTEXITCODE -ne 0) { throw 'Annotation tool failed; saved CSV annotations are preserved.' }
