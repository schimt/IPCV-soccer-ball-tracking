param([switch]$Priority, [switch]$ListOnly)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$reviewArgs = @('-m', 'balltrack.test2_review')
if ($Priority) { $reviewArgs += '--priority' }
if ($ListOnly) { $reviewArgs += '--list-only' }
& '.\.venv\Scripts\python.exe' @reviewArgs
if ($LASTEXITCODE -ne 0) { throw 'Quality review stopped; saved annotations are preserved.' }
