param(
    [string]$Output = "models\reinvent4\mol2mol_medium_similarity.prior"
)

$ErrorActionPreference = "Stop"
$url = "https://zenodo.org/api/records/15641297/files/mol2mol_medium_similarity.prior/content"
$target = [IO.Path]::GetFullPath($Output)
$parent = Split-Path -Parent $target
New-Item -ItemType Directory -Path $parent -Force | Out-Null

if (Test-Path $target) {
    Write-Host "Prior already exists: $target"
    exit 0
}

Invoke-WebRequest -Uri $url -OutFile $target -UseBasicParsing
Write-Host "Downloaded official REINVENT4 Mol2Mol prior to $target"
