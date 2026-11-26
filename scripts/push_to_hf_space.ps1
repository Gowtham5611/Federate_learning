# Push ONLY backend + Dockerfile + Space README.md to Hugging Face Spaces
param(
    [Parameter(Mandatory = $true)][string]$HfUsername,
    [Parameter(Mandatory = $true)][string]$SpaceName,
    [Parameter(Mandatory = $true)][string]$HfToken
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$StageDir = Join-Path ([System.IO.Path]::GetTempPath()) "hf_space_stage_$(Get-Random)"

Write-Host "Staging backend-only files to $StageDir ..."
New-Item -ItemType Directory -Path $StageDir -Force | Out-Null

Copy-Item (Join-Path $RepoRoot "Dockerfile") $StageDir
Copy-Item (Join-Path $RepoRoot ".dockerignore") $StageDir
Copy-Item (Join-Path $RepoRoot "asgi.py") $StageDir
Copy-Item (Join-Path $RepoRoot "requirements.txt") $StageDir
Copy-Item (Join-Path $RepoRoot "deploy\hf_space\README.md") (Join-Path $StageDir "README.md")

# Copy backend excluding heavy raw/archive/test folders not needed at runtime
$BackendDest = Join-Path $StageDir "backend"
robocopy (Join-Path $RepoRoot "backend") $BackendDest /E /XD __pycache__ .pytest_cache raw archive scratch experiments tests "colab work and results" "Colab Results" | Out-Null

Push-Location $StageDir
try {
    git init -b main
    git config user.name $HfUsername
    git config user.email "$HfUsername@users.noreply.huggingface.co"
    git add -A
    git commit -m "Deploy FedSecureAI Backend to Hugging Face Spaces"
    $RemoteUrl = "https://${HfUsername}:${HfToken}@huggingface.co/spaces/${HfUsername}/${SpaceName}"
    git push --force $RemoteUrl main
    Write-Host "Successfully pushed to https://huggingface.co/spaces/${HfUsername}/${SpaceName}"
} finally {
    Pop-Location
    Remove-Item -Recurse -Force $StageDir -ErrorAction SilentlyContinue
}
