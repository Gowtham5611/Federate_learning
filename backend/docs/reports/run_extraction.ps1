# PowerShell Automation Runner for FedSecureAI Results Extraction
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "FedSecureAI - Extract and Audit All Results from Both Colab Folders" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

$PythonExe = ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

Write-Host "Executing extraction engine using: $PythonExe" -ForegroundColor Yellow
& $PythonExe results_extracted_audit/extract_results.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nExtraction and audit successfully completed!" -ForegroundColor Green
    $OutDir = Resolve-Path results_extracted_audit
    Write-Host "Output Directory: $OutDir" -ForegroundColor Green
    Write-Host "To open dashboard in default browser:" -ForegroundColor Cyan
    Write-Host "  Start-Process results_extracted_audit\results_dashboard.html" -ForegroundColor White
} else {
    Write-Host "`nExtraction encountered an error. Exit code: $LASTEXITCODE" -ForegroundColor Red
}

