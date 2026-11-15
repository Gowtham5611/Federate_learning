Add-Type -AssemblyName System.Runtime.WindowsRuntime
Add-Type -AssemblyName System.Drawing

$null = [Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime]
$null = [Windows.Data.Pdf.PdfDocument, Windows.Data.Pdf, ContentType = WindowsRuntime]

function AwaitOp($asyncOp, $type) {
    $asTask = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
        $_.Name -eq 'AsTask' -and 
        $_.GetParameters().Count -eq 1 -and 
        $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
    }
    $method = $asTask.MakeGenericMethod($type)
    $task = $method.Invoke($null, @($asyncOp))
    $task.Wait()
    return $task.Result
}

function AwaitAction($asyncAction) {
    $asTask = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
        $_.Name -eq 'AsTask' -and 
        $_.GetParameters().Count -eq 1 -and 
        $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncAction'
    }
    $task = $asTask.Invoke($null, @($asyncAction))
    $task.Wait()
}

$pdfPath = (Resolve-Path "results/uploads/final_demo_report_pred_1794301880_85a48325.pdf").Path
$storageFile = AwaitOp ([Windows.Storage.StorageFile]::GetFileFromPathAsync($pdfPath)) ([Windows.Storage.StorageFile])
$pdfDoc = AwaitOp ([Windows.Data.Pdf.PdfDocument]::LoadFromFileAsync($storageFile)) ([Windows.Data.Pdf.PdfDocument])

Write-Output "PDF Page Count: $($pdfDoc.PageCount)"

$outDir = "results/pdf_pages"
if (-not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Path $outDir | Out-Null
}

for ($i = 0; $i -lt $pdfDoc.PageCount; $i++) {
    $page = $pdfDoc.GetPage($i)
    $memStream = New-Object Windows.Storage.Streams.InMemoryRandomAccessStream
    AwaitAction ($page.RenderToStreamAsync($memStream))
    
    $netStream = [System.IO.WindowsRuntimeStreamExtensions]::AsStreamForRead($memStream)
    $img = [System.Drawing.Image]::FromStream($netStream)
    $outPath = Join-Path $outDir "page_$($i + 1).png"
    $img.Save($outPath, [System.Drawing.Imaging.ImageFormat]::Png)
    $img.Dispose()
    $netStream.Dispose()
    $memStream.Dispose()
    $page.Dispose()
    Write-Output "Saved page $($i + 1) to $outPath"
}
