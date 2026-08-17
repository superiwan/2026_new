param(
    [Parameter(Mandatory = $true)]
    [string]$Onnx,
    [Parameter(Mandatory = $true)]
    [string]$Calibration,
    [string]$Output = "output/poker_corner_conversion_640",
    [ValidateRange(20, 200)]
    [int]$CalibrationCount = 100,
    [ValidatePattern('^[A-Za-z0-9._-]+$')]
    [string]$ArtifactName = "poker_corner_yolo11n_640_int8_v1"
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$stagingName = "poker_corner_conversion_input_$PID"
$staging = Join-Path $repo "output/$stagingName"
$calibrationTarget = Join-Path $staging "calibration"
$outputPath = Join-Path $repo $Output

New-Item -ItemType Directory -Force -Path $staging | Out-Null
New-Item -ItemType Directory -Force -Path $calibrationTarget | Out-Null
New-Item -ItemType Directory -Force -Path $outputPath | Out-Null
Copy-Item -LiteralPath (Resolve-Path $Onnx).Path `
    -Destination (Join-Path $staging "model.onnx")
Get-ChildItem -LiteralPath (Resolve-Path $Calibration).Path -File |
    Where-Object { $_.Extension -match '^\.(jpg|jpeg|png)$' } |
    Sort-Object Name |
    Select-Object -First $CalibrationCount |
    Copy-Item -Destination $calibrationTarget

$calibrationCount = (Get-ChildItem -LiteralPath $calibrationTarget -File).Count
if ($calibrationCount -lt 20) {
    throw "At least 20 calibration images are required; found $calibrationCount"
}

$repoDocker = $repo -replace '\\', '/'
docker run --privileged --rm `
    --volume "${repoDocker}:/workspace" `
    --workdir /workspace `
    sophgo/tpuc_dev:latest `
    bash -lc "pip install -q tpu_mlir onnxsim && tr -d '\r' < scripts/convert_poker_corner_cv181x.sh > /tmp/convert_poker_corner_cv181x.sh && ARTIFACT_NAME='$ArtifactName' CALIBRATION_COUNT=$calibrationCount bash /tmp/convert_poker_corner_cv181x.sh output/$stagingName/model.onnx output/$stagingName/calibration '$Output'"

if ($LASTEXITCODE -ne 0) {
    throw "CV181x conversion failed with exit code $LASTEXITCODE"
}
