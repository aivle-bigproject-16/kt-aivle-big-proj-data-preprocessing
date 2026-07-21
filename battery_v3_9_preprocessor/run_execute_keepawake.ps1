<#
  run_execute_keepawake.ps1
  run_pipeline.ps1 로 위임하는 얇은 래퍼다. approve-selection 이 먼저 끝나 있어야 한다.
  상세 설명은 run_pipeline.ps1 참고.
#>
param(
    [Parameter(Mandatory = $true)] [string]$RawRoot,
    [Parameter(Mandatory = $true)] [string]$WorkDir,
    [Parameter(Mandatory = $true)] [string]$Output,
    [int]$Seed = 42,
    [int]$Jobs = 8,
    [switch]$KeepDisplay
)
$forward = @{ Stage = "execute"; RawRoot = $RawRoot; WorkDir = $WorkDir; Output = $Output; Seed = $Seed; Jobs = $Jobs }
if ($KeepDisplay) { $forward.KeepDisplay = $true }
& (Join-Path $PSScriptRoot "run_pipeline.ps1") @forward
exit $LASTEXITCODE