<#
  run_dryrun_keepawake.ps1
  run_pipeline.ps1 로 위임하는 얇은 래퍼다. 실행 로직을 한 곳에 두어야
  버전이 올라갈 때 한쪽만 고쳐지는 사고를 막는다. 상세 설명은 run_pipeline.ps1 참고.
#>
param(
    [Parameter(Mandatory = $true)] [string]$RawRoot,
    [Parameter(Mandatory = $true)] [string]$WorkDir,
    [int]$Seed = 42,
    [int]$Jobs = 8,
    [switch]$KeepDisplay
)
$forward = @{ Stage = "dry-run"; RawRoot = $RawRoot; WorkDir = $WorkDir; Seed = $Seed; Jobs = $Jobs }
if ($KeepDisplay) { $forward.KeepDisplay = $true }
& (Join-Path $PSScriptRoot "run_pipeline.ps1") @forward
exit $LASTEXITCODE
