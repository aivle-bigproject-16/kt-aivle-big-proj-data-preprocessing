<#
  run_execute_keepawake.ps1
  execute(실제 데이터셋 생성)를 도는 동안만 시스템 절전을 막는다. keepawake는 임시·비관리자·자동해제.
  approve-selection은 이 스크립트 전에 별도로 끝나 있어야 한다(approval.json 필요).

  사용:
    powershell -ExecutionPolicy Bypass -File .\run_execute_keepawake.ps1 `
      -RawRoot "..." -WorkDir "..." -Output "..." -Jobs 8
#>
param(
    [Parameter(Mandatory = $true)] [string]$RawRoot,
    [Parameter(Mandatory = $true)] [string]$WorkDir,
    [Parameter(Mandatory = $true)] [string]$Output,
    [int]$Seed = 42,
    [int]$Jobs = 8,
    [string]$Python = "py",
    [string]$PythonArgs = "-3.12 -X utf8",
    [switch]$KeepDisplay
)
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Add-Type -Namespace Win32 -Name PowerExec -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("kernel32.dll", SetLastError = true)]
public static extern uint SetThreadExecutionState(uint esFlags);
'@
$ES_CONTINUOUS       = [uint32]2147483648
$ES_SYSTEM_REQUIRED  = [uint32]1
$ES_DISPLAY_REQUIRED = [uint32]2
$flags = $ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED
if ($KeepDisplay) { $flags = $flags -bor $ES_DISPLAY_REQUIRED }

$logDir = Join-Path $WorkDir "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log = Join-Path $logDir "execute_$stamp.log"
$start = Get-Date
"[keepawake] execute 절전 방지 ON : $start" | Tee-Object -FilePath $log -Append

try {
    [Win32.PowerExec]::SetThreadExecutionState($flags) | Out-Null
    $argList = @()
    if ($PythonArgs) { $argList += $PythonArgs.Split(" ") }
    $argList += @("-m", "battery_v3_6.cli", "execute",
                  "--raw-root", $RawRoot,
                  "--work-dir", $WorkDir,
                  "--output", $Output,
                  "--seed", "$Seed",
                  "--jobs", "$Jobs")
    "[keepawake] 실행: $Python $($argList -join ' ')" | Tee-Object -FilePath $log -Append
    & $Python @argList 2>&1 | Tee-Object -FilePath $log -Append
    $exit = $LASTEXITCODE
}
finally {
    [Win32.PowerExec]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
    $end = Get-Date
    "[keepawake] execute 절전 방지 OFF : $end (총 {0:hh\:mm\:ss}, exit=$exit)" -f ($end - $start) | Tee-Object -FilePath $log -Append
}
if ($exit -ne 0) { Write-Host "execute 실패 exit=$exit. 로그: $log" -ForegroundColor Red; exit $exit }
Write-Host "execute 완료. 로그: $log" -ForegroundColor Green
