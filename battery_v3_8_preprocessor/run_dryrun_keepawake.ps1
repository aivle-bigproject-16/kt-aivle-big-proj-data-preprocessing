<#
  run_dryrun_keepawake.ps1
  직렬 dry-run을 실행하는 동안만 시스템 절전(sleep) 진입을 막는다.
  - SetThreadExecutionState 사용: 비관리자, 임시. 스크립트가 살아있는 동안만 유효, 종료 시 자동 해제.
  - 전력 계획을 영구 변경하지 않는다.
  - 로그와 시작/종료 타임스탬프를 남겨 실제 dry-run 소요시간(장당 실측)을 확보한다.

  사용:
    powershell -ExecutionPolicy Bypass -File .\run_dryrun_keepawake.ps1 `
      -RawRoot "E:\103.배터리 불량 이미지 데이터\3.개방데이터\1.데이터" `
      -WorkDir "E:\battery_v3_work"

  주의:
    * 기본 인터프리터는 패키지 폴더의 .venv 이다. README 절차대로 venv를 먼저 만들어야 한다.
      venv 없이 글로벌 런처로 돌리려면 -Python py -PythonArgs "-3.12 -X utf8" 을 지정한다.
    * 노트북은 AC 전원 연결 필수(배터리면 정책상 잠들 수 있음).
    * 덮개(lid) 닫으면 SetThreadExecutionState로도 못 막는다 → 덮개 열어두거나
      제어판에서 "덮개를 닫을 때: 아무 것도 안 함"으로 설정.
    * 디스플레이는 꺼져도 된다(시스템만 깨어있으면 됨). 화면도 켜두려면 -KeepDisplay 지정.
#>

param(
    [Parameter(Mandatory = $true)] [string]$RawRoot,
    [Parameter(Mandatory = $true)] [string]$WorkDir,
    [int]$Seed = 42,
    [int]$Jobs = 1,
    [string]$Python = (Join-Path $PSScriptRoot ".venv\Scripts\python.exe"),
    [string]$PythonArgs = "-X utf8",
    [switch]$KeepDisplay
)

$ErrorActionPreference = "Stop"

# `-m battery_v3_3.cli` 는 패키지 폴더가 cwd여야 import된다. 항상 스크립트 폴더에서 실행.
Set-Location -Path $PSScriptRoot

# --- SetThreadExecutionState P/Invoke ---
Add-Type -Namespace Win32 -Name Power -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("kernel32.dll", SetLastError = true)]
public static extern uint SetThreadExecutionState(uint esFlags);
'@

# NOTE: PowerShell은 0x80000000을 int32(-2147483648)로 파싱해 uint32 캐스트가 실패한다.
#       십진수 리터럴(2147483648 > int32 max)을 쓰면 int64 양수로 파싱돼 안전하다.
$ES_CONTINUOUS       = [uint32]2147483648   # 0x80000000
$ES_SYSTEM_REQUIRED  = [uint32]1            # 0x00000001
$ES_DISPLAY_REQUIRED = [uint32]2            # 0x00000002

$flags = $ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED
if ($KeepDisplay) { $flags = $flags -bor $ES_DISPLAY_REQUIRED }

# 로그 준비
New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
$logDir = Join-Path $WorkDir "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log = Join-Path $logDir "dryrun_$stamp.log"

$start = Get-Date
"[keepawake] 절전 방지 ON (system awake$(if($KeepDisplay){' + display'})) : $start" | Tee-Object -FilePath $log -Append

try {
    [Win32.Power]::SetThreadExecutionState($flags) | Out-Null

    $pyExe = $Python
    $argList = @()
    if ($PythonArgs) { $argList += $PythonArgs.Split(" ") }
    $argList += @("-m", "battery_v3_8.cli", "dry-run",
                  "--raw-root", $RawRoot,
                  "--work-dir", $WorkDir,
                  "--seed", "$Seed",
                  "--jobs", "$Jobs")

    "[keepawake] 실행: $pyExe $($argList -join ' ')" | Tee-Object -FilePath $log -Append

    # stdout+stderr 로그로. 콘솔에도 표시.
    & $pyExe @argList 2>&1 | Tee-Object -FilePath $log -Append
    $exit = $LASTEXITCODE
}
finally {
    # 절전 정책 원복(ES_CONTINUOUS만 남겨 플래그 해제)
    [Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
    $end = Get-Date
    $dur = $end - $start
    "[keepawake] 절전 방지 OFF : $end" | Tee-Object -FilePath $log -Append
    "[keepawake] 총 소요: {0:hh\:mm\:ss} (exit=$exit)" -f $dur | Tee-Object -FilePath $log -Append
}

if ($exit -ne 0) {
    Write-Host "dry-run 실패 (exit=$exit). 로그: $log" -ForegroundColor Red
    exit $exit
}
Write-Host "dry-run 완료. 로그: $log" -ForegroundColor Green
Write-Host "다음: reports\dryrun_warnings.csv (헤더만=0건) 확인 후 approve-selection" -ForegroundColor Cyan
