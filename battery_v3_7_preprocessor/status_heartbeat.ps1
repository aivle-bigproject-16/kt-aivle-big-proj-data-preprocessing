<#
  status_heartbeat.ps1
  배터리 전처리 파이프라인 상태를 30분마다 PIPELINE_STATUS.md에 스냅샷으로 남긴다(사람이 아무 때나 확인).
  파이프라인 로직은 건드리지 않는다. 읽기·기록만. output 최종 폴더가 생기면 DONE 남기고 종료.
#>
param(
    [Parameter(Mandatory = $true)] [string]$WorkDir,
    [Parameter(Mandatory = $true)] [string]$Output,
    [int]$IntervalSec = 1800,
    [int]$MaxHours = 10
)
$ErrorActionPreference = "SilentlyContinue"
$statusFile = Join-Path $WorkDir "PIPELINE_STATUS.md"
$reports = Join-Path $WorkDir "reports"
$deadline = (Get-Date).AddHours($MaxHours)

function Snapshot {
    $now = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $sb = New-Object System.Text.StringBuilder
    [void]$sb.AppendLine("### $now")
    # 프로세스
    $procs = Get-Process python, py -ErrorAction SilentlyContinue
    if ($procs) {
        foreach ($p in $procs) { [void]$sb.AppendLine("- pid $($p.Id): CPU $([math]::Round($p.CPU,0))s, WS $([math]::Round($p.WS/1GB,2))GB") }
        [void]$sb.AppendLine("- python 프로세스 수: $($procs.Count) (>3 이면 워커 가동 중)")
    } else {
        [void]$sb.AppendLine("- python 프로세스 없음 (단계 사이 or 완료 or 중단)")
    }
    # 단계 판정
    $exLog = Get-ChildItem (Join-Path $WorkDir "logs") -Filter "execute_*.log" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1
    $stage = if (Test-Path $Output) { "DONE (output 존재)" } elseif ($exLog) { "execute 진행/완료" } else { "dry-run 진행" }
    [void]$sb.AppendLine("- 단계: $stage")
    # reports
    if (Test-Path $reports) {
        $rc = (Get-ChildItem $reports -File -ErrorAction SilentlyContinue | Measure-Object).Count
        [void]$sb.AppendLine("- reports 파일 수: $rc")
        $dw = Join-Path $reports "dryrun_warnings.csv"
        if (Test-Path $dw) {
            $lines = (Get-Content $dw | Measure-Object -Line).Lines
            [void]$sb.AppendLine("- dryrun_warnings.csv: $([math]::Max(0,$lines-1)) 건 (헤더 제외)")
        }
        $ss = Join-Path $reports "scan_summary.md"
        if (Test-Path $ss) { [void]$sb.AppendLine("- scan_summary.md 생성됨 (dry-run 스캔+선택 완료)") }
    } else {
        [void]$sb.AppendLine("- reports 아직 없음 (스캔/열거 진행 중)")
    }
    # 최신 로그 tail 1줄
    $latest = Get-ChildItem (Join-Path $WorkDir "logs") -Filter "*.log" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1
    if ($latest) {
        $tail = (Get-Content $latest.FullName -Tail 1)
        if ($tail) { [void]$sb.AppendLine("- 로그 마지막줄: $tail") }
    }
    [void]$sb.AppendLine("")
    Add-Content -Path $statusFile -Value $sb.ToString()
}

"# 파이프라인 상태 하트비트 (30분 간격)`n시작: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')`n" | Set-Content -Path $statusFile
Snapshot
while ((Get-Date) -lt $deadline) {
    if (Test-Path $Output) { Add-Content -Path $statusFile -Value "### DONE $(Get-Date -Format 'HH:mm:ss') — output 폴더 생성됨. 하트비트 종료.`n"; break }
    Start-Sleep -Seconds $IntervalSec
    Snapshot
}
