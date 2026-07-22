<#
  run_pipeline.ps1
  전처리 파이프라인의 단일 실행 진입점이다. 버전이 올라가도 수정 없이 동작한다.

  ## 버전 비종속

  모듈 이름을 하드코딩하지 않고 스크립트 폴더에서 `battery_v*` 패키지를 찾아
  쓴다. v3.6에서 v3.7로 올릴 때 `run_dryrun_keepawake.ps1`이 v3.3 시절 모듈을
  계속 부르다 실행 즉시 실패한 사고가 있었다. 폴더를 읽어 결정하면 그 종류의
  사고가 재발하지 않는다.

  ## 반드시 pwsh(PowerShell 7)로 실행한다

  Windows PowerShell 5.1은 BOM 없는 UTF-8 파일을 ANSI로 읽는다. 이 저장소의
  스크립트와 경로에는 한글이 들어가므로 5.1에서는 파싱 단계에서 깨진다.
  아래 가드가 5.1 실행을 즉시 차단한다.

  ## 장시간 실행은 반드시 분리(detached)한다

  에이전트 세션의 자식 프로세스는 툴 호출이 끝나면 정리된다. dry-run과 execute는
  30분에서 2시간이 걸리므로 부모 세션에 묶이면 조용히 사라진다. `-Detached` 를
  붙이면 스크립트가 스스로를 독립 프로세스로 다시 띄우고 PID 만 남기고 반환한다.

  경로에 공백이 있으면 `-File` 인자가 잘린다. `Start-Process -ArgumentList` 는
  배열 원소가 공백을 포함해도 인용하지 않고 공백으로 이어 붙이므로, 위 재실행
  로직은 공백을 가질 수 있는 값을 모두 이중 인용해서 넘긴다. "배열로 넘기면
  안전하다"는 믿음으로 인용을 생략했다가 실제로 잘린 적이 있다.

  ## 진행 상황 확인

  단계마다 `<WorkDir>\pipeline.status`에 시작과 종료를 기록한다. 마지막 줄이
  `finished ... exit=0`이면 정상 완료다.

  ## 사용 예

      pwsh -File .\run_pipeline.ps1 -Stage dry-run -RawRoot "..." -WorkDir "..." -Detached
      pwsh -File .\run_pipeline.ps1 -Stage approve -WorkDir "..." -ApprovedBy "홍길동"
      pwsh -File .\run_pipeline.ps1 -Stage execute -RawRoot "..." -WorkDir "..." -Output "..." -Detached
      pwsh -File .\run_pipeline.ps1 -Stage upload  -WorkDir "..." -Output "..." -Remote "gdrive:..." -Detached
#>
param(
    [ValidateSet("dry-run", "approve", "execute", "upload")]
    [Parameter(Mandatory = $true)] [string]$Stage,
    [string]$RawRoot,
    [string]$WorkDir,
    [string]$Output,
    [string]$Remote,
    [string]$ApprovedBy,
    [int]$Seed = 42,
    [int]$Jobs = 8,
    [switch]$KeepDisplay,
    [switch]$Detached
)

$ErrorActionPreference = "Stop"

# Long stages must not be children of the calling session: an agent session
# reaps its children when a tool call ends, which killed two dry-runs after
# roughly 35 minutes each with no log to explain it. Relaunch detached instead.
#
# Start-Process -ArgumentList does NOT quote array elements that contain spaces:
# it joins them with a single space, so a path like "...\Between Laptop\..." gets
# split and -File sees only "...\Between". Every value that can contain a space
# must therefore be wrapped in double quotes here before it is handed over.
if ($Detached) {
    function Quote([string]$value) { '"' + $value + '"' }
    $forward = @("-ExecutionPolicy", "Bypass", "-NonInteractive", "-File", (Quote $PSCommandPath), "-Stage", $Stage)
    foreach ($name in @("RawRoot", "WorkDir", "Output", "Remote", "ApprovedBy")) {
        $value = Get-Variable -Name $name -ValueOnly -ErrorAction SilentlyContinue
        if ($value) { $forward += @("-$name", (Quote $value)) }
    }
    $forward += @("-Seed", "$Seed", "-Jobs", "$Jobs")
    if ($KeepDisplay) { $forward += "-KeepDisplay" }
    $child = Start-Process -FilePath (Get-Process -Id $PID).Path -ArgumentList $forward -WindowStyle Hidden -PassThru
    Write-Host "$Stage 를 분리 실행으로 시작했다. PID $($child.Id)" -ForegroundColor Cyan
    Write-Host "진행 상황: $WorkDir\pipeline.status" -ForegroundColor Cyan
    exit 0
}

if ($PSVersionTable.PSVersion.Major -lt 7) {
    Write-Error "pwsh(PowerShell 7) 필요. Windows PowerShell 5.1은 BOM 없는 UTF-8 한글을 깨뜨린다."
    exit 2
}

Set-Location -Path $PSScriptRoot

$package = Get-ChildItem -Path $PSScriptRoot -Directory -Filter "battery_v*" |
    Where-Object { Test-Path (Join-Path $_.FullName "cli.py") } |
    Sort-Object Name -Descending |
    Select-Object -First 1
if (-not $package) {
    Write-Error "battery_v* 패키지를 $PSScriptRoot 에서 찾지 못했다."
    exit 2
}
$module = $package.Name

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "가상환경이 없다. README의 설치 절차를 먼저 수행한다: $python"
    exit 2
}

if (-not $WorkDir) { Write-Error "-WorkDir 는 모든 단계에서 필요하다."; exit 2 }
New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
$logDir = Join-Path $WorkDir "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log = Join-Path $logDir "$($Stage -replace '-', '')_$stamp.log"
$status = Join-Path $WorkDir "pipeline.status"

switch ($Stage) {
    "dry-run" {
        if (-not $RawRoot) { Write-Error "-RawRoot 필요"; exit 2 }
        $cliArgs = @("dry-run", "--raw-root", $RawRoot, "--work-dir", $WorkDir, "--seed", "$Seed", "--jobs", "$Jobs")
    }
    "approve" {
        if (-not $ApprovedBy) { Write-Error "-ApprovedBy 필요"; exit 2 }
        $cliArgs = @("approve-selection", "--work-dir", $WorkDir, "--approved-by", $ApprovedBy, "--seed", "$Seed")
    }
    "execute" {
        if (-not $RawRoot -or -not $Output) { Write-Error "-RawRoot 와 -Output 필요"; exit 2 }
        $cliArgs = @("execute", "--raw-root", $RawRoot, "--work-dir", $WorkDir, "--output", $Output, "--seed", "$Seed", "--jobs", "$Jobs")
    }
    "upload" {
        if (-not $Output -or -not $Remote) { Write-Error "-Output 과 -Remote 필요"; exit 2 }
        $cliArgs = $null
    }
}

function Test-ZipClosed([string]$path) {
    # 중앙 디렉터리가 기록됐는지만 확인한다. testzip 은 전체를 압축 해제하므로
    # 수십 GB 파일에는 쓰지 않는다. 19.7GB 파일에서 이 방식은 2.5초면 끝난다.
    & $python -c "import sys,zipfile
try:
    with zipfile.ZipFile(sys.argv[1]) as archive: archive.infolist()
except Exception: sys.exit(1)
sys.exit(0)" $path 2>$null | Out-Null
    return ($LASTEXITCODE -eq 0)
}

function Invoke-Upload {
    # 순서: reports/·소스·루트 → CT zip → RGB(EXT) zip.
    # reports 는 게이트 근거와 감사 기록이라 가장 먼저 올려 팀이 선정 결과를 바로
    # 검토할 수 있게 한다(약 259MB). CT(약 18GB)가 EXT(약 71GB)보다 작으므로 CT 를
    # 먼저 올린다. 아래 zip 목록은 출력 폴더에서 실제 파일명을 찾아 정렬하므로
    # (battery_CT_* < battery_EXT_*), zip 이름이 버전마다 달라도(v3_·v4_1_ 등) 동작한다.
    # 종전에는 이름을 하드코딩(battery_CT_v3_*)해 두어, v4.1이 output.py 에서 zip 을
    # battery_CT_v4_1_* 로 바꾸자 업로드 루프가 4개 모두 건너뛰고 zip 을 하나도 올리지
    # 못했다.
    #
    # 부가 산출물에 --exclude "*.zip" 을 쓰지 않는 이유: 그 옵션은 zip 만 뺄 뿐
    # CT/ 와 EXT/ 원본 폴더로 재귀해 zip 안에 이미 든 40만 장(약 91GB)을 통째로
    # 재업로드한다. 그래서 필요한 항목만 명시적으로 지정한다.
    "[upload] 1/2 부가 산출물 (reports/·소스·루트 파일)" | Tee-Object -FilePath $log -Append
    & rclone copy $Output $Remote `
        --include "reports/**" --include "battery_v*/**" `
        --include "README.md" --include "requirements.lock" --include "prepare_training_view.py" `
        --transfers 8 --retries 5 --stats 30s --stats-one-line `
        --log-level INFO --log-file $log
    if ($LASTEXITCODE -ne 0) { return $LASTEXITCODE }
    "[upload] 부가 산출물 완료" | Tee-Object -FilePath $log -Append

    $zips = Get-ChildItem $Output -Filter "*.zip" | Sort-Object Name
    if (-not $zips) { "[upload] zip 없음 — 산출물 확인 필요" | Tee-Object -FilePath $log -Append; return 1 }
    "[upload] 2/2 데이터셋 zip (CT -> EXT): $($zips.Count)개" | Tee-Object -FilePath $log -Append
    foreach ($zip in $zips) {
        $name = $zip.Name
        $path = $zip.FullName
        $previous = -1L
        while ($true) {
            $size = (Get-Item $path).Length
            if ($size -gt 0 -and $size -eq $previous -and (Test-ZipClosed $path)) { break }
            $previous = $size
            Start-Sleep -Seconds 20
        }
        $gb = [math]::Round((Get-Item $path).Length / 1GB, 2)
        "[upload] 시작: $name ($gb GB)" | Tee-Object -FilePath $log -Append
        & rclone copyto $path "$Remote/$name" --drive-chunk-size 128M --transfers 1 `
            --retries 5 --low-level-retries 20 --stats 30s --stats-one-line `
            --log-level INFO --log-file $log
        if ($LASTEXITCODE -ne 0) { return $LASTEXITCODE }
        "[upload] 완료: $name" | Tee-Object -FilePath $log -Append
    }
    return $LASTEXITCODE
}

Add-Type -Namespace Win32 -Name PipelinePower -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("kernel32.dll", SetLastError = true)]
public static extern uint SetThreadExecutionState(uint esFlags);
'@
$ES_CONTINUOUS       = [uint32]2147483648
$ES_SYSTEM_REQUIRED  = [uint32]1
$ES_DISPLAY_REQUIRED = [uint32]2
$flags = $ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED
if ($KeepDisplay) { $flags = $flags -bor $ES_DISPLAY_REQUIRED }

$start = Get-Date
"started $Stage $($start.ToString('yyyy-MM-dd HH:mm:ss')) module=$module pid=$PID" | Add-Content -Path $status
"[run_pipeline] $Stage / module $module / jobs $Jobs / seed $Seed" | Tee-Object -FilePath $log -Append

$exit = 1
try {
    [Win32.PipelinePower]::SetThreadExecutionState($flags) | Out-Null
    if ($Stage -eq "upload") {
        $exit = Invoke-Upload
    }
    else {
        & $python -X utf8 -u -m "$module.cli" @cliArgs *>&1 | Tee-Object -FilePath $log -Append
        $exit = $LASTEXITCODE
    }
}
catch {
    $_ | Out-String | Tee-Object -FilePath $log -Append
    $exit = 1
}
finally {
    [Win32.PipelinePower]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
    $end = Get-Date
    $duration = $end - $start
    "finished $Stage $($end.ToString('yyyy-MM-dd HH:mm:ss')) elapsed=$($duration.ToString('hh\:mm\:ss')) exit=$exit" |
        Add-Content -Path $status
}

if ($exit -ne 0) {
    Write-Host "$Stage 실패 (exit=$exit). 로그: $log" -ForegroundColor Red
    exit $exit
}
Write-Host "$Stage 완료. 로그: $log" -ForegroundColor Green
exit 0
