$ErrorActionPreference = 'Stop'

$projectRoot = $PSScriptRoot
$runtimeDirectory = Join-Path $projectRoot '.runtime'
$stdoutLog = Join-Path $runtimeDirectory 'dev.stdout.log'
$stderrLog = Join-Path $runtimeDirectory 'dev.stderr.log'
$demoUrl = 'http://127.0.0.1:3000/'

function Test-DemoReady {
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri $demoUrl -TimeoutSec 2
        return $response.StatusCode -eq 200
    }
    catch {
        return $false
    }
}

if (Test-DemoReady) {
    Write-Host 'WM-VLA POC 结果页已经运行。' -ForegroundColor Green
    Start-Process $demoUrl
    exit 0
}

$npmCommand = Get-Command npm.cmd -ErrorAction SilentlyContinue
if (-not $npmCommand) {
    Write-Error '没有找到 npm。请先安装 Node.js 22 或更高版本。'
}

if (-not (Test-Path -LiteralPath (Join-Path $projectRoot 'node_modules'))) {
    Write-Error '项目依赖尚未安装。请先在项目目录执行 npm install。'
}

New-Item -ItemType Directory -Path $runtimeDirectory -Force | Out-Null

$process = Start-Process `
    -FilePath $npmCommand.Source `
    -ArgumentList @('run', 'dev') `
    -WorkingDirectory $projectRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -PassThru

Write-Host "正在启动 WM-VLA POC 结果页（PID $($process.Id)）..." -ForegroundColor Cyan

$deadline = [DateTime]::UtcNow.AddSeconds(30)
do {
    Start-Sleep -Milliseconds 500
    if (Test-DemoReady) {
        Write-Host "启动成功：$demoUrl" -ForegroundColor Green
        Start-Process $demoUrl
        exit 0
    }
} while ([DateTime]::UtcNow -lt $deadline -and -not $process.HasExited)

if ($process.HasExited) {
    Write-Error "服务进程提前退出。请检查日志：$stderrLog"
}

Write-Error "服务启动超过 30 秒。请检查日志：$stderrLog"
