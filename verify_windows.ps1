<#
.SYNOPSIS
    One-shot Windows verification for llmops-semantic-cache. Writes windows_verification.log.

.DESCRIPTION
    Run from the project folder:

        powershell -ExecutionPolicy Bypass -File .\verify_windows.ps1

    Optional switches: -SkipNeural  -SkipDocker  -SkipFreshClone  -KeepTemp

    SAFETY (by design, reviewable below):
      * Never runs git push, gh, or any GitHub login/credential command, and never adds or
        changes remotes (it only lists them with the read-only `git remote -v`).
      * Never creates commits if the Git history can be restored from _setup\*.bundle. Only if
        there is no .git AND no bundle AND your Git identity is configured, it creates one local
        commit under YOUR identity. It never changes git config.
      * Never deletes project files. An existing .venv is renamed to .venv-old-<timestamp>
        (git-ignored); _setup is moved to %TEMP%\llmops-verify-backup-<timestamp> after a
        successful restore. Only regenerable tool caches (.pytest_cache, .mypy_cache,
        .ruff_cache, __pycache__) are removed from the project at the end.
      * Everything else it creates lives in %TEMP%\llmops-verify-<timestamp>, which is deleted
        at the end unless -KeepTemp is given.
      * Servers it starts are local (127.0.0.1) and stopped at the end. API keys are never
        needed: the mock LLM is used throughout.
#>
[CmdletBinding()]
param(
    [switch]$SkipNeural,
    [switch]$SkipDocker,
    [switch]$SkipFreshClone,
    [switch]$KeepTemp
)

$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'
$ProjectRoot = $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

$Stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$LogFile = Join-Path $ProjectRoot 'windows_verification.log'
$TempRoot = Join-Path $env:TEMP "llmops-verify-$Stamp"
$BackupRoot = Join-Path $env:TEMP "llmops-verify-backup-$Stamp"
New-Item -ItemType Directory -Force -Path $TempRoot | Out-Null
$Results = New-Object System.Collections.Generic.List[object]
$Servers = New-Object System.Collections.Generic.List[object]
$script:LastOutput = ''
$StartTime = Get-Date

Set-Content -LiteralPath $LogFile -Encoding UTF8 -Value @(
    'llmops-semantic-cache - Windows verification log',
    "started:  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')",
    "project:  $ProjectRoot",
    "temp dir: $TempRoot"
)

# ----------------------------------------------------------------------------- helpers
function Write-Log([string]$Text) {
    Add-Content -LiteralPath $LogFile -Encoding UTF8 -Value $Text
    Write-Host $Text
}

function Write-Section([string]$Name) {
    Write-Log ''
    Write-Log ('=' * 50)
    Write-Log $Name
    Write-Log ('=' * 50)
}

function Add-Result([string]$Check, [string]$Status, [string]$Detail = '') {
    $Results.Add([pscustomobject]@{ Check = $Check; Status = $Status; Detail = $Detail })
    Write-Log ("[{0}] {1}{2}" -f $Status, $Check, $(if ($Detail) { "  -- $Detail" } else { '' }))
}

function Invoke-Logged {
    param([string]$Exe, [string[]]$Arguments = @(), [string]$WorkDir = $ProjectRoot)
    Write-Log ("PS> {0} {1}" -f $Exe, ($Arguments -join ' '))
    Write-Log ("    (in {0})" -f $WorkDir)
    $code = -1
    $output = @()
    Push-Location -LiteralPath $WorkDir
    try {
        $global:LASTEXITCODE = 0
        $output = & $Exe @Arguments 2>&1 | ForEach-Object { "$_" }
        $code = $LASTEXITCODE
        if ($null -eq $code) { $code = 0 }
    } catch {
        $output = @("ERROR: $($_.Exception.Message)")
        $code = -1
    } finally {
        Pop-Location
    }
    if ($output) { Write-Log (($output) -join [Environment]::NewLine) }
    Write-Log "exit code: $code"
    $script:LastOutput = ($output -join "`n")
    return $code
}

function Get-ToolPath([string]$Name) {
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($cmd) { return $cmd.Source }
    return $null
}

function Test-PortFree([int]$Port) {
    try {
        $listener = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Loopback, $Port)
        $listener.Start()
        $listener.Stop()
        return $true
    } catch {
        return $false
    }
}

function Get-FreePort([int]$Start = 8000) {
    for ($p = $Start; $p -lt ($Start + 200); $p++) {
        if (Test-PortFree $p) { return $p }
    }
    return 0
}

function Start-Api {
    param([string]$PythonExe, [string]$WorkDir, [int]$Port, [hashtable]$EnvVars, [string]$Name)
    $saved = @{}
    foreach ($key in $EnvVars.Keys) {
        $saved[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
        [Environment]::SetEnvironmentVariable($key, [string]$EnvVars[$key], 'Process')
    }
    $out = Join-Path $TempRoot "$Name.stdout.log"
    $err = Join-Path $TempRoot "$Name.stderr.log"
    Write-Log ("starting API '{0}': {1} -m llmops_cache  (port {2}, cwd {3})" -f $Name, $PythonExe, $Port, $WorkDir)
    Write-Log ("  env: " + (($EnvVars.Keys | Sort-Object | ForEach-Object { "$_=$($EnvVars[$_])" }) -join ' '))
    $proc = $null
    try {
        $proc = Start-Process -FilePath $PythonExe -ArgumentList @('-m', 'llmops_cache') `
            -WorkingDirectory $WorkDir -RedirectStandardOutput $out -RedirectStandardError $err `
            -WindowStyle Hidden -PassThru
    } catch {
        Write-Log "  Start-Process failed: $($_.Exception.Message)"
    } finally {
        foreach ($key in $saved.Keys) { [Environment]::SetEnvironmentVariable($key, $saved[$key], 'Process') }
    }
    $server = [pscustomobject]@{ Name = $Name; Process = $proc; Out = $out; Err = $err; Up = $false; Port = $Port }
    if ($null -eq $proc) { return $server }
    $Servers.Add($server)
    $deadline = (Get-Date).AddSeconds(60)
    while ((Get-Date) -lt $deadline) {
        if ($proc.HasExited) { break }
        try {
            $r = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2
            if ($r.StatusCode -eq 200) { $server.Up = $true; break }
        } catch { Start-Sleep -Milliseconds 500 }
    }
    Write-Log ("  API '{0}' up: {1}" -f $Name, $server.Up)
    return $server
}

function Stop-Api($Server, [string]$Reason = '') {
    if ($null -eq $Server -or $null -eq $Server.Process) { return }
    if (-not $Server.Process.HasExited) {
        Stop-Process -Id $Server.Process.Id -Force -ErrorAction SilentlyContinue
        $Server.Process.WaitForExit(10000) | Out-Null
    }
    Write-Log ("stopped API '{0}' {1}" -f $Server.Name, $Reason)
    foreach ($file in @($Server.Out, $Server.Err)) {
        if (Test-Path -LiteralPath $file) {
            Write-Log "----- $(Split-Path $file -Leaf) -----"
            Write-Log ((Get-Content -LiteralPath $file -ErrorAction SilentlyContinue) -join [Environment]::NewLine)
        }
    }
    [void]$Servers.Remove($Server)
}

function Get-HttpStatus([string]$Url) {
    try {
        return (Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 10).StatusCode
    } catch {
        if ($_.Exception.Response) { return [int]$_.Exception.Response.StatusCode }
        return 0
    }
}

function Invoke-Smoke([string]$PythonExe, [string]$Url, [string]$Phase, [string]$Prefix, [string]$WorkDir = $ProjectRoot) {
    $code = Invoke-Logged -Exe $PythonExe -Arguments @('scripts\smoke_test.py', '--url', $Url, '--phase', $Phase) -WorkDir $WorkDir
    foreach ($line in ($script:LastOutput -split "`n")) {
        if ($line -match '^(PASS|FAIL)\s+(.+?)(\s+--\s+(.*))?$') {
            Add-Result "$Prefix$($Matches[2].Trim())" $Matches[1] $Matches[4]
        }
    }
    return $code
}

function Get-PytestSummary([string]$Text) {
    $line = ($Text -split "`n" | Where-Object { $_ -match '\d+ (passed|failed|error)' } | Select-Object -Last 1)
    if ($line) { return $line.Trim(' ', '=') }
    return 'no summary line found'
}

# ============================================================================= ENVIRONMENT
Write-Section 'WINDOWS ENVIRONMENT'
try {
    $os = Get-CimInstance Win32_OperatingSystem
    Write-Log ("OS: {0} {1} (build {2}) {3}" -f $os.Caption, $os.Version, $os.BuildNumber, $os.OSArchitecture)
    Add-Result 'Windows' 'PASS' "$($os.Caption) $($os.Version)"
} catch {
    Write-Log ("OS: {0}" -f [Environment]::OSVersion.VersionString)
    Add-Result 'Windows' 'PASS' ([Environment]::OSVersion.VersionString)
}
Write-Log ("PowerShell: {0}" -f $PSVersionTable.PSVersion)
Write-Log ("CPU cores: {0}" -f [Environment]::ProcessorCount)

# Python: prefer the py launcher (3.13 -> 3.11), then python on PATH. Skips the Store stub.
$PyExe = $null
$PyPrefix = @()
$pyLauncher = Get-ToolPath 'py'
$candidates = @()
if ($pyLauncher) { foreach ($v in @('-3.13', '-3.12', '-3.11')) { $candidates += , @($pyLauncher, $v) } }
$pythonOnPath = Get-ToolPath 'python'
if ($pythonOnPath) { $candidates += , @($pythonOnPath) }
foreach ($cand in $candidates) {
    $exe = $cand[0]
    $pre = @()
    if ($cand.Count -gt 1) { $pre = @($cand[1]) }
    $probe = & $exe @pre -c "import sys; print(sys.version_info >= (3, 11)); print(sys.version.split()[0]); print(sys.executable)" 2>$null
    if ($LASTEXITCODE -eq 0 -and $probe -and $probe[0] -eq 'True') {
        $PyExe = $exe
        $PyPrefix = $pre
        Write-Log ("Python: {0} at {1} (via {2} {3})" -f $probe[1], $probe[2], $exe, ($pre -join ' '))
        Add-Result 'Python 3.11+' 'PASS' $probe[1]
        break
    } elseif ($probe) {
        Write-Log ("ignored {0} {1}: version {2} (< 3.11)" -f $exe, ($pre -join ' '), $probe[1])
    }
}
if (-not $PyExe) {
    Add-Result 'Python 3.11+' 'FAIL' 'no Python 3.11+ found (py launcher or python on PATH)'
}

$GitExe = Get-ToolPath 'git'
if ($GitExe) {
    Invoke-Logged -Exe $GitExe -Arguments @('--version') | Out-Null
    Add-Result 'Git' 'PASS' $script:LastOutput.Trim()
} else {
    Add-Result 'Git' 'NOT AVAILABLE' 'git not on PATH'
}

$DockerExe = Get-ToolPath 'docker'
$DockerUsable = $false
if ($DockerExe) {
    Invoke-Logged -Exe $DockerExe -Arguments @('--version') | Out-Null
    $dockerVersion = $script:LastOutput.Trim()
    $composeCode = Invoke-Logged -Exe $DockerExe -Arguments @('compose', 'version')
    $infoCode = Invoke-Logged -Exe $DockerExe -Arguments @('info', '--format', '{{.ServerVersion}}')
    if ($infoCode -eq 0 -and $composeCode -eq 0) {
        $DockerUsable = $true
        Add-Result 'Docker' 'PASS' "$dockerVersion; engine running"
    } elseif ($infoCode -ne 0) {
        Add-Result 'Docker' 'NOT AVAILABLE' 'docker CLI found but the engine is not running (start Docker Desktop)'
    } else {
        Add-Result 'Docker' 'NOT AVAILABLE' 'docker compose plugin not available'
    }
} else {
    Add-Result 'Docker' 'NOT AVAILABLE' 'docker not on PATH (Docker Desktop not installed?)'
}
Write-Log ("port 8000 free: {0}" -f (Test-PortFree 8000))

# ============================================================================= GIT
Write-Section 'GIT'
$GitOk = $false
if ($GitExe) {
    $globalName = (& $GitExe config --get user.name 2>$null)
    $globalEmail = (& $GitExe config --get user.email 2>$null)
    Write-Log ("configured git identity: name='{0}' email='{1}'" -f $globalName, $globalEmail)
    $identityOk = [bool]$globalName -and [bool]$globalEmail
    $bundle = Join-Path $ProjectRoot '_setup\llmops-semantic-cache.bundle'
    $setupCi = Join-Path $ProjectRoot '_setup\ci.yml'
    $ciPath = Join-Path $ProjectRoot '.github\workflows\ci.yml'

    if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot '.git'))) {
        if (Test-Path -LiteralPath $bundle) {
            Write-Log 'No .git yet: restoring history from _setup bundle (no new commits are created).'
            if ((Test-Path -LiteralPath $setupCi) -and -not (Test-Path -LiteralPath $ciPath)) {
                New-Item -ItemType Directory -Force -Path (Split-Path $ciPath) | Out-Null
                Copy-Item -LiteralPath $setupCi -Destination $ciPath
                Write-Log 'copied _setup\ci.yml -> .github\workflows\ci.yml'
            }
            $c1 = Invoke-Logged -Exe $GitExe -Arguments @('init')
            $c2 = Invoke-Logged -Exe $GitExe -Arguments @('symbolic-ref', 'HEAD', 'refs/heads/main')
            $c3 = Invoke-Logged -Exe $GitExe -Arguments @('bundle', 'verify', $bundle)
            $c4 = Invoke-Logged -Exe $GitExe -Arguments @('fetch', $bundle, 'main')
            $c5 = Invoke-Logged -Exe $GitExe -Arguments @('reset', 'FETCH_HEAD')
            if (($c1 + $c2 + $c3 + $c4 + $c5) -eq 0) {
                Add-Result 'Git history restored from bundle' 'PASS'
                New-Item -ItemType Directory -Force -Path $BackupRoot | Out-Null
                Move-Item -LiteralPath (Join-Path $ProjectRoot '_setup') -Destination $BackupRoot
                Write-Log "moved _setup to $BackupRoot (backup; not deleted)"
            } else {
                Add-Result 'Git history restored from bundle' 'FAIL' 'see git output above'
            }
        } elseif ($identityOk) {
            Write-Log 'No .git and no bundle: creating a local repository with ONE commit under your identity.'
            Invoke-Logged -Exe $GitExe -Arguments @('init') | Out-Null
            Invoke-Logged -Exe $GitExe -Arguments @('symbolic-ref', 'HEAD', 'refs/heads/main') | Out-Null
            Invoke-Logged -Exe $GitExe -Arguments @('add', '-A') | Out-Null
            Invoke-Logged -Exe $GitExe -Arguments @('diff', '--cached', '--check') | Out-Null
            $cc = Invoke-Logged -Exe $GitExe -Arguments @('commit', '-m', 'feat: build semantic LLM cache')
            Add-Result 'Git repository initialised' $(if ($cc -eq 0) { 'PASS' } else { 'FAIL' })
        } else {
            Add-Result 'Git repository initialised' 'FAIL' 'no .git, no bundle, and git user.name/user.email not configured - not committing'
        }
    } else {
        Write-Log '.git already exists: not modifying history.'
        if (Test-Path -LiteralPath (Join-Path $ProjectRoot '_setup')) {
            Write-Log 'WARNING: _setup still exists next to an existing .git; leaving it untouched.'
        }
    }

    if (Test-Path -LiteralPath (Join-Path $ProjectRoot '.git')) {
        $GitOk = $true
        $probe = (& $GitExe status 2>&1 | ForEach-Object { "$_" }) -join ' '
        if ($probe -match 'dubious ownership') {
            Add-Result 'Git can read the repository' 'FAIL' 'git reports dubious ownership (common on FAT/exFAT drives); this script does not change git config - see the command git prints above'
            Write-Log $probe
            $GitOk = $false
        }
    }
    if ($GitOk) {
        Invoke-Logged -Exe $GitExe -Arguments @('rev-parse', '--abbrev-ref', 'HEAD') | Out-Null
        $branch = $script:LastOutput.Trim()
        Add-Result 'Git branch is main' $(if ($branch -eq 'main') { 'PASS' } else { 'FAIL' }) "branch=$branch"
        Invoke-Logged -Exe $GitExe -Arguments @('log', '--oneline', '--decorate', '--graph', '--all') | Out-Null
        Invoke-Logged -Exe $GitExe -Arguments @('log', '--format=%h | author: %an <%ae> | committer: %cn <%ce> | %s') | Out-Null
        $authors = $script:LastOutput
        Invoke-Logged -Exe $GitExe -Arguments @('log', '--format=%B') | Out-Null
        $messages = $script:LastOutput
        $aiIdentity = ($authors + $messages) -match '(?i)(co-authored-by|anthropic|noreply@anthropic|claude)'
        Add-Result 'Git history has no AI author/co-author' $(if ($aiIdentity) { 'FAIL' } else { 'PASS' })
        Write-Log "repository identity used for any future commit: '$globalName <$globalEmail>'"
        Add-Result 'Git identity configured' $(if ($identityOk) { 'PASS' } else { 'FAIL' }) $(if ($identityOk) { "$globalName <$globalEmail>" } else { 'set user.name and user.email before committing' })
        Invoke-Logged -Exe $GitExe -Arguments @('remote', '-v') | Out-Null
        Write-Log '(remote listing only - this script never adds remotes or pushes)'
        $ciTracked = (& $GitExe ls-files -- '.github/workflows/ci.yml' 2>$null)
        Add-Result 'CI workflow at .github/workflows/ci.yml (tracked)' $(if ($ciTracked) { 'PASS' } else { 'FAIL' })
        Invoke-Logged -Exe $GitExe -Arguments @('status', '--porcelain=v1', '--untracked-files=all') | Out-Null
        $porcelain = $script:LastOutput.Trim()
        Add-Result 'Git working tree clean before tests' $(if (-not $porcelain) { 'PASS' } else { 'FAIL' }) $porcelain
    }
} else {
    Add-Result 'Git checks' 'NOT VERIFIED' 'git not available'
}

# ============================================================================= PYTHON
Write-Section 'PYTHON INSTALLATION'
$VenvPy = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$InstallOk = $false
if ($PyExe) {
    if (Test-Path -LiteralPath (Join-Path $ProjectRoot '.venv')) {
        $old = ".venv-old-$Stamp"
        Rename-Item -LiteralPath (Join-Path $ProjectRoot '.venv') -NewName $old
        Write-Log "existing .venv renamed to $old (kept, git-ignored)"
    }
    $c = Invoke-Logged -Exe $PyExe -Arguments ($PyPrefix + @('-m', 'venv', '.venv'))
    Add-Result 'Create fresh .venv' $(if ($c -eq 0 -and (Test-Path -LiteralPath $VenvPy)) { 'PASS' } else { 'FAIL' })
    if (Test-Path -LiteralPath $VenvPy) {
        Invoke-Logged -Exe $VenvPy -Arguments @('-c', "import sys; print('isolated venv:', sys.prefix != sys.base_prefix, sys.prefix)") | Out-Null
        Invoke-Logged -Exe $VenvPy -Arguments @('-m', 'pip', 'install', '--upgrade', 'pip') | Out-Null
        $c = Invoke-Logged -Exe $VenvPy -Arguments @('-m', 'pip', 'install', '-e', '.[dev]')
        $InstallOk = ($c -eq 0)
        Add-Result 'pip install -e ".[dev]"' $(if ($InstallOk) { 'PASS' } else { 'FAIL' })
        $c = Invoke-Logged -Exe $VenvPy -Arguments @('-m', 'pip', 'check')
        Add-Result 'pip check (dependency conflicts)' $(if ($c -eq 0) { 'PASS' } else { 'FAIL' })
        Invoke-Logged -Exe $VenvPy -Arguments @('-m', 'pip', 'list') | Out-Null
    }
} else {
    Add-Result 'Python installation' 'NOT VERIFIED' 'no Python 3.11+'
}

# ============================================================================= TESTS
Write-Section 'TESTS'
if ($InstallOk) {
    $c = Invoke-Logged -Exe $VenvPy -Arguments @('-m', 'pytest', '-ra')
    Add-Result 'pytest (full suite)' $(if ($c -eq 0) { 'PASS' } else { 'FAIL' }) (Get-PytestSummary $script:LastOutput)
} else { Add-Result 'pytest' 'NOT VERIFIED' 'install failed' }

Write-Section 'LINT'
if ($InstallOk) {
    $c = Invoke-Logged -Exe $VenvPy -Arguments @('-m', 'ruff', 'check', '.')
    Add-Result 'ruff check .' $(if ($c -eq 0) { 'PASS' } else { 'FAIL' })
    $c = Invoke-Logged -Exe $VenvPy -Arguments @('-m', 'ruff', 'format', '--check', '.')
    Add-Result 'ruff format --check .' $(if ($c -eq 0) { 'PASS' } else { 'FAIL' })
} else { Add-Result 'ruff' 'NOT VERIFIED' 'install failed' }

Write-Section 'TYPE CHECK'
if ($InstallOk) {
    $c = Invoke-Logged -Exe $VenvPy -Arguments @('-m', 'mypy', '.')
    Add-Result 'mypy . (strict: src, scripts, examples)' $(if ($c -eq 0) { 'PASS' } else { 'FAIL' })
    $c = Invoke-Logged -Exe $VenvPy -Arguments @('-m', 'mypy')
    Add-Result 'mypy (package)' $(if ($c -eq 0) { 'PASS' } else { 'FAIL' })
} else { Add-Result 'mypy' 'NOT VERIFIED' 'install failed' }

# ============================================================================= SECURITY
Write-Section 'SECURITY'
if ($InstallOk) {
    $c = Invoke-Logged -Exe $VenvPy -Arguments @('-m', 'pip', 'install', 'pip-audit')
    if ($c -eq 0) {
        $c = Invoke-Logged -Exe $VenvPy -Arguments @('-m', 'pip_audit', '--skip-editable')
        if ($c -eq 0) { Add-Result 'pip-audit (known CVEs in installed deps)' 'PASS' }
        elseif ($script:LastOutput -match '(?i)(connection|timed out|resolve|network|ssl)') { Add-Result 'pip-audit' 'NOT VERIFIED' 'vulnerability database unreachable' }
        else { Add-Result 'pip-audit (known CVEs in installed deps)' 'FAIL' 'see output above' }
    } else { Add-Result 'pip-audit' 'NOT VERIFIED' 'could not install pip-audit' }
}
if ($GitOk) {
    $tracked = @(& $GitExe ls-files 2>$null)
    $envTracked = $tracked | Where-Object { ($_ -match '(^|/)\.env$') -or (($_ -match '(^|/)\.env\.') -and ($_ -notmatch '\.env\.example$')) }
    Add-Result '.env not tracked' $(if (-not $envTracked) { 'PASS' } else { 'FAIL' }) ($envTracked -join ', ')
    $patterns = @(
        'sk-[A-Za-z0-9_-]{20,}', 'AKIA[0-9A-Z]{16}', 'ghp_[A-Za-z0-9]{30,}', 'github_pat_[A-Za-z0-9_]{30,}',
        'xox[baprs]-[A-Za-z0-9-]{10,}', '-----BEGIN [A-Z ]*PRIVATE KEY-----', 'AIza[0-9A-Za-z_-]{35}'
    )
    $absPath = '(?i)\b[A-Z]:\\(Users|Documents and Settings)\\|/home/[a-z]+/|/Users/[A-Za-z]+/'
    $secretHits = @()
    $pathHits = @()
    foreach ($file in $tracked) {
        if ($file -eq 'verify_windows.ps1') { continue }  # contains the patterns themselves
        $full = Join-Path $ProjectRoot $file
        if (-not (Test-Path -LiteralPath $full -PathType Leaf)) { continue }
        foreach ($pat in $patterns) {
            $m = Select-String -LiteralPath $full -Pattern $pat -ErrorAction SilentlyContinue
            if ($m) { $secretHits += ($m | ForEach-Object { "$($file):$($_.LineNumber)" }) }
        }
        $m = Select-String -LiteralPath $full -Pattern $absPath -ErrorAction SilentlyContinue
        if ($m) { $pathHits += ($m | ForEach-Object { "$($file):$($_.LineNumber): $($_.Line.Trim())" }) }
    }
    Add-Result 'No secrets in tracked files' $(if (-not $secretHits) { 'PASS' } else { 'FAIL' }) ($secretHits -join '; ')
    Add-Result 'No absolute personal paths in tracked files' $(if (-not $pathHits) { 'PASS' } else { 'FAIL' }) ($pathHits -join '; ')
    Write-Log ("tracked files scanned: {0}" -f $tracked.Count)
}

# ============================================================================= APPLICATION
Write-Section 'APPLICATION'
$Port = Get-FreePort 8000
$StoreDir = Join-Path $TempRoot 'store'
# The server runs from an empty folder so a personal .env in the project (API keys, a real
# LLM provider...) cannot change what is being tested. The package is installed, so the cwd
# does not matter for imports.
$AppCwd = Join-Path $TempRoot 'app-cwd'
New-Item -ItemType Directory -Force -Path $AppCwd | Out-Null
if (Test-Path -LiteralPath (Join-Path $ProjectRoot '.env')) {
    Write-Log 'note: a .env exists in the project; it is NOT used for these checks (not read, not logged).'
}
$AppEnv = @{
    HOST = '127.0.0.1'; PORT = "$Port"; APP_ENV = 'development'; LLM_PROVIDER = 'mock';
    EMBEDDING_PROVIDER = 'hashing'; MOCK_LLM_LATENCY_MS = '300'; VECTOR_STORE_PATH = $StoreDir;
    CACHE_SNAPSHOT_INTERVAL_SECONDS = '1'; LOG_LEVEL = 'INFO'
}
$App = $null
$Url = "http://127.0.0.1:$Port"
if ($InstallOk -and $Port -gt 0) {
    $App = Start-Api -PythonExe $VenvPy -WorkDir $AppCwd -Port $Port -EnvVars $AppEnv -Name 'app'
    Add-Result 'Application startup (python -m llmops_cache)' $(if ($App.Up) { 'PASS' } else { 'FAIL' }) $Url
}
if ($App -and $App.Up) {
    foreach ($path in @('/', '/health', '/docs', '/redoc', '/openapi.json', '/v1/cache/stats', '/v1/metrics')) {
        $status = Get-HttpStatus "$Url$path"
        Add-Result "GET $path (PowerShell Invoke-WebRequest)" $(if ($status -eq 200) { 'PASS' } else { 'FAIL' }) "HTTP $status"
    }
    try {
        $health = Invoke-RestMethod -Uri "$Url/health"
        Write-Log ("/health -> " + ($health | ConvertTo-Json -Compress))
        $body = @{ message = 'PowerShell native request: what is an API?' } | ConvertTo-Json
        $resp = Invoke-RestMethod -Method Post -Uri "$Url/v1/chat" -ContentType 'application/json' -Body $body
        Write-Log ("POST /v1/chat (Invoke-RestMethod) -> " + ($resp | ConvertTo-Json -Compress))
        Add-Result 'POST /v1/chat from PowerShell' $(if ($null -ne $resp.response) { 'PASS' } else { 'FAIL' }) "cache_hit=$($resp.cache_hit) source=$($resp.source)"
        $bad = $null
        try { Invoke-RestMethod -Method Post -Uri "$Url/v1/chat" -ContentType 'application/json' -Body '{"message": ""}' | Out-Null }
        catch { $bad = [int]$_.Exception.Response.StatusCode }
        Add-Result 'Invalid input rejected with 422' $(if ($bad -eq 422) { 'PASS' } else { 'FAIL' }) "HTTP $bad"
    } catch {
        Add-Result 'POST /v1/chat from PowerShell' 'FAIL' $_.Exception.Message
    }
    Invoke-Smoke -PythonExe $VenvPy -Url $Url -Phase 'endpoints' -Prefix 'HTTP ' | Out-Null
    $c = Invoke-Logged -Exe $VenvPy -Arguments @('-m', 'llmops_cache.cli', '--url', $Url, 'health')
    Add-Result 'CLI: python -m llmops_cache.cli health' $(if ($c -eq 0) { 'PASS' } else { 'FAIL' })
    $c = Invoke-Logged -Exe $VenvPy -Arguments @('-m', 'llmops_cache.cli', '--url', $Url, 'stats')
    Add-Result 'CLI: python -m llmops_cache.cli stats' $(if ($c -eq 0) { 'PASS' } else { 'FAIL' })
}

Write-Section 'CACHE MISS/HIT'
if ($App -and $App.Up) {
    Invoke-Smoke -PythonExe $VenvPy -Url $Url -Phase 'cache' -Prefix '' | Out-Null
    $c = Invoke-Logged -Exe $VenvPy -Arguments @('-m', 'llmops_cache.cli', '--url', $Url, 'demo')
    $demoOk = ($c -eq 0) -and ($script:LastOutput -match 'CACHE MISS[\s\S]*CACHE HIT[\s\S]*CACHE MISS')
    Add-Result 'CLI demo against server: MISS, HIT, MISS' $(if ($demoOk) { 'PASS' } else { 'FAIL' })
    $c = Invoke-Logged -Exe $VenvPy -Arguments @('-m', 'llmops_cache.cli', '--url', $Url, 'metrics')
    Add-Result 'GET /v1/metrics via CLI' $(if ($c -eq 0) { 'PASS' } else { 'FAIL' })
    $c = Invoke-Logged -Exe $VenvPy -Arguments @('-m', 'llmops_cache.cli', '--url', $Url, 'clear-cache')
    Add-Result 'DELETE /v1/cache via CLI' $(if ($c -eq 0) { 'PASS' } else { 'FAIL' })
} else { Add-Result 'Cache MISS/HIT' 'NOT VERIFIED' 'application did not start' }

Write-Section 'TTL'
if ($App -and $App.Up) {
    Invoke-Smoke -PythonExe $VenvPy -Url $Url -Phase 'ttl' -Prefix '' | Out-Null
} else { Add-Result 'TTL' 'NOT VERIFIED' 'application did not start' }

Write-Section 'CONCURRENCY'
if ($App -and $App.Up) {
    Invoke-Smoke -PythonExe $VenvPy -Url $Url -Phase 'concurrency' -Prefix '' | Out-Null
} else { Add-Result 'Concurrency' 'NOT VERIFIED' 'application did not start' }

Write-Section 'PERSISTENCE'
if ($App -and $App.Up) {
    Write-Log 'Documented behaviour: with VECTOR_STORE_PATH set, the index is snapshotted every'
    Write-Log 'CACHE_SNAPSHOT_INTERVAL_SECONDS (1 s here) and on graceful shutdown. Windows cannot send'
    Write-Log 'Ctrl+C to a hidden process, so this test HARD-KILLS the server: only the periodic'
    Write-Log 'snapshot can save the entry. (Graceful shutdown is covered by the pytest suite.)'
    Invoke-Smoke -PythonExe $VenvPy -Url $Url -Phase 'persist-seed' -Prefix '' | Out-Null
    Start-Sleep -Seconds 3
    $files = @(Get-ChildItem -LiteralPath $StoreDir -ErrorAction SilentlyContinue | ForEach-Object { "$($_.Name) ($($_.Length) bytes)" })
    Write-Log ("store directory after snapshot: " + ($files -join ', '))
    Add-Result 'Snapshot files written (entries.json + vectors.npy)' $(if ($files.Count -ge 2) { 'PASS' } else { 'FAIL' }) ($files -join ', ')
    Stop-Api $App '(hard kill for persistence test)'
    $App = Start-Api -PythonExe $VenvPy -WorkDir $AppCwd -Port $Port -EnvVars $AppEnv -Name 'app-restarted'
    if ($App.Up) {
        Invoke-Smoke -PythonExe $VenvPy -Url $Url -Phase 'persist-check' -Prefix '' | Out-Null
    } else { Add-Result 'Persistence: restart' 'FAIL' 'server did not restart' }
    Stop-Api $App '(end of application checks)'
    $App = $null
} else { Add-Result 'Persistence' 'NOT VERIFIED' 'application did not start' }

Write-Section 'DEMO, THRESHOLD AND BENCHMARK'
if ($InstallOk) {
    $c = Invoke-Logged -Exe $VenvPy -Arguments @('examples\demo.py', '--latency-ms', '50')
    $default = $script:LastOutput
    $line3 = ($default -split "`n" | Select-String -Pattern 'Request 3' -Context 0, 1 | ForEach-Object { $_.Context.PostContext } | Select-Object -First 1)
    Add-Result 'Demo (threshold 0.90): request 2 HIT, request 3 MISS' $(if ($c -eq 0 -and $default -match "Request 2[^\n]*\n\s*CACHE HIT" -and "$line3" -match 'CACHE MISS') { 'PASS' } else { 'FAIL' })
    $c = Invoke-Logged -Exe $VenvPy -Arguments @('examples\demo.py', '--threshold', '0.8', '--latency-ms', '0')
    $line3 = ($script:LastOutput -split "`n" | Select-String -Pattern 'Request 3' -Context 0, 1 | ForEach-Object { $_.Context.PostContext } | Select-Object -First 1)
    Add-Result 'Demo --threshold 0.8 shows the documented false positive (request 3 HIT)' $(if ($c -eq 0 -and "$line3" -match 'CACHE HIT') { 'PASS' } else { 'FAIL' }) ("$line3".Trim())
    $c = Invoke-Logged -Exe $VenvPy -Arguments @('scripts\evaluate_threshold.py')
    Add-Result 'Threshold evaluation script (0.70-0.95)' $(if ($c -eq 0) { 'PASS' } else { 'FAIL' })
    $c = Invoke-Logged -Exe $VenvPy -Arguments @('scripts\benchmark.py')
    Add-Result 'Benchmark (exact vs reworded vs false positive)' $(if ($c -eq 0) { 'PASS' } else { 'FAIL' })
}

# ============================================================================= NEURAL
Write-Section 'NEURAL EMBEDDING'
if ($SkipNeural) {
    Add-Result 'Neural embedding' 'NOT VERIFIED' 'skipped with -SkipNeural'
} elseif (-not $PyExe) {
    Add-Result 'Neural embedding' 'NOT VERIFIED' 'no Python'
} else {
    $NeuralVenv = Join-Path $TempRoot 'neural-venv'
    $NeuralPy = Join-Path $NeuralVenv 'Scripts\python.exe'
    $savedHf = $env:HF_HOME
    $env:HF_HOME = Join-Path $TempRoot 'hf-cache'   # keep the model out of your user profile
    Write-Log "Separate venv: $NeuralVenv (so PyTorch never touches .venv). HF_HOME=$env:HF_HOME"
    Invoke-Logged -Exe $PyExe -Arguments ($PyPrefix + @('-m', 'venv', $NeuralVenv)) | Out-Null
    Invoke-Logged -Exe $NeuralPy -Arguments @('-m', 'pip', 'install', '--upgrade', 'pip') | Out-Null
    $c = Invoke-Logged -Exe $NeuralPy -Arguments @('-m', 'pip', 'install', '-e', '.[sentence-transformers,dev]')
    if ($c -ne 0) {
        Add-Result 'pip install -e ".[sentence-transformers]"' 'FAIL' 'see output above'
        Add-Result 'NEURAL MODEL' 'NOT VERIFIED' 'install failed'
    } else {
        Add-Result 'pip install -e ".[sentence-transformers]"' 'PASS'
        $c = Invoke-Logged -Exe $NeuralPy -Arguments @('scripts\neural_check.py')
        if ($c -eq 0) {
            Add-Result 'NEURAL MODEL: real model loaded and similarities measured' 'PASS' 'see table above'
            $env:RUN_NEURAL_TESTS = '1'
            $c = Invoke-Logged -Exe $NeuralPy -Arguments @('-m', 'pytest', '-m', 'neural', '-ra', '-s')
            Remove-Item Env:RUN_NEURAL_TESTS -ErrorAction SilentlyContinue
            Add-Result 'pytest -m neural (real model)' $(if ($c -eq 0) { 'PASS' } else { 'FAIL' }) (Get-PytestSummary $script:LastOutput)
            $c = Invoke-Logged -Exe $NeuralPy -Arguments @('scripts\evaluate_threshold.py', '--embedding-provider', 'sentence-transformers', '--show-pairs')
            Add-Result 'Threshold evaluation with neural embedder' $(if ($c -eq 0) { 'PASS' } else { 'FAIL' })
            $c = Invoke-Logged -Exe $NeuralPy -Arguments @('scripts\benchmark.py', '--embedding-provider', 'sentence-transformers', '--latency-ms', '0')
            Add-Result 'Benchmark with neural embedder' $(if ($c -eq 0) { 'PASS' } else { 'FAIL' })
        } elseif ($c -eq 2) {
            Add-Result 'NEURAL MODEL' 'NOT VERIFIED' 'MODEL DOWNLOAD UNAVAILABLE (see error above)'
        } else {
            Add-Result 'NEURAL MODEL' 'FAIL' "neural_check.py exit code $c"
        }
    }
    $env:HF_HOME = $savedHf
}

# ============================================================================= DOCKER
Write-Section 'DOCKER'
if ($SkipDocker) {
    Add-Result 'Docker' 'NOT VERIFIED' 'skipped with -SkipDocker'
} elseif (-not $DockerUsable) {
    Add-Result 'DOCKER' 'NOT VERIFIED' 'DOCKER NOT AVAILABLE'
} elseif (-not (Test-PortFree 8000)) {
    Add-Result 'DOCKER' 'NOT VERIFIED' 'port 8000 is in use; docker-compose.yml publishes 8000'
} else {
    $proj = @('compose', '-p', 'llmopsverify')
    $c = Invoke-Logged -Exe $DockerExe -Arguments @('compose', 'config', '--quiet')
    Add-Result 'docker compose config --quiet (valid; values not printed)' $(if ($c -eq 0) { 'PASS' } else { 'FAIL' })
    $c = Invoke-Logged -Exe $DockerExe -Arguments ($proj + @('build'))
    Add-Result 'docker compose build (image)' $(if ($c -eq 0) { 'PASS' } else { 'FAIL' })
    if ($c -eq 0) {
        $c = Invoke-Logged -Exe $DockerExe -Arguments ($proj + @('up', '-d'))
        $up = $false
        $deadline = (Get-Date).AddSeconds(90)
        while ((Get-Date) -lt $deadline) {
            if ((Get-HttpStatus 'http://127.0.0.1:8000/health') -eq 200) { $up = $true; break }
            Start-Sleep -Seconds 2
        }
        Add-Result 'Container healthy (GET /health)' $(if ($up) { 'PASS' } else { 'FAIL' })
        if ($up) {
            Invoke-Smoke -PythonExe $VenvPy -Url 'http://127.0.0.1:8000' -Phase 'basic' -Prefix 'Docker: ' | Out-Null
            Invoke-Smoke -PythonExe $VenvPy -Url 'http://127.0.0.1:8000' -Phase 'persist-seed' -Prefix 'Docker: ' | Out-Null
            Invoke-Logged -Exe $DockerExe -Arguments ($proj + @('restart')) | Out-Null
            $up = $false
            $deadline = (Get-Date).AddSeconds(90)
            while ((Get-Date) -lt $deadline) {
                if ((Get-HttpStatus 'http://127.0.0.1:8000/health') -eq 200) { $up = $true; break }
                Start-Sleep -Seconds 2
            }
            if ($up) {
                Invoke-Smoke -PythonExe $VenvPy -Url 'http://127.0.0.1:8000' -Phase 'persist-check' -Prefix 'Docker (after restart): ' | Out-Null
            } else { Add-Result 'Docker restart' 'FAIL' 'not healthy after restart' }
        }
        Invoke-Logged -Exe $DockerExe -Arguments ($proj + @('logs', '--no-color')) | Out-Null
        Invoke-Logged -Exe $DockerExe -Arguments ($proj + @('down', '-v')) | Out-Null
        Write-Log '(containers and the llmopsverify_cache-data volume removed; the built image is kept)'
    }
}

# ============================================================================= FRESH CLONE
Write-Section 'FRESH CLONE'
if ($SkipFreshClone) {
    Add-Result 'Fresh clone' 'NOT VERIFIED' 'skipped with -SkipFreshClone'
} elseif (-not ($GitOk -and $PyExe)) {
    Add-Result 'Fresh clone' 'NOT VERIFIED' 'needs git repository and Python'
} else {
    $Fresh = Join-Path $TempRoot 'fresh-clone'
    $c = Invoke-Logged -Exe $GitExe -Arguments @('clone', '--quiet', $ProjectRoot, $Fresh)
    Add-Result 'Fresh clone: git clone' $(if ($c -eq 0) { 'PASS' } else { 'FAIL' })
    if ($c -eq 0) {
        Invoke-Logged -Exe $GitExe -Arguments @('-C', $Fresh, 'ls-files') | Out-Null
        $FreshPy = Join-Path $Fresh '.venv\Scripts\python.exe'
        $c = Invoke-Logged -Exe $PyExe -Arguments ($PyPrefix + @('-m', 'venv', '.venv')) -WorkDir $Fresh
        $c = Invoke-Logged -Exe $FreshPy -Arguments @('-m', 'pip', 'install', '--upgrade', 'pip') -WorkDir $Fresh
        $c = Invoke-Logged -Exe $FreshPy -Arguments @('-m', 'pip', 'install', '-e', '.[dev]') -WorkDir $Fresh
        Add-Result 'Fresh clone: venv + pip install -e ".[dev]"' $(if ($c -eq 0) { 'PASS' } else { 'FAIL' })
        Copy-Item -LiteralPath (Join-Path $Fresh '.env.example') -Destination (Join-Path $Fresh '.env')
        Write-Log 'copied .env.example -> .env (README step)'
        $c = Invoke-Logged -Exe $FreshPy -Arguments @('-m', 'pytest', '-q') -WorkDir $Fresh
        Add-Result 'Fresh clone: pytest' $(if ($c -eq 0) { 'PASS' } else { 'FAIL' }) (Get-PytestSummary $script:LastOutput)
        $FreshPort = Get-FreePort 8100
        $FreshApp = Start-Api -PythonExe $FreshPy -WorkDir $Fresh -Port $FreshPort -EnvVars @{ PORT = "$FreshPort"; MOCK_LLM_LATENCY_MS = '200' } -Name 'fresh-app'
        Add-Result 'Fresh clone: server starts with .env from .env.example' $(if ($FreshApp.Up) { 'PASS' } else { 'FAIL' })
        if ($FreshApp.Up) {
            Invoke-Smoke -PythonExe $FreshPy -Url "http://127.0.0.1:$FreshPort" -Phase 'basic' -Prefix 'Fresh clone: ' -WorkDir $Fresh | Out-Null
            $c = Invoke-Logged -Exe $FreshPy -Arguments @('-m', 'llmops_cache.cli', '--url', "http://127.0.0.1:$FreshPort", 'metrics') -WorkDir $Fresh
            Add-Result 'Fresh clone: observe metrics (CLI)' $(if ($c -eq 0) { 'PASS' } else { 'FAIL' })
        }
        Stop-Api $FreshApp '(fresh clone)'
        $c = Invoke-Logged -Exe $FreshPy -Arguments @('examples\demo.py', '--latency-ms', '50') -WorkDir $Fresh
        Add-Result 'Fresh clone: python examples/demo.py' $(if ($c -eq 0 -and $script:LastOutput -match 'CACHE HIT') { 'PASS' } else { 'FAIL' })
        Invoke-Logged -Exe $GitExe -Arguments @('-C', $Fresh, 'status', '--porcelain=v1') | Out-Null
        Add-Result 'Fresh clone: running everything leaves no untracked/unignored files' $(if (-not $script:LastOutput.Trim()) { 'PASS' } else { 'FAIL' }) $script:LastOutput.Trim()
    }
}

# ============================================================================= CLEANUP + FINAL
Write-Section 'FINAL RESULT'
foreach ($s in @($Servers)) { Stop-Api $s '(cleanup)' }
foreach ($name in @('.pytest_cache', '.mypy_cache', '.ruff_cache')) {
    $dir = Join-Path $ProjectRoot $name
    if (Test-Path -LiteralPath $dir) { Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction SilentlyContinue }
}
Get-ChildItem -LiteralPath $ProjectRoot -Recurse -Directory -Filter '__pycache__' -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notmatch '\\\.venv' } |
    ForEach-Object { Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue }
Write-Log 'removed regenerable caches from the project (.pytest_cache, .mypy_cache, .ruff_cache, __pycache__)'

if ($GitOk) {
    Invoke-Logged -Exe $GitExe -Arguments @('status', '--porcelain=v1', '--untracked-files=all') | Out-Null
    $final = $script:LastOutput.Trim()
    Add-Result 'Git working tree clean at the end' $(if (-not $final) { 'PASS' } else { 'FAIL' }) $final
    Invoke-Logged -Exe $GitExe -Arguments @('log', '--oneline', '--decorate', '-n', '10') | Out-Null
}
Add-Result 'GitHub push' 'NOT PERFORMED' 'this script never pushes'

$tempSafe = $TempRoot.StartsWith($env:TEMP) -and ($TempRoot -match 'llmops-verify-\d{8}-\d{6}$')
if ($KeepTemp) {
    Write-Log "temp dir kept: $TempRoot"
} elseif ($tempSafe -and (Test-Path -LiteralPath $TempRoot)) {
    Remove-Item -LiteralPath $TempRoot -Recurse -Force -ErrorAction SilentlyContinue
    Write-Log "temp dir removed: $TempRoot"
}

Write-Log ''
Write-Log ('{0,-14} {1}' -f 'STATUS', 'CHECK')
Write-Log ('{0,-14} {1}' -f '------', '-----')
foreach ($r in $Results) {
    Write-Log ('{0,-14} {1}{2}' -f $r.Status, $r.Check, $(if ($r.Detail) { "  -- $($r.Detail)" } else { '' }))
}
$failed = @($Results | Where-Object { $_.Status -eq 'FAIL' })
$unverified = @($Results | Where-Object { $_.Status -in @('NOT VERIFIED', 'NOT AVAILABLE') })
Write-Log ''
Write-Log ("checks: {0}   PASS: {1}   FAIL: {2}   NOT VERIFIED / NOT AVAILABLE: {3}" -f $Results.Count, @($Results | Where-Object { $_.Status -eq 'PASS' }).Count, $failed.Count, $unverified.Count)
if ($failed.Count -eq 0) {
    Write-Log 'OVERALL: NO FAILURES (items marked NOT VERIFIED / NOT AVAILABLE were not tested)'
} else {
    Write-Log 'OVERALL: FAILURES PRESENT'
}
Write-Log ("finished: {0}  (duration {1:n0} s)" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'), ((Get-Date) - $StartTime).TotalSeconds)
Write-Log "log file: $LogFile"
