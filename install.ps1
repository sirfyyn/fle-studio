<#
.SYNOPSIS
Installs FLE Studio on Windows (Windows PowerShell 5.1 or PowerShell 7).

.DESCRIPTION
1. Installs uv (https://docs.astral.sh/uv/) if it is missing.
2. Installs FLE Studio as a uv tool. uv brings its own Python 3.12, so no Python install is needed.
3. Adds a Start Menu shortcut named "FLE Studio".
4. Starts FLE Studio (skip this with -NoStart).

No administrator rights are needed. Everything goes below your user profile.

.PARAMETER NoStart
Install only; do not start FLE Studio afterwards. With "irm | iex" you cannot pass parameters, so setting
$env:FLE_STUDIO_NO_START = '1' does the same.

.PARAMETER Ref
Git tag, branch or commit of FLE Studio to install. Default: $env:FLE_STUDIO_REF, else v0.4.1.

.PARAMETER Source
Install from this folder or wheel file instead of GitHub (offline installs, CI); git is not needed then and -Ref is
ignored. Default: $env:FLE_STUDIO_SOURCE.

.EXAMPLE
irm https://raw.githubusercontent.com/sirfyyn/fle-studio/main/install.ps1 | iex

.EXAMPLE
& ([scriptblock]::Create((irm https://raw.githubusercontent.com/sirfyyn/fle-studio/main/install.ps1))) -NoStart
#>
param(
    [switch]$NoStart,
    [string]$Ref = $(if ($env:FLE_STUDIO_REF) { $env:FLE_STUDIO_REF } else { 'v0.4.1' }),
    [string]$Source = $env:FLE_STUDIO_SOURCE
)

# Everything runs inside one script block, so that "irm | iex" leaves no helper functions or variables behind
# in your PowerShell session (only the two parameters above).
& {
    $ErrorActionPreference = 'Stop'

    $repoUrl = 'https://github.com/sirfyyn/fle-studio'
    $uvInstallerUrl = 'https://astral.sh/uv/install.ps1'

    function Write-Step {
        param([string]$Text)
        Write-Host ('==> ' + $Text) -ForegroundColor Cyan
    }

    function Update-SessionPath {
        # Pick up PATH entries that installers wrote to the registry after this window was opened.
        # Entries already in this session stay where they are; new ones are appended.
        $known = @{}
        $parts = New-Object System.Collections.Generic.List[string]
        $sources = @(
            $env:Path,
            [Environment]::GetEnvironmentVariable('Path', 'User'),
            [Environment]::GetEnvironmentVariable('Path', 'Machine')
        )
        foreach ($source in $sources) {
            if (-not $source) { continue }
            foreach ($entry in $source.Split(';')) {
                $clean = [Environment]::ExpandEnvironmentVariables($entry.Trim())
                if (-not $clean) { continue }
                $key = $clean.TrimEnd('\').ToLowerInvariant()
                if (-not $known.ContainsKey($key)) {
                    $known[$key] = $true
                    $parts.Add($clean)
                }
            }
        }
        $env:Path = ($parts -join ';')
    }

    function Test-PathContains {
        param([string]$List, [string]$Dir)
        if (-not $List) { return $false }
        $want = $Dir.TrimEnd('\').ToLowerInvariant()
        foreach ($entry in $List.Split(';')) {
            $clean = [Environment]::ExpandEnvironmentVariables($entry.Trim()).TrimEnd('\').ToLowerInvariant()
            if ($clean -and ($clean -eq $want)) { return $true }
        }
        return $false
    }

    function Find-Uv {
        # uv can exist without being on PATH (a fresh install only changes the registry, not this window).
        if (Get-Command uv -CommandType Application -ErrorAction SilentlyContinue) { return $true }
        $dirs = @($env:UV_INSTALL_DIR, (Join-Path $env:USERPROFILE '.local\bin'), (Join-Path $env:USERPROFILE '.cargo\bin'))
        foreach ($dir in $dirs) {
            if ($dir -and (Test-Path -LiteralPath (Join-Path $dir 'uv.exe'))) {
                $env:Path = $dir + ';' + $env:Path
                return $true
            }
        }
        return $false
    }

    function Invoke-Live {
        # Runs a program with its output shown in this window. Returns the exit code.
        # ErrorActionPreference is relaxed for the call, so that a failing program is reported through
        # the return value (and our own message) instead of aborting with a PowerShell error.
        param([string]$File, [string[]]$Arguments)
        $previous = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            & $File @Arguments | Out-Host
            return $LASTEXITCODE
        } finally {
            $ErrorActionPreference = $previous
        }
    }

    function Invoke-Capture {
        # Runs a program and returns its exit code and its text output (stdout and stderr together).
        param([string]$File, [string[]]$Arguments)
        $previous = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            $lines = & $File @Arguments 2>&1
            $code = $LASTEXITCODE
        } finally {
            $ErrorActionPreference = $previous
        }
        $text = (($lines | ForEach-Object { "$_" }) -join "`n").Trim()
        return [pscustomobject]@{ ExitCode = $code; Text = $text }
    }

    function Install-Uv {
        Write-Step 'uv is not installed. Installing it from astral.sh (no administrator rights needed).'
        # The official one-liner is: irm https://astral.sh/uv/install.ps1 | iex
        # It runs in a child Windows PowerShell here, so that it cannot close this window and does not
        # depend on this machine's script execution policy.
        $powershell = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
        if (-not (Test-Path -LiteralPath $powershell)) { $powershell = 'powershell.exe' }
        $command = 'irm ' + $uvInstallerUrl + ' | iex'
        $code = Invoke-Live -File $powershell -Arguments @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', $command)
        if ($code -ne 0) {
            throw 'The uv installer failed (see the messages above). Install uv by hand (https://docs.astral.sh/uv/getting-started/installation/) and run this installer again.'
        }
    }

    function New-StartMenuShortcut {
        # Returns the path of the shortcut, or $null if it could not be created (not fatal).
        param([string]$Target)
        try {
            $programs = [Environment]::GetFolderPath('Programs')
            $link = Join-Path $programs 'FLE Studio.lnk'
            $shell = New-Object -ComObject WScript.Shell
            $shortcut = $shell.CreateShortcut($link)
            $shortcut.TargetPath = $Target
            $shortcut.WorkingDirectory = $env:USERPROFILE
            $shortcut.Description = 'FLE Studio: run large MoE models on your own machine'
            $shortcut.WindowStyle = 1
            $shortcut.Save()
            return $link
        } catch {
            Write-Host ('Could not create the Start Menu shortcut: ' + $_.Exception.Message) -ForegroundColor Yellow
            return $null
        }
    }

    try {
        if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
            throw 'This installer is for Windows. On macOS and Linux use install.sh (see the README).'
        }
        if ($Ref -notmatch '^[A-Za-z0-9._/+-]+$') {
            throw ('The release name may only contain letters, digits and . _ / + - (got: ' + $Ref + ').')
        }
        $startApp = -not ([bool]$NoStart -or ($env:FLE_STUDIO_NO_START -eq '1'))

        $fromSource = [bool]$Source
        if ($fromSource) {
            if (-not (Test-Path -LiteralPath $Source)) {
                throw ('The source does not exist: ' + $Source)
            }
            $Source = (Resolve-Path -LiteralPath $Source).ProviderPath
            Write-Step ('FLE Studio installer (source: ' + $Source + ')')
        } else {
            Write-Step ('FLE Studio installer (release: ' + $Ref + ')')
        }
        Update-SessionPath

        if (-not $fromSource -and -not (Get-Command git -CommandType Application -ErrorAction SilentlyContinue)) {
            throw "Git is needed, because uv fetches FLE Studio from GitHub with it. Install it with 'winget install --id Git.Git -e' or from https://git-scm.com/download/win, then open a new PowerShell window and run this installer again."
        }

        if (Find-Uv) {
            $uvVersion = (Invoke-Capture -File 'uv' -Arguments @('--version')).Text
            Write-Step ('Found ' + $uvVersion + '.')
        } else {
            Install-Uv
            Update-SessionPath
            if (-not (Find-Uv)) {
                throw 'uv was installed, but this window cannot find it. Open a new PowerShell window and run the installer again.'
            }
            $uvVersion = (Invoke-Capture -File 'uv' -Arguments @('--version')).Text
            Write-Step ('Installed ' + $uvVersion + '.')
        }

        if ($fromSource) {
            Write-Step ('Installing FLE Studio from ' + $Source + ' (uv sets up Python 3.12 for it if needed) ...')
            $spec = $Source
        } else {
            Write-Step 'Installing FLE Studio from GitHub (uv sets up Python 3.12 for it if needed) ...'
            $spec = 'git+' + $repoUrl + '@' + $Ref
        }
        $code = Invoke-Live -File 'uv' -Arguments @('tool', 'install', '--python', '3.12', '--force', $spec)
        if ($code -ne 0) {
            throw ("Installing FLE Studio failed (see the messages above). Common causes: no internet access, the release '" + $Ref + "' does not exist (try -Ref main), or FLE Studio is running right now (close it and run this installer again).")
        }

        $binDir = ''
        $binQuery = Invoke-Capture -File 'uv' -Arguments @('tool', 'dir', '--bin')
        if (($binQuery.ExitCode -eq 0) -and $binQuery.Text) { $binDir = $binQuery.Text.Split("`n")[0].Trim() }
        if (-not $binDir) { $binDir = Join-Path $env:USERPROFILE '.local\bin' }
        $launcher = Join-Path $binDir 'fle-studio.exe'
        if (-not (Test-Path -LiteralPath $launcher)) {
            throw ('FLE Studio was installed, but its launcher is missing in ' + $binDir + ". 'uv tool list' shows what uv installed.")
        }
        $check = Invoke-Capture -File $launcher -Arguments @('--version')
        if ($check.ExitCode -ne 0) {
            throw ('FLE Studio was installed, but "' + $launcher + ' --version" failed: ' + $check.Text)
        }

        $shortcut = New-StartMenuShortcut -Target $launcher

        $dataRoot = $env:LOCALAPPDATA
        if (-not $dataRoot) { $dataRoot = Join-Path $env:USERPROFILE 'AppData\Local' }
        $dataDir = Join-Path $dataRoot 'FLE Studio'
        if ($env:FLE_STUDIO_HOME) { $dataDir = $env:FLE_STUDIO_HOME }

        Write-Host ''
        Write-Step ($check.Text + ' is installed.')
        Write-Host ''
        if ($shortcut) {
            Write-Host '  Start:      fle-studio   (or "FLE Studio" in the Start Menu)'
        } else {
            Write-Host '  Start:      fle-studio'
        }
        Write-Host '  Update:     uv tool upgrade fle-studio'
        if ($fromSource) {
            Write-Host ('              (this install came from ' + $Source + '; run the installer again to update it)')
        } else {
            Write-Host ('              (this install follows ' + $Ref + '; to move to another release, run the installer again')
            Write-Host '               with -Ref <tag>, or set $env:FLE_STUDIO_REF before "irm | iex")'
        }
        Write-Host '  Uninstall:  uv tool uninstall fle-studio'
        if ($shortcut) {
            Write-Host ('              and delete the Start Menu shortcut: ' + $shortcut)
        }
        Write-Host ''
        Write-Host 'Settings, chats and downloaded models live in'
        Write-Host ('  ' + $dataDir)
        Write-Host 'and stay there when you uninstall. Delete that folder to remove them as well.'

        $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
        $machinePath = [Environment]::GetEnvironmentVariable('Path', 'Machine')
        $persisted = (Test-PathContains -List $userPath -Dir $binDir) -or (Test-PathContains -List $machinePath -Dir $binDir)
        if (-not $persisted) {
            Write-Host ''
            Write-Host ('Note: ' + $binDir + ' is not on your PATH. Run "uv tool update-shell" and open a new PowerShell')
            Write-Host 'window to start FLE Studio by typing fle-studio. The Start Menu shortcut works either way.'
        }
        Write-Host ''

        if ($startApp) {
            Write-Step 'Starting FLE Studio. It opens in your browser. Press Ctrl+C in this window to stop it.'
            & $launcher
        }
    } catch {
        Write-Host ''
        Write-Host ('ERROR: ' + $_.Exception.Message) -ForegroundColor Red
        # "exit" would close a window that ran "irm | iex"; only a real script file reports a failure code.
        if ($PSCommandPath) { exit 1 }
    }
}
