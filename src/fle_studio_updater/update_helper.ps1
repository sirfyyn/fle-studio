# Written out by FLE Studio when it updates itself on Windows.
#
# The running program file of FLE Studio is locked on Windows, so uv cannot replace it while FLE Studio is open.
# This script waits for FLE Studio to end, installs the new version with uv, and starts FLE Studio again: the new
# version when the install worked, the old one when it failed. It writes what happened to the log and to the
# result file, which the next FLE Studio shows once.
param([string]$Job)
$ErrorActionPreference = 'Continue'
$j = Get-Content -LiteralPath $Job -Raw -Encoding UTF8 | ConvertFrom-Json

function Note($text) {
    $line = "[{0}] {1}" -f (Get-Date -Format s), $text
    Add-Content -LiteralPath $j.log -Value $line -Encoding UTF8
}

Note "helper started; waiting for FLE Studio (process $($j.pid)) to end"
try { Wait-Process -Id $j.pid -Timeout 90 -ErrorAction Stop } catch { }

$uv = @($j.uv)
$uvArgs = @()
if ($uv.Count -gt 1) { $uvArgs = $uv[1..($uv.Count - 1)] }
& $uv[0] @uvArgs tool install --python $j.python --force $j.spec *>> $j.log
$code = $LASTEXITCODE
if ($code -eq 0) { $status = 'ok'; Note 'uv finished' } else { $status = 'failed'; Note "uv exited with code $code" }

$result = @{ status = $status; tag = $j.tag; code = $code; time = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds() }
$result | ConvertTo-Json | Set-Content -LiteralPath $j.result -Encoding UTF8

$cmd = @($j.start)
$quoted = @()
if ($cmd.Count -gt 1) {
    foreach ($a in $cmd[1..($cmd.Count - 1)]) {
        if ($a -match '\s') { $quoted += '"' + $a + '"' } else { $quoted += $a }
    }
}
Note "starting FLE Studio again: $($cmd -join ' ')"
if ($quoted.Count -gt 0) {
    Start-Process -FilePath $cmd[0] -ArgumentList $quoted
} else {
    Start-Process -FilePath $cmd[0]
}
