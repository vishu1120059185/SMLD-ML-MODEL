# Drive e2 in bounded bursts; each burst resumes from the checkpoint.
# Progress is guaranteed even if this script is interrupted, because
# e2 saves results after every completed model.
$ErrorActionPreference = 'Continue'
$burstSeconds = 300
$maxBursts = 14
$wd = "C:\Users\AIML-DL-18\Desktop\AIML - 99\SMLD ML MODEL\stat-twin"
Set-Location $wd
$env:PYTHONIOENCODING = 'utf-8'

for ($b = 1; $b -le $maxBursts; $b++) {
    $proc = Start-Process -FilePath python `
        -ArgumentList "-u", "-m", "stattwin.experiments.e2_model_comparison",
                      "--ds", "FD001", "--profile", "fast" `
        -WorkingDirectory $wd -PassThru -NoNewWindow `
        -RedirectStandardOutput "e2_burst.log" -RedirectStandardError "e2_burst.err"

    $finished = $proc.WaitForExit($burstSeconds * 1000)
    if ($finished) {
        Write-Output "=== burst $b : process exited ($($proc.ExitCode)) ==="
    } else {
        Write-Output "=== burst $b : window elapsed, stopping child ==="
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    }

    Get-Content e2_burst.log -ErrorAction SilentlyContinue |
        Select-String -Pattern "Evaluating|Done in|Resuming|Skipping|Completed|Models evaluated" |
        ForEach-Object { "   $_" }

    $state = python -c @"
import json, pathlib
d = json.loads(pathlib.Path('results/e2_model_comparison/e2_results.json').read_text(encoding='utf-8'))
print('PARTIAL', d.get('partial'), 'N', len(d['models']), [m['model'] for m in d['models']])
"@
    Write-Output $state
    if ($state -match "PARTIAL False") { Write-Output "ALL DONE"; break }
    if (-not $finished -and $b -eq $maxBursts) { Write-Output "burst budget exhausted" }
}
