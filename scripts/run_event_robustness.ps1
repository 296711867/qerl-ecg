# Run the two extra RL seeds and the quality-state ablation after run_event_v2.ps1.
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
$pythonExe = 'C:\Asoftware\anaconda3\python.exe'
$variant = 'v2b_ce_sqrt'

function Invoke-Python($arguments) {
    & $pythonExe @arguments
    if ($LASTEXITCODE -ne 0) { throw "Python failed: $($arguments -join ' ')" }
}

foreach ($seed in @(43, 44)) {
    foreach ($fold in 0..4) {
        $tag = "event_rl_f${fold}_${variant}_s${seed}"
        if (-not (Test-Path "outputs/checkpoints/$tag.pt")) {
            Invoke-Python @('src/train_rl.py',
                '--train-traj', "data/processed/event_traj_train_f${fold}_${variant}.npz",
                '--val-traj', "data/processed/event_traj_val_f${fold}_${variant}.npz",
                '--abstain-penalty', '1.0', '--min-coverage', '0.9',
                '--seed', "$seed", '--tag', $tag)
        }
    }
}
foreach ($fold in 0..4) {
    $tag = "event_rl_noquality_f${fold}_${variant}"
    if (-not (Test-Path "outputs/checkpoints/$tag.pt")) {
        Invoke-Python @('src/train_rl.py',
            '--train-traj', "data/processed/event_traj_train_f${fold}_${variant}.npz",
            '--val-traj', "data/processed/event_traj_val_f${fold}_${variant}.npz",
            '--abstain-penalty', '1.0', '--min-coverage', '0.9',
            '--no-quality-state', '--tag', $tag)
    }
}
Invoke-Python @('scripts/eval_event_robustness.py')
