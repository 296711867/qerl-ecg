# Reuse the v2b classifier trajectories; retrain only policies without future event length.
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
$pythonExe = 'C:\Asoftware\anaconda3\python.exe'
$variant = 'v2b_ce_sqrt'
$tagVariant = "${variant}_causal"

function Invoke-Python($arguments) {
    & $pythonExe @arguments
    if ($LASTEXITCODE -ne 0) { throw "Python failed: $($arguments -join ' ')" }
}

foreach ($fold in 0..4) {
    $tag = "event_rl_f${fold}_${tagVariant}"
    if (-not (Test-Path "outputs/checkpoints/$tag.pt")) {
        Invoke-Python @('src/train_rl.py',
            '--train-traj', "data/processed/event_traj_train_f${fold}_${variant}.npz",
            '--val-traj', "data/processed/event_traj_val_f${fold}_${variant}.npz",
            '--algo', 'dddqn', '--abstain-penalty', '1.0',
            '--min-coverage', '0.9', '--causal-state', '--tag', $tag)
    }
}
Invoke-Python @('scripts/eval_event_cv.py', $variant, 'causal')

foreach ($seed in @(43, 44)) {
    foreach ($fold in 0..4) {
        $tag = "event_rl_f${fold}_${tagVariant}_s${seed}"
        if (-not (Test-Path "outputs/checkpoints/$tag.pt")) {
            Invoke-Python @('src/train_rl.py',
                '--train-traj', "data/processed/event_traj_train_f${fold}_${variant}.npz",
                '--val-traj', "data/processed/event_traj_val_f${fold}_${variant}.npz",
                '--abstain-penalty', '1.0', '--min-coverage', '0.9',
                '--causal-state', '--seed', "$seed", '--tag', $tag)
        }
    }
}
foreach ($fold in 0..4) {
    $tag = "event_rl_noquality_f${fold}_${tagVariant}"
    if (-not (Test-Path "outputs/checkpoints/$tag.pt")) {
        Invoke-Python @('src/train_rl.py',
            '--train-traj', "data/processed/event_traj_train_f${fold}_${variant}.npz",
            '--val-traj', "data/processed/event_traj_val_f${fold}_${variant}.npz",
            '--abstain-penalty', '1.0', '--min-coverage', '0.9',
            '--causal-state', '--no-quality-state', '--tag', $tag)
    }
}
Invoke-Python @('scripts/eval_event_robustness.py', 'causal')
