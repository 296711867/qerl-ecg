# Reproduce the forward event-onset experiment from the saved annotations and folds.
param([ValidateSet('v2', 'v2b_ce_sqrt')][string]$Variant = 'v2b_ce_sqrt')
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
$pythonExe = 'C:\Asoftware\anaconda3\python.exe'

function Invoke-Python($arguments) {
    & $pythonExe @arguments
    if ($LASTEXITCODE -ne 0) { throw "Python failed: $($arguments -join ' ')" }
}

Invoke-Python @('src/data/build_event_episodes.py')
foreach ($fold in 0..4) {
    $classifier = "outputs/checkpoints/event_cls_f${fold}_${Variant}.pt"
    if (-not (Test-Path $classifier)) {
        $loss = if ($Variant -eq 'v2') { 'focal' } else { 'ce' }
        Invoke-Python @('src/train_event_classifier.py', '--fold', "$fold",
            '--loss', $loss, '--sampler', 'sqrt', '--variant', $Variant)
    }
    foreach ($part in @('train', 'val', 'test')) {
        $trajectory = "data/processed/event_traj_${part}_f${fold}_${Variant}.npz"
        if (-not (Test-Path $trajectory)) {
            Invoke-Python @('src/compute_event_trajectories.py', '--fold', "$fold",
                '--part', $part, '--variant', $Variant)
        }
    }
    $policy = "outputs/checkpoints/event_rl_f${fold}_${Variant}.pt"
    if (-not (Test-Path $policy)) {
        Invoke-Python @('src/train_rl.py',
            '--train-traj', "data/processed/event_traj_train_f${fold}_${Variant}.npz",
            '--val-traj', "data/processed/event_traj_val_f${fold}_${Variant}.npz",
            '--algo', 'dddqn', '--abstain-penalty', '1.0',
            '--min-coverage', '0.9',
            '--tag', "event_rl_f${fold}_${Variant}")
    }
}
Invoke-Python @('scripts/eval_event_cv.py', $Variant)
