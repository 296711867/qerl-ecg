# v3 robustness: extra classifier seeds on the frozen split (E1) and two re-drawn
# patient-disjoint partitions (E2), all under the frozen v2b_ce_sqrt + causal-DQN protocol.
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
$pythonExe = 'C:\Asoftware\anaconda3\python.exe'

function Invoke-Python($arguments) {
    & $pythonExe @arguments
    if ($LASTEXITCODE -ne 0) { throw "Python failed: $($arguments -join ' ')" }
}

foreach ($p in @(1, 2)) {
    if (-not (Test-Path "data/splits/folds_p${p}.json")) {
        Invoke-Python @('src/data/make_splits.py', '--seed', "$p")
    }
}

# variant -> (split file, classifier seed offset)
$runs = [ordered]@{
    'v3_cs1' = @('folds', 100)
    'v3_cs2' = @('folds', 200)
    'v3_p1'  = @('folds_p1', 0)
    'v3_p2'  = @('folds_p2', 0)
}
foreach ($variant in $runs.Keys) {
    $split, $offset = $runs[$variant]
    foreach ($fold in 0..4) {
        if (-not (Test-Path "outputs/checkpoints/event_cls_f${fold}_${variant}.pt")) {
            Invoke-Python @('src/train_event_classifier.py', '--fold', "$fold",
                '--loss', 'ce', '--sampler', 'sqrt', '--variant', $variant,
                '--split', $split, '--seed-offset', "$offset")
        }
        foreach ($part in @('train', 'val', 'test')) {
            if (-not (Test-Path "data/processed/event_traj_${part}_f${fold}_${variant}.npz")) {
                Invoke-Python @('src/compute_event_trajectories.py', '--fold', "$fold",
                    '--part', $part, '--variant', $variant, '--split', $split)
            }
        }
        $tag = "event_rl_f${fold}_${variant}_causal"
        if (-not (Test-Path "outputs/checkpoints/$tag.pt")) {
            Invoke-Python @('src/train_rl.py',
                '--train-traj', "data/processed/event_traj_train_f${fold}_${variant}.npz",
                '--val-traj', "data/processed/event_traj_val_f${fold}_${variant}.npz",
                '--algo', 'dddqn', '--abstain-penalty', '1.0',
                '--min-coverage', '0.9', '--causal-state', '--tag', $tag)
        }
    }
    Invoke-Python @('scripts/eval_event_cv.py', $variant, 'causal')
}
Write-Output 'V3_RUNS_DONE'
