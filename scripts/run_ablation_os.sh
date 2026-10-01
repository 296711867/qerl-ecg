#!/bin/bash
# OS 版消融：去质量/去弃权/λ扫描（DDDQN，OS 轨迹）
set -e
cd "$(dirname "$0")/.."
PY="C:/Asoftware/anaconda3/python.exe"
export KMP_DUPLICATE_LIB_OK=TRUE

TT='data/processed/traj_train_f0_9class_w10_f0_os.npz'
VT='data/processed/traj_val_f0_9class_w10_f0_os.npz'

$PY src/train_rl.py --train-traj $TT --val-traj $VT --algo dddqn \
    --no-quality-state --tag abl_noquality_os > outputs/logs/abl_noquality_os.log 2>&1
echo "[os-abl] no-quality done"
$PY src/train_rl.py --train-traj $TT --val-traj $VT --algo dddqn \
    --no-abstain --tag abl_noabstain_os > outputs/logs/abl_noabstain_os.log 2>&1
echo "[os-abl] no-abstain done"
for LAM in 0.00 0.01 0.05 0.10; do
  $PY src/train_rl.py --train-traj $TT --val-traj $VT --algo dddqn \
      --wait-cost $LAM --tag rl_lambda${LAM}_os > outputs/logs/rl_lambda${LAM}_os.log 2>&1
  echo "[os-abl] lambda=$LAM done"
done
echo "[os-abl] ALL DONE"
