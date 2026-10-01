#!/bin/bash
# 阶段3：消融实验
#   A: 去质量特征   B: 去弃权   C: 等待成本 λ 扫描（0.03 已有 rl_dddqn_f0）
set -e
cd "$(dirname "$0")/.."
PY="C:/Asoftware/anaconda3/python.exe"

TRAIN_TRAJ="data/processed/traj_train_f0_9class_w10_f0.npz"
VAL_TRAJ="data/processed/traj_val_f0_9class_w10_f0.npz"

$PY src/train_rl.py --train-traj $TRAIN_TRAJ --val-traj $VAL_TRAJ --algo dddqn \
    --no-quality-state --tag abl_noquality_f0 > outputs/logs/abl_noquality.log 2>&1
echo "[ablation] no-quality done"

$PY src/train_rl.py --train-traj $TRAIN_TRAJ --val-traj $VAL_TRAJ --algo dddqn \
    --no-abstain --tag abl_noabstain_f0 > outputs/logs/abl_noabstain.log 2>&1
echo "[ablation] no-abstain done"

for LAM in 0.00 0.01 0.05 0.10; do
  $PY src/train_rl.py --train-traj $TRAIN_TRAJ --val-traj $VAL_TRAJ --algo dddqn \
      --wait-cost $LAM --tag rl_lambda${LAM}_f0 > outputs/logs/rl_lambda${LAM}.log 2>&1
  echo "[ablation] lambda=$LAM done"
done

$PY scripts/eval_ablation.py > outputs/logs/eval_ablation.log 2>&1
echo "[ablation] ALL DONE"
