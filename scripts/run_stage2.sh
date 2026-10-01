#!/bin/bash
# 阶段2：等轨迹预计算完成 → 训练 4 个 RL 变体 → 全策略对比评估
set -e
cd "$(dirname "$0")/.."
PY="C:/Asoftware/anaconda3/python.exe"

echo "[stage2] waiting for trajectories..."
while ! grep -q "ALL_TRAJ_DONE" outputs/logs/traj_f0.log 2>/dev/null; do
  sleep 30
done
echo "[stage2] trajectories ready"

TRAIN_TRAJ="data/processed/traj_train_f0_9class_w10_f0.npz"
VAL_TRAJ="data/processed/traj_val_f0_9class_w10_f0.npz"

for ALGO in dqn double dueling dddqn; do
  $PY src/train_rl.py --train-traj $TRAIN_TRAJ --val-traj $VAL_TRAJ \
      --algo $ALGO --episodes 12 --tag rl_${ALGO}_f0 \
      > outputs/logs/rl_${ALGO}.log 2>&1
  echo "[stage2] RL $ALGO done"
done

$PY scripts/eval_policies.py > outputs/logs/eval_policies.log 2>&1
echo "[stage2] ALL DONE"
