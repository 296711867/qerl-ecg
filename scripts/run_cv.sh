#!/bin/bash
# 5 折全 CV 主实验：fold 1-4（fold 0 已有）
# 每折顺序: w10分类器 → 轨迹(train/val/test) → RL-DDDQN → 其余窗口(1/2/3/5)
# 核心链路优先，保证中断时最重要的数据先完成
set -e
cd "$(dirname "$0")/.."
PY="C:/Asoftware/anaconda3/python.exe"
export KMP_DUPLICATE_LIB_OK=TRUE

for FOLD in 1 2 3 4; do
  # --- 核心链路: w10 → 轨迹 → RL ---
  $PY src/train_classifier.py --task 9class --window 10 --fold $FOLD \
      > outputs/logs/cv_f${FOLD}_9class_w10.log 2>&1
  echo "[cv] fold $FOLD: w10 classifier done"

  CKPT="outputs/checkpoints/9class_w10_f${FOLD}.pt"
  for PART in train val test; do
    $PY src/compute_trajectories.py --ckpt $CKPT --fold $FOLD --part $PART --no-embed \
        >> outputs/logs/cv_f${FOLD}_traj.log 2>&1
  done
  echo "[cv] fold $FOLD: trajectories done"

  $PY src/train_rl.py \
      --train-traj data/processed/traj_train_f${FOLD}_9class_w10_f${FOLD}.npz \
      --val-traj   data/processed/traj_val_f${FOLD}_9class_w10_f${FOLD}.npz \
      --algo dddqn --tag rl_dddqn_f${FOLD} \
      > outputs/logs/cv_f${FOLD}_rl.log 2>&1
  echo "[cv] fold $FOLD: RL done"

  # --- Table 2 补全: 其余窗口 ---
  for W in 1 2 3 5; do
    $PY src/train_classifier.py --task 9class --window $W --fold $FOLD \
        > outputs/logs/cv_f${FOLD}_9class_w${W}.log 2>&1
  done
  echo "[cv] fold $FOLD: all windows done"
done

$PY scripts/eval_cv.py > outputs/logs/eval_cv.log 2>&1
echo "[cv] ALL DONE"
