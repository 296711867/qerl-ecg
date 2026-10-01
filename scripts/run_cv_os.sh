#!/bin/bash
# 5 折过采样版 CV（sqrt 采样器）：fold 0 的 w10 已训好（9class_w10_f0_os）
# 每折: w10_os 分类器 → 轨迹 → RL-DDDQN_os → 其余窗口；最后汇总 cv_summary_os.md
set -e
cd "$(dirname "$0")/.."
PY="C:/Asoftware/anaconda3/python.exe"
export KMP_DUPLICATE_LIB_OK=TRUE

for FOLD in 0 1 2 3 4; do
  if [ ! -f "outputs/checkpoints/9class_w10_f${FOLD}_os.pt" ]; then
    $PY src/train_classifier.py --task 9class --window 10 --fold $FOLD \
        --sampler sqrt --tag 9class_w10_f${FOLD}_os \
        > outputs/logs/os_f${FOLD}_9class_w10.log 2>&1
  fi
  echo "[os-cv] fold $FOLD: w10 classifier done"

  if [ ! -f "data/processed/traj_test_f${FOLD}_9class_w10_f${FOLD}_os.npz" ]; then
    for PART in train val test; do
      $PY src/compute_trajectories.py \
          --ckpt outputs/checkpoints/9class_w10_f${FOLD}_os.pt \
          --fold $FOLD --part $PART --no-embed \
          >> outputs/logs/os_f${FOLD}_traj.log 2>&1
    done
  fi
  echo "[os-cv] fold $FOLD: trajectories done"

  if [ ! -f "outputs/checkpoints/rl_dddqn_f${FOLD}_os.pt" ]; then
    $PY src/train_rl.py \
        --train-traj data/processed/traj_train_f${FOLD}_9class_w10_f${FOLD}_os.npz \
        --val-traj   data/processed/traj_val_f${FOLD}_9class_w10_f${FOLD}_os.npz \
        --algo dddqn --tag rl_dddqn_f${FOLD}_os \
        > outputs/logs/os_f${FOLD}_rl.log 2>&1
  fi
  echo "[os-cv] fold $FOLD: RL done"

  for W in 1 2 3 5; do
    if [ ! -f "outputs/checkpoints/9class_w${W}_f${FOLD}_os.pt" ]; then
      $PY src/train_classifier.py --task 9class --window $W --fold $FOLD \
          --sampler sqrt --tag 9class_w${W}_f${FOLD}_os \
          > outputs/logs/os_f${FOLD}_9class_w${W}.log 2>&1
    fi
  done
  echo "[os-cv] fold $FOLD: all windows done"
done

$PY scripts/eval_cv.py os > outputs/logs/eval_cv_os.log 2>&1
echo "[os-cv] ALL DONE"
