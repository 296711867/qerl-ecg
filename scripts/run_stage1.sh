#!/bin/bash
# 阶段1全量实验：等下载完成 → 二分类 + 9分类窗口扫描（fold 0）
set -e
cd "$(dirname "$0")/.."
PY="C:/Asoftware/anaconda3/python.exe"

echo "[stage1] waiting for download to finish..."
while ! grep -q "^Done:" outputs/logs/download.log 2>/dev/null; do
  sleep 30
done
echo "[stage1] download finished: $(tail -1 outputs/logs/download.log)"

# 1a. 二分类 baseline（10s）
$PY src/train_classifier.py --task binary --window 10 --fold 0 \
  > outputs/logs/train_binary_w10.log 2>&1
echo "[stage1] binary w10 done"

# 1b/1c. 9 分类窗口扫描 1/2/3/5/10s
for W in 1 2 3 5 10; do
  $PY src/train_classifier.py --task 9class --window $W --fold 0 \
    > outputs/logs/train_9class_w${W}.log 2>&1
  echo "[stage1] 9class w${W} done"
done
echo "[stage1] ALL DONE"
