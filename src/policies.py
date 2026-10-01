# -*- coding: utf-8 -*-
"""早期识别策略仿真与指标。

策略在预计算轨迹上运行（probs/entropy/quality, 每步 = 多观察 1 秒）。
动作语义（与 RL 一致）:
  WAIT   继续观察（若已达 w_max 则强制 STOP）
  STOP   输出当前 argmax 分类
  ABSTAIN 拒绝判断（不计入分类指标，计 coverage/selective risk）

指标（chatGPT建议3 第23-25节）:
  macro_f1 / balanced_acc   非弃权 episode 上的分类性能
  mean_decision_time / earliness  全部 episode 的决策时刻与 t_stop/T_max
  mean_decision_time_covered     非弃权 episode 的决策时刻
  coverage                  非弃权比例
  selective_risk            非弃权样本上的错误率
  badq_severe_alarm_rate    坏质量 episode 中预测危急类的比例（非误报率）
  badq_false_alarm_rate     坏质量且真实非危急类中预测危急类的比例
"""
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

CLASSES = ["N", "AFIB/AFL", "AVB", "SND", "SR-mPAC-BT", "SR-mPVC-BT", "SVTA", "VT", "MAT"]
T_MAX = 10
# 动作编码
WAIT, STOP, ABSTAIN = 0, 1, 2

# 奖励参数（configs/project.yaml reward 节）
R = dict(correct=1.0, wrong=-2.0, wait_per_second=-0.03,
         appropriate_abstain=0.3, unnecessary_abstain=-0.4)


def step_probs(traj, i, w_idx):
    """第 i 个 episode、前缀索引 w_idx（0-based, 即观察 w_idx+1 秒）的模型输出。"""
    return traj["probs"][i, w_idx], traj["entropy"][i, w_idx], traj["quality"][i, w_idx]


def simulate_threshold(traj, tau=0.95, stop_at_max=True):
    """置信度阈值策略：max prob ≥ tau 即停，否则观察到最大长度。"""
    n = len(traj["labels9"])
    stop_w = np.ones(n, dtype=np.int64)  # 1-based 观察秒数
    for i in range(n):
        wmax = traj["wmax"][i]
        for j in range(wmax):
            p = traj["probs"][i, j]
            stop_w[i] = j + 1
            if p.max() >= tau:
                break
    return dict(stop_w=stop_w, abstain=np.zeros(n, dtype=bool))


def simulate_time_threshold(traj, tau0, slope):
    """Early exit when confidence clears a linearly time-varying threshold."""
    n = len(traj["labels9"])
    stop_w = traj["wmax"].astype(np.int64).copy()
    confidence = traj["probs"].max(axis=2)
    for w in range(1, T_MAX + 1):
        threshold = np.clip(tau0 + slope * (w - 1) / (T_MAX - 1), 0.01, 0.99)
        eligible = (traj["wmax"] >= w) & (stop_w == traj["wmax"])
        stop_w[eligible & (confidence[:, w - 1] >= threshold)] = w
    return dict(stop_w=stop_w, abstain=np.zeros(n, dtype=bool))


def simulate_fixed(traj, w_sec: int):
    n = len(traj["labels9"])
    return dict(stop_w=np.minimum(traj["wmax"], min(w_sec, T_MAX)).astype(np.int64),
                abstain=np.zeros(n, dtype=bool))


def decisions_from_sim(traj, sim):
    """根据停止时刻输出预测。"""
    stop_w, abstain = sim["stop_w"], sim["abstain"]
    n = len(stop_w)
    pred = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        if abstain[i]:
            continue
        pred[i] = traj["probs"][i, stop_w[i] - 1].argmax()
    return pred


def episode_returns(traj, sim, r=R):
    """逐 episode 回报（供策略比较与 RL 校验）。"""
    stop_w, abstain = sim["stop_w"], sim["abstain"]
    y = traj["labels9"]
    qtrue = traj["qtrue"]
    out = np.zeros(len(y))
    for i in range(len(y)):
        rew = (stop_w[i] - 1) * r["wait_per_second"]
        if abstain[i]:
            observed_q = (traj["qtrue_steps"][i, stop_w[i] - 1]
                          if "qtrue_steps" in traj else qtrue[i])
            rew += (r["appropriate_abstain"] if observed_q > 0.5
                    else r["unnecessary_abstain"])
        else:
            pred = traj["probs"][i, stop_w[i] - 1].argmax()
            rew += r["correct"] if pred == y[i] else r["wrong"]
        out[i] = rew
    return out


def evaluate_sim(traj, sim, r=R):
    stop_w, abstain = sim["stop_w"], sim["abstain"]
    y = traj["labels9"]
    keep = ~abstain
    pred_all = decisions_from_sim(traj, sim)
    pred = pred_all[keep]
    yk = y[keep]
    m = dict(
        coverage=float(keep.mean()),
        mean_decision_time=float(stop_w.mean()),
        mean_decision_time_covered=float(stop_w[keep].mean()),
        earliness=float((stop_w / T_MAX).mean()),
        macro_f1=float(f1_score(yk, pred, average="macro", zero_division=0)),
        macro_f1_all=float(f1_score(y, pred_all, labels=list(range(9)),
                                    average="macro", zero_division=0)),
        balanced_acc=float(balanced_accuracy_score(yk, pred)),
        accuracy=float((yk == pred).mean()),
        selective_risk=float((yk != pred).mean()),
        mean_return=float(episode_returns(traj, sim, r).mean()),
    )
    rec = recall_score(yk, pred, average=None, labels=list(range(9)), zero_division=0)
    m["per_class_recall"] = {CLASSES[i]: float(rec[i]) for i in range(9)}
    # 危急报警集合为 VT/SVTA/AVB；SND 不纳入此终点。
    bad = traj["qtrue"] > 0.5
    if bad.sum() > 0:
        severe_ids = [CLASSES.index(c) for c in ["VT", "SVTA", "AVB"]]
        severe_pred = np.isin(pred_all, severe_ids)
        severe_true = np.isin(y, severe_ids)
        m["badq_severe_alarm_rate"] = float(severe_pred[bad].mean())
        negative = bad & ~severe_true
        positive = bad & severe_true
        m["badq_false_alarm_rate"] = (float(severe_pred[negative].mean())
                                      if negative.any() else float("nan"))
        m["badq_severe_recall"] = (float(severe_pred[positive].mean())
                                    if positive.any() else float("nan"))
        m["badq_n"] = int(bad.sum())
        m["badq_false_alarm_n"] = int((severe_pred & negative).sum())
    return m
