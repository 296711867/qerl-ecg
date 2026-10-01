"""Minimal ALERT-inspired offline DDQN trigger on frozen event trajectories.

State: predicted-class one-hot, max probability, top-two margin, elapsed/10.
Actions: wait one second or stop. No quality or future event length in state.
"""
import json
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.policies import evaluate_sim

VARIANT = "v2b_ce_sqrt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GAMMA, LR, TAU, HIDDEN, BATCH = 1.0, 1e-4, 3e-3, 32, 512
REWARD = dict(correct=1.0, wrong=-2.0, wait_per_second=-0.03,
              appropriate_abstain=0.3, unnecessary_abstain=-1.0)


def load(fold, part):
    return dict(np.load(ROOT / f"data/processed/event_traj_{part}_f{fold}_{VARIANT}.npz"))


def state(probs, steps):
    top = np.partition(probs, -2, axis=1)[:, -2:]
    pred = probs.argmax(axis=1)
    onehot = np.eye(9, dtype=np.float32)[pred]
    return np.column_stack((onehot, top[:, 1], top[:, 1] - top[:, 0],
                            steps / 10.0)).astype(np.float32)


class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(12, HIDDEN), nn.LayerNorm(HIDDEN),
                                 nn.ReLU(), nn.Linear(HIDDEN, 2))

    def forward(self, x):
        return self.net(x)


def make_buffer(traj):
    """Enumerate wait/stop transitions; wait at event end forces classification."""
    states, actions, rewards, next_states, dones, next_masks = [], [], [], [], [], []
    probs, labels = traj["probs"], traj["labels9"]
    for i, nsteps in enumerate(traj["wmax"]):
        for j in range(int(nsteps)):
            s = state(probs[i:i + 1, j], np.array([j + 1]))[0]
            cls_reward = 1.0 if probs[i, j].argmax() == labels[i] else -2.0
            states.append(s); actions.append(1); rewards.append(cls_reward)
            next_states.append(s); dones.append(True); next_masks.append([False, False])
            if j + 1 < nsteps:
                s2 = state(probs[i:i + 1, j + 1], np.array([j + 2]))[0]
                wait_reward, done, mask = -0.03, False, [True, True]
            else:
                # The simulator ends at the labeled segment boundary without exposing it in state.
                s2 = s
                wait_reward, done, mask = cls_reward, True, [False, False]
            states.append(s); actions.append(0); rewards.append(wait_reward)
            next_states.append(s2); dones.append(done); next_masks.append(mask)
    return tuple(torch.as_tensor(a, device=DEVICE) for a in (
        np.asarray(states, np.float32), np.asarray(actions, np.int64),
        np.asarray(rewards, np.float32), np.asarray(next_states, np.float32),
        np.asarray(dones, np.float32), np.asarray(next_masks, bool)))


@torch.no_grad()
def rollout(traj, model):
    probs, wmax = traj["probs"], traj["wmax"]
    n = len(wmax)
    step_idx = np.zeros(n, dtype=np.int64)
    stop_w = np.zeros(n, dtype=np.int64)
    live = np.ones(n, dtype=bool)
    for _ in range(10):
        idx = np.flatnonzero(live)
        if not len(idx):
            break
        p = probs[idx, step_idx[idx]]
        s = torch.as_tensor(state(p, step_idx[idx] + 1), device=DEVICE)
        action = model(s).argmax(1).cpu().numpy()
        stop = action == 1
        stop_w[idx[stop]] = step_idx[idx[stop]] + 1
        live[idx[stop]] = False
        wait_idx = idx[~stop]
        if len(wait_idx):
            ended = ((step_idx[wait_idx] + 1 >= wmax[wait_idx]) |
                     (step_idx[wait_idx] == 9))
            stop_w[wait_idx[ended]] = step_idx[wait_idx[ended]] + 1
            live[wait_idx[ended]] = False
            step_idx[wait_idx[~ended]] += 1
    assert np.all(stop_w > 0)
    return dict(stop_w=stop_w, abstain=np.zeros(n, dtype=bool))


def train_fold(fold, tr, va, epochs, tag):
    transitions = make_buffer(tr)
    online, target = QNet().to(DEVICE), QNet().to(DEVICE)
    target.load_state_dict(online.state_dict())
    opt = torch.optim.Adam(online.parameters(), lr=LR)
    n = len(transitions[0])
    best_score, best_epoch, best_state = -float("inf"), 0, None
    history = []
    for epoch in range(1, epochs + 1):
        online.train()
        order = torch.randperm(n, device=DEVICE)
        losses = []
        for ids in order.split(BATCH):
            s, a, r, s2, done, mask2 = (x[ids] for x in transitions)
            q = online(s).gather(1, a[:, None]).squeeze(1)
            with torch.no_grad():
                q_online = online(s2).masked_fill(~mask2, -1e9)
                a2 = q_online.argmax(1)
                q_next = target(s2).gather(1, a2[:, None]).squeeze(1)
                q_next = q_next.masked_fill(~mask2.any(1), 0)
                y = r + GAMMA * (1 - done) * q_next
            loss = (q - y).square().mean()
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                for tp, p in zip(target.parameters(), online.parameters()):
                    tp.lerp_(p, TAU)
            losses.append(float(loss.detach()))
        online.eval()
        val_metrics = evaluate_sim(va, rollout(va, online), r=REWARD)
        history.append(dict(epoch=epoch, loss=float(np.mean(losses)),
                            val_return=val_metrics["mean_return"],
                            val_f1=val_metrics["macro_f1_all"],
                            val_time=val_metrics["mean_decision_time"]))
        if val_metrics["mean_return"] > best_score:
            best_score, best_epoch = val_metrics["mean_return"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in online.state_dict().items()}
    ckpt_path = ROOT / f"outputs/checkpoints/{tag}_f{fold}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=best_state, best_epoch=best_epoch, history=history,
                    state_features="argmax_onehot+maxprob+margin+elapsed",
                    actions=["wait1", "stop"], reward=REWARD), ckpt_path)
    model = QNet().to(DEVICE); model.load_state_dict(best_state); model.eval()
    return model, dict(fold=fold, n_train=len(tr["labels9"]), n_val=len(va["labels9"]),
                       transitions=n, best_epoch=best_epoch,
                       best_val_return=best_score,
                       checkpoint=str(ckpt_path.relative_to(ROOT)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--tag", default=f"alert_inspired_{VARIANT}")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    torch.manual_seed(42); np.random.seed(42)
    folds = json.loads((ROOT / "data/splits/folds.json").read_text())
    models, selections, trainval_cases = [], [], []
    for fold in range(5):
        tr, va = load(fold, "train"), load(fold, "val")
        train_cases, val_cases = set(tr["case_id"]), set(va["case_id"])
        assert train_cases <= set(folds[f"fold_{fold}"]["train"])
        assert val_cases <= set(folds[f"fold_{fold}"]["val"])
        assert train_cases.isdisjoint(val_cases)
        assert len(tr["labels9"]) and len(va["labels9"])
        model, selected = train_fold(fold, tr, va, args.epochs, args.tag)
        models.append(model); selections.append(selected)
        trainval_cases.append((train_cases, val_cases))
        print(f"fold={fold} selected_epoch={selected['best_epoch']} "
              f"val_return={selected['best_val_return']:.4f}", flush=True)

    # Test trajectories are first loaded after every fold's checkpoint is selected.
    results = []
    for fold, model in enumerate(models):
        te = load(fold, "test")
        test_cases = set(te["case_id"])
        assert test_cases <= set(folds[f"fold_{fold}"]["test"])
        assert test_cases.isdisjoint(trainval_cases[fold][0] | trainval_cases[fold][1])
        metrics = evaluate_sim(te, rollout(te, model), r=REWARD)
        results.append(dict(fold=fold, n_test=len(te["labels9"]), metrics=metrics))
        print(f"test fold={fold} F1={metrics['macro_f1_all']:.4f} "
              f"time={metrics['mean_decision_time']:.2f}s", flush=True)

    keys = ("macro_f1_all", "mean_decision_time", "coverage", "mean_return")
    summary = {k: dict(mean=float(np.mean([r["metrics"][k] for r in results])),
                       sd=float(np.std([r["metrics"][k] for r in results], ddof=1)))
               for k in keys}
    def json_safe(value):
        if isinstance(value, dict):
            return {k: json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [json_safe(v) for v in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        if isinstance(value, np.integer):
            return int(value)
        return value

    out = dict(protocol="ALERT-inspired minimal same-trajectory baseline",
               variant=VARIANT, seed=42, device=DEVICE,
               state="9-way argmax one-hot + max probability + margin + elapsed/10; no wmax",
               actions=["wait one second", "stop"],
               reward=REWARD, training="exhaustive static transitions + offline DDQN",
               model_selection="maximum validation mean return; test not loaded until all selections finish",
               epochs=args.epochs, tag=args.tag,
               test_used_for_selection=False, folds=selections, test_results=results,
               test_summary=summary)
    out = json_safe(out)
    table = ROOT / f"outputs/tables/{args.tag}.json"
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text(json.dumps(out, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# ALERT-inspired 同轨迹基线", "",
             "v2b_ce_sqrt 预计算事件轨迹；患者级五折。状态仅含 argmax one-hot、max probability、margin、elapsed/10，不含 wmax；动作 wait1/stop。训练使用穷举静态转移和离线 DDQN，按验证集平均回报选 epoch。所有五折 checkpoint 选定后才载入各自 test；测试集不参与选择。此为 ALERT-inspired 最小实现，不是官方 ALERT 复现。", "",
             "| 指标 | 五折均值±SD |", "|---|---:|"]
    names = {"macro_f1_all": "全体 macro-F1", "mean_decision_time": "平均观察秒数",
             "coverage": "coverage", "mean_return": "平均回报"}
    for k in keys:
        lines.append(f"| {names[k]} | {summary[k]['mean']:.4f}±{summary[k]['sd']:.4f} |")
    lines += ["", "| Fold | Test N | Val最佳epoch | Test macro-F1 | Test观察秒 | Test回报 |",
              "|---:|---:|---:|---:|---:|---:|"]
    for sel, res in zip(selections, results):
        m = res["metrics"]
        lines.append(f"| {res['fold']} | {res['n_test']} | {sel['best_epoch']} | "
                     f"{m['macro_f1_all']:.4f} | {m['mean_decision_time']:.3f} | {m['mean_return']:.4f} |")
    md = ROOT / f"outputs/tables/{args.tag}.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {table.relative_to(ROOT)} and {md.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
