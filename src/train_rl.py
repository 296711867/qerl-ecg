# -*- coding: utf-8 -*-
"""阶段2b/2c：在预计算轨迹上训练 DQN / Double-Dueling-DQN。

流程: 收集(ε-greedy, 向量化) → replay → Double-Dueling 更新 → 验证轨迹贪心评估
用法:
  python src/train_rl.py --train-traj traj_train_f0_<ckpt>.npz --val-traj traj_val_f0_<ckpt>.npz --tag rl_dddqn_f0
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.policies import evaluate_sim
from src.rl.agent import DQNAgent, Replay
from src.rl.env import ABSTAIN, STOP, STATE_DIM, VecEarlyEnv


def collect_and_train(env, agent, replay, episode_idx, batch_env, updates, warmup=1000):
    """一遍训练集：收集经验 + 在线更新。"""
    perm = np.random.permutation(episode_idx)
    losses = []
    for s0 in range(0, len(perm), batch_env):
        ii = perm[s0:s0 + batch_env]
        s, mask = env.reset(ii)
        while not env.done.all():
            a = agent.act(s, mask)
            a[env.done] = STOP
            live = ~env.done  # 步进前仍存活的 episode
            s2, r, newly, mask2 = env.step(a)
            for k in np.flatnonzero(live):
                replay.push(s[k], a[k], r[k], s2[k], newly[k], mask2[k])
            if len(replay.buf) >= warmup:
                for _ in range(updates):
                    losses.append(agent.update(replay.sample(128)))
            s, mask = s2, mask2
    return float(np.mean(losses)) if losses else 0.0


def simulate_agent(traj, agent, batch=4096, env_kw=None):
    """贪心策略仿真，返回 policies.evaluate_sim 兼容的 sim dict。"""
    env = VecEarlyEnv(traj, **(env_kw or {}))
    idx = np.arange(len(traj["labels9"]))
    stop_w = np.zeros(len(idx), dtype=np.int64)
    abstain = np.zeros(len(idx), dtype=bool)
    for s0 in range(0, len(idx), batch):
        ii = idx[s0:s0 + batch]
        s, mask = env.reset(ii)
        while not env.done.all():
            a = agent.act(s, mask, greedy=True)
            a[env.done] = STOP
            s, r, newly, mask = env.step(a)
        stop_w[ii] = env.w
        abstain[ii] = env.terminal == ABSTAIN
    return dict(stop_w=stop_w, abstain=abstain)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-traj", required=True)
    ap.add_argument("--val-traj", required=True)
    ap.add_argument("--algo", default="dddqn",
                    choices=["dqn", "double", "dueling", "dddqn"],
                    help="dqn=vanilla, double=Double DQN, dueling=Dueling DQN, dddqn=Double-Dueling")
    ap.add_argument("--episodes", type=int, default=12, help="完整遍历训练集的轮数")
    ap.add_argument("--batch-env", type=int, default=512)
    ap.add_argument("--updates-per-step", type=int, default=2)
    ap.add_argument("--no-quality-state", action="store_true", help="消融: 状态去质量特征")
    ap.add_argument("--no-abstain", action="store_true", help="消融: 屏蔽弃权动作")
    ap.add_argument("--causal-state", action="store_true",
                    help="不向策略或动作掩码暴露未来标注段长")
    ap.add_argument("--wait-cost", type=float, default=0.03, help="reward 等待成本 λ")
    ap.add_argument("--abstain-penalty", type=float, default=0.4,
                    help="干净信号弃权的惩罚幅度")
    ap.add_argument("--min-coverage", type=float, default=0.0,
                    help="验证集 checkpoint 的最低覆盖率")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--random-horizon", action="store_true",
                    help="每轮训练随机提前截断各事件；验证轨迹保持原样")
    ap.add_argument("--tag", default="rl_dddqn")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    tr = dict(np.load(ROOT / args.train_traj))
    va = dict(np.load(ROOT / args.val_traj))
    env_kw = dict(use_quality=not args.no_quality_state,
                  allow_abstain=not args.no_abstain,
                  reveal_wmax=not args.causal_state,
                  reward=dict(correct=1.0, wrong=-2.0,
                              wait_per_second=-args.wait_cost,
                              appropriate_abstain=0.3,
                              unnecessary_abstain=-args.abstain_penalty))
    env = VecEarlyEnv(tr, **env_kw)
    horizon_rng = np.random.default_rng(args.seed + 1000)
    agent = DQNAgent(state_dim=STATE_DIM if args.causal_state else 16,
                     double=args.algo in ("double", "dddqn"),
                     dueling=args.algo in ("dueling", "dddqn"))
    replay = Replay(100000)

    best_score, hist = -1e9, []
    for ep in range(1, args.episodes + 1):
        if args.random_horizon:
            env.t = dict(tr, wmax=horizon_rng.integers(1, tr["wmax"] + 1))
        loss = collect_and_train(env, agent, replay, np.arange(len(tr["labels9"])),
                                 args.batch_env, args.updates_per_step)
        agent.sync_target()
        m = evaluate_sim(va, simulate_agent(va, agent, env_kw=env_kw),
                         r=env_kw["reward"])
        hist.append(dict(ep=ep, loss=loss, **m))
        print(f"ep{ep:02d} loss={loss:.4f} F1={m['macro_f1']:.4f} "
              f"bal={m['balanced_acc']:.4f} dt={m['mean_decision_time']:.2f}s "
              f"cov={m['coverage']:.3f} ret={m['mean_return']:.3f}", flush=True)
        if m["coverage"] >= args.min_coverage and m["mean_return"] > best_score:
            best_score = m["mean_return"]
            torch.save(dict(model=agent.q.state_dict(), hist=hist),
                       ROOT / f"outputs/checkpoints/{args.tag}.pt")

    (ROOT / "outputs/tables").mkdir(exist_ok=True, parents=True)
    (ROOT / f"outputs/tables/{args.tag}_metrics.json").write_text(json.dumps(hist, indent=1))
    if best_score == -1e9:
        raise RuntimeError(f"No checkpoint met validation coverage >= {args.min_coverage}")
    print(f"best mean return: {best_score:.3f}")


if __name__ == "__main__":
    main()
