# -*- coding: utf-8 -*-
"""早期识别 MDP：向量化环境（在预计算轨迹上运行，极快）。

MDP:
  状态 s_t = [9类概率, max_prob, top1-top2 margin, 熵, 质量预测, 已观察比例, bias]
  动作 a_t ∈ {WAIT_1S, WAIT_2S, STOP, ABSTAIN}
  奖励 r_t = -0.03/s 等待成本; STOP: +1 对 / -2 错; ABSTAIN: +0.3 真坏质量 / -0.4 否则
  终止: STOP / ABSTAIN / 离线标注段结束时强制 STOP
  默认不暴露未来段长；历史 checkpoint 必须显式设 reveal_wmax=True。
"""
import numpy as np

WAIT1, WAIT2, STOP, ABSTAIN = 0, 1, 2, 3
N_ACTIONS = 4
T_MAX = 10
STATE_DIM = 15  # causal state: 9 probs + max + margin + entropy + quality + elapsed + bias


def featurize(probs, entropy, quality, w, wmax=None):
    """批量状态特征。probs: (B,9) 其余: (B,)"""
    top2 = np.partition(probs, -2, axis=1)[:, -2:]
    pieces = [
        probs,
        top2[:, -1:],
        (top2[:, -1] - top2[:, -2])[:, None],
        entropy[:, None],
        quality[:, None],
        (w / T_MAX)[:, None],
    ]
    if wmax is not None:  # retained only to load and audit historical policies
        pieces.append((wmax / T_MAX)[:, None])
    pieces.append(np.ones((len(probs), 1), dtype=np.float32))
    return np.concatenate(pieces, axis=1).astype(np.float32)


class VecEarlyEnv:
    """B 个并行 episode；已完成 episode 免疫步进（reward=0, 状态不变）。

    消融开关:
      use_quality=False   状态中质量特征置零（质量感知消融）
      allow_abstain=False 屏蔽 ABSTAIN 动作（弃权机制消融）
    """

    def __init__(self, traj, reward=None, use_quality=True, allow_abstain=True,
                 reveal_wmax=False):
        self.t = traj
        self.r = reward or dict(correct=1.0, wrong=-2.0, wait_per_second=-0.03,
                                 appropriate_abstain=0.3, unnecessary_abstain=-0.4)
        self.use_quality = use_quality
        self.allow_abstain = allow_abstain
        self.reveal_wmax = reveal_wmax

    def reset(self, idx: np.ndarray):
        self.idx = np.asarray(idx)
        self.B = len(self.idx)
        self.w = np.ones(self.B, dtype=np.int64)      # 当前观察秒数（1-based）
        self.wmax = self.t["wmax"][self.idx]
        self.done = np.zeros(self.B, dtype=bool)
        self.terminal = np.full(self.B, -1, dtype=np.int64)  # 终止动作（STOP/ABSTAIN）
        self.cum_r = np.zeros(self.B, dtype=np.float32)
        return self._states(), self._valid_actions()

    def _states(self):
        j = self.w - 1
        quality = self.t["quality"][self.idx, j] if self.use_quality \
            else np.zeros(self.B, dtype=np.float32)
        return featurize(self.t["probs"][self.idx, j], self.t["entropy"][self.idx, j],
                         quality, self.w.astype(np.float64),
                         self.wmax.astype(np.float64) if self.reveal_wmax else None)

    def _valid_actions(self):
        m = np.ones((self.B, N_ACTIONS), dtype=bool)
        if self.reveal_wmax:
            m[:, WAIT1] = self.w + 1 <= self.wmax
            m[:, WAIT2] = self.w + 2 <= self.wmax
        else:
            m[:, WAIT1] = self.w < T_MAX
            m[:, WAIT2] = self.w < T_MAX
        m[:, ABSTAIN] = self.allow_abstain
        return m

    def step(self, actions: np.ndarray):
        """执行动作。返回 (next_states, rewards, dones_new, valid_actions)。"""
        live = ~self.done
        a = np.where(live, actions, STOP)
        r = np.zeros(self.B, dtype=np.float32)

        wait_mask = np.isin(a, [WAIT1, WAIT2]) & live
        adv = np.where(a == WAIT2, 2, 1)
        adv = np.where(wait_mask, np.minimum(adv, self.wmax - self.w), 0)
        r += adv * self.r["wait_per_second"]
        w_before = self.w.copy()
        self.w = np.where(wait_mask, self.w + adv, self.w)

        # 主动 STOP，或等待推进到 wmax 强制 STOP
        stop = live & ((a == STOP) | ((self.w >= self.wmax) & wait_mask))
        abst = live & (a == ABSTAIN)

        if stop.any():
            # 决策依据：主动 STOP 用当时观察；等待到头用推进后的观察
            j = np.where(w_before[stop] == self.w[stop], w_before[stop] - 1,
                         self.w[stop] - 1)
            pred = self.t["probs"][self.idx[stop], j].argmax(1)
            y = self.t["labels9"][self.idx[stop]]
            r[stop] += np.where(pred == y, self.r["correct"], self.r["wrong"])
            self.terminal[stop] = STOP
        if abst.any():
            if "qtrue_steps" in self.t:
                bad = self.t["qtrue_steps"][self.idx[abst], self.w[abst] - 1] > 0.5
            else:
                bad = self.t["qtrue"][self.idx[abst]] > 0.5
            r[abst] += np.where(bad, self.r["appropriate_abstain"],
                                self.r["unnecessary_abstain"])
            self.terminal[abst] = ABSTAIN

        newly_done = stop | abst
        self.done = self.done | newly_done
        self.cum_r += r
        return self._states(), r, newly_done, self._valid_actions()
