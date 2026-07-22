#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gear_reward_demo.py
===================
论文概念复现 / Conceptual reproduction:
  Copy Less, Ground More: Overcoming Repetitive Copying in Long-Context Reasoning
  via Evidence-Aware Reinforcement Learning (GEAR)
  arXiv: 2607.19345

本脚本用一个**可控合成任务**演示 GEAR 的核心机制——证据感知奖励塑形，不依赖真实 LLM：
  GEAR 奖励 = 准确率奖励
            + λ_g · 接地奖励(与关键证据重叠)
            - λ_d · 惩罚项(与干扰材料重叠)

任务设定 / Task:
  一篇长文档被切成 N 个句子，其中一小部分是"关键证据"(evidence)，其余是"干扰材料"(distractor)。
  一个玩具策略对每个句子决定是否"抄进推理"(copy)。为正确作答需要抄到关键证据；
  但把干扰句也大量抄进来 = "重复复制(repetitive copying)"这一失效模式。
  我们对比两种奖励：
    (1) 仅准确率奖励 (accuracy-only)  —— 对抄不抄干扰句没有信号，容易保留大量无用复制；
    (2) GEAR 奖励                     —— 显式奖励接地、惩罚抄干扰，从而"少抄、多接地"。

运行 / Run:
  python gear_reward_demo.py
依赖 / Deps: torch (CPU 可跑)
"""

import torch
import torch.nn.functional as F

torch.manual_seed(0)

# ----------------------------------------------------------------------------
# 文档与超参 / Document & hyper-parameters
# ----------------------------------------------------------------------------
N = 40                      # 文档句子总数 / total sentences (模拟长上下文 long context)
N_EVIDENCE = 4              # 关键证据句数 / number of key-evidence sentences
GROUP = 64                  # 每步采样的 rollout 数 / rollouts per step
STEPS = 300
LR = 0.1
LAMBDA_G = 1.0             # 接地奖励权重 / grounding-reward weight
LAMBDA_D = 1.0             # 干扰惩罚权重 / distractor-penalty weight

# 关键证据在前 N_EVIDENCE 个位置，其余为干扰 / evidence = first indices, rest = distractors
evidence_mask = torch.zeros(N)
evidence_mask[:N_EVIDENCE] = 1.0
distractor_mask = 1.0 - evidence_mask


def rollout_metrics(copied: torch.Tensor):
    """给定一批'是否抄入'的 0/1 决策 [G, N]，计算 GEAR 所需的三类量。
    Compute the quantities GEAR needs from a batch of copy decisions."""
    # 接地重叠：抄到的关键证据 / 全部关键证据  ∈ [0,1]
    grounding = (copied * evidence_mask).sum(dim=1) / evidence_mask.sum()
    # 干扰重叠：抄到的干扰句 / 全部干扰句      ∈ [0,1]
    distractor = (copied * distractor_mask).sum(dim=1) / distractor_mask.sum()
    # "推理长度"≈ 抄入的句子总数（越长越可能是重复复制）/ reasoning length proxy
    length = copied.sum(dim=1)
    # 准确率：接地越好越可能答对（需要证据才能作答）/ accuracy grows with grounding
    acc_prob = grounding                          # 简化：答对概率 = 接地程度
    accuracy = torch.bernoulli(acc_prob)          # 采样出可验证的 0/1 正确性
    return grounding, distractor, length, accuracy


def gear_reward(grounding, distractor, accuracy):
    """GEAR 三合一奖励 / the three-in-one GEAR reward."""
    return accuracy + LAMBDA_G * grounding - LAMBDA_D * distractor


def accuracy_only_reward(grounding, distractor, accuracy):
    """对照：仅准确率奖励（对抄干扰句没有直接信号）/ baseline: accuracy only."""
    return accuracy


def train(reward_fn, name):
    print("=" * 72)
    print(f"[{name}]")
    print("=" * 72)
    # 策略参数：每个句子一个 logit，sigmoid 后为'抄入该句'的概率
    # Policy: one logit per sentence; sigmoid = probability of copying that sentence.
    logits = torch.nn.Parameter(torch.zeros(N))   # 初始 p=0.5，先验倾向"什么都抄一点"
    opt = torch.optim.Adam([logits], lr=LR)

    for step in range(STEPS):
        probs = torch.sigmoid(logits)                       # [N] 每句抄入概率
        dist = torch.distributions.Bernoulli(probs=probs)
        copied = dist.sample((GROUP,))                       # [G, N] 采样抄入决策

        grounding, distractor, length, accuracy = rollout_metrics(copied)
        rewards = reward_fn(grounding, distractor, accuracy) # [G]

        # 组内归一化优势（类似 GRPO）/ group-normalized advantage
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # REINFORCE：log π(copied) = Σ_i Bernoulli 对数似然 / policy-gradient objective
        logp = dist.log_prob(copied).sum(dim=1)              # [G]
        loss = -(adv.detach() * logp).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 60 == 0 or step == STEPS - 1:
            print(f"  step {step:3d} | 接地 grounding: {grounding.mean():.3f} "
                  f"| 抄干扰 distractor_copy: {distractor.mean():.3f} "
                  f"| 推理长度 length: {length.mean():4.1f} "
                  f"| 准确率 acc: {accuracy.mean():.3f}")

    with torch.no_grad():
        probs = torch.sigmoid(logits)
        copied = (probs > 0.5).float().unsqueeze(0)
        g, d, l, _ = rollout_metrics(copied)
    print(f"  => 最终(贪心) grounding={g.item():.3f} distractor_copy={d.item():.3f} length={l.item():.1f}\n")
    return g.item(), d.item(), l.item()


if __name__ == "__main__":
    print("\n>>> 复现目标 / Goal: 对比'仅准确率'与'GEAR'两种奖励对'重复复制'的影响。\n")
    g_base, d_base, l_base = train(accuracy_only_reward, "对照：仅准确率奖励 / Accuracy-only reward")
    g_gear, d_gear, l_gear = train(gear_reward, "GEAR：证据感知奖励塑形 / Evidence-aware reward")

    print("=" * 72)
    print("对比 / Summary")
    print("=" * 72)
    print(f"  仅准确率 / Accuracy-only : grounding={g_base:.3f}  distractor_copy={d_base:.3f}  length={l_base:.1f}")
    print(f"  GEAR                    : grounding={g_gear:.3f}  distractor_copy={d_gear:.3f}  length={l_gear:.1f}")
    print()
    print("  预期现象 / Expected: GEAR 在保持/提升接地(grounding)的同时，显著降低对干扰句的")
    print("  复制(distractor_copy↓)并缩短推理长度(length↓)，即论文所说的 'Copy Less, Ground More'。")
    print("  Expected: GEAR keeps/raises grounding while cutting distractor copying and length —")
    print("  exactly the paper's 'Copy Less, Ground More'.")
    print()
    print("  注意 / Note: 概念级复现；真实实现中接地/惩罚基于文本重叠或语义相似度，")
    print("  并需防范用改写规避 n-gram 重叠的 reward hacking。")
