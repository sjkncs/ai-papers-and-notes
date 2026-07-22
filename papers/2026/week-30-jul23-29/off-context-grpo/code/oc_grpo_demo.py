#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
oc_grpo_demo.py
================
论文概念复现 / Conceptual reproduction:
  Off-Context GRPO: Learning to Reason on Hard Problems using Privileged Information
  arXiv: 2607.19313

本脚本用一个**可控合成任务**演示论文的核心机制，不依赖任何真实 LLM：
  1) "学习悬崖"(learning cliff)：在极难题上，vanilla GRPO 从'自身策略 p'采样，组内
     奖励全为 0 → 组归一化优势(advantage)全为 0 → 梯度≈0，模型学不到任何东西。
  2) Off-Context GRPO：改从带"特权信息/提示"的**引导策略 q** 采样(off-context rollout)，
     此时能采到正确解、奖励非零、优势非零；再用**重要性校正**把梯度指回"无提示目标 p"。

一个重要的洞察 / A key insight (脚本会实证给出):
  朴素、精确的重要性权重 w = p/q 会因为 IS 恒等式而把梯度**还原成 vanilla 的零梯度**
  (E_{a~q}[ (p/q) A ∇logp ] = E_{a~p}[ A ∇logp ]，而后者在悬崖上正是 0)。
  因此论文强调要"避免未校正训练的不稳定"——实践中需要对权重做**截断/有界化**
  (truncated importance sampling) 与方差控制。本脚本用有界权重 w∈[w_min,w_max]
  作为该稳定化手段的最小 stand-in，从而让学习信号既非零又低方差。

任务 / Task:
  单步"推理"题：从 K 个候选中选 1 个，只有 CORRECT 那个 reward=1。
  无提示策略 p 初始给正确答案的概率极低(=难题)；引导策略 q 在正确答案 logit 上加提示加成。
  真正要优化/评估的是无提示策略 p 本身。

运行 / Run:  python oc_grpo_demo.py
依赖 / Deps: torch (CPU 可跑)
"""

import torch
import torch.nn.functional as F

torch.manual_seed(0)

# ----------------------------------------------------------------------------
# 超参数 / Hyper-parameters
# ----------------------------------------------------------------------------
K = 20                # 候选答案数 / candidates
CORRECT = 7           # 正确答案索引 / correct index
GROUP = 32            # GRPO 组大小 G / group size
STEPS = 300
LR = 0.5
BASE_CORRECT = -4.0   # 无提示策略下正确答案的初始 logit(很低=难题)/ low logit => hard problem
HINT_BONUS = 8.0      # 特权提示在正确 logit 上的加成 / privileged hint bonus
W_MIN, W_MAX = 0.25, 5.0   # 有界(截断)重要性权重 / bounded (truncated) importance weight


def make_policy_logits():
    """构造'难题'初始策略：正确答案 logit 很低 → p(correct)≈0.001。"""
    logits = torch.zeros(K)
    logits[CORRECT] = BASE_CORRECT
    return torch.nn.Parameter(logits + 0.01 * torch.randn(K))


def group_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """GRPO 组内归一化优势 A_i=(r_i-mean)/(std+eps)。
    若组内奖励全相同(全 0) → A 全为 0 → 这正是'学习悬崖'的数学根源。"""
    return (rewards - rewards.mean()) / (rewards.std() + 1e-8)


def sample_and_reward(probs: torch.Tensor):
    """采样一组动作并给可验证奖励(==CORRECT →1 否则 0)。"""
    dist = torch.distributions.Categorical(probs=probs)
    actions = dist.sample((GROUP,))
    rewards = (actions == CORRECT).float()
    return actions, rewards


# ============================================================================
# 方案 A：Vanilla GRPO —— 从自身策略 p 采样，在难题上撞'学习悬崖'
# Scheme A: Vanilla GRPO — samples from p itself, hits the cliff
# ============================================================================
def train_vanilla_grpo():
    print("=" * 72)
    print("[A] Vanilla GRPO —— 演示'学习悬崖' / demonstrating the learning cliff")
    print("=" * 72)
    logits = make_policy_logits()
    opt = torch.optim.SGD([logits], lr=LR)
    for step in range(STEPS):
        probs = F.softmax(logits, dim=-1)
        actions, rewards = sample_and_reward(probs.detach())   # 从 p 自身采样 / sample from p
        adv = group_advantages(rewards)                        # 难题上 rewards 常全 0 → adv 全 0
        logp = torch.log(probs[actions] + 1e-12)
        loss = -(adv * logp).mean()
        opt.zero_grad(); loss.backward()
        gnorm = logits.grad.norm().item()
        opt.step()
        if step % 60 == 0 or step == STEPS - 1:
            print(f"  step {step:3d} | 组内成功/{GROUP}: {int(rewards.sum()):2d} "
                  f"| grad_norm: {gnorm:.2e} | p(correct): {F.softmax(logits,-1)[CORRECT].item():.4f}")
    pc = F.softmax(logits, -1)[CORRECT].item()
    print(f"  => 难题上组内几乎从不成功 → 优势≈0 → 梯度≈0 → p(correct) 几乎不动。")
    print(f"     rewards ~all-zero → advantage ~0 → gradient ~0 → no learning.\n")
    return pc


# ============================================================================
# 方案 B：Off-Context GRPO —— 引导策略 q 采样 + 有界重要性校正
# Scheme B: OC-GRPO — guided rollouts from q + bounded importance correction
# ============================================================================
def train_oc_grpo():
    print("=" * 72)
    print("[B] Off-Context GRPO —— 特权提示引导 + 有界重要性校正")
    print("    privileged hint (off-context rollouts) + bounded importance correction")
    print("=" * 72)
    logits = make_policy_logits()          # 待优化的无提示目标策略 p_theta
    opt = torch.optim.SGD([logits], lr=LR)
    for step in range(STEPS):
        p_probs = F.softmax(logits, dim=-1)                    # 无提示目标策略 p

        # 构造引导(带特权提示)策略 q：正确答案 logit 上加 HINT_BONUS
        guided_logits = logits.detach().clone()
        guided_logits[CORRECT] += HINT_BONUS
        q_probs = F.softmax(guided_logits, dim=-1)

        # 从 q 采样(off-context rollout)：现在能采到正确解、奖励非零
        actions, rewards = sample_and_reward(q_probs)
        adv = group_advantages(rewards)                        # rewards 含 1 → adv 不再全 0

        # 有界(截断)重要性权重 w=clip(p/q, [w_min,w_max])：把 q-rollout 的更新指回 p，
        # 同时控制方差、避免朴素 p/q 因 IS 恒等式而'退化回零梯度'。
        p_a, q_a = p_probs[actions], q_probs[actions]
        w = (p_a.detach() / (q_a.detach() + 1e-12)).clamp(W_MIN, W_MAX)

        logp = torch.log(p_probs[actions] + 1e-12)             # 对无提示策略 p 求 log-prob
        loss = -(w * adv * logp).mean()                        # OC-GRPO 目标
        opt.zero_grad(); loss.backward()
        gnorm = logits.grad.norm().item()
        opt.step()
        if step % 60 == 0 or step == STEPS - 1:
            print(f"  step {step:3d} | 组内成功/{GROUP}: {int(rewards.sum()):2d} "
                  f"| grad_norm: {gnorm:.2e} | p(correct): {F.softmax(logits,-1)[CORRECT].item():.4f}")
    pc = F.softmax(logits, -1)[CORRECT].item()
    print(f"  => 引导采样提供非零奖励与非零优势；有界重要性校正把梯度指回无提示目标 p，")
    print(f"     p(correct) 稳步上升，跨过'学习悬崖'。")
    print(f"     Guided rollouts give non-zero advantage; bounded IS correction steers the")
    print(f"     gradient back to the unguided target — p(correct) climbs, cliff overcome.\n")
    return pc


# ============================================================================
# 诊断：朴素、无界的 p/q 会因 IS 恒等式退化回 vanilla 的零梯度
# Diagnostic: naive UNBOUNDED p/q collapses back to vanilla's zero gradient
# ============================================================================
def diagnostic_naive_is():
    print("=" * 72)
    print("[诊断/Diagnostic] 朴素无界 w=p/q → 退化回零梯度 / naive unbounded IS collapses")
    print("=" * 72)
    logits = make_policy_logits()
    opt = torch.optim.SGD([logits], lr=LR)
    for step in range(STEPS):
        p_probs = F.softmax(logits, dim=-1)
        guided_logits = logits.detach().clone(); guided_logits[CORRECT] += HINT_BONUS
        q_probs = F.softmax(guided_logits, dim=-1)
        actions, rewards = sample_and_reward(q_probs)
        adv = group_advantages(rewards)
        w = (p_probs[actions].detach() / (q_probs[actions].detach() + 1e-12))  # 无截断 / no clip
        logp = torch.log(p_probs[actions] + 1e-12)
        loss = -(w * adv * logp).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    pc = F.softmax(logits, -1)[CORRECT].item()
    print(f"  无界 IS 最终 p(correct): {pc:.4f}  (≈ vanilla，几乎不学 / barely learns)\n")
    return pc


if __name__ == "__main__":
    print("\n>>> 复现目标 / Goal: vanilla GRPO 在难题上梯度消失；OC-GRPO(有界校正)恢复学习信号。\n")
    p_vanilla = train_vanilla_grpo()
    p_oc = train_oc_grpo()
    p_naive = diagnostic_naive_is()

    print("=" * 72)
    print("对比 / Summary")
    print("=" * 72)
    print(f"  Vanilla GRPO           最终 p(correct): {p_vanilla:.4f}")
    print(f"  OC-GRPO (bounded IS)   最终 p(correct): {p_oc:.4f}")
    print(f"  Naive unbounded IS     最终 p(correct): {p_naive:.4f}")
    print(f"  OC-GRPO vs Vanilla 提升: {p_oc / max(p_vanilla, 1e-9):.1f}x")
    print("\n注意 / Note: 概念级复现。真实实现需在 token 级做重要性校正与 clip/KL 正则，")
    print("并采用更强的方差控制；此处以有界权重 w∈[w_min,w_max] 作为稳定化的最小 stand-in。")
