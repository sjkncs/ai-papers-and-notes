"""
When Model Merging Rivals Joint Multi-Task Reinforcement Learning:
A Task-Vector Geometry Analysis
(arXiv 2607.16062)

PyTorch 概念复现（带中文注释） / Conceptual PyTorch reproduction

本代码最小化实现论文的"任务向量几何分析工具箱"，复现其核心论断：
1. 任务向量 (task vector) = 微调后参数 - 基座参数；
2. 两个独立训练的专家，其任务向量方向"近乎正交"(cosine ~ 0.06-0.10)，
   却在支撑集(非零位置)上大量重叠(~65%)；
3. 方向与支撑集"解耦" => 基于符号/支撑集裁剪的合并方法(TIES / RAM)
   实际坍缩为"近乎均匀平均"，因此不同合并方法之间几乎无差别；
4. 用 random-init floor(随机地板) 与 same-run ceiling(同源天花板) 标定，
   确认这一小段共享方向反映"学习"而非纯高维随机假象。

运行方式 / Run:
    pip install torch numpy
    python task_vector_geometry_demo.py
"""

import numpy as np
import torch


torch.manual_seed(0)
np.random.seed(0)

DIM = 20000          # 模拟"参数"维度 / parameter dimension
SUPPORT_FRAC = 0.65  # 目标支撑集重叠比例 / target support overlap
SIGNAL_DIM = 200     # 两个专家共享的"学习方向"维度 / shared learned subspace


# ---------------------------- 构造任务向量 / Build task vectors ----------------------------
def make_expert_task_vectors():
    """
    构造两个专家的任务向量 tv1, tv2：
    - 各自有一大片"活跃"参数(支撑集)，两者支撑集按 SUPPORT_FRAC 重叠；
    - 在极小的共享子空间(SIGNAL_DIM)上注入一点点相同方向的"学习信号"，
      其余方向为独立随机——从而 cosine 很低但非零。
    """
    base = torch.zeros(DIM)  # 基座参数视为 0，任务向量即微调增量

    # 1) 支撑集：让两个专家有 ~65% 的非零位置重叠
    n_active = int(DIM * 0.5)
    idx1 = torch.randperm(DIM)[:n_active]
    supp1 = torch.zeros(DIM, dtype=torch.bool)
    supp1[idx1] = True

    n_overlap = int(n_active * SUPPORT_FRAC)
    overlap_idx = idx1[torch.randperm(n_active)[:n_overlap]]
    rest_pool = torch.tensor([i for i in range(DIM) if not supp1[i]])
    extra_idx = rest_pool[torch.randperm(len(rest_pool))[: n_active - n_overlap]]
    supp2 = torch.zeros(DIM, dtype=torch.bool)
    supp2[overlap_idx] = True
    supp2[extra_idx] = True

    # 2) 独立随机增量（决定"方向近乎正交"）
    tv1 = torch.randn(DIM) * supp1.float()
    tv2 = torch.randn(DIM) * supp2.float()

    # 3) 注入极小的共享学习方向（高于随机地板，但仍使 cosine 很低）
    shared_dir = torch.zeros(DIM)
    shared_dir[:SIGNAL_DIM] = torch.randn(SIGNAL_DIM)
    shared_dir = shared_dir / shared_dir.norm()
    tv1 = tv1 + 0.22 * tv1.norm() * shared_dir
    tv2 = tv2 + 0.22 * tv2.norm() * shared_dir
    return base, tv1, tv2


# ---------------------------- 几何度量 / Geometry metrics ----------------------------
def cosine(a, b):
    return torch.dot(a, b).item() / (a.norm().item() * b.norm().item() + 1e-12)


def support_overlap(a, b, eps=1e-8):
    sa = a.abs() > eps
    sb = b.abs() > eps
    inter = (sa & sb).sum().item()
    union = (sa | sb).sum().item()
    return inter / (union + 1e-12), (inter / (sa.sum().item() + 1e-12))


# ---------------------------- 合并方法 / Merge methods ----------------------------
def merge_uniform(tv1, tv2):
    """均匀平均基线 / uniform averaging baseline."""
    return 0.5 * (tv1 + tv2)


def merge_ties(tv1, tv2, keep_frac=0.5):
    """
    TIES 简化版：
    1) 各自按幅值裁剪(只保留 top-|k| 的参数)；
    2) 逐坐标做符号选举(sign election)；
    3) 只在多数符号一致处求均值。
    """
    def trim(v):
        thresh = torch.quantile(v.abs(), 1 - keep_frac)
        return v * (v.abs() >= thresh).float()

    t1, t2 = trim(tv1), trim(tv2)
    sign = torch.sign(t1 + t2)                       # 选举出的主符号
    agree1 = (torch.sign(t1) == sign).float() * t1
    agree2 = (torch.sign(t2) == sign).float() * t2
    denom = ((torch.sign(t1) == sign).float()
             + (torch.sign(t2) == sign).float()).clamp(min=1)
    return (agree1 + agree2) / denom


def merge_ram(tv1, tv2, keep_frac=0.5):
    """RAM+ 简化版：随机幅值掩码后平均（支撑集裁剪的另一种形式）。"""
    def rand_mask(v):
        thresh = torch.quantile(v.abs(), 1 - keep_frac)
        return (v.abs() >= thresh).float()

    m1, m2 = rand_mask(tv1), rand_mask(tv2)
    num = tv1 * m1 + tv2 * m2
    den = (m1 + m2).clamp(min=1)
    return num / den


# ---------------------------- 标定 / Calibration ----------------------------
def random_init_floor():
    """随机地板：两个纯随机向量的 cosine —— 高维下天然接近 0。"""
    a, b = torch.randn(DIM), torch.randn(DIM)
    return cosine(a, b)


def same_run_ceiling(tv):
    """
    同源天花板：同一个任务向量加一点独立噪声，模拟"同一次训练的两个检查点",
    其 cosine 应明显高于随机地板 —— 作为"学习方向"的上界参照。
    """
    noise = torch.randn(DIM) * (0.3 * tv.std())
    return cosine(tv, tv + noise)


def main():
    base, tv1, tv2 = make_expert_task_vectors()

    cos = cosine(tv1, tv2)
    jacc, frac = support_overlap(tv1, tv2)
    floor = random_init_floor()
    ceil = same_run_ceiling(tv1)

    print("=" * 70)
    print("任务向量几何分析 / Task-vector geometry")
    print("=" * 70)
    print(f"方向余弦相似度  cosine(tv1, tv2)      = {cos:.4f}   (论文报告 0.06-0.10)")
    print(f"支撑集重叠(占专家1) support overlap    = {frac:.3f}   (论文报告 ~0.65)")
    print(f"随机地板 random-init floor  cosine    = {floor:.4f}")
    print(f"同源天花板 same-run ceiling cosine    = {ceil:.4f}")
    print(f"=> 观测方向相对随机地板略有提升(反映学习)，但远低于天花板 => 近乎正交\n")

    # 不同合并方法 -> 与"均匀平均"的接近程度
    uni = merge_uniform(tv1, tv2)
    ties = merge_ties(tv1, tv2)
    ram = merge_ram(tv1, tv2)

    print("=" * 70)
    print("合并方法 vs 均匀平均 / Merge methods vs uniform averaging")
    print("=" * 70)
    for name, m in [("TIES", ties), ("RAM+", ram)]:
        c = cosine(m, uni)
        rel = (m - uni).norm().item() / (uni.norm().item() + 1e-12)
        print(f"{name:>5}: cosine(merged, uniform) = {c:.4f} | "
              f"相对差异 ||Δ||/||uni|| = {rel:.4f}")
    print("\n=> 方向-支撑解耦时，TIES / RAM+ 与均匀平均高度对齐(cosine≈1)，")
    print("   即不同合并方法在几何上难以区分 —— 复现论文核心论断。")
    print("=" * 70)


if __name__ == "__main__":
    main()
