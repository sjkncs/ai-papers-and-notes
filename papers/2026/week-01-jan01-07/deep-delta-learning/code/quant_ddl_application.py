"""
Deep Delta Learning × 量化金融应用
DDL for Quantitative Finance Applications
==========================================

基于论文 DDL (arXiv:2601.00417) 的量化金融应用实现。

三个核心应用 / Three Core Applications:
1. DDL时序预测器: 用DDL残差重写能力处理市场regime切换
2. MoE因子路由器: 借鉴MiMo的MoE架构做动态因子组合
3. 端到端投资组合优化: DDL网络直接输出投资组合权重

运行: python quant_ddl_application.py
依赖: pip install torch numpy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict


# ============================================================
# 1. DDL时序预测器 / DDL Time-Series Forecaster
# ============================================================
# 核心思想: 金融时序中市场regime会切换(动量→均值回归→震荡)
# 标准残差连接只能累加特征，无法主动"遗忘"过时regime信号
# DDL允许每层选择性重写，适配regime切换

class MarketRegimeGate(nn.Module):
    """
    市场状态门控 / Market Regime Gate

    学习识别当前市场状态(momentum/mean-reversion/volatile),
    并选择性重写隐状态中过时的regime信号。

    量化意义: 当市场从动量切换到均值回归时，
    旧的动量特征需要被主动覆写而非继续累加。
    """

    def __init__(self, d_model: int, num_regimes: int = 3):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

        # 门控: 决定哪些维度需要被重写 / Gate: which dimensions to rewrite
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

        # 目标值: 当前regime下的"正确"特征值
        # Target: the "correct" feature value under current regime
        self.target = nn.Linear(d_model, d_model)

        # Regime分类头 (辅助训练) / Regime classification head (auxiliary)
        self.regime_head = nn.Linear(d_model, num_regimes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, d_model] 时序隐状态

        Returns:
            delta_correction: [batch, seq_len, d_model] 重写修正量
            regime_logits: [batch, seq_len, num_regimes] 状态分类
        """
        normed = self.norm(x)
        g = self.gate(normed)           # [B, T, D] 门控值 ∈ [0,1]
        t = self.target(normed)         # [B, T, D] 目标值

        # Delta修正 = gate * (target - current)
        # 含义: 按gate的开度，将current向target拉近
        delta = g * (t - x)

        regime_logits = self.regime_head(normed)

        return delta, regime_logits


class DDLTimeSeriesForecaster(nn.Module):
    """
    DDL时序预测模型 / DDL Time-Series Forecasting Model

    用于金融时序预测，核心改进:
    1. 每层Transformer Block使用DDL残差替代标准残差
    2. 加入MarketRegimeGate进行regime-aware重写
    3. 输出层预测未来收益率

    与标准时序Transformer的对比:
    - 标准: x_{l+1} = x_l + f_l(x_l)  (只能累加，regime信号过时后无法消除)
    - DDL:  x_{l+1} = x_l + g*(target-x_l) + f_l(x_l)  (可重写，主动适应regime切换)
    """

    def __init__(
        self,
        num_features: int = 50,       # 输入因子数 / Number of input factors
        d_model: int = 128,           # 隐藏维度
        n_heads: int = 4,             # 注意力头数
        n_layers: int = 3,            # Transformer层数
        num_regimes: int = 3,         # 市场状态数 (动量/均值回归/震荡)
        forecast_horizon: int = 5,    # 预测未来N天收益
        dropout: float = 0.1,
    ):
        super().__init__()

        # 因子嵌入 / Factor embedding
        self.factor_proj = nn.Linear(num_features, d_model)

        # 时间位置编码 / Temporal positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, 252, d_model) * 0.02)

        # DDL Transformer Blocks
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(nn.ModuleDict({
                # 标准attention + FFN
                'norm1': nn.LayerNorm(d_model),
                'attn': nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
                'norm2': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model),
                ),
                # DDL门控修正 / DDL gated correction
                'regime_gate': MarketRegimeGate(d_model, num_regimes),
            }))

        # 预测头 / Forecast head
        self.forecast_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, forecast_horizon),
        )

        self.num_features = num_features
        self.forecast_horizon = forecast_horizon

    def forward(
        self,
        factor_data: torch.Tensor,
        regime_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            factor_data: [batch, seq_len, num_features] 因子数据 (如: 50个因子的日频数据)
            regime_labels: [batch, seq_len] 可选的regime标签 (用于辅助训练)

        Returns:
            dict with:
                'returns': [batch, forecast_horizon] 预测的未来收益
                'regime_logits': [batch, seq_len, num_regimes] 各层regime预测
                'attention_weights': 注意力权重 (可选)
        """
        B, T, F = factor_data.shape

        # 因子投影 / Factor projection
        x = self.factor_proj(factor_data)  # [B, T, d_model]

        # 加位置编码 / Add positional encoding
        x = x + self.pos_embed[:, :T, :]

        all_regime_logits = []

        for block in self.blocks:
            # --- DDL Step 1: Regime-aware delta correction ---
            delta, regime_logits = block['regime_gate'](x)
            all_regime_logits.append(regime_logits)

            # --- DDL Step 2: Standard attention ---
            normed = block['norm1'](x)
            attn_out, _ = block['attn'](normed, normed, normed)

            # --- DDL Step 3: Combine (delta + attention) ---
            x = x + delta + attn_out

            # --- Standard FFN with residual ---
            x = x + block['ffn'](block['norm2'](x))

        # 取最后一个时间步的表示做预测
        # Use last timestep representation for forecasting
        last_hidden = x[:, -1, :]  # [B, d_model]
        returns = self.forecast_head(last_hidden)  # [B, forecast_horizon]

        # 汇总regime预测 / Aggregate regime predictions
        regime_stack = torch.stack(all_regime_logits, dim=0)  # [n_layers, B, T, num_regimes]
        avg_regime = regime_stack.mean(dim=0)  # [B, T, num_regimes]

        # Regime分类损失 (如果有标签) / Regime loss (if labels provided)
        regime_loss = None
        if regime_labels is not None:
            regime_loss = F.cross_entropy(
                avg_regime.view(-1, avg_regime.size(-1)),
                regime_labels.view(-1),
            )

        return {
            'returns': returns,
            'regime_logits': avg_regime,
            'regime_loss': regime_loss,
        }


# ============================================================
# 2. MoE因子路由器 / MoE Factor Router
# ============================================================
# 核心思想: 借鉴MiMo-V2-Flash的MoE架构
# 不同expert对应不同市场regime的因子策略
# Router根据当前市场状态动态选择因子组合

class MoEFactorRouter(nn.Module):
    """
    MoE因子路由器 / MoE Factor Router

    借鉴MiMo-V2-Flash的MoE设计:
    - 309B总参数但仅15B激活 → 多个因子expert但每次只激活top-K
    - Routing network学习根据市场状态动态选择

    量化意义:
    - Expert 1: 动量因子专家 (适用于趋势市)
    - Expert 2: 价值因子专家 (适用于均值回归市)
    - Expert 3: 波动率因子专家 (适用于高波动市)
    - Router: 根据当前市场状态选择激活哪些expert
    """

    def __init__(
        self,
        num_factors: int = 50,
        num_experts: int = 8,
        top_k: int = 2,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # 路由网络 / Routing network: 决定激活哪些expert
        self.router = nn.Sequential(
            nn.Linear(num_factors, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
        )

        # 因子专家 / Factor experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_factors, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),  # 输出: 单因子收益预测
            )
            for _ in range(num_experts)
        ])

        # 负载均衡损失辅助 / Load balancing auxiliary loss
        self.register_buffer('expert_counts', torch.zeros(num_experts))

    def forward(self, factor_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            factor_input: [batch, num_factors] 当前因子快照

        Returns:
            dict with:
                'prediction': [batch, 1] 综合预测
                'expert_weights': [batch, num_experts] expert权重
                'load_balance_loss': 负载均衡损失标量
        """
        B = factor_input.shape[0]

        # 路由: 计算每个expert的激活概率
        router_logits = self.router(factor_input)  # [B, num_experts]

        # Top-K选择 (类似MiMo的sparse activation)
        top_k_logits, top_k_indices = router_logits.topk(self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)  # [B, top_k]

        # 只激活选中的expert / Only activate selected experts
        expert_outputs = torch.zeros(B, self.num_experts, device=factor_input.device)

        for k in range(self.top_k):
            idx = top_k_indices[:, k]  # [B]
            for b in range(B):
                expert_idx = idx[b].item()
                expert_out = self.experts[expert_idx](factor_input[b:b+1])  # [1, 1]
                expert_outputs[b, expert_idx] = expert_out.squeeze()

        # 加权聚合 / Weighted aggregation
        sparse_weights = torch.zeros_like(router_logits)
        for k in range(self.top_k):
            sparse_weights.scatter_(1, top_k_indices[:, k:k+1], top_k_weights[:, k:k+1])

        prediction = (expert_outputs * sparse_weights).sum(dim=-1, keepdim=True)  # [B, 1]

        # 负载均衡损失 / Load balancing loss (确保expert均匀使用)
        expert_freq = sparse_weights.mean(dim=0)  # [num_experts]
        ideal_freq = torch.ones_like(expert_freq) / self.num_experts
        load_balance_loss = ((expert_freq - ideal_freq) ** 2).sum()

        return {
            'prediction': prediction,
            'expert_weights': sparse_weights,
            'load_balance_loss': load_balance_loss,
        }


# ============================================================
# 3. 端到端投资组合优化 / End-to-End Portfolio Optimization
# ============================================================

class DDLPortfolioOptimizer(nn.Module):
    """
    DDL端到端投资组合优化器 / DDL End-to-End Portfolio Optimizer

    输入: 过去N天的因子数据
    输出: 投资组合权重 (满足: 权重之和=1, 权重≥0)

    使用DDL残差连接的时序编码器 + 约束投影层

    量化意义:
    - 传统: 先预测收益 → 再求解均值-方差优化 (两步分离)
    - 本方法: 端到端直接从因子数据到最优权重 (联合优化)
    """

    def __init__(
        self,
        num_assets: int = 30,         # 资产数量
        num_factors: int = 50,        # 每资产的因子数
        lookback: int = 60,           # 回望窗口(交易日)
        d_model: int = 128,
        n_layers: int = 2,
        max_position: float = 0.1,    # 单资产最大仓位
    ):
        super().__init__()
        self.num_assets = num_assets
        self.max_position = max_position

        # DDL时序编码器 (预测未来收益)
        self.forecaster = DDLTimeSeriesForecaster(
            num_features=num_factors,
            d_model=d_model,
            n_layers=n_layers,
            forecast_horizon=1,  # 预测1天
        )

        # 组合权重网络 / Portfolio weight network
        self.weight_net = nn.Sequential(
            nn.Linear(d_model + num_assets, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_assets),
        )

    def forward(
        self,
        factor_data: torch.Tensor,
        cov_matrix: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            factor_data: [batch, lookback, num_assets, num_factors]
                每资产每日的因子数据
            cov_matrix: [batch, num_assets, num_assets] 协方差矩阵 (可选)

        Returns:
            dict with:
                'weights': [batch, num_assets] 投资组合权重
                'expected_return': [batch, 1] 预期组合收益
                'risk': [batch, 1] 组合风险 (如有cov_matrix)
        """
        B, T, A, F = factor_data.shape

        # 对每个资产独立预测 (可批量化优化)
        asset_returns = []
        for a in range(A):
            asset_factors = factor_data[:, :, a, :]  # [B, T, F]
            result = self.forecaster(asset_factors)
            asset_returns.append(result['returns'])  # [B, 1]

        predicted_returns = torch.cat(asset_returns, dim=-1)  # [B, num_assets]

        # 构建组合权重 / Construct portfolio weights
        # 输入: 预测收益 + 收益均值(作为特征)
        weight_input = torch.cat([predicted_returns, predicted_returns], dim=-1)
        raw_weights = self.weight_net(weight_input[:, :self.forecaster.forecast_head[1].out_features + A])

        # 约束投影 / Constraint projection
        # 1. Softmax确保权重之和=1且非负
        weights = F.softmax(raw_weights, dim=-1)

        # 2. 截断: 单资产不超过max_position
        weights = torch.clamp(weights, max=self.max_position)
        weights = weights / weights.sum(dim=-1, keepdim=True)  # 重新归一化

        # 组合预期收益 / Portfolio expected return
        expected_return = (weights * predicted_returns).sum(dim=-1, keepdim=True)

        # 组合风险 / Portfolio risk
        risk = None
        if cov_matrix is not None:
            # w^T * Σ * w
            risk = torch.bmm(
                weights.unsqueeze(1),
                torch.bmm(cov_matrix, weights.unsqueeze(-1)),
            ).squeeze()

        return {
            'weights': weights,
            'expected_return': expected_return,
            'risk': risk,
            'predicted_returns': predicted_returns,
        }


# ============================================================
# 训练与回测示例 / Training & Backtest Example
# ============================================================

def simulate_market_data(
    num_samples: int = 1000,
    seq_len: int = 60,
    num_assets: int = 10,
    num_factors: int = 20,
) -> Dict[str, torch.Tensor]:
    """
    生成模拟金融数据 / Generate simulated financial data

    模拟三种市场regime:
    - Regime 0: 动量 (趋势延续)
    - Regime 1: 均值回归 (震荡)
    - Regime 2: 高波动 (crash/rally)
    """
    # 随机regime序列
    regimes = torch.randint(0, 3, (num_samples, seq_len))

    # 因子数据 (简化: 正态分布)
    factor_data = torch.randn(num_samples, seq_len, num_assets, num_factors)

    # 收益数据 (根据regime不同有不同统计特征)
    returns = torch.zeros(num_samples, seq_len, num_assets)
    for s in range(num_samples):
        for t in range(seq_len):
            r = regimes[s, t].item()
            if r == 0:  # 动量
                returns[s, t] = torch.randn(num_assets) * 0.01 + 0.001
            elif r == 1:  # 均值回归
                returns[s, t] = torch.randn(num_assets) * 0.005
            else:  # 高波动
                returns[s, t] = torch.randn(num_assets) * 0.03

    return {
        'factors': factor_data,
        'returns': returns,
        'regimes': regimes,
    }


def train_and_evaluate():
    """完整训练与评估流程 / Full training & evaluation pipeline"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- 模型初始化 ---
    NUM_ASSETS = 10
    NUM_FACTORS = 20

    forecaster = DDLTimeSeriesForecaster(
        num_features=NUM_FACTORS,
        d_model=64,
        n_heads=4,
        n_layers=2,
        forecast_horizon=5,
    ).to(device)

    moe_router = MoEFactorRouter(
        num_factors=NUM_FACTORS,
        num_experts=6,
        top_k=2,
    ).to(device)

    # --- 模拟数据 ---
    data = simulate_market_data(
        num_samples=500,
        seq_len=60,
        num_assets=NUM_ASSETS,
        num_factors=NUM_FACTORS,
    )

    # --- 训练一步 (DDL Forecaster) ---
    optimizer = torch.optim.AdamW(forecaster.parameters(), lr=1e-4)

    factor_slice = data['factors'][:, :, 0, :].to(device)  # 单资产
    return_target = data['returns'][:, -5:, 0].to(device)  # 未来5天

    forecaster.train()
    result = forecaster(factor_slice, regime_labels=data['regimes'].to(device))

    # 复合损失: 收益预测 + regime分类
    pred_loss = F.mse_loss(result['returns'], return_target.mean(dim=-1, keepdim=True).expand_as(result['returns']))
    regime_loss = result['regime_loss'] if result['regime_loss'] is not None else 0
    total_loss = pred_loss + 0.1 * regime_loss

    total_loss.backward()
    optimizer.step()

    print(f"\n{'='*60}")
    print(f"DDL Time-Series Forecaster Training")
    print(f"{'='*60}")
    print(f"  Prediction Loss: {pred_loss.item():.4f}")
    print(f"  Regime Loss: {regime_loss if isinstance(regime_loss, float) else regime_loss.item():.4f}")
    print(f"  Total Loss: {total_loss.item():.4f}")
    print(f"  Parameters: {sum(p.numel() for p in forecaster.parameters()):,}")

    # --- 训练一步 (MoE Router) ---
    moe_optimizer = torch.optim.AdamW(moe_router.parameters(), lr=1e-4)

    factor_snapshot = data['factors'][:, -1, 0, :].to(device)  # 最新因子快照

    moe_router.train()
    moe_result = moe_router(factor_snapshot)

    moe_loss = F.mse_loss(
        moe_result['prediction'],
        data['returns'][:, -1, 0:1].to(device),
    ) + 0.01 * moe_result['load_balance_loss']

    moe_loss.backward()
    moe_optimizer.step()

    print(f"\n{'='*60}")
    print(f"MoE Factor Router Training")
    print(f"{'='*60}")
    print(f"  MSE Loss: {moe_loss.item():.4f}")
    print(f"  Load Balance Loss: {moe_result['load_balance_loss'].item():.4f}")
    print(f"  Expert weights (sample): {moe_result['expert_weights'][0].detach().cpu().numpy().round(3)}")

    # --- Regime检测结果 ---
    print(f"\n{'='*60}")
    print(f"Market Regime Detection")
    print(f"{'='*60}")
    forecaster.eval()
    with torch.no_grad():
        test_result = forecaster(factor_slice[:5])
        regime_probs = F.softmax(test_result['regime_logits'][:, -1, :], dim=-1)
        regime_names = ['Momentum 动量', 'Mean-Reversion 均值回归', 'High-Vol 高波动']
        for i in range(5):
            dominant = regime_probs[i].argmax().item()
            print(f"  Sample {i}: {regime_names[dominant]} (probs: {regime_probs[i].numpy().round(3)})")


if __name__ == "__main__":
    print("=" * 60)
    print("DDL × Quantitative Finance Applications")
    print("量化金融应用代码复现")
    print("=" * 60)

    train_and_evaluate()
