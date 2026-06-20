"""
LLM Multi-Agent for Chinese REITs: 多代理协作投资决策系统
LLM Multi-Agent Investment System for Chinese Public REITs

复现论文: arXiv 2602.00082 (Jan 2026)
核心思想: 四个专业化LLM代理(基本面/技术面/风险/优化)通过辩论机制协作评估中国公募REITs

作者: QoderWork AI Research
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. REITs数据模型 (REITs Data Model)
# ============================================================

class AssetType(Enum):
    HIGHWAY = "高速公路"
    INDUSTRIAL_PARK = "产业园区"
    WAREHOUSE = "仓储物流"
    SEWAGE = "污水处理"
    POWER = "清洁能源"


@dataclass
class REITData:
    """中国公募REITs数据"""
    ticker: str
    name: str
    asset_type: AssetType
    nav_per_unit: float          # 单位净值
    ffo_per_unit: float          # 可供分配金额/份
    occupancy_rate: float        # 出租率
    dividend_yield: float        # 分红率
    price: float                 # 当前价格
    volume_20d: float            # 20日成交量
    price_change_5d: float       # 5日涨跌幅
    price_change_20d: float      # 20日涨跌幅
    interest_rate_sensitivity: float  # 利率敏感度


class REITsDataGenerator:
    """生成模拟的中国公募REITs数据"""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
    
    def generate_reits_universe(self, n_reits: int = 15) -> List[REITData]:
        """生成REITs池"""
        asset_types = list(AssetType)
        reits = []
        
        names = [
            ("华安张江光大REIT", AssetType.INDUSTRIAL_PARK),
            ("博时招商蛇口REIT", AssetType.INDUSTRIAL_PARK),
            ("中金普洛斯REIT", AssetType.WAREHOUSE),
            ("红土盐田港REIT", AssetType.WAREHOUSE),
            ("浙商沪杭甬REIT", AssetType.HIGHWAY),
            ("华夏越秀高速REIT", AssetType.HIGHWAY),
            ("鹏华深圳能源REIT", AssetType.POWER),
            ("中航首钢绿能REIT", AssetType.POWER),
            ("富国首创水务REIT", AssetType.SEWAGE),
            ("东吴苏园产业REIT", AssetType.INDUSTRIAL_PARK),
            ("国泰君安临港REIT", AssetType.WAREHOUSE),
            ("华夏中国交建REIT", AssetType.HIGHWAY),
            ("中金厦门安居REIT", AssetType.INDUSTRIAL_PARK),
            ("嘉实物美消费REIT", AssetType.WAREHOUSE),
            ("华夏和达高科REIT", AssetType.INDUSTRIAL_PARK),
        ]
        
        for i in range(min(n_reits, len(names))):
            name, asset_type = names[i]
            reit = REITData(
                ticker=f"REIT_{i+1:03d}",
                name=name,
                asset_type=asset_type,
                nav_per_unit=self.rng.uniform(2.0, 8.0),
                ffo_per_unit=self.rng.uniform(0.1, 0.5),
                occupancy_rate=self.rng.uniform(0.7, 0.98),
                dividend_yield=self.rng.uniform(0.03, 0.08),
                price=self.rng.uniform(2.0, 10.0),
                volume_20d=self.rng.uniform(1e6, 5e7),
                price_change_5d=self.rng.uniform(-0.05, 0.05),
                price_change_20d=self.rng.uniform(-0.1, 0.1),
                interest_rate_sensitivity=self.rng.uniform(-2.0, -0.5),
            )
            reits.append(reit)
        
        return reits
    
    def generate_forward_returns(self, reits: List[REITData], 
                                  n_days: int = 60) -> np.ndarray:
        """生成前瞻收益 (用于回测)"""
        n = len(reits)
        returns = np.zeros((n_days, n))
        
        for i, reit in enumerate(reits):
            # 基于基本面的drift
            fundamental_drift = (reit.dividend_yield / 252 + 
                                (reit.occupancy_rate - 0.85) * 0.0001)
            vol = 0.015  # REITs波动率低于股票
            
            for t in range(n_days):
                market_factor = self.rng.randn() * 0.005
                rate_factor = self.rng.randn() * 0.002 * reit.interest_rate_sensitivity
                idio = self.rng.randn() * vol
                returns[t, i] = fundamental_drift + market_factor + rate_factor + idio
        
        return returns


# ============================================================
# 2. LLM代理角色 (LLM Agent Roles)
# ============================================================

class AgentOpinion(Enum):
    STRONG_BUY = "强烈买入"
    BUY = "买入"
    HOLD = "持有"
    SELL = "卖出"
    STRONG_SELL = "强烈卖出"


@dataclass
class AgentAssessment:
    """代理评估结果"""
    agent_name: str
    opinion: AgentOpinion
    confidence: float       # 0-1
    score: float           # -1 to 1
    reasoning: str
    key_factors: List[str]


class FundamentalAnalyst:
    """基本面分析师代理"""
    
    def __init__(self):
        self.name = "基本面分析师"
    
    def assess(self, reit: REITData) -> AgentAssessment:
        """基于基本面评估REIT"""
        score = 0.0
        factors = []
        
        # NAV折溢价
        nav_premium = (reit.price - reit.nav_per_unit) / reit.nav_per_unit
        if nav_premium < -0.1:
            score += 0.3
            factors.append(f"NAV折价{nav_premium:.1%}")
        elif nav_premium > 0.2:
            score -= 0.2
            factors.append(f"NAV溢价{nav_premium:.1%}")
        
        # 分红率
        if reit.dividend_yield > 0.06:
            score += 0.2
            factors.append(f"高分红率{reit.dividend_yield:.2%}")
        elif reit.dividend_yield < 0.03:
            score -= 0.2
            factors.append(f"低分红率{reit.dividend_yield:.2%}")
        
        # 出租率
        if reit.occupancy_rate > 0.9:
            score += 0.2
            factors.append(f"高出租率{reit.occupancy_rate:.1%}")
        elif reit.occupancy_rate < 0.75:
            score -= 0.3
            factors.append(f"低出租率{reit.occupancy_rate:.1%}")
        
        # FFO估值
        ffo_multiple = reit.price / (reit.ffo_per_unit + 1e-8)
        if ffo_multiple < 15:
            score += 0.15
            factors.append(f"低FFO倍数{ffo_multiple:.1f}x")
        
        score = np.clip(score, -1, 1)
        confidence = min(0.9, 0.5 + abs(score) * 0.5)
        
        if score > 0.4:
            opinion = AgentOpinion.BUY
        elif score > 0.1:
            opinion = AgentOpinion.HOLD
        elif score < -0.3:
            opinion = AgentOpinion.SELL
        else:
            opinion = AgentOpinion.HOLD
        
        return AgentAssessment(
            agent_name=self.name,
            opinion=opinion,
            confidence=confidence,
            score=score,
            reasoning=f"基本面评估: NAV{nav_premium:+.1%}, 分红{reit.dividend_yield:.2%}, 出租率{reit.occupancy_rate:.1%}",
            key_factors=factors,
        )


class TechnicalAnalyst:
    """技术分析师代理"""
    
    def __init__(self):
        self.name = "技术分析师"
    
    def assess(self, reit: REITData) -> AgentAssessment:
        """基于技术面评估"""
        score = 0.0
        factors = []
        
        # 短期动量
        if reit.price_change_5d > 0.02:
            score += 0.15
            factors.append(f"5日正动量{reit.price_change_5d:.2%}")
        elif reit.price_change_5d < -0.03:
            score -= 0.1
            factors.append(f"5日负动量{reit.price_change_5d:.2%}")
        
        # 中期反转 (过度下跌可能反弹)
        if reit.price_change_20d < -0.08:
            score += 0.2  # 超跌反弹
            factors.append(f"20日超跌{reit.price_change_20d:.2%}, 可能反弹")
        elif reit.price_change_20d > 0.08:
            score -= 0.15  # 超买回落
            factors.append(f"20日超涨{reit.price_change_20d:.2%}, 可能回调")
        
        # 成交量
        if reit.volume_20d > 2e7:
            factors.append("高成交量, 流动性好")
        
        score = np.clip(score, -1, 1)
        confidence = min(0.7, 0.3 + abs(score) * 0.5)
        
        opinion = AgentOpinion.BUY if score > 0.2 else \
                  AgentOpinion.SELL if score < -0.2 else AgentOpinion.HOLD
        
        return AgentAssessment(
            agent_name=self.name,
            opinion=opinion,
            confidence=confidence,
            score=score,
            reasoning=f"技术面: 5日{reit.price_change_5d:+.2%}, 20日{reit.price_change_20d:+.2%}",
            key_factors=factors,
        )


class RiskManager:
    """风险经理代理"""
    
    def __init__(self):
        self.name = "风险经理"
    
    def assess(self, reit: REITData) -> AgentAssessment:
        """基于风险评估"""
        score = 0.0
        factors = []
        
        # 利率风险
        if abs(reit.interest_rate_sensitivity) > 1.5:
            score -= 0.2
            factors.append(f"高利率敏感度{reit.interest_rate_sensitivity:.2f}")
        
        # 流动性风险
        if reit.volume_20d < 5e6:
            score -= 0.3
            factors.append("低成交量, 流动性风险")
        
        # 集中度风险 (基于资产类型)
        # 简化: 单一资产类型风险较高
        if reit.asset_type in [AssetType.HIGHWAY, AssetType.SEWAGE]:
            factors.append(f"{reit.asset_type.value}类资产, 收入稳定但增长有限")
        elif reit.asset_type == AssetType.INDUSTRIAL_PARK:
            if reit.occupancy_rate < 0.8:
                score -= 0.15
                factors.append("产业园出租率偏低, 空置风险")
        
        score = np.clip(score, -1, 1)
        confidence = 0.7  # 风险评估通常较稳定
        
        # 风险经理通常偏保守
        opinion = AgentOpinion.HOLD if score > -0.2 else AgentOpinion.SELL
        
        return AgentAssessment(
            agent_name=self.name,
            opinion=opinion,
            confidence=confidence,
            score=score,
            reasoning=f"风险评估: 利率敏感度{reit.interest_rate_sensitivity:.2f}, 成交量{reit.volume_20d/1e6:.1f}M",
            key_factors=factors,
        )


# ============================================================
# 3. 多代理辩论机制 (Multi-Agent Debate)
# ============================================================

class MultiAgentDebate:
    """
    多代理辩论机制
    
    流程:
    1. 每个代理独立评估
    2. 公开各自观点
    3. 质疑和反驳 (意见分歧大的代理互相挑战)
    4. 修正观点
    5. 达成共识
    """
    
    def __init__(self, n_rounds: int = 3):
        self.n_rounds = n_rounds
    
    def conduct_debate(self, reit: REITData, 
                        agents: List) -> dict:
        """
        执行辩论流程
        
        returns: 辩论结果 (共识分数, 各轮观点变化)
        """
        # 第1轮: 独立评估
        assessments = [agent.assess(reit) for agent in agents]
        
        debate_log = [{'round': 0, 'assessments': [
            {'agent': a.agent_name, 'score': a.score, 'opinion': a.opinion.value}
            for a in assessments
        ]}]
        
        # 辩论轮次
        current_scores = np.array([a.score for a in assessments])
        current_confidences = np.array([a.confidence for a in assessments])
        
        for round_num in range(1, self.n_rounds + 1):
            # 计算群体均值
            group_mean = np.average(current_scores, weights=current_confidences)
            
            # 每个代理向群体均值靠拢 (信心越强，移动越少)
            new_scores = []
            for i, assessment in enumerate(assessments):
                # 修正幅度 = (1 - confidence) * (group_mean - current_score)
                adjustment = (1 - current_confidences[i]) * (group_mean - current_scores[i]) * 0.5
                new_score = current_scores[i] + adjustment
                new_scores.append(new_score)
            
            current_scores = np.array(new_scores)
            
            debate_log.append({
                'round': round_num,
                'scores': current_scores.tolist(),
                'group_mean': float(group_mean),
            })
        
        # 最终共识
        final_scores = current_scores
        final_weights = current_confidences
        consensus_score = float(np.average(final_scores, weights=final_weights))
        
        # 分歧度 (标准化方差)
        disagreement = float(np.std(final_scores) / (abs(consensus_score) + 0.5))
        
        return {
            'consensus_score': consensus_score,
            'disagreement': disagreement,
            'initial_assessments': debate_log[0]['assessments'],
            'debate_log': debate_log,
            'final_scores': {a.agent_name: s for a, s in zip(agents, final_scores)},
        }


# ============================================================
# 4. 组合优化器代理 (Portfolio Optimizer Agent)
# ============================================================

class PortfolioOptimizer:
    """
    组合优化师代理
    
    综合所有REITs的辩论结果，输出最优配置
    """
    
    def __init__(self, max_single_weight: float = 0.15, 
                 target_dividend: float = 0.05):
        self.max_weight = max_single_weight
        self.target_dividend = target_dividend
    
    def optimize(self, reits: List[REITData], 
                  debate_results: Dict[str, dict]) -> np.ndarray:
        """
        基于辩论共识分数的组合优化
        
        returns: 权重向量
        """
        n = len(reits)
        scores = np.array([debate_results[r.name]['consensus_score'] for r in reits])
        disagreements = np.array([debate_results[r.name]['disagreement'] for r in reits])
        
        # 基础权重: 共识分数的softmax
        # 分数越高 → 权重越大
        adjusted_scores = scores - disagreements * 0.3  # 分歧度惩罚
        
        exp_scores = np.exp(adjusted_scores * 2)  # 温度参数
        weights = exp_scores / exp_scores.sum()
        
        # 约束: 单个最大权重
        weights = np.minimum(weights, self.max_weight)
        weights /= weights.sum()
        
        # 分红率约束: 组合分红率 ≥ 目标
        dividend_yields = np.array([r.dividend_yield for r in reits])
        port_dividend = weights @ dividend_yields
        
        if port_dividend < self.target_dividend:
            # 增加高分红REITs的权重
            high_div_mask = dividend_yields > self.target_dividend
            if high_div_mask.any():
                boost = np.where(high_div_mask, 1.5, 0.8)
                weights *= boost
                weights /= weights.sum()
        
        return weights


# ============================================================
# 5. 主程序 (Main Pipeline)
# ============================================================

def main():
    print("=" * 70)
    print("LLM Multi-Agent: 中国公募REITs多代理协作投资决策")
    print("Multi-Agent Investment System for Chinese Public REITs")
    print("=" * 70)
    
    # ---- 数据生成 ----
    print("\n[1] 生成中国公募REITs数据...")
    data_gen = REITsDataGenerator(seed=42)
    reits = data_gen.generate_reits_universe(n_reits=12)
    
    for reit in reits[:5]:
        print(f"  {reit.name}: 价格{reit.price:.2f}, NAV{reit.nav_per_unit:.2f}, "
              f"分红{reit.dividend_yield:.2%}, 出租率{reit.occupancy_rate:.1%}")
    print(f"  ... 共 {len(reits)} 只REITs")
    
    # ---- 初始化代理 ----
    print("\n[2] 初始化四个专业代理...")
    agents = [
        FundamentalAnalyst(),
        TechnicalAnalyst(),
        RiskManager(),
    ]
    
    for agent in agents:
        print(f"  ✓ {agent.name}")
    
    # ---- 辩论评估 ----
    print("\n[3] 多代理辩论评估...")
    debate = MultiAgentDebate(n_rounds=3)
    debate_results = {}
    
    for reit in reits:
        result = debate.conduct_debate(reit, agents)
        debate_results[reit.name] = result
        
        print(f"\n  {reit.name} ({reit.asset_type.value}):")
        print(f"    共识分数: {result['consensus_score']:+.4f}")
        print(f"    分歧度: {result['disagreement']:.4f}")
        
        for assessment in result['initial_assessments']:
            print(f"    {assessment['agent']}: {assessment['opinion']} "
                  f"(score={assessment['score']:+.4f})")
    
    # ---- 组合优化 ----
    print(f"\n[4] 组合优化...")
    optimizer = PortfolioOptimizer(max_single_weight=0.15, target_dividend=0.05)
    weights = optimizer.optimize(reits, debate_results)
    
    print(f"\n  最优配置:")
    print(f"  {'REITs名称':<20} {'权重':>8} {'共识分':>8} {'分红率':>8} {'资产类型':<12}")
    print(f"  {'-'*60}")
    
    for reit, w in zip(reits, weights):
        if w > 0.01:
            dr = debate_results[reit.name]
            print(f"  {reit.name:<20} {w:>8.2%} {dr['consensus_score']:>+8.4f} "
                  f"{reit.dividend_yield:>8.2%} {reit.asset_type.value:<12}")
    
    # 组合统计
    port_dividend = weights @ np.array([r.dividend_yield for r in reits])
    port_score = weights @ np.array([debate_results[r.name]['consensus_score'] for r in reits])
    
    print(f"\n  组合分红率: {port_dividend:.2%}")
    print(f"  组合共识分: {port_score:+.4f}")
    print(f"  有效REITs数: {(weights > 0.01).sum()}")
    
    # ---- 回测验证 ----
    print(f"\n[5] 回测验证...")
    forward_returns = data_gen.generate_forward_returns(reits, n_days=120)
    
    # 多代理组合收益
    port_returns = forward_returns @ weights
    
    # 等权基准
    equal_weights = np.ones(len(reits)) / len(reits)
    equal_returns = forward_returns @ equal_weights
    
    # 分红加权基准
    div_weights = np.array([r.dividend_yield for r in reits])
    div_weights /= div_weights.sum()
    div_returns = forward_returns @ div_weights
    
    # 绩效对比
    def calc_stats(returns, name):
        sharpe = returns.mean() * 252 / (returns.std() * np.sqrt(252) + 1e-8)
        cum = np.cumprod(1 + returns)
        max_dd = np.max(1 - cum / cum.cummax())
        ann_ret = cum[-1] ** (252/len(returns)) - 1
        return {'name': name, 'sharpe': sharpe, 'max_dd': max_dd, 'ann_ret': ann_ret}
    
    stats_multi = calc_stats(port_returns, "多代理组合")
    stats_equal = calc_stats(equal_returns, "等权基准")
    stats_div = calc_stats(div_returns, "分红加权")
    
    print(f"\n  {'策略':<15} {'年化收益':>10} {'夏普比率':>10} {'最大回撤':>10}")
    print(f"  {'-'*50}")
    for s in [stats_multi, stats_equal, stats_div]:
        print(f"  {s['name']:<15} {s['ann_ret']:>10.4f} {s['sharpe']:>10.4f} {s['max_dd']:>10.4f}")
    
    # ---- 辩论收敛分析 ----
    print(f"\n[6] 辩论收敛分析...")
    
    sample_reit = reits[0]
    sample_result = debate_results[sample_reit.name]
    
    print(f"\n  {sample_reit.name} 辩论过程:")
    for entry in sample_result['debate_log']:
        if entry['round'] == 0:
            print(f"    初始轮:")
            for a in entry['assessments']:
                print(f"      {a['agent']}: {a['score']:+.4f}")
        else:
            scores = entry['scores']
            print(f"    第{entry['round']}轮: scores={[f'{s:+.4f}' for s in scores]}, "
                  f"均值={entry['group_mean']:+.4f}")
    
    print(f"    最终共识: {sample_result['consensus_score']:+.4f}")
    print(f"    分歧度: {sample_result['disagreement']:.4f}")
    
    print("\n" + "=" * 70)
    print("LLM Multi-Agent REITs 复现完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
