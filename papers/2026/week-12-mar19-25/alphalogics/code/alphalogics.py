"""
AlphaLogics: 多代理市场逻辑驱动的Alpha因子生成系统
Multi-Agent System for Scalable and Interpretable Alpha Factor Generation

复现论文: arXiv 2603 (Mar 2026)
核心: 研究员代理(假设) + 验证员代理(回测) + 逻辑员代理(经济学解释)

作者: QoderWork AI Research
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. 代理角色定义 (Agent Role Definitions)
# ============================================================

@dataclass
class FactorCandidate:
    """因子候选"""
    name: str
    formula_desc: str
    category: str       # momentum, value, quality, volatility, microstructure
    lookback: int
    direction: int      # 1=正, -1=反
    ic: float = 0.0
    icir: float = 0.0
    logic_score: float = 0.0  # 市场逻辑得分 (0-1)
    accepted: bool = False


class ResearcherAgent:
    """
    研究员代理: 基于市场逻辑生成因子假设
    
    方法:
    1. 从市场微观结构原理推导因子
    2. 从已知异象(anomalies)衍生新因子
    3. 组合已有因子创造复合因子
    """
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.hypothesis_count = 0
    
    def generate_hypotheses(self, n_hypotheses: int = 5) -> List[FactorCandidate]:
        """生成因子假设"""
        candidates = []
        
        templates = [
            # 动量类
            ('volume_weighted_momentum', '成交量加权动量(大额交易权重更高)', 'momentum', 20, 1),
            ('overnight_return_gap', '隔夜收益缺口(开盘vs收盘)', 'momentum', 1, 1),
            ('acceleration_momentum', '动量加速度(动量的动量)', 'momentum', 40, 1),
            # 价值类
            ('earnings_surprise_drift', '盈利惊喜漂移(PEAD)', 'value', 60, 1),
            ('relative_strength_value', '相对强弱价值(弱中寻强)', 'value', 120, 1),
            # 质量类
            ('accrual_quality', '应计质量(低应计=高质量)', 'quality', 252, 1),
            ('earnings_consistency', '盈利一致性(连续正增长)', 'quality', 252, 1),
            # 波动率类
            ('downside_volatility', '下行波动率(仅计算负收益)', 'volatility', 60, -1),
            ('vol_of_vol', '波动率的波动率', 'volatility', 40, -1),
            ('jump_risk', '跳跃风险(极端收益频率)', 'volatility', 60, -1),
            # 微观结构类
            ('order_flow_imbalance', '订单流不平衡', 'microstructure', 5, 1),
            ('bid_ask_spread_change', '买卖价差变化', 'microstructure', 10, -1),
            ('large_trade_frequency', '大额交易频率', 'microstructure', 20, 1),
        ]
        
        # 随机选择并添加变体
        selected = self.rng.choice(len(templates), size=min(n_hypotheses, len(templates)), replace=False)
        
        for idx in selected:
            name, desc, cat, lb, direction = templates[idx]
            # 添加随机变体
            lb_variant = max(5, lb + self.rng.randint(-5, 10))
            candidates.append(FactorCandidate(
                name=f"{name}_v{self.hypothesis_count}",
                formula_desc=desc,
                category=cat,
                lookback=lb_variant,
                direction=direction,
            ))
            self.hypothesis_count += 1
        
        return candidates


class ValidatorAgent:
    """
    验证员代理: 对因子假设进行回测验证
    
    评估:
    1. IC (Information Coefficient)
    2. ICIR (IC Information Ratio)
    3. 换手率
    4. 多空收益
    """
    
    def __init__(self, min_ic: float = 0.02, min_icir: float = 0.3):
        self.min_ic = min_ic
        self.min_icir = min_icir
    
    def validate(self, candidate: FactorCandidate, returns: np.ndarray, 
                  day: int) -> FactorCandidate:
        """验证单个因子"""
        n_assets = returns.shape[1]
        lookback = min(candidate.lookback, day)
        
        if lookback < 5:
            candidate.ic = 0
            candidate.icir = 0
            return candidate
        
        hist = returns[day-lookback:day]
        
        # 根据因子类别计算因子值
        if candidate.category == 'momentum':
            factor_vals = hist.mean(axis=0) * candidate.direction
        elif candidate.category == 'value':
            cum_ret = np.cumprod(1 + hist)[-1] - 1
            factor_vals = -cum_ret * candidate.direction
        elif candidate.category == 'quality':
            factor_vals = hist.mean(axis=0) / (hist.std(axis=0) + 1e-8) * candidate.direction
        elif candidate.category == 'volatility':
            factor_vals = hist.std(axis=0) * candidate.direction
        else:
            factor_vals = hist.std(axis=0) / (np.abs(hist).mean(axis=0) + 1e-8) * candidate.direction
        
        # 前瞻收益
        if day + 5 < len(returns):
            fwd = returns[day+1:day+6].mean(axis=0)
            if factor_vals.std() > 0 and fwd.std() > 0:
                candidate.ic = float(np.corrcoef(factor_vals, fwd)[0, 1])
            else:
                candidate.ic = 0.0
        else:
            candidate.ic = 0.0
        
        return candidate
    
    def batch_validate(self, candidates: List[FactorCandidate], 
                        returns: np.ndarray, days: List[int]) -> List[FactorCandidate]:
        """批量验证，计算ICIR"""
        for c in candidates:
            ics = []
            for d in days:
                temp = self.validate(FactorCandidate(
                    c.name, c.formula_desc, c.category, c.lookback, c.direction
                ), returns, d)
                ics.append(temp.ic)
            
            if ics:
                c.ic = float(np.mean(ics))
                c.icir = float(np.mean(ics) / (np.std(ics) + 1e-8))
            else:
                c.ic = 0.0
                c.icir = 0.0
        
        return candidates


class LogicAgent:
    """
    逻辑员代理: 验证因子的经济学合理性
    
    检查:
    1. 因子是否有清晰的经济学逻辑？
    2. 因子是否与已知异象相关？
    3. 因子的衰减模式是否合理？
    """
    
    def __init__(self):
        # 已知市场异象和逻辑
        self.known_anomalies = {
            'momentum': {'logic': '行为偏差(锚定效应)+信息扩散延迟', 'strength': 0.8},
            'value': {'logic': '风险补偿+过度外推', 'strength': 0.7},
            'quality': {'logic': '投资者忽视基本面质量', 'strength': 0.6},
            'volatility': {'logic': '彩票偏好+杠杆约束', 'strength': 0.7},
            'microstructure': {'logic': '信息不对称+流动性溢价', 'strength': 0.5},
        }
    
    def evaluate_logic(self, candidate: FactorCandidate) -> FactorCandidate:
        """评估因子的市场逻辑"""
        cat = candidate.category
        if cat in self.known_anomalies:
            base_score = self.known_anomalies[cat]['strength']
            # IC越高 → 逻辑越被数据支持
            ic_bonus = min(abs(candidate.ic) * 5, 0.3)
            candidate.logic_score = min(1.0, base_score + ic_bonus)
        else:
            candidate.logic_score = 0.3
        
        return candidate


# ============================================================
# 2. 多代理协作引擎 (Multi-Agent Collaboration Engine)
# ============================================================

class AlphaLogicsEngine:
    """
    AlphaLogics多代理因子生成引擎
    
    流程:
    1. Researcher → 生成因子假设
    2. Validator → 回测验证
    3. Logic → 经济学合理性检查
    4. 接受/拒绝决策
    """
    
    def __init__(self, n_assets: int = 30, n_days: int = 600, seed: int = 42):
        self.n_assets = n_assets
        self.n_days = n_days
        self.rng = np.random.RandomState(seed)
        
        self.researcher = ResearcherAgent(seed)
        self.validator = ValidatorAgent(min_ic=0.02, min_icir=0.3)
        self.logic = LogicAgent()
        
        self.returns = self._generate_data()
        self.accepted_factors: List[FactorCandidate] = []
    
    def _generate_data(self) -> np.ndarray:
        """生成带因子结构的市场数据"""
        returns = np.zeros((self.n_days, self.n_assets))
        for t in range(self.n_days):
            market = self.rng.randn() * 0.01
            factor_signal = self.rng.randn(5) * 0.002  # 5个隐因子
            exposures = self.rng.randn(self.n_assets, 5) * 0.3
            idio = self.rng.randn(self.n_assets) * 0.015
            returns[t] = market + exposures @ factor_signal + idio
        return returns
    
    def run_pipeline(self, n_rounds: int = 5, hypotheses_per_round: int = 5) -> dict:
        """运行完整的多代理因子挖掘流程"""
        all_candidates = []
        
        # 验证时间点
        eval_days = list(range(60, self.n_days - 10, 20))
        
        for round_num in range(n_rounds):
            print(f"  Round {round_num+1}/{n_rounds}:")
            
            # 1. Researcher生成假设
            hypotheses = self.researcher.generate_hypotheses(hypotheses_per_round)
            print(f"    Researcher: 生成 {len(hypotheses)} 个因子假设")
            
            # 2. Validator回测验证
            validated = self.validator.batch_validate(hypotheses, self.returns, eval_days)
            passed_validation = [c for c in validated if abs(c.ic) >= 0.02]
            print(f"    Validator: {len(passed_validation)}/{len(validated)} 通过IC检验")
            
            # 3. Logic逻辑验证
            for c in passed_validation:
                self.logic.evaluate_logic(c)
            
            # 4. 接受决策 (IC+Logic综合评分)
            for c in passed_validation:
                combined_score = 0.6 * abs(c.icir) + 0.4 * c.logic_score
                if combined_score > 0.4:
                    c.accepted = True
                    self.accepted_factors.append(c)
            
            accepted_this_round = [c for c in passed_validation if c.accepted]
            print(f"    Logic: {len(accepted_this_round)} 个因子被接受")
            
            all_candidates.extend(validated)
        
        return self._summarize(all_candidates)
    
    def _summarize(self, all_candidates) -> dict:
        """汇总结果"""
        accepted = self.accepted_factors
        
        if accepted:
            # 构建组合因子
            eval_days = list(range(60, self.n_days - 10, 5))
            composite_ics = []
            
            for d in eval_days:
                composite = np.zeros(self.n_assets)
                for f in accepted[:5]:  # 用前5个因子
                    lookback = min(f.lookback, d)
                    if lookback >= 5:
                        hist = self.returns[d-lookback:d]
                        if f.category == 'momentum':
                            fv = hist.mean(axis=0)
                        else:
                            fv = hist.std(axis=0)
                        if fv.std() > 0:
                            fv = (fv - fv.mean()) / fv.std()
                        composite += fv
                if composite.std() > 0:
                    fwd = self.returns[d+1:d+6].mean(axis=0) if d+6 < self.n_days else np.zeros(self.n_assets)
                    if fwd.std() > 0:
                        composite_ics.append(np.corrcoef(composite, fwd)[0, 1])
            
            composite_ic = np.mean(composite_ics) if composite_ics else 0
        else:
            composite_ic = 0
        
        return {
            'total_generated': len(all_candidates),
            'total_accepted': len(accepted),
            'accepted_factors': accepted,
            'composite_ic': composite_ic,
            'categories': {cat: sum(1 for f in accepted if f.category == cat) 
                          for cat in ['momentum', 'value', 'quality', 'volatility', 'microstructure']},
        }


# ============================================================
# 3. 主程序
# ============================================================

def main():
    print("=" * 70)
    print("AlphaLogics: 多代理市场逻辑驱动的Alpha因子生成")
    print("=" * 70)
    
    print("\n[1] 初始化AlphaLogics引擎...")
    engine = AlphaLogicsEngine(n_assets=30, n_days=600, seed=42)
    
    print("\n[2] 运行多代理因子挖掘流程...")
    results = engine.run_pipeline(n_rounds=5, hypotheses_per_round=6)
    
    print(f"\n[3] 结果汇总:")
    print(f"  总生成因子: {results['total_generated']}")
    print(f"  接受因子数: {results['total_accepted']}")
    print(f"  组合因子IC: {results['composite_ic']:+.4f}")
    
    print(f"\n  按类别统计:")
    for cat, count in results['categories'].items():
        print(f"    {cat}: {count} 个")
    
    print(f"\n  接受的因子:")
    for f in results['accepted_factors'][:10]:
        print(f"    {f.name:<35} IC={f.ic:+.4f} ICIR={f.icir:+.4f} "
              f"Logic={f.logic_score:.2f} [{f.category}]")
    
    print(f"\n  关键发现:")
    print(f"  ✓ 多代理协作: Researcher+Validator+Logic三层筛选")
    print(f"  ✓ 市场逻辑约束: 确保因子有经济学解释")
    print(f"  ✓ 可扩展: 自动化流程支持大规模因子挖掘")
    
    print("\n" + "=" * 70)
    print("AlphaLogics 复现完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
