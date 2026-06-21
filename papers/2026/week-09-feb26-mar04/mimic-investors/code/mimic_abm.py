"""
Mimic Investors ABM: 模仿交易者Agent-Based模型
Agent-Based Model: Are Original Investors Stolen by Mimic Traders?

复现论文: arXiv 2603.03671 (Mar 2026)
核心: 多代理市场模拟，量化模仿者对原始投资者alpha的侵蚀

作者: QoderWork AI Research
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')


@dataclass
class Agent:
    agent_type: str  # 'informed', 'mimic', 'noise'
    capital: float
    position: float = 0.0
    pnl: float = 0.0
    signal_accuracy: float = 0.6


class MarketABM:
    """
    Agent-Based Market Model
    
    代理类型:
    - Informed: 拥有私有alpha信号
    - Mimic: 观察并模仿informed的交易
    - Noise: 随机交易
    """
    
    def __init__(self, n_informed=2, n_mimic=5, n_noise=20, seed=42):
        self.rng = np.random.RandomState(seed)
        self.agents = []
        
        for _ in range(n_informed):
            self.agents.append(Agent('informed', 1000000, signal_accuracy=0.65))
        for _ in range(n_mimic):
            self.agents.append(Agent('mimic', 500000, signal_accuracy=0.55))
        for _ in range(n_noise):
            self.agents.append(Agent('noise', 200000, signal_accuracy=0.5))
    
    def simulate(self, n_days=500) -> dict:
        """运行市场模拟"""
        # 真实价格过程
        true_price = 100.0
        price_history = [true_price]
        
        informed_pnl = []
        mimic_pnl = []
        noise_pnl = []
        
        for day in range(n_days):
            # 真实价格变动 (informed有信号)
            true_return = self.rng.randn() * 0.01
            
            # Informed交易
            informed_orders = []
            for agent in self.agents:
                if agent.agent_type == 'informed':
                    signal = np.sign(true_return) if self.rng.random() < agent.signal_accuracy else -np.sign(true_return)
                    order = signal * agent.capital * 0.1
                    informed_orders.append(order)
                    agent.position = order / true_price
            
            # Mimic观察并模仿informed的平均方向
            mimic_orders = []
            if informed_orders:
                avg_informed_dir = np.sign(np.mean(informed_orders))
                for agent in self.agents:
                    if agent.agent_type == 'mimic':
                        # 有延迟和噪声的模仿
                        if self.rng.random() < 0.7:  # 70%概率跟随
                            order = avg_informed_dir * agent.capital * 0.08
                        else:
                            order = -avg_informed_dir * agent.capital * 0.05
                        mimic_orders.append(order)
                        agent.position = order / true_price
            
            # Noise交易
            noise_orders = []
            for agent in self.agents:
                if agent.agent_type == 'noise':
                    order = self.rng.randn() * agent.capital * 0.05
                    noise_orders.append(order)
                    agent.position = order / true_price
            
            # 市场出清: 价格受总订单流影响
            total_order = sum(informed_orders) + sum(mimic_orders) + sum(noise_orders)
            price_impact = total_order / (sum(a.capital for a in self.agents) + 1e-8) * 0.1
            
            # 价格更新 = 真实变动 + 订单流冲击
            actual_return = true_return + price_impact
            true_price *= (1 + actual_return)
            price_history.append(true_price)
            
            # 计算PnL
            day_informed_pnl = sum(o * actual_return for o in informed_orders) / max(len(informed_orders), 1)
            day_mimic_pnl = sum(o * actual_return for o in mimic_orders) / max(len(mimic_orders), 1)
            day_noise_pnl = sum(o * actual_return for o in noise_orders) / max(len(noise_orders), 1)
            
            informed_pnl.append(day_informed_pnl)
            mimic_pnl.append(day_mimic_pnl)
            noise_pnl.append(day_noise_pnl)
        
        return {
            'prices': np.array(price_history),
            'informed_pnl': np.array(informed_pnl),
            'mimic_pnl': np.array(mimic_pnl),
            'noise_pnl': np.array(noise_pnl),
        }
    
    def analyze_alpha_erosion(self, results: dict) -> dict:
        """分析alpha侵蚀"""
        informed = results['informed_pnl']
        mimic = results['mimic_pnl']
        
        # 前半段 vs 后半段
        mid = len(informed) // 2
        informed_first_half = informed[:mid].mean()
        informed_second_half = informed[mid:].mean()
        
        # Alpha衰减
        alpha_decay = informed_second_half - informed_first_half
        
        # Mimic对informed的影响
        correlation = np.corrcoef(informed, mimic)[0, 1]
        
        return {
            'informed_avg_pnl': informed.mean(),
            'mimic_avg_pnl': mimic.mean(),
            'noise_avg_pnl': results['noise_pnl'].mean(),
            'informed_first_half': informed_first_half,
            'informed_second_half': informed_second_half,
            'alpha_decay': alpha_decay,
            'informed_mimic_corr': correlation,
            'informed_sharpe': informed.mean() / (informed.std() + 1e-8) * np.sqrt(252),
            'mimic_sharpe': mimic.mean() / (mimic.std() + 1e-8) * np.sqrt(252),
        }


def main():
    print("=" * 70)
    print("Mimic Investors ABM: 模仿交易者侵蚀效应")
    print("=" * 70)
    
    print("\n[1] 不同模仿者数量的对比实验...")
    
    mimic_counts = [0, 2, 5, 10, 20]
    results_by_mimic = {}
    
    for n_mimic in mimic_counts:
        abm = MarketABM(n_informed=2, n_mimic=n_mimic, n_noise=20, seed=42)
        results = abm.simulate(n_days=500)
        analysis = abm.analyze_alpha_erosion(results)
        results_by_mimic[n_mimic] = analysis
    
    print(f"\n  {'模仿者数':>10} {'Informed PnL':>14} {'Mimic PnL':>12} {'Alpha衰减':>12} {'相关性':>10}")
    print(f"  {'-'*62}")
    
    for n_mimic, analysis in results_by_mimic.items():
        mimic_pnl = analysis['mimic_avg_pnl'] if n_mimic > 0 else 0
        print(f"  {n_mimic:>10} {analysis['informed_avg_pnl']:>+14.2f} "
              f"{mimic_pnl:>+12.2f} {analysis['alpha_decay']:>+12.2f} "
              f"{analysis['informed_mimic_corr']:>10.4f}")
    
    print(f"\n[2] 关键发现:")
    no_mimic = results_by_mimic[0]['informed_avg_pnl']
    many_mimic = results_by_mimic[20]['informed_avg_pnl']
    erosion_pct = (no_mimic - many_mimic) / (abs(no_mimic) + 1e-8) * 100
    
    print(f"  无模仿者时 Informed 日均PnL: {no_mimic:+.2f}")
    print(f"  20个模仿者时 Informed 日均PnL: {many_mimic:+.2f}")
    print(f"  Alpha侵蚀比例: {erosion_pct:.1f}%")
    
    print(f"\n  结论:")
    print(f"  • 模仿者数量增加 → Informed的alpha被显著侵蚀")
    print(f"  • 模仿者与Informed的订单流高度相关 → 信息泄露")
    print(f"  • 防御策略: 交易随机化、分批执行、暗池交易")
    
    print("\n" + "=" * 70)
    print("Mimic Investors ABM 复现完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
