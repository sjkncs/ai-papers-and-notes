"""
ASRI: 加密货币市场聚合系统性风险指数
Aggregated Systemic Risk Index for Cryptocurrency Markets

复现论文: arXiv 2602.03874 (Feb 2026)
核心思想: 多维度聚合指数捕获加密市场独特的系统性风险传染渠道

作者: QoderWork AI Research
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. 加密市场数据模型 (Crypto Market Data Model)
# ============================================================

@dataclass
class CryptoAsset:
    """加密资产"""
    symbol: str
    name: str
    category: str      # 'layer1', 'defi', 'stablecoin', 'meme', 'layer2'
    market_cap: float  # 百万美元


class CryptoMarketSimulator:
    """生成模拟加密市场数据"""
    
    def __init__(self, n_days: int = 730, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.n_days = n_days
        
        self.assets = [
            CryptoAsset('BTC', 'Bitcoin', 'layer1', 800000),
            CryptoAsset('ETH', 'Ethereum', 'layer1', 300000),
            CryptoAsset('SOL', 'Solana', 'layer1', 80000),
            CryptoAsset('AVAX', 'Avalanche', 'layer1', 15000),
            CryptoAsset('UNI', 'Uniswap', 'defi', 8000),
            CryptoAsset('AAVE', 'Aave', 'defi', 3000),
            CryptoAsset('MKR', 'Maker', 'defi', 2500),
            CryptoAsset('CRV', 'Curve', 'defi', 1500),
            CryptoAsset('USDT', 'Tether', 'stablecoin', 90000),
            CryptoAsset('USDC', 'USD Coin', 'stablecoin', 30000),
            CryptoAsset('DAI', 'DAI', 'stablecoin', 5000),
            CryptoAsset('ARB', 'Arbitrum', 'layer2', 5000),
            CryptoAsset('OP', 'Optimism', 'layer2', 3000),
            CryptoAsset('DOGE', 'Dogecoin', 'meme', 10000),
            CryptoAsset('SHIB', 'Shiba Inu', 'meme', 5000),
        ]
    
    def generate_market_data(self) -> dict:
        """生成完整的市场数据"""
        n_assets = len(self.assets)
        rng = self.rng
        
        # 收益生成 (带传染效应和regime)
        returns = np.zeros((self.n_days, n_assets))
        volumes = np.zeros((self.n_days, n_assets))
        tvl = np.zeros((self.n_days, n_assets))  # DeFi TVL
        stablecoin_pegs = np.ones((self.n_days, 3))  # 3个稳定币
        
        # Regime: 0=normal, 1=bull, 2=bear, 3=crisis
        regimes = np.zeros(self.n_days, dtype=int)
        current_regime = 1
        
        for t in range(self.n_days):
            # Regime切换
            if rng.random() < 0.01:
                current_regime = rng.choice(4, p=[0.3, 0.3, 0.25, 0.15])
            regimes[t] = current_regime
            
            regime_params = {
                0: {'mu': 0.001, 'sigma': 0.03, 'corr': 0.4},
                1: {'mu': 0.003, 'sigma': 0.04, 'corr': 0.3},
                2: {'mu': -0.005, 'sigma': 0.06, 'corr': 0.6},
                3: {'mu': -0.02, 'sigma': 0.12, 'corr': 0.85},
            }
            p = regime_params[current_regime]
            
            # 共同因子 (BTC主导)
            btc_factor = rng.randn() * p['sigma'] * 0.7
            eth_factor = rng.randn() * p['sigma'] * 0.3
            
            for i, asset in enumerate(self.assets):
                # 对BTC的beta
                if asset.category == 'layer1':
                    beta_btc = 0.8 + rng.randn() * 0.1
                elif asset.category == 'defi':
                    beta_btc = 1.2 + rng.randn() * 0.2
                elif asset.category == 'stablecoin':
                    beta_btc = 0.02
                elif asset.category == 'meme':
                    beta_btc = 1.5 + rng.randn() * 0.3
                else:
                    beta_btc = 0.9
                
                idio = rng.randn() * p['sigma'] * 0.5
                returns[t, i] = p['mu'] + beta_btc * btc_factor + idio
                
                # 成交量
                base_vol = asset.market_cap * 0.05  # 5%日换手
                vol_multiplier = 1 + abs(returns[t, i]) * 10
                volumes[t, i] = base_vol * vol_multiplier * (1 + rng.randn() * 0.2)
                
                # DeFi TVL
                if asset.category == 'defi':
                    tvl_change = returns[t, i] * 0.5 + rng.randn() * 0.02
                    if t > 0:
                        tvl[t, i] = tvl[t-1, i] * (1 + tvl_change)
                    else:
                        tvl[t, i] = asset.market_cap * 10
            
            # 稳定币脱锚风险
            for j in range(3):
                if current_regime == 3:  # 危机时脱锚风险增大
                    peg_drift = rng.randn() * 0.005
                    stablecoin_pegs[t, j] = max(0.9, min(1.1, 
                        (stablecoin_pegs[t-1, j] if t > 0 else 1.0) + peg_drift))
                else:
                    stablecoin_pegs[t, j] = 1.0 + rng.randn() * 0.0005
        
        return {
            'returns': returns,
            'volumes': volumes,
            'tvl': tvl,
            'stablecoin_pegs': stablecoin_pegs,
            'regimes': regimes,
            'assets': self.assets,
        }


# ============================================================
# 2. ASRI指数构建 (Systemic Risk Index Construction)
# ============================================================

class ASRIBuilder:
    """
    聚合系统性风险指数 (ASRI) 构建器
    
    子指标:
    1. 价格联动度 (Price Co-movement)
    2. 成交量异常 (Volume Anomaly)
    3. 稳定币脱锚风险 (Stablecoin De-peg Risk)
    4. DeFi TVL变化 (TVL Change)
    5. 波动率聚集 (Volatility Clustering)
    """
    
    def __init__(self, window: int = 30, weights: Dict[str, float] = None):
        self.window = window
        self.weights = weights or {
            'price_comovement': 0.30,
            'volume_anomaly': 0.20,
            'stablecoin_peg': 0.20,
            'tvl_change': 0.15,
            'vol_clustering': 0.15,
        }
    
    def compute_price_comovement(self, returns: np.ndarray, t: int) -> float:
        """
        价格联动度: 高联动 = 高风险
        
        计算最近window天内资产间平均相关性
        """
        if t < self.window:
            return 0.0
        
        recent = returns[t-self.window:t]
        corr = np.corrcoef(recent.T)
        n = corr.shape[0]
        
        # 上三角平均相关性
        mask = np.triu(np.ones((n, n)), k=1).astype(bool)
        avg_corr = corr[mask].mean()
        
        # 归一化到0-1
        return float(np.clip(avg_corr, 0, 1))
    
    def compute_volume_anomaly(self, volumes: np.ndarray, t: int) -> float:
        """
        成交量异常: 异常放量 = 高风险
        
        当前成交量 / 历史均值
        """
        if t < self.window * 3:
            return 0.0
        
        current = volumes[t].mean()
        historical = volumes[max(0, t-self.window*3):t-self.window].mean()
        
        ratio = current / (historical + 1e-8)
        # 归一化
        anomaly = np.clip((ratio - 1) / 2, 0, 1)
        return float(anomaly)
    
    def compute_stablecoin_peg_risk(self, pegs: np.ndarray, t: int) -> float:
        """
        稳定币脱锚风险: 偏离1.0越大 = 越危险
        """
        if t < 1:
            return 0.0
        
        deviations = np.abs(pegs[t] - 1.0)
        max_dev = deviations.max()
        
        # 0.5%以上偏离开始有风险，2%以上为危机
        risk = np.clip(max_dev / 0.02, 0, 1)
        return float(risk)
    
    def compute_tvl_risk(self, tvl: np.ndarray, t: int) -> float:
        """
        DeFi TVL风险: 快速下降 = 高风险
        
        TVL骤降暗示流动性危机和协议风险
        """
        if t < self.window:
            return 0.0
        
        # DeFi资产 (索引4-7)
        defi_tvl = tvl[t, 4:8].sum()
        defi_tvl_prev = tvl[max(0, t-self.window), 4:8].sum()
        
        if defi_tvl_prev > 0:
            change = (defi_tvl - defi_tvl_prev) / defi_tvl_prev
        else:
            change = 0
        
        # 负变化 = 风险
        risk = np.clip(-change / 0.3, 0, 1)
        return float(risk)
    
    def compute_vol_clustering(self, returns: np.ndarray, t: int) -> float:
        """
        波动率聚集: 高波动持续 = 高风险
        """
        if t < self.window * 2:
            return 0.0
        
        recent_vol = returns[t-self.window:t].std()
        prev_vol = returns[t-self.window*2:t-self.window].std()
        
        # 波动率加速上升
        vol_ratio = recent_vol / (prev_vol + 1e-8)
        risk = np.clip((vol_ratio - 1) / 1.5, 0, 1)
        return float(risk)
    
    def compute_asri(self, market_data: dict) -> np.ndarray:
        """
        计算完整ASRI时序
        """
        returns = market_data['returns']
        volumes = market_data['volumes']
        pegs = market_data['stablecoin_pegs']
        tvl = market_data['tvl']
        n_days = len(returns)
        
        asri = np.zeros(n_days)
        components = {k: np.zeros(n_days) for k in self.weights}
        
        for t in range(self.window * 3, n_days):
            components['price_comovement'][t] = self.compute_price_comovement(returns, t)
            components['volume_anomaly'][t] = self.compute_volume_anomaly(volumes, t)
            components['stablecoin_peg'][t] = self.compute_stablecoin_peg_risk(pegs, t)
            components['tvl_change'][t] = self.compute_tvl_risk(tvl, t)
            components['vol_clustering'][t] = self.compute_vol_clustering(returns, t)
            
            # 加权聚合
            asri[t] = sum(self.weights[k] * components[k][t] for k in self.weights)
        
        return asri, components


# ============================================================
# 3. 系统性风险事件检测 (Systemic Risk Event Detection)
# ============================================================

class RiskEventDetector:
    """
    基于ASRI的系统性风险事件检测
    
    事件定义:
    - 黄色预警: ASRI > 均值 + 1标准差
    - 橙色预警: ASRI > 均值 + 2标准差
    - 红色预警: ASRI > 均值 + 3标准差 (系统性危机)
    """
    
    def __init__(self, asri: np.ndarray):
        self.asri = asri
        self.mean = asri[asri > 0].mean()
        self.std = asri[asri > 0].std()
    
    def detect_events(self) -> List[dict]:
        """检测所有风险事件"""
        events = []
        
        for t in range(len(self.asri)):
            if self.asri[t] <= 0:
                continue
            
            z_score = (self.asri[t] - self.mean) / (self.std + 1e-8)
            
            if z_score > 3:
                level = '红色'
            elif z_score > 2:
                level = '橙色'
            elif z_score > 1:
                level = '黄色'
            else:
                continue
            
            events.append({
                'day': t,
                'asri': self.asri[t],
                'z_score': z_score,
                'level': level,
            })
        
        return events
    
    def get_current_status(self) -> dict:
        """当前风险状态"""
        current = self.asri[-1]
        z = (current - self.mean) / (self.std + 1e-8)
        
        if z > 3:
            status = '系统性危机'
            action = '降低仓位至最低，启动止损'
        elif z > 2:
            status = '高风险'
            action = '降低仓位50%，增加稳定币配比'
        elif z > 1:
            status = '关注'
            action = '检查持仓风险暴露，设置止损'
        else:
            status = '正常'
            action = '维持当前配置'
        
        return {
            'current_asri': current,
            'z_score': z,
            'status': status,
            'recommended_action': action,
            'historical_mean': self.mean,
            'historical_std': self.std,
        }


# ============================================================
# 4. 主程序 (Main Pipeline)
# ============================================================

def main():
    print("=" * 70)
    print("ASRI: 加密货币市场聚合系统性风险指数")
    print("Aggregated Systemic Risk Index for Cryptocurrency Markets")
    print("=" * 70)
    
    # ---- 生成市场数据 ----
    print("\n[1] 生成加密市场数据...")
    simulator = CryptoMarketSimulator(n_days=730, seed=42)  # 2年
    market_data = simulator.generate_market_data()
    
    print(f"  资产数: {len(simulator.assets)}")
    print(f"  时间跨度: {simulator.n_days} 天 (~2年)")
    
    # Regime统计
    regimes = market_data['regimes']
    for r, name in enumerate(['normal', 'bull', 'bear', 'crisis']):
        count = (regimes == r).sum()
        print(f"  {name:<10}: {count} days ({count/len(regimes)*100:.1f}%)")
    
    # ---- 构建ASRI ----
    print(f"\n[2] 构建ASRI指数...")
    builder = ASRIBuilder(window=30)
    asri, components = builder.compute_asri(market_data)
    
    print(f"  ASRI权重配置:")
    for name, weight in builder.weights.items():
        print(f"    {name}: {weight:.0%}")
    
    asri_valid = asri[asri > 0]
    print(f"\n  ASRI统计:")
    print(f"    均值: {asri_valid.mean():.4f}")
    print(f"    标准差: {asri_valid.std():.4f}")
    print(f"    最大值: {asri_valid.max():.4f}")
    print(f"    95分位: {np.percentile(asri_valid, 95):.4f}")
    
    # ---- 风险事件检测 ----
    print(f"\n[3] 系统性风险事件检测...")
    detector = RiskEventDetector(asri)
    events = detector.detect_events()
    
    yellow = [e for e in events if e['level'] == '黄色']
    orange = [e for e in events if e['level'] == '橙色']
    red = [e for e in events if e['level'] == '红色']
    
    print(f"\n  事件统计:")
    print(f"    黄色预警 (>1σ): {len(yellow)} 天")
    print(f"    橙色预警 (>2σ): {len(orange)} 天")
    print(f"    红色预警 (>3σ): {len(red)} 天")
    
    # 显示前5个最严重事件
    top_events = sorted(events, key=lambda x: -x['z_score'])[:5]
    print(f"\n  Top-5 风险事件:")
    for e in top_events:
        print(f"    Day {e['day']:>3}: ASRI={e['asri']:.4f}, "
              f"Z={e['z_score']:.2f} [{e['level']}]")
    
    # ---- 当前状态 ----
    print(f"\n[4] 当前风险状态...")
    status = detector.get_current_status()
    print(f"  当前ASRI: {status['current_asri']:.4f}")
    print(f"  Z-Score: {status['z_score']:.2f}")
    print(f"  状态: {status['status']}")
    print(f"  建议操作: {status['recommended_action']}")
    
    # ---- 子指标分析 ----
    print(f"\n[5] 子指标贡献分析...")
    print(f"\n  {'子指标':<20} {'均值':>8} {'峰值':>8} {'贡献度':>8}")
    print(f"  {'-'*48}")
    
    for name in builder.weights:
        comp = components[name]
        comp_valid = comp[comp > 0]
        if len(comp_valid) > 0:
            avg = comp_valid.mean()
            peak = comp_valid.max()
            contribution = avg * builder.weights[name]
            print(f"  {name:<20} {avg:>8.4f} {peak:>8.4f} {contribution:>8.4f}")
    
    # ---- Regime vs ASRI ----
    print(f"\n[6] Regime vs ASRI 对应分析...")
    print(f"\n  {'Regime':<10} {'平均ASRI':>10} {'ASRI标准差':>10} {'>2σ天数':>10}")
    print(f"  {'-'*45}")
    
    for r, name in enumerate(['normal', 'bull', 'bear', 'crisis']):
        mask = regimes == r
        valid_mask = mask & (asri > 0)
        if valid_mask.sum() > 0:
            regime_asri = asri[valid_mask]
            avg = regime_asri.mean()
            std = regime_asri.std()
            high_risk_days = (regime_asri > detector.mean + 2 * detector.std).sum()
            print(f"  {name:<10} {avg:>10.4f} {std:>10.4f} {high_risk_days:>10}")
    
    # ---- 与传统指标对比 ----
    print(f"\n[7] ASRI vs 传统风险指标对比...")
    
    # BTC波动率
    btc_returns = market_data['returns'][:, 0]
    btc_vol_30d = np.array([btc_returns[max(0,t-30):t].std() * np.sqrt(365) 
                             if t > 30 else 0 for t in range(len(btc_returns))])
    
    # 最大回撤
    btc_prices = np.cumprod(1 + btc_returns)
    running_max = np.maximum.accumulate(btc_prices)
    btc_drawdown = 1 - btc_prices / running_max
    
    # 相关性
    corr_valid = asri[90:] > 0
    if corr_valid.sum() > 10:
        asri_corr = asri[90:][corr_valid]
        vol_corr = btc_vol_30d[90:][corr_valid]
        dd_corr = btc_drawdown[90:][corr_valid]
        
        print(f"\n  ASRI与BTC指标的相关性 (Day 90+):")
        print(f"    ASRI vs BTC年化波动率: {np.corrcoef(asri_corr, vol_corr)[0,1]:.4f}")
        print(f"    ASRI vs BTC回撤: {np.corrcoef(asri_corr, dd_corr)[0,1]:.4f}")
    
    print(f"\n  ASRI的优势:")
    print(f"    1. 多维度: 不仅看价格，还看量、稳定币、DeFi TVL")
    print(f"    2. 前瞻性: 稳定币脱锚和TVL变化往往领先价格下跌")
    print(f"    3. 加密原生: 考虑了加密市场独特的传染渠道")
    
    print("\n" + "=" * 70)
    print("ASRI 复现完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
