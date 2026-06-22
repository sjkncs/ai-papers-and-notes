"""
Regret-Driven Portfolios: LLM引导的智能聚类组合优化
Regret-Driven Portfolios: LLM-Guided Smart Clustering for Optimal Allocation

复现论文: arXiv 2601.17021 (Jan 2026)
核心思想: 利用LLM语义理解对资产进行叙事聚类，结合遗憾最小化进行动态配置

作者: QoderWork AI Research
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, fcluster
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. 模拟LLM语义Embedding (Simulated LLM Semantic Embedding)
# ============================================================

class LLMSemanticEngine:
    """
    模拟LLM对金融资产的语义理解
    
    实际应用中替换为真实的LLM API调用:
    - 输入: 资产描述、近期新闻标题
    - 输出: 语义embedding向量
    """
    
    def __init__(self, n_assets: int, embed_dim: int = 64, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.n_assets = n_assets
        self.embed_dim = embed_dim
        
        # 模拟资产的行业/主题标签
        self.sectors = [
            'tech', 'tech', 'tech', 'tech', 'tech',      # 科技
            'finance', 'finance', 'finance', 'finance',    # 金融
            'health', 'health', 'health',                  # 医疗
            'energy', 'energy', 'energy',                  # 能源
            'consumer', 'consumer',                        # 消费
            'industrial', 'industrial',                    # 工业
            'materials',                                   # 材料
        ][:n_assets]
        
        # 生成语义embedding (同行业更相近)
        self.embeddings = self._generate_embeddings()
    
    def _generate_embeddings(self) -> np.ndarray:
        """生成语义embedding，同行业资产更相近"""
        unique_sectors = list(set(self.sectors))
        sector_centroids = {s: self.rng.randn(self.embed_dim) * 0.5 
                           for s in unique_sectors}
        
        embeddings = np.zeros((self.n_assets, self.embed_dim))
        for i in range(self.n_assets):
            centroid = sector_centroids[self.sectors[i]]
            # 行业中心 + 个体噪声
            embeddings[i] = centroid + self.rng.randn(self.embed_dim) * 0.2
        
        # L2归一化
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        
        return embeddings
    
    def get_asset_embeddings(self) -> np.ndarray:
        """获取所有资产的语义embedding"""
        return self.embeddings
    
    def generate_sentiment_scores(self, t: int, total_t: int) -> np.ndarray:
        """
        生成时变情绪分数
        
        模拟LLM对每日新闻的情绪评分
        [-1, 1]: -1=极度悲观, 0=中性, 1=极度乐观
        """
        # 基础情绪 + 行业效应 + 时序漂移
        base_sentiment = 0.1  # 略偏乐观
        sector_effects = np.array([
            0.15, 0.1, 0.2, -0.05, 0.05,    # tech: 偏乐观
            0.0, -0.1, 0.05, 0.1,            # finance: 中性
            0.1, 0.05, -0.1,                  # health: 中性偏乐观
            -0.15, -0.1, -0.05,               # energy: 偏悲观
            0.1, 0.05,                         # consumer: 乐观
            0.0, -0.05,                        # industrial: 中性
            -0.1,                              # materials: 偏悲观
        ])[:self.n_assets]
        
        # 时序成分 (regime切换)
        regime = np.sin(2 * np.pi * t / 252) * 0.3  # 年度周期
        if t > total_t * 0.7:  # 后30%模拟熊市
            regime -= 0.4
        
        # 个体噪声
        noise = self.rng.randn(self.n_assets) * 0.1
        
        sentiment = base_sentiment + sector_effects + regime + noise
        sentiment = np.clip(sentiment, -1, 1)
        
        return sentiment


# ============================================================
# 2. 智能聚类 (Smart Clustering)
# ============================================================

class SmartClusterer:
    """
    基于LLM语义的智能资产聚类
    
    与传统方法的对比:
    1. 行业分类: 基于固定分类体系 (GICS)
    2. 收益相关性: 基于历史收益的相关矩阵
    3. LLM语义聚类: 基于叙事相似性的动态聚类
    """
    
    def __init__(self, method: str = 'semantic', n_clusters: int = 5):
        """
        method: 'semantic', 'correlation', 'industry'
        """
        self.method = method
        self.n_clusters = n_clusters
    
    def cluster(self, embeddings: np.ndarray = None, 
                returns: np.ndarray = None,
                sectors: List[str] = None) -> np.ndarray:
        """
        执行聚类，返回每个资产的簇标签
        
        returns: cluster_labels (n_assets,)
        """
        if self.method == 'semantic':
            return self._semantic_cluster(embeddings)
        elif self.method == 'correlation':
            return self._correlation_cluster(returns)
        elif self.method == 'industry':
            return self._industry_cluster(sectors)
    
    def _semantic_cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """基于语义embedding的层次聚类"""
        # 余弦距离
        dist_matrix = cdist(embeddings, embeddings, metric='cosine')
        
        # 层次聚类
        condensed = dist_matrix[np.triu_indices(len(dist_matrix), k=1)]
        Z = linkage(condensed, method='ward')
        labels = fcluster(Z, self.n_clusters, criterion='maxclust')
        
        return labels - 1  # 0-indexed
    
    def _correlation_cluster(self, returns: np.ndarray) -> np.ndarray:
        """基于收益相关性的聚类"""
        corr = np.corrcoef(returns.T)
        dist_matrix = 1 - corr  # 相关性距离
        
        condensed = dist_matrix[np.triu_indices(len(dist_matrix), k=1)]
        Z = linkage(condensed, method='ward')
        labels = fcluster(Z, self.n_clusters, criterion='maxclust')
        
        return labels - 1
    
    def _industry_cluster(self, sectors: List[str]) -> np.ndarray:
        """基于行业分类的聚类"""
        unique_sectors = list(set(sectors))
        sector_to_label = {s: i for i, s in enumerate(unique_sectors)}
        labels = np.array([sector_to_label[s] for s in sectors])
        return labels


# ============================================================
# 3. 遗憾最小化配置 (Regret-Minimization Allocation)
# ============================================================

@dataclass
class RegretConfig:
    """遗憾最小化配置"""
    learning_rate: float = 0.01      # 学习率
    risk_aversion: float = 2.0       # 风险厌恶系数
    max_weight: float = 0.3          # 单个资产最大权重
    rebalance_freq: int = 20         # 再平衡频率 (交易日)
    max_drawdown_constraint: float = 0.15  # 最大回撤约束


class RegretMinimizer:
    """
    Follow-the-Leader + 遗憾最小化资产配置
    
    核心思想:
    - 在每个聚类内部，跟踪每个资产的历史表现
    - 使用指数加权平均(Hedge算法)分配权重
    - 遗憾值衡量与最优事后策略的差距
    """
    
    def __init__(self, n_assets: int, config: RegretConfig):
        self.n_assets = n_assets
        self.cfg = config
        
        # 累积收益 (用于FTL)
        self.cumulative_returns = np.zeros(n_assets)
        # Hedge权重 (指数加权)
        self.hedge_weights = np.ones(n_assets) / n_assets
        # 历史遗憾值
        self.regret_history = []
    
    def get_weights(self, current_returns: np.ndarray = None) -> np.ndarray:
        """
        计算当前权重
        
        结合Follow-the-Leader和Hedge算法:
        w_i = softmax(η * R_i) / Z
        其中 R_i 是资产i的累积收益
        """
        if current_returns is not None:
            self.cumulative_returns += current_returns
        
        # Follow-the-Leader成分
        ftl_scores = self.cfg.learning_rate * self.cumulative_returns
        
        # Hedge更新
        if current_returns is not None:
            self.hedge_weights *= np.exp(self.cfg.learning_rate * current_returns)
            self.hedge_weights /= self.hedge_weights.sum()
        
        # 混合FTL和Hedge
        ftl_weights = np.exp(ftl_scores)
        ftl_weights /= ftl_weights.sum()
        
        weights = 0.5 * ftl_weights + 0.5 * self.hedge_weights
        
        # 约束: 最大权重
        weights = np.minimum(weights, self.cfg.max_weight)
        weights /= weights.sum()
        
        return weights
    
    def update_regret(self, returns: np.ndarray, weights: np.ndarray):
        """
        更新遗憾值
        
        Regret = max_i Σr_i - Σ(w·r)
        即最优单资产收益 - 组合收益
        """
        port_return = weights @ returns
        best_asset_return = returns.max()
        regret = best_asset_return - port_return
        
        self.regret_history.append(regret)
    
    def get_cumulative_regret(self) -> float:
        """累积遗憾"""
        return sum(self.regret_history)


# ============================================================
# 4. 情绪增强信号 (Sentiment-Enhanced Signals)
# ============================================================

class SentimentSignal:
    """
    LLM情绪信号处理
    
    将LLM生成的情绪分数转化为交易信号:
    1. 情绪趋势 → 动量信号补充
    2. 情绪极端值 → 再平衡触发
    3. 情绪分歧 → 风险信号
    """
    
    def __init__(self, window: int = 20, extreme_threshold: float = 0.6):
        self.window = window
        self.extreme_threshold = extreme_threshold
        self.sentiment_history = []
    
    def update(self, sentiment: np.ndarray):
        """更新情绪历史"""
        self.sentiment_history.append(sentiment.copy())
    
    def get_momentum_signal(self) -> np.ndarray:
        """
        情绪动量信号: 近期情绪变化方向
        
        正值: 情绪改善 (看多)
        负值: 情绪恶化 (看空)
        """
        if len(self.sentiment_history) < self.window:
            return np.zeros(len(self.sentiment_history[-1]))
        
        recent = np.array(self.sentiment_history[-self.window:])
        # 线性趋势斜率
        t = np.arange(self.window)
        signals = np.zeros(recent.shape[1])
        for i in range(recent.shape[1]):
            signals[i] = np.polyfit(t, recent[:, i], 1)[0]
        
        return signals
    
    def check_rebalance_trigger(self) -> Tuple[bool, str]:
        """
        检查是否需要触发再平衡
        
        触发条件:
        1. 平均情绪超过极端阈值
        2. 情绪分歧 (标准差) 异常高
        """
        if len(self.sentiment_history) < 5:
            return False, ""
        
        current = self.sentiment_history[-1]
        mean_sentiment = current.mean()
        std_sentiment = current.std()
        
        if abs(mean_sentiment) > self.extreme_threshold:
            direction = "极度乐观" if mean_sentiment > 0 else "极度悲观"
            return True, f"情绪极端: {direction} ({mean_sentiment:.2f})"
        
        if std_sentiment > 0.5:
            return True, f"情绪分歧过大: σ={std_sentiment:.2f}"
        
        return False, ""
    
    def get_risk_signal(self) -> float:
        """
        风险信号: 基于情绪分歧的波动率调整因子
        
        高分歧 → 高风险 → 降低仓位
        """
        if len(self.sentiment_history) < 5:
            return 1.0
        
        recent = np.array(self.sentiment_history[-5:])
        avg_std = recent.std(axis=1).mean()
        
        # 分歧越大，风险因子越低
        risk_factor = np.exp(-2 * avg_std)
        return float(np.clip(risk_factor, 0.3, 1.0))


# ============================================================
# 5. 完整回测引擎 (Full Backtesting Engine)
# ============================================================

class RegretPortfolioBacktest:
    """
    Regret-Driven Portfolio 完整回测
    
    流程:
    1. 每月重新聚类 (基于最新语义embedding)
    2. 每日在每个聚类内进行遗憾最小化配置
    3. 情绪极端时触发额外再平衡
    4. 跟踪组合收益、风险、遗憾值
    """
    
    def __init__(self, n_assets: int = 20, n_days: int = 504, seed: int = 42):
        self.n_assets = n_assets
        self.n_days = n_days
        self.rng = np.random.RandomState(seed)
        
        # 组件初始化
        self.llm_engine = LLMSemanticEngine(n_assets, seed=seed)
        self.config = RegretConfig()
        self.sentiment = SentimentSignal()
        
        # 生成模拟市场数据
        self.returns = self._generate_market_data()
    
    def _generate_market_data(self) -> np.ndarray:
        """生成模拟市场数据 (带regime切换)"""
        returns = np.zeros((self.n_days, self.n_assets))
        
        # Regime参数
        regimes = {
            'bull': {'mu': 0.0005, 'sigma': 0.012, 'prob': 0.5},
            'bear': {'mu': -0.001, 'sigma': 0.02, 'prob': 0.25},
            'sideways': {'mu': 0.0001, 'sigma': 0.008, 'prob': 0.25},
        }
        
        current_regime = 'bull'
        for t in range(self.n_days):
            # Regime切换 (Markov链)
            if self.rng.random() < 0.02:  # 2%概率切换
                current_regime = self.rng.choice(list(regimes.keys()))
            
            params = regimes[current_regime]
            
            # 行业效应
            sector_effects = np.zeros(self.n_assets)
            if current_regime == 'bull':
                sector_effects[:5] = 0.0003   # tech领涨
                sector_effects[5:9] = 0.0001  # finance跟随
            elif current_regime == 'bear':
                sector_effects[:5] = -0.0005  # tech领跌
                sector_effects[13:16] = 0.0002  # energy抗跌
            
            # 截面相关性 (通过共同因子)
            common_factor = self.rng.randn() * params['sigma'] * 0.5
            idio = self.rng.randn(self.n_assets) * params['sigma'] * 0.8
            
            returns[t] = params['mu'] + sector_effects + common_factor + idio
        
        return returns
    
    def run(self, cluster_method: str = 'semantic') -> dict:
        """
        运行回测
        
        cluster_method: 'semantic', 'correlation', 'industry'
        """
        print(f"\n  运行回测: 聚类方法={cluster_method}, "
              f"资产数={self.n_assets}, 天数={self.n_days}")
        
        clusterer = SmartClusterer(method=cluster_method, n_clusters=5)
        
        # 初始化每个聚类一个RegretMinimizer
        cluster_minimizers = {}
        
        # 记录
        portfolio_returns = []
        daily_weights = []
        rebalance_events = []
        
        # 初始聚类
        embeddings = self.llm_engine.get_asset_embeddings()
        labels = clusterer.cluster(
            embeddings=embeddings,
            returns=self.returns[:60],
            sectors=self.llm_engine.sectors,
        )
        
        for c in range(len(set(labels))):
            cluster_assets = np.where(labels == c)[0]
            cluster_minimizers[c] = RegretMinimizer(len(cluster_assets), self.config)
        
        last_rebalance = 0
        
        for t in range(self.n_days):
            # 获取当日收益
            day_returns = self.returns[t]
            
            # 生成情绪
            sentiment = self.llm_engine.generate_sentiment_scores(t, self.n_days)
            self.sentiment.update(sentiment)
            
            # 检查再平衡触发
            should_rebalance, reason = self.sentiment.check_rebalance_trigger()
            
            # 定期再平衡聚类
            if t - last_rebalance > 60:  # 每季度
                if t > 60:
                    labels = clusterer.cluster(
                        embeddings=embeddings,
                        returns=self.returns[max(0,t-60):t],
                        sectors=self.llm_engine.sectors,
                    )
                    # 重建minimizers
                    cluster_minimizers = {}
                    for c in range(len(set(labels))):
                        cluster_assets = np.where(labels == c)[0]
                        cluster_minimizers[c] = RegretMinimizer(
                            len(cluster_assets), self.config)
                    last_rebalance = t
                    rebalance_events.append((t, "定期再聚类"))
            
            if should_rebalance:
                rebalance_events.append((t, reason))
            
            # 计算权重
            weights = np.zeros(self.n_assets)
            for c in cluster_minimizers:
                cluster_assets = np.where(labels == c)[0]
                if len(cluster_assets) == 0:
                    continue
                
                minimizer = cluster_minimizers[c]
                cluster_returns = day_returns[cluster_assets]
                
                # 获取聚类内权重
                cluster_weights = minimizer.get_weights(cluster_returns)
                
                # 风险信号调整
                risk_factor = self.sentiment.get_risk_signal()
                cluster_weights *= risk_factor
                
                # 分配到全局权重
                # 聚类权重按等权分配 (可优化为情绪加权)
                cluster_allocation = 1.0 / len(cluster_minimizers)
                for idx, asset_idx in enumerate(cluster_assets):
                    weights[asset_idx] = cluster_allocation * cluster_weights[idx]
                
                # 更新遗憾
                minimizer.update_regret(cluster_returns, cluster_weights)
            
            # 权重归一化
            if weights.sum() > 0:
                weights /= weights.sum()
            else:
                weights = np.ones(self.n_assets) / self.n_assets
            
            # 组合收益
            port_return = weights @ day_returns
            portfolio_returns.append(port_return)
            daily_weights.append(weights.copy())
        
        portfolio_returns = np.array(portfolio_returns)
        daily_weights = np.array(daily_weights)
        
        # 计算绩效指标
        results = self._compute_metrics(portfolio_returns, daily_weights, rebalance_events)
        return results
    
    def _compute_metrics(self, returns: np.ndarray, weights: np.ndarray,
                         rebalance_events: list) -> dict:
        """计算绩效指标"""
        # 累积收益曲线
        cum_returns = np.cumprod(1 + returns)
        
        # 年化收益
        annual_return = (cum_returns[-1]) ** (252 / len(returns)) - 1
        
        # 年化波动率
        annual_vol = returns.std() * np.sqrt(252)
        
        # 夏普比率 (无风险利率2%)
        sharpe = (annual_return - 0.02) / (annual_vol + 1e-8)
        
        # 最大回撤
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = 1 - cum_returns / running_max
        max_drawdown = drawdowns.max()
        
        # Calmar比率
        calmar = annual_return / (max_drawdown + 1e-8)
        
        # 胜率
        win_rate = (returns > 0).mean()
        
        # 下行风险
        downside_returns = returns[returns < 0]
        downside_risk = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # Sortino比率
        sortino = (annual_return - 0.02) / (downside_risk + 1e-8)
        
        return {
            'annual_return': annual_return,
            'annual_vol': annual_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'win_rate': win_rate,
            'n_rebalances': len(rebalance_events),
            'cumulative_return': cum_returns[-1] - 1,
            'rebalance_events': rebalance_events,
        }


# ============================================================
# 6. 主程序 (Main Pipeline)
# ============================================================

def main():
    print("=" * 70)
    print("Regret-Driven Portfolios: LLM引导的智能聚类组合优化")
    print("LLM-Guided Smart Clustering for Optimal Allocation")
    print("=" * 70)
    
    # ---- 回测不同聚类方法 ----
    backtest = RegretPortfolioBacktest(n_assets=20, n_days=504, seed=42)
    
    methods = ['semantic', 'correlation', 'industry']
    all_results = {}
    
    for method in methods:
        print(f"\n{'='*50}")
        print(f"聚类方法: {method}")
        print(f"{'='*50}")
        
        results = backtest.run(cluster_method=method)
        all_results[method] = results
        
        print(f"\n  年化收益: {results['annual_return']:.4f}")
        print(f"  年化波动率: {results['annual_vol']:.4f}")
        print(f"  夏普比率: {results['sharpe_ratio']:.4f}")
        print(f"  索提诺比率: {results['sortino_ratio']:.4f}")
        print(f"  最大回撤: {results['max_drawdown']:.4f}")
        print(f"  Calmar比率: {results['calmar_ratio']:.4f}")
        print(f"  胜率: {results['win_rate']:.4f}")
        print(f"  再平衡次数: {results['n_rebalances']}")
    
    # ---- 与基准对比 ----
    print(f"\n{'='*50}")
    print("基准对比")
    print(f"{'='*50}")
    
    # 等权基准
    equal_returns = backtest.returns.mean(axis=1)
    equal_cum = np.cumprod(1 + equal_returns)
    equal_annual = equal_cum[-1] ** (252/len(equal_returns)) - 1
    equal_vol = equal_returns.std() * np.sqrt(252)
    equal_sharpe = (equal_annual - 0.02) / (equal_vol + 1e-8)
    equal_dd = (1 - equal_cum / np.maximum.accumulate(equal_cum)).max()
    
    print(f"\n  等权基准:")
    print(f"    年化收益: {equal_annual:.4f}")
    print(f"    夏普比率: {equal_sharpe:.4f}")
    print(f"    最大回撤: {equal_dd:.4f}")
    
    # 汇总对比
    print(f"\n  {'方法':<15} {'年化收益':>10} {'夏普比率':>10} {'最大回撤':>10}")
    print(f"  {'-'*50}")
    print(f"  {'等权':<15} {equal_annual:>10.4f} {equal_sharpe:>10.4f} {equal_dd:>10.4f}")
    for method, res in all_results.items():
        print(f"  {method:<15} {res['annual_return']:>10.4f} "
              f"{res['sharpe_ratio']:>10.4f} {res['max_drawdown']:>10.4f}")
    
    # ---- 情绪信号分析 ----
    print(f"\n{'='*50}")
    print("情绪信号分析")
    print(f"{'='*50}")
    
    best_method = max(all_results, key=lambda m: all_results[m]['sharpe_ratio'])
    best_results = all_results[best_method]
    
    if best_results['rebalance_events']:
        print(f"\n  情绪触发的再平衡事件:")
        for t, reason in best_results['rebalance_events'][:5]:
            print(f"    Day {t}: {reason}")
        if len(best_results['rebalance_events']) > 5:
            print(f"    ... 共 {len(best_results['rebalance_events'])} 次")
    
    print("\n" + "=" * 70)
    print("Regret-Driven Portfolios 复现完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
