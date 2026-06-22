"""
Generative AI for Stock Selection: LLM+RAG自动化因子生成
Generative AI for Stock Selection

复现论文: arXiv 2602.00196 (Feb 2026)
核心思想: 利用LLM+RAG从SEC文件、期权数据、分析师报告中自动生成alpha因子

作者: QoderWork AI Research
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. 模拟知识库 (Simulated Knowledge Base)
# ============================================================

@dataclass
class Document:
    """知识库文档"""
    doc_id: str
    doc_type: str  # 'sec_filing', 'options_data', 'analyst_report', 'earnings_call'
    ticker: str
    content: str
    embedding: np.ndarray = None


class KnowledgeBase:
    """
    模拟RAG知识库
    
    实际应用中替换为真实的向量数据库 (FAISS, Pinecone等)
    """
    
    def __init__(self, n_stocks: int = 50, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.n_stocks = n_stocks
        self.embed_dim = 64
        
        self.tickers = [f'STOCK_{i:03d}' for i in range(n_stocks)]
        self.documents = self._generate_documents()
    
    def _generate_documents(self) -> List[Document]:
        """为每只股票生成模拟文档"""
        docs = []
        
        doc_types = ['sec_filing', 'options_data', 'analyst_report', 'earnings_call']
        
        for ticker in self.tickers:
            for doc_type in doc_types:
                content = self._generate_content(ticker, doc_type)
                doc = Document(
                    doc_id=f"{ticker}_{doc_type}",
                    doc_type=doc_type,
                    ticker=ticker,
                    content=content,
                    embedding=self.rng.randn(self.embed_dim) * 0.3,
                )
                docs.append(doc)
        
        return docs
    
    def _generate_content(self, ticker: str, doc_type: str) -> str:
        """生成模拟文档内容"""
        templates = {
            'sec_filing': f"{ticker}: Revenue growth {self.rng.uniform(-5, 25):.1f}%, "
                         f"margin {self.rng.uniform(10, 40):.1f}%, "
                         f"debt-to-equity {self.rng.uniform(0.1, 2.0):.2f}",
            'options_data': f"{ticker}: IV skew {self.rng.uniform(-0.1, 0.3):.3f}, "
                           f"put-call ratio {self.rng.uniform(0.5, 2.0):.2f}, "
                           f"unusual activity detected",
            'analyst_report': f"{ticker}: Rating {'Buy' if self.rng.random() > 0.3 else 'Hold'}, "
                             f"target upside {self.rng.uniform(-10, 30):.1f}%, "
                             f"key risk: competition",
            'earnings_call': f"{ticker}: Management sentiment {'positive' if self.rng.random() > 0.4 else 'cautious'}, "
                            f"guidance {'raised' if self.rng.random() > 0.5 else 'maintained'}, "
                            f"capex outlook {'expansion' if self.rng.random() > 0.5 else 'stable'}",
        }
        return templates.get(doc_type, "No data")
    
    def retrieve(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Document]:
        """检索最相关的文档"""
        scores = []
        for doc in self.documents:
            similarity = np.dot(query_embedding, doc.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc.embedding) + 1e-8)
            scores.append((similarity, doc))
        
        scores.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scores[:top_k]]
    
    def retrieve_by_ticker(self, ticker: str) -> List[Document]:
        """按股票代码检索"""
        return [doc for doc in self.documents if doc.ticker == ticker]


# ============================================================
# 2. LLM因子生成器 (LLM Factor Generator)
# ============================================================

class LLMFactorGenerator:
    """
    模拟LLM+RAG因子生成
    
    实际应用中替换为真实的LLM API (GPT-4, Claude等)
    """
    
    def __init__(self, knowledge_base: KnowledgeBase, seed: int = 42):
        self.kb = knowledge_base
        self.rng = np.random.RandomState(seed)
        
        # 预定义的因子模板
        self.factor_templates = [
            'revenue_momentum', 'margin_change', 'iv_skew', 
            'sentiment_score', 'analyst_revision', 'options_flow',
            'guidance_surprise', 'debt_quality', 'capex_efficiency',
            'insider_activity',
        ]
    
    def generate_factor(self, ticker: str, factor_name: str) -> float:
        """
        为单个股票生成单个因子值
        
        模拟LLM根据检索到的文档计算因子
        """
        # 检索相关文档
        docs = self.kb.retrieve_by_ticker(ticker)
        
        # 模拟LLM处理: 从文档内容中提取信号
        signal = 0.0
        
        for doc in docs:
            if doc.doc_type == 'sec_filing' and factor_name in ['revenue_momentum', 'margin_change']:
                # 从SEC文件中提取财务因子
                parts = doc.content.split(',')
                for part in parts:
                    if 'growth' in part.lower():
                        try:
                            val = float(part.split()[-1].rstrip('%'))
                            signal += val * 0.01
                        except:
                            pass
                    if 'margin' in part.lower():
                        try:
                            val = float(part.split()[-1].rstrip('%'))
                            signal += val * 0.005
                        except:
                            pass
            
            elif doc.doc_type == 'options_data' and factor_name in ['iv_skew', 'options_flow']:
                parts = doc.content.split(',')
                for part in parts:
                    if 'skew' in part.lower():
                        try:
                            val = float(part.split()[-1])
                            signal += val
                        except:
                            pass
            
            elif doc.doc_type == 'analyst_report' and factor_name == 'analyst_revision':
                if 'Buy' in doc.content:
                    signal += 0.5
                elif 'Hold' in doc.content:
                    signal += 0.0
                else:
                    signal -= 0.3
            
            elif doc.doc_type == 'earnings_call' and factor_name == 'sentiment_score':
                if 'positive' in doc.content:
                    signal += 0.3
                elif 'cautious' in doc.content:
                    signal -= 0.2
        
        # 添加噪声 (LLM不完美)
        signal += self.rng.randn() * 0.1
        
        return float(signal)
    
    def generate_all_factors(self, n_timesteps: int = 60) -> np.ndarray:
        """
        为所有股票生成所有AI因子
        
        returns: (n_timesteps, n_stocks, n_factors) 因子矩阵
        """
        n_stocks = self.kb.n_stocks
        n_factors = len(self.factor_templates)
        
        factors = np.zeros((n_timesteps, n_stocks, n_factors))
        
        for t in range(n_timesteps):
            for s, ticker in enumerate(self.kb.tickers):
                for f, factor_name in enumerate(self.factor_templates):
                    base = self.generate_factor(ticker, factor_name)
                    # 添加时序变化
                    drift = self.rng.randn() * 0.05
                    factors[t, s, f] = base + drift
        
        return factors


# ============================================================
# 3. 因子评估 (Factor Evaluation)
# ============================================================

class FactorEvaluator:
    """
    因子评估框架
    
    指标:
    - IC (Information Coefficient): 因子与未来收益的相关性
    - ICIR (IC Information Ratio): IC的稳定性
    - 因子正交性: 与传统因子的相关性
    - 组合回测: 因子在组合中的表现
    """
    
    @staticmethod
    def compute_ic(factors: np.ndarray, forward_returns: np.ndarray) -> dict:
        """
        计算信息系数 (IC)
        
        factors: (T, N) 因子值
        forward_returns: (T, N) 前瞻收益
        """
        T, N = factors.shape
        ics = []
        
        for t in range(T):
            if forward_returns[t].std() > 0 and factors[t].std() > 0:
                ic = np.corrcoef(factors[t], forward_returns[t])[0, 1]
                ics.append(ic)
            else:
                ics.append(0.0)
        
        ics = np.array(ics)
        return {
            'mean_ic': ics.mean(),
            'ic_std': ics.std(),
            'icir': ics.mean() / (ics.std() + 1e-8),
            'ic_positive_rate': (ics > 0).mean(),
        }
    
    @staticmethod
    def factor_orthogonality(ai_factors: np.ndarray, 
                              traditional_factors: np.ndarray) -> dict:
        """
        因子正交性: AI因子与传统因子的相关性
        
        低相关性意味着AI因子提供了增量信息
        """
        # 展平时序维度
        T, N, F_ai = ai_factors.shape
        _, _, F_trad = traditional_factors.shape
        
        ai_flat = ai_factors.reshape(-1, F_ai)
        trad_flat = traditional_factors.reshape(-1, F_trad)
        
        # 计算每对因子的相关性
        correlations = []
        for i in range(F_ai):
            for j in range(F_trad):
                if ai_flat[:, i].std() > 0 and trad_flat[:, j].std() > 0:
                    corr = abs(np.corrcoef(ai_flat[:, i], trad_flat[:, j])[0, 1])
                    correlations.append(corr)
        
        correlations = np.array(correlations)
        return {
            'mean_abs_correlation': correlations.mean(),
            'max_abs_correlation': correlations.max(),
            'low_correlation_rate': (correlations < 0.3).mean(),  # 低相关比例
        }
    
    @staticmethod
    def long_short_backtest(factors: np.ndarray, returns: np.ndarray,
                             n_quantiles: int = 5) -> dict:
        """
        分层回测: 按因子值分组，比较多空组合
        
        returns: (T, N) 同期收益
        """
        T, N = factors.shape
        
        quantile_returns = {q: [] for q in range(n_quantiles)}
        
        for t in range(T):
            # 排序分组
            factor_vals = factors[t]
            sorted_indices = np.argsort(factor_vals)
            group_size = N // n_quantiles
            
            for q in range(n_quantiles):
                start = q * group_size
                end = (q + 1) * group_size if q < n_quantiles - 1 else N
                group_stocks = sorted_indices[start:end]
                quantile_returns[q].append(returns[t, group_stocks].mean())
        
        # 多空收益 = 最高分位 - 最低分位
        long_returns = np.array(quantile_returns[n_quantiles - 1])
        short_returns = np.array(quantile_returns[0])
        ls_returns = long_returns - short_returns
        
        return {
            'long_annual_return': long_returns.mean() * 252,
            'short_annual_return': short_returns.mean() * 252,
            'long_short_annual_return': ls_returns.mean() * 252,
            'long_sharpe': long_returns.mean() * 252 / (long_returns.std() * np.sqrt(252) + 1e-8),
            'ls_sharpe': ls_returns.mean() * 252 / (ls_returns.std() * np.sqrt(252) + 1e-8),
            'ls_max_drawdown': np.max(1 - np.cumprod(1+ls_returns) / 
                                       np.maximum.accumulate(np.cumprod(1+ls_returns))),
        }


# ============================================================
# 4. 传统因子基准 (Traditional Factor Benchmarks)
# ============================================================

def generate_traditional_factors(n_timesteps: int, n_stocks: int, 
                                  returns: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    生成传统因子作为基准
    
    1. 动量因子 (12-1 month momentum)
    2. 价值因子 (模拟: 1/price)
    3. 波动率因子 (realized volatility)
    4. 质量因子 (模拟: ROE proxy)
    """
    rng = np.random.RandomState(seed)
    n_factors = 4
    factors = np.zeros((n_timesteps, n_stocks, n_factors))
    
    for t in range(n_timesteps):
        lookback = min(t, 20)
        if lookback > 1:
            hist = returns[max(0, t-lookback):t]
            
            # 动量
            factors[t, :, 0] = hist.mean(axis=0) * 252
            # 波动率
            factors[t, :, 1] = hist.std(axis=0) * np.sqrt(252)
            # 价值 (模拟)
            factors[t, :, 2] = rng.randn(n_stocks) * 0.1
            # 质量 (模拟)
            factors[t, :, 3] = rng.randn(n_stocks) * 0.1
    
    return factors


# ============================================================
# 5. 主程序 (Main Pipeline)
# ============================================================

def main():
    print("=" * 70)
    print("Generative AI for Stock Selection: LLM+RAG自动化因子生成")
    print("LLM+RAG Generated Factors for US Equity Selection")
    print("=" * 70)
    
    # ---- 知识库构建 ----
    print("\n[1] 构建知识库...")
    kb = KnowledgeBase(n_stocks=50, seed=42)
    print(f"  股票数量: {kb.n_stocks}")
    print(f"  文档总数: {len(kb.documents)}")
    print(f"  文档类型: sec_filing, options_data, analyst_report, earnings_call")
    
    # ---- AI因子生成 ----
    print("\n[2] LLM+RAG生成AI因子...")
    generator = LLMFactorGenerator(kb, seed=42)
    
    n_timesteps = 252  # 1年日频
    ai_factors = generator.generate_all_factors(n_timesteps=n_timesteps)
    print(f"  因子矩阵形状: {ai_factors.shape}")
    print(f"  因子数量: {len(generator.factor_templates)}")
    print(f"  因子列表: {generator.factor_templates}")
    
    # ---- 生成模拟市场数据 ----
    print("\n[3] 生成模拟市场数据...")
    rng = np.random.RandomState(42)
    
    # 带因子暴露的收益
    true_factor_returns = rng.randn(n_timesteps, len(generator.factor_templates)) * 0.001
    factor_exposures = rng.randn(kb.n_stocks, len(generator.factor_templates)) * 0.5
    
    market_returns = np.zeros((n_timesteps, kb.n_stocks))
    for t in range(n_timesteps):
        market_factor = rng.randn() * 0.01
        for s in range(kb.n_stocks):
            alpha = 0  # 因子暴露 * 因子收益
            for f in range(len(generator.factor_templates)):
                alpha += factor_exposures[s, f] * true_factor_returns[t, f]
            idio = rng.randn() * 0.02
            market_returns[t, s] = market_factor + alpha + idio
    
    print(f"  收益矩阵: {market_returns.shape}")
    print(f"  平均日收益: {market_returns.mean():.6f}")
    
    # ---- 因子评估 ----
    print("\n[4] AI因子评估...")
    evaluator = FactorEvaluator()
    
    # 对每个因子单独评估
    forward_returns = np.roll(market_returns, -1, axis=0)[:-1]
    ai_factors_trimmed = ai_factors[:-1]
    
    print(f"\n  单因子IC分析:")
    print(f"  {'因子':<20} {'Mean IC':>10} {'ICIR':>10} {'IC>0比例':>10}")
    print(f"  {'-'*55}")
    
    for f, name in enumerate(generator.factor_templates):
        ic_stats = evaluator.compute_ic(ai_factors_trimmed[:, :, f], forward_returns)
        print(f"  {name:<20} {ic_stats['mean_ic']:>10.4f} "
              f"{ic_stats['icir']:>10.4f} {ic_stats['ic_positive_rate']:>10.4f}")
    
    # ---- 传统因子基准 ----
    print("\n[5] 与传统因子对比...")
    trad_factors = generate_traditional_factors(
        n_timesteps-1, kb.n_stocks, market_returns[:-1], seed=42)
    
    # 正交性分析
    ortho = evaluator.factor_orthogonality(ai_factors_trimmed, trad_factors)
    print(f"\n  AI因子 vs 传统因子正交性:")
    print(f"    平均绝对相关性: {ortho['mean_abs_correlation']:.4f}")
    print(f"    最大绝对相关性: {ortho['max_abs_correlation']:.4f}")
    print(f"    低相关比例 (<0.3): {ortho['low_correlation_rate']:.4f}")
    
    if ortho['low_correlation_rate'] > 0.5:
        print(f"    → AI因子与传统因子有较好的正交性，提供增量信息")
    
    # ---- 组合因子回测 ----
    print("\n[6] 组合因子回测...")
    
    # AI因子组合 (等权合并所有AI因子)
    ai_combined = ai_factors_trimmed.mean(axis=2)
    ai_backtest = evaluator.long_short_backtest(ai_combined, market_returns[:-1])
    
    # 传统因子组合
    trad_combined = trad_factors.mean(axis=2)
    trad_backtest = evaluator.long_short_backtest(trad_combined, market_returns[:-1])
    
    # 混合因子 (AI + 传统)
    blended = 0.5 * ai_combined + 0.5 * trad_combined
    blended_backtest = evaluator.long_short_backtest(blended, market_returns[:-1])
    
    print(f"\n  {'策略':<20} {'年化收益':>12} {'夏普比率':>10} {'最大回撤':>10}")
    print(f"  {'-'*55}")
    print(f"  {'AI因子多空':<20} {ai_backtest['long_short_annual_return']:>12.4f} "
          f"{ai_backtest['ls_sharpe']:>10.4f} {ai_backtest['ls_max_drawdown']:>10.4f}")
    print(f"  {'传统因子多空':<20} {trad_backtest['long_short_annual_return']:>12.4f} "
          f"{trad_backtest['ls_sharpe']:>10.4f} {trad_backtest['ls_max_drawdown']:>10.4f}")
    print(f"  {'混合因子多空':<20} {blended_backtest['long_short_annual_return']:>12.4f} "
          f"{blended_backtest['ls_sharpe']:>10.4f} {blended_backtest['ls_max_drawdown']:>10.4f}")
    
    # ---- 因子衰减分析 ----
    print("\n[7] 因子衰减分析...")
    
    for f, name in enumerate(generator.factor_templates[:5]):  # 前5个因子
        ic_by_horizon = []
        for horizon in [1, 5, 10, 20]:
            fwd = np.roll(market_returns, -horizon, axis=0)
            valid_len = min(len(fwd)-horizon, len(ai_factors_trimmed))
            if valid_len > 0:
                ic = evaluator.compute_ic(
                    ai_factors_trimmed[:valid_len, :, f], fwd[:valid_len])
                ic_by_horizon.append(ic['mean_ic'])
            else:
                ic_by_horizon.append(0.0)
        
        print(f"  {name:<20}: IC@1d={ic_by_horizon[0]:+.4f}, "
              f"IC@5d={ic_by_horizon[1]:+.4f}, "
              f"IC@10d={ic_by_horizon[2]:+.4f}, "
              f"IC@20d={ic_by_horizon[3]:+.4f}")
    
    print("\n" + "=" * 70)
    print("Generative AI for Stock Selection 复现完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
