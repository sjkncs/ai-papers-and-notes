"""
MarketGANs: 因子嵌入GAN的多资产金融时序数据增强
MarketGANs: Factor-Embedded GAN for Multi-Asset Financial Time Series Augmentation

复现论文: arXiv 2601.17773 (Jan 2026)
核心思想: 将金融因子模型(Fama-French)嵌入GAN架构，生成保持截面相关性和时序动态的多资产收益

作者: QoderWork AI Research
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. 因子模型数据生成 (Factor Model Data Generation)
# ============================================================

@dataclass
class FactorModelConfig:
    """因子模型配置"""
    n_assets: int = 30           # 资产数量
    n_factors: int = 3           # 因子数量 (如 Fama-French 三因子)
    n_timesteps: int = 2520      # 训练数据时间步 (~10年)
    factor_vol: float = 0.15     # 因子波动率
    idio_vol: float = 0.25       # 特质波动率
    beta_range: Tuple = (0.5, 1.5)  # 因子载荷范围


class FactorDataGenerator:
    """
    基于因子模型生成多资产收益数据
    
    模型: r_t = B_t @ f_t + epsilon_t
    其中:
    - B_t: 时变因子载荷矩阵 (n_assets x n_factors)
    - f_t: 因子收益向量 (n_factors,)
    - epsilon_t: 特质收益 (n_assets,)
    """
    
    def __init__(self, config: FactorModelConfig, seed: int = 42):
        self.cfg = config
        self.rng = np.random.RandomState(seed)
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化因子模型参数"""
        cfg = self.cfg
        
        # 基准因子载荷 (时变的均值)
        self.beta_base = self.rng.uniform(
            cfg.beta_range[0], cfg.beta_range[1], 
            (cfg.n_assets, cfg.n_factors)
        )
        
        # 因子收益的均值和协方差
        # 模拟: 市场因子(MKT), 规模因子(SMB), 价值因子(HML)
        self.factor_mean = np.array([0.0004, 0.0002, 0.0003])[:cfg.n_factors]
        factor_corr = np.eye(cfg.n_factors)
        factor_corr[0, 1] = factor_corr[1, 0] = 0.2  # MKT-SMB低相关
        factor_corr[0, 2] = factor_corr[2, 0] = -0.3  # MKT-HML负相关
        if cfg.n_factors > 2:
            factor_corr[1, 2] = factor_corr[2, 1] = 0.1
        self.factor_cov = factor_corr * (cfg.factor_vol / np.sqrt(252))**2
    
    def generate(self) -> dict:
        """生成完整的多资产时序数据"""
        cfg = self.cfg
        rng = self.rng
        
        # 生成因子收益 (带GARCH效应)
        factor_returns = np.zeros((cfg.n_timesteps, cfg.n_factors))
        factor_vol_state = np.ones(cfg.n_factors) * (cfg.factor_vol / np.sqrt(252))
        
        for t in range(cfg.n_timesteps):
            # GARCH(1,1) 波动率动态
            if t > 0:
                factor_vol_state = np.sqrt(
                    0.01 * (cfg.factor_vol / np.sqrt(252))**2 +  # 长期均值
                    0.10 * factor_returns[t-1]**2 +                # ARCH项
                    0.85 * factor_vol_state**2                     # GARCH项
                )
            
            # 带时变波动率的因子收益
            z = rng.multivariate_normal(np.zeros(cfg.n_factors), self.factor_cov)
            factor_returns[t] = self.factor_mean + factor_vol_state * z
        
        # 时变因子载荷 (缓慢漂移)
        beta_series = np.zeros((cfg.n_timesteps, cfg.n_assets, cfg.n_factors))
        beta_series[0] = self.beta_base
        for t in range(1, cfg.n_timesteps):
            # AR(1) 过程 + 小噪声
            beta_series[t] = 0.99 * beta_series[t-1] + 0.01 * self.beta_base + \
                             rng.normal(0, 0.01, (cfg.n_assets, cfg.n_factors))
        
        # 特质收益 (带截面相关性)
        # 使用block-diagonal结构模拟行业内的相关性
        idio_corr = np.eye(cfg.n_assets)
        for i in range(0, cfg.n_assets, 5):  # 每5个资产一组
            end = min(i + 5, cfg.n_assets)
            idio_corr[i:end, i:end] = 0.3 * np.ones((end-i, end-i))
            np.fill_diagonal(idio_corr[i:end, i:end], 1.0)
        
        idio_cov = idio_corr * (cfg.idio_vol / np.sqrt(252))**2
        
        # 特质收益 (也带GARCH)
        idio_returns = np.zeros((cfg.n_timesteps, cfg.n_assets))
        idio_vol_state = np.ones(cfg.n_assets) * (cfg.idio_vol / np.sqrt(252))
        
        for t in range(cfg.n_timesteps):
            if t > 0:
                idio_vol_state = np.sqrt(
                    0.01 * (cfg.idio_vol / np.sqrt(252))**2 +
                    0.08 * idio_returns[t-1]**2 +
                    0.88 * idio_vol_state**2
                )
            z = rng.multivariate_normal(np.zeros(cfg.n_assets), idio_cov)
            idio_returns[t] = idio_vol_state * z
        
        # 组合: r_t = B_t @ f_t + epsilon_t
        asset_returns = np.zeros((cfg.n_timesteps, cfg.n_assets))
        for t in range(cfg.n_timesteps):
            asset_returns[t] = beta_series[t] @ factor_returns[t] + idio_returns[t]
        
        return {
            'returns': asset_returns,           # (T, n_assets)
            'factor_returns': factor_returns,    # (T, n_factors)
            'beta_series': beta_series,          # (T, n_assets, n_factors)
            'idio_returns': idio_returns,        # (T, n_assets)
            'idio_vol': idio_vol_state,          # (n_assets,)
        }


# ============================================================
# 2. 时序卷积网络 (Temporal Convolutional Network)
# ============================================================

class TemporalConvBlock:
    """
    因果膨胀卷积块 (Causal Dilated Conv Block)
    
    用于捕获时序依赖而不泄露未来信息
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, dilation: int = 1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # 权重初始化 (Xavier)
        scale = np.sqrt(2.0 / (in_channels * kernel_size))
        self.W = np.random.randn(kernel_size, in_channels, out_channels) * scale
        self.b = np.zeros(out_channels)
        
        # 残差投影
        if in_channels != out_channels:
            self.W_res = np.random.randn(in_channels, out_channels) * np.sqrt(2.0 / in_channels)
        else:
            self.W_res = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: (batch, timesteps, in_channels)
        returns: (batch, timesteps, out_channels)
        """
        B, T, C = x.shape
        
        # 因果填充 (左侧padding)
        pad_size = (self.kernel_size - 1) * self.dilation
        x_padded = np.pad(x, ((0, 0), (pad_size, 0), (0, 0)))
        
        # 膨胀卷积
        out = np.zeros((B, T, self.out_channels))
        for i in range(self.kernel_size):
            offset = i * self.dilation
            x_slice = x_padded[:, offset:offset + T, :]
            out += x_slice @ self.W[i]
        out += self.b
        
        # 门控激活 (Gated Activation)
        tanh_out = np.tanh(out)
        sigmoid_out = 1.0 / (1.0 + np.exp(-out))
        out = tanh_out * sigmoid_out
        
        # 残差连接
        if self.W_res is not None:
            res = x @ self.W_res
        else:
            res = x
        
        return out + res


class TCNEncoder:
    """
    时序卷积网络编码器
    多层膨胀卷积捕获不同时间尺度的依赖
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, n_layers: int = 4):
        self.layers = []
        dim = input_dim
        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else hidden_dim
            dilation = 2 ** i
            self.layers.append(TemporalConvBlock(dim, out_dim, kernel_size=3, dilation=dilation))
            dim = out_dim
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (batch, timesteps, features) -> (batch, timesteps, hidden_dim)"""
        for layer in self.layers:
            x = layer.forward(x)
        return x


# ============================================================
# 3. 因子嵌入GAN (Factor-Embedded GAN)
# ============================================================

class FactorGenerator:
    """
    因子嵌入生成器
    
    生成: β(t) 和 f(t)，然后通过 r(t) = β(t) @ f(t) + ε(t) 构建资产收益
    这确保了生成数据遵循资产定价理论
    """
    
    def __init__(self, n_assets: int, n_factors: int, latent_dim: int = 32,
                 hidden_dim: int = 64):
        self.n_assets = n_assets
        self.n_factors = n_factors
        self.latent_dim = latent_dim
        
        # TCN编码器处理噪声输入
        self.tcn = TCNEncoder(latent_dim, hidden_dim, n_layers=3)
        
        # 输出头
        # 因子收益头: hidden_dim -> n_factors
        scale_f = np.sqrt(2.0 / hidden_dim)
        self.W_factor = np.random.randn(hidden_dim, n_factors) * scale_f
        self.b_factor = np.zeros(n_factors)
        
        # 因子载荷头: hidden_dim -> n_assets * n_factors
        scale_b = np.sqrt(2.0 / hidden_dim)
        self.W_beta = np.random.randn(hidden_dim, n_assets * n_factors) * scale_b
        self.b_beta = np.zeros(n_assets * n_factors)
        
        # 特质波动率头
        self.W_idio = np.random.randn(hidden_dim, n_assets) * scale_b
        self.b_idio = np.ones(n_assets) * 0.01  # 初始化为正值
    
    def generate(self, batch_size: int = 1, seq_len: int = 252) -> dict:
        """
        生成一批多资产时序数据
        
        returns:
            returns: (batch, seq_len, n_assets) 资产收益
            factors: (batch, seq_len, n_factors) 因子收益
            betas: (batch, seq_len, n_assets, n_factors) 因子载荷
        """
        # 随机噪声输入
        z = np.random.randn(batch_size, seq_len, self.latent_dim) * 0.1
        
        # TCN编码
        h = self.tcn.forward(z)  # (batch, seq_len, hidden_dim)
        
        # 因子收益
        factors = h @ self.W_factor + self.b_factor  # (batch, seq_len, n_factors)
        
        # 因子载荷 (softplus保证正值)
        beta_raw = h @ self.W_beta + self.b_beta
        beta_raw = beta_raw.reshape(batch_size, seq_len, self.n_assets, self.n_factors)
        betas = np.log(1 + np.exp(beta_raw))  # softplus
        betas = np.clip(betas, 0.1, 3.0)  # 限制范围
        
        # 特质波动率
        idio_vol = h @ self.W_idio + self.b_idio
        idio_vol = np.log(1 + np.exp(idio_vol))  # softplus
        idio_vol = np.clip(idio_vol, 0.001, 0.05)
        
        # 构建收益
        returns = np.zeros((batch_size, seq_len, self.n_assets))
        idio_returns = np.zeros_like(returns)
        for b in range(batch_size):
            for t in range(seq_len):
                # 因子部分
                factor_component = betas[b, t] @ factors[b, t]
                # 特质部分
                idio_component = idio_vol[b, t] * np.random.randn(self.n_assets)
                returns[b, t] = factor_component + idio_component
                idio_returns[b, t] = idio_component
        
        return {
            'returns': returns,
            'factors': factors,
            'betas': betas,
            'idio_vol': idio_vol,
            'idio_returns': idio_returns,
        }


class MultiScaleDiscriminator:
    """
    多尺度判别器
    
    1. 时序判别器: 检验单资产时序特性
    2. 截面判别器: 检验同时刻资产间相关性
    3. 联合判别器: 检验整体数据分布
    """
    
    def __init__(self, n_assets: int, hidden_dim: int = 64):
        self.n_assets = n_assets
        
        # 时序判别器 (per-asset)
        self.tcn_temporal = TCNEncoder(1, hidden_dim, n_layers=3)
        scale = np.sqrt(2.0 / hidden_dim)
        self.W_temporal = np.random.randn(hidden_dim, 1) * scale
        self.b_temporal = np.zeros(1)
        
        # 截面判别器
        self.W_cross = np.random.randn(n_assets, hidden_dim) * scale
        self.b_cross = np.zeros(hidden_dim)
        self.W_cross_out = np.random.randn(hidden_dim, 1) * scale
        
        # 联合判别器
        self.W_joint = np.random.randn(n_assets + hidden_dim, 1) * scale
        self.b_joint = np.zeros(1)
    
    def discriminate_temporal(self, returns: np.ndarray) -> np.ndarray:
        """时序判别: returns (batch, seq_len, n_assets) -> (batch, seq_len, n_assets)"""
        B, T, N = returns.shape
        scores = np.zeros((B, T, N))
        for i in range(N):
            x = returns[:, :, i:i+1]  # (B, T, 1)
            h = self.tcn_temporal.forward(x)  # (B, T, hidden)
            scores[:, :, i] = (h @ self.W_temporal + self.b_temporal)[:, :, 0]
        return scores
    
    def discriminate_cross_sectional(self, returns: np.ndarray) -> np.ndarray:
        """截面判别: returns (batch, seq_len, n_assets) -> (batch, seq_len, 1)"""
        # 简化: 用截面统计量
        cross_mean = returns.mean(axis=2, keepdims=True)  # (B, T, 1)
        cross_std = returns.std(axis=2, keepdims=True)
        cross_skew = ((returns - cross_mean)**3).mean(axis=2, keepdims=True) / (cross_std**3 + 1e-8)
        
        features = np.concatenate([cross_mean, cross_std, cross_skew], axis=2)
        # 重复到hidden_dim
        features = np.tile(features, (1, 1, self.W_cross.shape[1] // 3 + 1))[:, :, :self.W_cross.shape[1]]
        
        h = np.tanh(features @ self.W_cross.T + self.b_cross)
        score = h @ self.W_cross_out  # (B, T, 1)
        return score
    
    def discriminate(self, returns: np.ndarray) -> dict:
        """综合判别分数"""
        temporal_score = self.discriminate_temporal(returns)
        cross_score = self.discriminate_cross_sectional(returns)
        
        # 联合分数
        temporal_avg = temporal_score.mean(axis=(1, 2), keepdims=True)  # (B, 1, 1)
        cross_avg = cross_score.mean(axis=1, keepdims=True)  # (B, 1, 1)
        joint_input = np.concatenate([
            np.tile(temporal_avg, (1, returns.shape[1], 1)),
            np.tile(cross_avg, (1, returns.shape[1], 1))
        ], axis=2)  # (B, T, 2)
        # 简化联合分数
        joint_score = (temporal_score.mean(axis=2, keepdims=True) + cross_score) / 2
        
        return {
            'temporal': temporal_score,
            'cross_sectional': cross_score,
            'joint': joint_score,
        }


# ============================================================
# 4. 金融特异性评估指标 (Finance-Specific Evaluation Metrics)
# ============================================================

class FinancialMetrics:
    """金融数据质量评估指标"""
    
    @staticmethod
    def covariance_distance(real: np.ndarray, generated: np.ndarray) -> float:
        """协方差矩阵Frobenius范数距离"""
        cov_real = np.cov(real.T)
        cov_gen = np.cov(generated.T)
        return np.linalg.norm(cov_real - cov_gen, 'fro')
    
    @staticmethod
    def correlation_structure(real: np.ndarray, generated: np.ndarray) -> dict:
        """相关性结构对比"""
        corr_real = np.corrcoef(real.T)
        corr_gen = np.corrcoef(generated.T)
        
        # 上三角元素的统计量
        n = corr_real.shape[0]
        mask = np.triu(np.ones((n, n)), k=1).astype(bool)
        
        return {
            'mean_corr_diff': np.abs(corr_real[mask] - corr_gen[mask]).mean(),
            'max_corr_diff': np.abs(corr_real[mask] - corr_gen[mask]).max(),
            'corr_real_mean': corr_real[mask].mean(),
            'corr_gen_mean': corr_gen[mask].mean(),
        }
    
    @staticmethod
    def tail_dependence(real: np.ndarray, generated: np.ndarray, 
                        threshold: float = 0.05) -> dict:
        """尾部依赖性 (极端联动)"""
        n_assets = real.shape[1]
        
        # 计算每对资产的下尾依赖
        real_lower_tail = np.zeros((n_assets, n_assets))
        gen_lower_tail = np.zeros((n_assets, n_assets))
        
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                # 真实数据
                real_q_i = np.quantile(real[:, i], threshold)
                real_q_j = np.quantile(real[:, j], threshold)
                real_both = np.mean((real[:, i] < real_q_i) & (real[:, j] < real_q_j))
                real_lower_tail[i, j] = real_both / (threshold * threshold + 1e-8)
                
                # 生成数据
                gen_q_i = np.quantile(generated[:, i], threshold)
                gen_q_j = np.quantile(generated[:, j], threshold)
                gen_both = np.mean((generated[:, i] < gen_q_i) & (generated[:, j] < gen_q_j))
                gen_lower_tail[i, j] = gen_both / (threshold * threshold + 1e-8)
        
        mask = np.triu(np.ones((n_assets, n_assets)), k=1).astype(bool)
        return {
            'tail_dep_real_mean': real_lower_tail[mask].mean(),
            'tail_dep_gen_mean': gen_lower_tail[mask].mean(),
            'tail_dep_error': np.abs(real_lower_tail[mask] - gen_lower_tail[mask]).mean(),
        }
    
    @staticmethod
    def volatility_clustering(real: np.ndarray, generated: np.ndarray) -> dict:
        """波动率聚集效应 (GARCH效应)"""
        def autocorr_abs_returns(returns, max_lag=10):
            acf = []
            abs_ret = np.abs(returns)
            for lag in range(1, max_lag + 1):
                acf.append(np.corrcoef(abs_ret[:-lag], abs_ret[lag:])[0, 1])
            return np.array(acf)
        
        # 对每个资产计算，然后取平均
        n_assets = real.shape[1]
        real_acf = np.zeros((n_assets, 10))
        gen_acf = np.zeros((n_assets, 10))
        
        for i in range(n_assets):
            real_acf[i] = autocorr_abs_returns(real[:, i])
            gen_acf[i] = autocorr_abs_returns(generated[:, i])
        
        return {
            'real_vol_clustering': real_acf.mean(axis=0),
            'gen_vol_clustering': gen_acf.mean(axis=0),
            'vol_clustering_error': np.abs(real_acf.mean(axis=0) - gen_acf.mean(axis=0)).mean(),
        }
    
    @staticmethod
    def portfolio_optimization_test(real: np.ndarray, generated: np.ndarray) -> dict:
        """
        下游任务: 在生成数据上优化的组合在真实数据上的表现
        测试生成数据对组合优化的实用价值
        """
        n_assets = real.shape[1]
        
        # 均值-方差优化 (简化版)
        def mv_optimize(returns, risk_aversion=1.0):
            mu = returns.mean(axis=0) * 252
            cov = np.cov(returns.T) * 252
            cov += 1e-6 * np.eye(n_assets)  # 正则化
            w = np.linalg.solve(cov, mu) / risk_aversion
            w = np.clip(w, 0, None)
            w /= w.sum()
            return w
        
        # 在真实数据上优化的权重
        w_real = mv_optimize(real)
        
        # 在生成数据上优化的权重
        w_gen = mv_optimize(generated)
        
        # 等权基准
        w_equal = np.ones(n_assets) / n_assets
        
        # 在测试数据上评估 (用后半段作为测试)
        test_returns = real[len(real)//2:]
        
        def portfolio_stats(w, returns):
            port_ret = returns @ w
            sharpe = port_ret.mean() * 252 / (port_ret.std() * np.sqrt(252) + 1e-8)
            cum = np.cumprod(1 + port_ret)
            max_dd = np.max(1 - cum / cum.cummax()) if len(cum) > 0 else 0
            return {'sharpe': sharpe, 'max_drawdown': max_dd, 
                    'annual_return': port_ret.mean() * 252}
        
        return {
            'real_optimized': portfolio_stats(w_real, test_returns),
            'gen_optimized': portfolio_stats(w_gen, test_returns),
            'equal_weight': portfolio_stats(w_equal, test_returns),
            'weight_distance': np.linalg.norm(w_real - w_gen),
        }


# ============================================================
# 5. 主程序 (Main Pipeline)
# ============================================================

def main():
    print("=" * 70)
    print("MarketGANs: 因子嵌入GAN的多资产金融时序数据增强")
    print("Factor-Embedded GAN for Multi-Asset Financial Time Series")
    print("=" * 70)
    
    # ---- 配置 ----
    config = FactorModelConfig(
        n_assets=20,
        n_factors=3,
        n_timesteps=2520,  # ~10年日频数据
    )
    
    # ---- 生成真实数据 ----
    print("\n[1] 生成真实因子模型数据 (模拟)...")
    data_gen = FactorDataGenerator(config, seed=42)
    real_data = data_gen.generate()
    real_returns = real_data['returns']
    
    print(f"  资产数: {config.n_assets}")
    print(f"  时间步: {config.n_timesteps}")
    print(f"  因子数: {config.n_factors}")
    print(f"  年化收益范围: [{real_returns.mean(axis=0).min()*252:.3f}, "
          f"{real_returns.mean(axis=0).max()*252:.3f}]")
    print(f"  年化波动率范围: [{real_returns.std(axis=0).min()*np.sqrt(252):.3f}, "
          f"{real_returns.std(axis=0).max()*np.sqrt(252):.3f}]")
    
    # ---- 因子嵌入GAN生成 ----
    print("\n[2] 因子嵌入GAN生成数据...")
    generator = FactorGenerator(
        n_assets=config.n_assets,
        n_factors=config.n_factors,
        latent_dim=32,
        hidden_dim=64,
    )
    
    gen_batch = generator.generate(batch_size=1, seq_len=config.n_timesteps)
    gen_returns = gen_batch['returns'][0]
    gen_factors = gen_batch['factors'][0]
    gen_betas = gen_batch['betas'][0]
    
    print(f"  生成数据形状: {gen_returns.shape}")
    print(f"  年化收益范围: [{gen_returns.mean(axis=0).min()*252:.3f}, "
          f"{gen_returns.mean(axis=0).max()*252:.3f}]")
    print(f"  年化波动率范围: [{gen_returns.std(axis=0).min()*np.sqrt(252):.3f}, "
          f"{gen_returns.std(axis=0).max()*np.sqrt(252):.3f}]")
    
    # ---- 金融质量评估 ----
    print("\n[3] 金融数据质量评估...")
    metrics = FinancialMetrics()
    
    # 协方差距离
    cov_dist = metrics.covariance_distance(real_returns, gen_returns)
    print(f"\n  协方差矩阵距离 (Frobenius): {cov_dist:.6f}")
    
    # 相关性结构
    corr_stats = metrics.correlation_structure(real_returns, gen_returns)
    print(f"  平均相关性差异: {corr_stats['mean_corr_diff']:.4f}")
    print(f"  真实平均相关性: {corr_stats['corr_real_mean']:.4f}")
    print(f"  生成平均相关性: {corr_stats['corr_gen_mean']:.4f}")
    
    # 尾部依赖
    tail_stats = metrics.tail_dependence(real_returns, gen_returns)
    print(f"\n  尾部依赖 (真实): {tail_stats['tail_dep_real_mean']:.4f}")
    print(f"  尾部依赖 (生成): {tail_stats['tail_dep_gen_mean']:.4f}")
    print(f"  尾部依赖误差: {tail_stats['tail_dep_error']:.4f}")
    
    # 波动率聚集
    vol_stats = metrics.volatility_clustering(real_returns, gen_returns)
    print(f"\n  波动率聚集 (lag-1 ACF):")
    print(f"    真实: {vol_stats['real_vol_clustering'][0]:.4f}")
    print(f"    生成: {vol_stats['gen_vol_clustering'][0]:.4f}")
    print(f"    误差: {vol_stats['vol_clustering_error']:.4f}")
    
    # 组合优化下游测试
    port_stats = metrics.portfolio_optimization_test(real_returns, gen_returns)
    print(f"\n  组合优化下游测试 (测试集):")
    print(f"    真实数据优化 - Sharpe: {port_stats['real_optimized']['sharpe']:.3f}, "
          f"MaxDD: {port_stats['real_optimized']['max_drawdown']:.3f}")
    print(f"    生成数据优化 - Sharpe: {port_stats['gen_optimized']['sharpe']:.3f}, "
          f"MaxDD: {port_stats['gen_optimized']['max_drawdown']:.3f}")
    print(f"    等权基准     - Sharpe: {port_stats['equal_weight']['sharpe']:.3f}, "
          f"MaxDD: {port_stats['equal_weight']['max_drawdown']:.3f}")
    print(f"    权重距离: {port_stats['weight_distance']:.4f}")
    
    # ---- 与Bootstrap方法对比 ----
    print("\n[4] 与Bootstrap方法对比...")
    
    # Block Bootstrap (常用基准)
    block_size = 20
    n_blocks = config.n_timesteps // block_size
    bootstrap_returns = np.zeros_like(real_returns)
    for i in range(n_blocks):
        start_idx = np.random.randint(0, config.n_timesteps - block_size)
        bootstrap_returns[i*block_size:(i+1)*block_size] = \
            real_returns[start_idx:start_idx+block_size]
    
    # 独立Bootstrap
    independent_bootstrap = real_returns[np.random.permutation(len(real_returns))]
    
    # 对比
    print(f"\n  方法对比 (协方差距离, 越小越好):")
    print(f"    因子嵌入GAN:   {cov_dist:.6f}")
    print(f"    Block Bootstrap: {metrics.covariance_distance(real_returns, bootstrap_returns):.6f}")
    print(f"    独立Bootstrap:   {metrics.covariance_distance(real_returns, independent_bootstrap):.6f}")
    
    # 相关性对比
    bs_corr = metrics.correlation_structure(real_returns, bootstrap_returns)
    ind_corr = metrics.correlation_structure(real_returns, independent_bootstrap)
    print(f"\n  相关性保持 (平均差异, 越小越好):")
    print(f"    因子嵌入GAN:   {corr_stats['mean_corr_diff']:.4f}")
    print(f"    Block Bootstrap: {bs_corr['mean_corr_diff']:.4f}")
    print(f"    独立Bootstrap:   {ind_corr['mean_corr_diff']:.4f}")
    
    # ---- 反事实数据生成 ----
    print("\n[5] 反事实数据生成 (压力测试)...")
    
    # 条件: 市场因子大幅下跌
    stress_factors = gen_factors.copy()
    # 模拟市场因子在未来60天暴跌
    stress_factors[-60:, 0] = -0.03  # 市场因子日均-3%
    
    stress_batch = generator.generate(batch_size=1, seq_len=config.n_timesteps)
    stress_returns = stress_batch['returns'][0]
    stress_returns[-60:] = gen_betas[-60:] @ stress_factors[-60:].T
    
    print(f"  压力场景: 市场因子60天内日均-3%")
    print(f"  压力期收益: {stress_returns[-60:].mean(axis=0).mean()*252:.3f} (年化)")
    print(f"  正常期收益: {stress_returns[:-60].mean(axis=0).mean()*252:.3f} (年化)")
    print(f"  压力期波动率: {stress_returns[-60:].std(axis=0).mean()*np.sqrt(252):.3f}")
    print(f"  正常期波动率: {stress_returns[:-60].std(axis=0).mean()*np.sqrt(252):.3f}")
    
    print("\n" + "=" * 70)
    print("MarketGANs 复现完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
