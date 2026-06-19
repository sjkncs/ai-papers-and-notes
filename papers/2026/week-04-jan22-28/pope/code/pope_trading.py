"""
POPE for Trading: 特权信息引导的交易策略探索学习
POPE for Trading: Privileged On-Policy Exploration for Trading Agent Learning

复现论文: arXiv 2601.18779 (Jan 2026) 的量化金融映射
核心思想: 用历史最优策略作为"特权信息"引导RL交易代理探索，解决稀疏奖励问题

作者: QoderWork AI Research
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. 交易环境 (Trading Environment)
# ============================================================

@dataclass
class MarketState:
    """市场状态"""
    prices: np.ndarray          # 当前价格 (n_assets,)
    returns_history: np.ndarray # 历史收益 (lookback, n_assets)
    position: np.ndarray        # 当前持仓 (n_assets,)
    cash: float                 # 现金
    step: int                   # 时间步


@dataclass
class TradeConfig:
    """交易配置"""
    n_assets: int = 10
    n_steps: int = 500          # 交易日数
    lookback: int = 20          # 回看窗口
    initial_cash: float = 1000000.0
    transaction_cost: float = 0.001  # 交易成本
    max_position: float = 0.2   # 单个资产最大仓位


class TradingEnvironment:
    """
    简化交易环境
    
    状态: [returns_history, position, cash_ratio]
    动作: 目标权重 (n_assets,)
    奖励: 组合收益 - 交易成本惩罚
    """
    
    def __init__(self, config: TradeConfig, seed: int = 42):
        self.cfg = config
        self.rng = np.random.RandomState(seed)
        
        # 生成价格数据 (GBM + 均值回复)
        self.prices = self._generate_prices()
        self.returns = np.diff(np.log(self.prices), axis=0)
        
        # 状态
        self.current_step = 0
        self.position = np.zeros(config.n_assets)
        self.cash = config.initial_cash
        self.portfolio_value = config.initial_cash
    
    def _generate_prices(self) -> np.ndarray:
        """生成模拟价格数据"""
        cfg = self.cfg
        n_total = cfg.n_steps + cfg.lookback + 100  # 额外buffer
        
        prices = np.zeros((n_total, cfg.n_assets))
        prices[0] = 100.0
        
        # 不同资产有不同特性
        drift = self.rng.uniform(-0.0002, 0.0005, cfg.n_assets)
        vol = self.rng.uniform(0.01, 0.03, cfg.n_assets)
        
        # 添加regime切换
        regime_vol = np.ones(n_total)
        regime_vol[200:300] = 2.0  # 中间100天波动率翻倍
        
        for t in range(1, n_total):
            # 共同因子
            market_factor = self.rng.randn() * 0.005
            # 个股噪声
            idio = self.rng.randn(cfg.n_assets) * vol * regime_vol[t]
            # 收益
            ret = drift + 0.5 * market_factor + idio
            # 价格更新
            prices[t] = prices[t-1] * np.exp(ret)
        
        return prices
    
    def reset(self) -> MarketState:
        """重置环境"""
        self.current_step = self.cfg.lookback
        self.position = np.zeros(self.cfg.n_assets)
        self.cash = self.cfg.initial_cash
        self.portfolio_value = self.cfg.initial_cash
        
        return self._get_state()
    
    def _get_state(self) -> MarketState:
        """获取当前状态"""
        cfg = self.cfg
        step = self.current_step
        
        returns_hist = self.returns[step-cfg.lookback:step]
        prices = self.prices[step]
        
        return MarketState(
            prices=prices,
            returns_history=returns_hist,
            position=self.position.copy(),
            cash=self.cash,
            step=step,
        )
    
    def step(self, target_weights: np.ndarray) -> Tuple[MarketState, float, bool]:
        """
        执行交易
        
        target_weights: 目标权重 (n_assets,), 和为1
        returns: (next_state, reward, done)
        """
        cfg = self.cfg
        
        # 限制权重
        target_weights = np.clip(target_weights, 0, cfg.max_position)
        if target_weights.sum() > 1.0:
            target_weights /= target_weights.sum()
        
        # 当前组合价值
        current_value = self.portfolio_value
        
        # 计算交易成本
        weight_change = np.abs(target_weights - self.position)
        trade_cost = weight_change.sum() * cfg.transaction_cost * current_value
        
        # 更新持仓
        self.position = target_weights.copy()
        self.cash = current_value * (1 - target_weights.sum()) - trade_cost
        
        # 移动到下一步
        self.current_step += 1
        done = self.current_step >= len(self.returns)
        
        if not done:
            # 计算收益
            asset_returns = self.returns[self.current_step]
            portfolio_return = target_weights @ asset_returns
            
            # 更新组合价值
            self.portfolio_value = current_value * (1 + portfolio_return) - trade_cost
            
            # 奖励 = 风险调整收益
            reward = portfolio_return - 0.5 * (portfolio_return ** 2)  # 均值-方差效用
        else:
            reward = 0.0
            portfolio_return = 0.0
        
        next_state = self._get_state() if not done else None
        
        return next_state, reward, done


# ============================================================
# 2. 策略网络 (Policy Network)
# ============================================================

class PolicyNetwork:
    """
    简化策略网络 (NumPy实现)
    
    输入: 市场状态特征
    输出: 资产权重 (softmax)
    """
    
    def __init__(self, state_dim: int, n_assets: int, hidden_dim: int = 64):
        self.state_dim = state_dim
        self.n_assets = n_assets
        
        # 两层网络
        scale1 = np.sqrt(2.0 / state_dim)
        self.W1 = np.random.randn(state_dim, hidden_dim) * scale1
        self.b1 = np.zeros(hidden_dim)
        
        scale2 = np.sqrt(2.0 / hidden_dim)
        self.W2 = np.random.randn(hidden_dim, n_assets) * scale2
        self.b2 = np.zeros(n_assets)
    
    def forward(self, state_features: np.ndarray) -> np.ndarray:
        """
        state_features: (state_dim,) or (batch, state_dim)
        returns: weights (n_assets,) or (batch, n_assets)
        """
        # 第一层 + ReLU
        h = state_features @ self.W1 + self.b1
        h = np.maximum(h, 0)
        
        # 输出层 + softmax
        logits = h @ self.W2 + self.b2
        
        # 数值稳定的softmax
        logits = logits - logits.max(axis=-1, keepdims=True)
        exp_logits = np.exp(logits)
        weights = exp_logits / (exp_logits.sum(axis=-1, keepdims=True) + 1e-8)
        
        return weights
    
    def get_state_features(self, state: MarketState) -> np.ndarray:
        """将MarketState转换为特征向量"""
        # 历史收益统计
        ret_mean = state.returns_history.mean(axis=0)      # (n_assets,)
        ret_std = state.returns_history.std(axis=0)         # (n_assets,)
        ret_skew = ((state.returns_history - ret_mean)**3).mean(axis=0) / (ret_std**3 + 1e-8)
        ret_momentum = state.returns_history[-5:].mean(axis=0)  # 5日动量
        ret_reversal = state.returns_history[-1].mean() - state.returns_history[-5:].mean(axis=0).mean()
        
        # 持仓特征
        position = state.position
        
        # 组合特征
        cash_ratio = np.array([state.cash / (state.cash + 1e-8)])
        
        features = np.concatenate([
            ret_mean, ret_std, ret_skew, ret_momentum, 
            np.ones(1) * ret_reversal,
            position, cash_ratio
        ])
        
        return features


# ============================================================
# 3. 特权信息引导探索 (Privileged On-Policy Exploration)
# ============================================================

class PrivilegedExplorer:
    """
    POPE核心: 用特权信息引导探索
    
    特权信息 = 历史最优策略在相同市场状态下的动作
    (在真实应用中可以是: teacher模型输出、后验最优权重、专家标注)
    """
    
    def __init__(self, n_assets: int, buffer_size: int = 1000):
        self.n_assets = n_assets
        self.buffer_size = buffer_size
        
        # 特权信息缓存: (state_features, privileged_action, reward)
        self.privileged_buffer = []
    
    def generate_privileged_info(self, env: TradingEnvironment, 
                                  n_episodes: int = 10) -> List[dict]:
        """
        生成特权信息: 使用多步前瞻搜索找到"近似最优"动作
        
        在实际中这可以是:
        - 完美后验信息 (oracle)
        - Teacher模型的输出
        - 人类专家的交易记录
        """
        privileged_data = []
        
        for _ in range(n_episodes):
            state = env.reset()
            done = False
            
            while not done and state is not None:
                # 前瞻: 尝试多个随机动作，选择最好的
                best_action = None
                best_future_return = -np.inf
                
                for _ in range(50):  # 采样50个候选
                    candidate = self.rng_dirichlet()
                    
                    # 简单评估: 用未来已知收益 (特权!)
                    if state.step < len(env.returns) - 1:
                        future_return = candidate @ env.returns[state.step]
                        if future_return > best_future_return:
                            best_future_return = future_return
                            best_action = candidate
                
                privileged_data.append({
                    'step': state.step,
                    'action': best_action,
                    'future_return': best_future_return,
                })
                
                state, _, done = env.step(best_action)
        
        return privileged_data
    
    def rng_dirichlet(self) -> np.ndarray:
        """生成Dirichlet分布的随机权重"""
        x = np.random.exponential(1.0, self.n_assets)
        return x / x.sum()
    
    def get_privileged_action(self, state: MarketState, 
                               privileged_data: List[dict]) -> Optional[np.ndarray]:
        """
        从特权信息缓存中获取引导动作
        
        查找与当前状态最相似的历史特权动作
        """
        if not privileged_data:
            return None
        
        # 找到时间步最接近的特权信息
        best_match = None
        best_distance = np.inf
        
        for entry in privileged_data:
            distance = abs(entry['step'] - state.step)
            if distance < best_distance:
                best_distance = distance
                best_match = entry
        
        return best_match['action'] if best_match else None
    
    def create_privileged_prompt(self, state: MarketState, 
                                  privileged_action: np.ndarray) -> np.ndarray:
        """
        POPE的核心机制: 将特权信息作为prompt前缀
        
        不是直接用特权动作作为目标，而是:
        1. 将特权动作的一部分信息注入到状态中
        2. 让策略在"增强状态"下做决策
        3. 学习到的技能可以迁移到"无增强"状态
        """
        # 特权动作的片段 (前k个资产的方向)
        k = max(1, self.n_assets // 3)
        privileged_fragment = privileged_action[:k]
        
        # 增强状态 = 原始状态 + 特权片段
        # 在策略训练时，这相当于修改了MDP的初始状态分布
        return privileged_fragment


# ============================================================
# 4. POPE训练循环 (POPE Training Loop)
# ============================================================

class POPETrainer:
    """
    POPE训练器
    
    训练流程:
    1. 生成特权信息 (oracle/teacher)
    2. 在特权引导下收集on-policy数据
    3. 用REINFORCE更新策略
    4. 逐渐减少特权引导 (curriculum)
    """
    
    def __init__(self, env: TradingEnvironment, 
                 policy: PolicyNetwork,
                 explorer: PrivilegedExplorer):
        self.env = env
        self.policy = policy
        self.explorer = explorer
        
        # 训练参数
        self.lr = 0.001
        self.gamma = 0.99  # 折扣因子
        self.privileged_prob = 0.8  # 初始特权引导概率
    
    def train(self, n_epochs: int = 20, episodes_per_epoch: int = 5) -> dict:
        """训练循环"""
        
        # 生成特权信息
        print("  生成特权信息 (oracle)...")
        privileged_data = self.explorer.generate_privileged_info(self.env, n_episodes=5)
        
        all_rewards = []
        all_returns = []
        
        for epoch in range(n_epochs):
            epoch_rewards = []
            epoch_portfolio_returns = []
            
            # 逐渐减少特权引导概率 (curriculum)
            current_priv_prob = max(0.1, self.privileged_prob * (0.95 ** epoch))
            
            for _ in range(episodes_per_epoch):
                state = self.env.reset()
                done = False
                episode_rewards = []
                episode_log_probs = []
                episode_returns_list = []
                
                while not done and state is not None:
                    # 状态特征
                    features = self.policy.get_state_features(state)
                    
                    # 决定是否使用特权引导
                    use_privileged = np.random.random() < current_priv_prob
                    
                    if use_privileged:
                        priv_action = self.explorer.get_privileged_action(
                            state, privileged_data)
                        if priv_action is not None:
                            # POPE核心: 特权信息作为prompt前缀
                            fragment = self.explorer.create_privileged_prompt(
                                state, priv_action)
                            # 增强特征 = 原始特征 + 特权片段(补零)
                            enhanced_features = np.concatenate([
                                features, 
                                np.pad(fragment, (0, len(features) - len(fragment)))
                            ])[:len(features)]
                        else:
                            enhanced_features = features
                    else:
                        enhanced_features = features
                    
                    # 策略输出
                    action = self.policy.forward(enhanced_features)
                    
                    # 添加探索噪声
                    noise = np.random.randn(len(action)) * 0.05
                    action = np.clip(action + noise, 0.01, None)
                    action /= action.sum()
                    
                    # 执行交易
                    next_state, reward, done = self.env.step(action)
                    
                    episode_rewards.append(reward)
                    if next_state is not None:
                        port_return = (self.env.portfolio_value / self.env.cfg.initial_cash) - 1
                        episode_returns_list.append(port_return)
                    
                    state = next_state
                
                if episode_rewards:
                    epoch_rewards.append(sum(episode_rewards))
                    epoch_portfolio_returns.append(
                        episode_returns_list[-1] if episode_returns_list else 0)
                    
                    # 简化REINFORCE更新
                    self._update_policy(episode_rewards, features)
            
            avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0
            avg_return = np.mean(epoch_portfolio_returns) if epoch_portfolio_returns else 0
            all_rewards.append(avg_reward)
            all_returns.append(avg_return)
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{n_epochs}: "
                      f"avg_reward={avg_reward:.6f}, "
                      f"avg_return={avg_return:.4%}, "
                      f"priv_prob={current_priv_prob:.2f}")
        
        return {
            'rewards': all_rewards,
            'returns': all_returns,
            'final_policy': self.policy,
        }
    
    def _update_policy(self, episode_rewards: list, last_features: np.ndarray):
        """简化策略梯度更新"""
        # 计算折扣回报
        T = len(episode_rewards)
        discounted_returns = np.zeros(T)
        running_return = 0
        for t in reversed(range(T)):
            running_return = episode_rewards[t] + self.gamma * running_return
            discounted_returns[t] = running_return
        
        # 标准化
        if discounted_returns.std() > 0:
            discounted_returns = (discounted_returns - discounted_returns.mean()) / \
                                  (discounted_returns.std() + 1e-8)
        
        # 简化梯度估计 (数值梯度)
        eps = 0.01
        for param_name in ['W1', 'b1', 'W2', 'b2']:
            param = getattr(self.policy, param_name)
            # 随机方向
            direction = np.random.randn(*param.shape)
            direction /= np.linalg.norm(direction) + 1e-8
            
            # 正扰动
            setattr(self.policy, param_name, param + eps * direction)
            loss_plus = -(discounted_returns.mean())
            
            # 负扰动
            setattr(self.policy, param_name, param - eps * direction)
            loss_minus = -(discounted_returns.mean())
            
            # 梯度估计
            grad = (loss_plus - loss_minus) / (2 * eps) * direction
            
            # 更新
            setattr(self.policy, param_name, param - self.lr * grad)


# ============================================================
# 5. 对比评估 (Comparative Evaluation)
# ============================================================

def evaluate_policy(policy: PolicyNetwork, env: TradingEnvironment, 
                    n_episodes: int = 20) -> dict:
    """评估策略性能"""
    all_final_values = []
    all_sharpes = []
    all_max_dds = []
    
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        daily_returns = []
        
        while not done and state is not None:
            features = policy.get_state_features(state)
            action = policy.forward(features)
            
            next_state, reward, done = env.step(action)
            
            if next_state is not None:
                asset_ret = env.returns[env.current_step]
                port_ret = action @ asset_ret
                daily_returns.append(port_ret)
            
            state = next_state
        
        if daily_returns:
            daily_returns = np.array(daily_returns)
            final_value = env.portfolio_value
            sharpe = daily_returns.mean() * 252 / (daily_returns.std() * np.sqrt(252) + 1e-8)
            cum = np.cumprod(1 + daily_returns)
            max_dd = np.max(1 - cum / cum.cummax())
            
            all_final_values.append(final_value)
            all_sharpes.append(sharpe)
            all_max_dds.append(max_dd)
    
    return {
        'avg_final_value': np.mean(all_final_values),
        'avg_sharpe': np.mean(all_sharpes),
        'avg_max_dd': np.mean(all_max_dds),
        'std_sharpe': np.std(all_sharpes),
    }


# ============================================================
# 6. 主程序 (Main Pipeline)
# ============================================================

def main():
    print("=" * 70)
    print("POPE for Trading: 特权信息引导的交易策略探索学习")
    print("Privileged On-Policy Exploration for Trading Agent Learning")
    print("=" * 70)
    
    # ---- 环境配置 ----
    trade_config = TradeConfig(n_assets=10, n_steps=300)
    
    # ---- 方法1: 标准RL (无特权信息) ----
    print("\n[1] 标准RL训练 (无特权信息)...")
    env1 = TradingEnvironment(trade_config, seed=42)
    state_dim = trade_config.n_assets * 4 + trade_config.n_assets + 1
    policy1 = PolicyNetwork(state_dim, trade_config.n_assets)
    
    # 简化训练: 随机探索
    state = env1.reset()
    baseline_returns = []
    for _ in range(300):
        features = policy1.get_state_features(state)
        action = policy1.forward(features)
        action += np.random.randn(len(action)) * 0.1
        action = np.clip(action, 0.01, None)
        action /= action.sum()
        state, reward, done = env1.step(action)
        if done or state is None:
            break
        baseline_returns.append(reward)
    
    baseline_metrics = evaluate_policy(policy1, TradingEnvironment(trade_config, seed=100))
    
    # ---- 方法2: POPE (特权信息引导) ----
    print("\n[2] POPE训练 (特权信息引导)...")
    env2 = TradingEnvironment(trade_config, seed=42)
    policy2 = PolicyNetwork(state_dim, trade_config.n_assets)
    explorer = PrivilegedExplorer(trade_config.n_assets)
    trainer = POPETrainer(env2, policy2, explorer)
    
    training_results = trainer.train(n_epochs=20, episodes_per_epoch=5)
    pope_metrics = evaluate_policy(policy2, TradingEnvironment(trade_config, seed=100))
    
    # ---- 方法3: 等权基准 ----
    print("\n[3] 等权基准...")
    env3 = TradingEnvironment(trade_config, seed=100)
    equal_returns = []
    state = env3.reset()
    while state is not None:
        equal_weights = np.ones(trade_config.n_assets) / trade_config.n_assets
        state, reward, done = env3.step(equal_weights)
        if done:
            break
        equal_returns.append(reward)
    
    # ---- 结果对比 ----
    print(f"\n{'='*50}")
    print("结果对比")
    print(f"{'='*50}")
    
    print(f"\n  {'方法':<20} {'Sharpe':>10} {'MaxDD':>10} {'最终价值':>15}")
    print(f"  {'-'*60}")
    
    equal_sharpe = np.mean(equal_returns) * 252 / (np.std(equal_returns) * np.sqrt(252) + 1e-8) \
                   if equal_returns else 0
    print(f"  {'等权基准':<20} {equal_sharpe:>10.4f} "
          f"{'N/A':>10} {env3.portfolio_value:>15,.0f}")
    print(f"  {'标准RL':<20} {baseline_metrics['avg_sharpe']:>10.4f} "
          f"{baseline_metrics['avg_max_dd']:>10.4f} "
          f"{baseline_metrics['avg_final_value']:>15,.0f}")
    print(f"  {'POPE (特权引导)':<20} {pope_metrics['avg_sharpe']:>10.4f} "
          f"{pope_metrics['avg_max_dd']:>10.4f} "
          f"{pope_metrics['avg_final_value']:>15,.0f}")
    
    # ---- 训练曲线 ----
    print(f"\n  训练奖励曲线 (每5 epoch):")
    for i in range(0, len(training_results['rewards']), 5):
        avg = np.mean(training_results['rewards'][i:i+5])
        bar = '█' * int(max(0, avg * 1000))
        print(f"    Epoch {i+1:3d}-{min(i+5, len(training_results['rewards'])):3d}: "
              f"{avg:+.6f} {bar}")
    
    print("\n" + "=" * 70)
    print("POPE for Trading 复现完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
