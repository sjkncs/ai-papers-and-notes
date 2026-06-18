"""
AlphaCFG: Alpha Factor Discovery via Grammar-Guided Learning and Search
复现论文: arXiv:2601.22119 (2026-01-15)

核心思想:
- 用上下文无关文法 (CFG) 定义合法 alpha 因子的语法结构
- 用蒙特卡洛树搜索 (MCTS) 在文法空间中搜索最优因子
- 语法感知的 value/policy 网络指导搜索方向

使用方法:
    python alphacfg.py --data sample_data.csv --n_iter 1000
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import math
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# 1. Alpha 因子文法定义 (Context-Free Grammar)
# ============================================================

class TokenType(Enum):
    """因子表达式的 token 类型"""
    FUNCTION = "function"
    OPERATOR = "operator"
    FEATURE = "feature"
    CONSTANT = "constant"
    LPAREN = "("
    RPAREN = ")"
    COMMA = ","


@dataclass
class GrammarRule:
    """文法规则: LHS -> RHS1 | RHS2 | ..."""
    lhs: str
    rhs_list: List[List[str]]


class AlphaGrammar:
    """
    Alpha 因子的上下文无关文法
    
    支持的因子表达式示例:
    - ts_mean(close, 20)
    - ts_rank(volume, 10) / ts_mean(volume, 10)
    - ts_corr(open, close, 5)
    - rank(ts_delta(high, 5))
    - ts_max(high, 20) - ts_min(low, 20)
    """
    
    # 时序函数
    TS_FUNCTIONS = [
        "ts_mean", "ts_std", "ts_rank", "ts_max", "ts_min",
        "ts_sum", "ts_delta", "ts_delay", "ts_corr", "ts_cov",
        "ts_argmax", "ts_argmin", "ts_skewness", "ts_kurtosis"
    ]
    
    # 截面函数
    CROSS_FUNCTIONS = ["rank", "zscore", "normalize", "demean"]
    
    # 数学运算
    OPERATORS = ["+", "-", "*", "/"]
    
    # 基础特征
    FEATURES = ["open", "high", "low", "close", "volume", "vwap",
                "returns", "log_returns", "turnover", "amplitude"]
    
    # 窗口参数
    WINDOWS = [5, 10, 20, 60, 120]
    
    def __init__(self):
        self.rules = self._build_grammar()
    
    def _build_grammar(self) -> Dict[str, List[List[str]]]:
        """构建文法规则"""
        rules = {
            "EXPR": [
                ["TS_FUNC_CALL"],
                ["CROSS_FUNC_CALL"],
                ["EXPR", "OP", "EXPR"],
                ["(", "EXPR", ")"],
            ],
            "TS_FUNC_CALL": [
                ["TS_FUNC1", "(", "FEATURE", ",", "WINDOW", ")"],
                ["TS_FUNC2", "(", "FEATURE", ",", "FEATURE", ",", "WINDOW", ")"],
            ],
            "CROSS_FUNC_CALL": [
                ["CROSS_FUNC", "(", "EXPR", ")"],
            ],
        }
        return rules
    
    def sample_factor(self, rng: np.random.Generator = None,
                      max_depth: int = 3) -> str:
        """随机采样一个合法的 alpha 因子表达式"""
        if rng is None:
            rng = np.random.default_rng()
        return self._expand("EXPR", rng, depth=0, max_depth=max_depth)
    
    def _expand(self, symbol: str, rng: np.random.Generator,
                depth: int, max_depth: int) -> str:
        """递归展开文法符号"""
        if depth >= max_depth:
            # 强制终止: 返回简单因子
            feature = rng.choice(self.FEATURES)
            window = rng.choice(self.WINDOWS)
            func = rng.choice(["ts_mean", "ts_rank", "ts_std"])
            return f"{func}({feature}, {window})"
        
        if symbol == "EXPR":
            if depth < max_depth - 1:
                choice = rng.choice(self.rules["EXPR"])
            else:
                # 浅层强制选简单表达式
                choice = rng.choice(self.rules["EXPR"][:2])
            return " ".join(
                self._expand(s, rng, depth + 1, max_depth) for s in choice
            )
        elif symbol == "TS_FUNC1":
            return str(rng.choice([f for f in self.TS_FUNCTIONS
                                   if f not in ["ts_corr", "ts_cov"]]))
        elif symbol == "TS_FUNC2":
            return str(rng.choice(["ts_corr", "ts_cov"]))
        elif symbol == "CROSS_FUNC":
            return str(rng.choice(self.CROSS_FUNCTIONS))
        elif symbol == "FEATURE":
            return str(rng.choice(self.FEATURES))
        elif symbol == "WINDOW":
            return str(int(rng.choice(self.WINDOWS)))
        elif symbol == "OP":
            return str(rng.choice(self.OPERATORS))
        elif symbol in ["(", ")", ","]:
            return symbol
        else:
            return symbol


# ============================================================
# 2. 因子计算引擎
# ============================================================

class FactorEngine:
    """
    根据因子表达式计算因子值
    输入: DataFrame (index=date, columns=stock features)
    """
    
    def __init__(self):
        self._cache = {}
    
    def compute(self, expr: str, data: pd.DataFrame) -> pd.Series:
        """计算因子表达式, 返回 factor 值序列"""
        try:
            result = self._eval_expr(expr, data)
            if isinstance(result, pd.DataFrame):
                result = result.mean(axis=1)
            return result
        except Exception as e:
            return pd.Series(np.nan, index=data.index)
    
    def _eval_expr(self, expr: str, data: pd.DataFrame):
        """解析并计算表达式 (简化版)"""
        expr = expr.strip()
        
        # 处理二元运算
        for op in ["+", "-", "*", "/"]:
            parts = self._split_by_op(expr, op)
            if len(parts) == 2:
                left = self._eval_expr(parts[0], data)
                right = self._eval_expr(parts[1], data)
                if op == "+": return left + right
                elif op == "-": return left - right
                elif op == "*": return left * right
                elif op == "/": return left / right.replace(0, np.nan)
        
        # 处理括号
        if expr.startswith("(") and expr.endswith(")"):
            return self._eval_expr(expr[1:-1], data)
        
        # 处理函数调用
        func_match = self._parse_func(expr)
        if func_match:
            func_name, args = func_match
            return self._eval_func(func_name, args, data)
        
        # 处理特征名
        if expr in data.columns:
            return data[expr]
        
        # 处理数字
        try:
            return float(expr)
        except ValueError:
            return pd.Series(np.nan, index=data.index)
    
    def _split_by_op(self, expr: str, op: str) -> List[str]:
        """在最外层分割表达式"""
        depth = 0
        for i, c in enumerate(expr):
            if c == "(": depth += 1
            elif c == ")": depth -= 1
            elif c == op and depth == 0 and i > 0:
                return [expr[:i].strip(), expr[i+1:].strip()]
        return [expr]
    
    def _parse_func(self, expr: str) -> Optional[Tuple[str, List[str]]]:
        """解析函数调用: func_name(arg1, arg2, ...)"""
        paren_idx = expr.find("(")
        if paren_idx < 0:
            return None
        func_name = expr[:paren_idx].strip()
        args_str = expr[paren_idx+1:-1]
        args = self._split_args(args_str)
        return func_name, args
    
    def _split_args(self, s: str) -> List[str]:
        """按逗号分割参数, 尊重括号嵌套"""
        args, depth, current = [], 0, ""
        for c in s:
            if c == "(": depth += 1
            elif c == ")": depth -= 1
            elif c == "," and depth == 0:
                args.append(current.strip())
                current = ""
                continue
            current += c
        if current.strip():
            args.append(current.strip())
        return args
    
    def _eval_func(self, name: str, args: List[str],
                   data: pd.DataFrame):
        """计算具体函数"""
        if name in ["ts_mean", "ts_std", "ts_rank", "ts_max", "ts_min",
                     "ts_sum", "ts_argmax", "ts_argmin"]:
            feature = args[0]
            window = int(args[1])
            col = self._eval_expr(feature, data)
            if isinstance(col, (int, float)):
                return col
            return self._ts_func(name, col, window)
        
        elif name in ["ts_delta", "ts_delay"]:
            feature = args[0]
            window = int(args[1])
            col = self._eval_expr(feature, data)
            if name == "ts_delta":
                return col.diff(window)
            return col.shift(window)
        
        elif name in ["ts_corr", "ts_cov"]:
            f1 = self._eval_expr(args[0], data)
            f2 = self._eval_expr(args[1], data)
            window = int(args[2])
            if name == "ts_corr":
                return f1.rolling(window).corr(f2)
            return f1.rolling(window).cov(f2)
        
        elif name == "rank":
            inner = self._eval_expr(args[0], data)
            return inner.rank(pct=True)
        
        elif name == "zscore":
            inner = self._eval_expr(args[0], data)
            return (inner - inner.mean()) / inner.std()
        
        elif name == "demean":
            inner = self._eval_expr(args[0], data)
            return inner - inner.mean()
        
        return pd.Series(np.nan, index=data.index)
    
    def _ts_func(self, name: str, col: pd.Series, window: int):
        """时序函数"""
        r = col.rolling(window, min_periods=1)
        if name == "ts_mean": return r.mean()
        elif name == "ts_std": return r.std()
        elif name == "ts_max": return r.max()
        elif name == "ts_min": return r.min()
        elif name == "ts_sum": return r.sum()
        elif name == "ts_argmax": return r.apply(np.argmax, raw=True)
        elif name == "ts_argmin": return r.apply(np.argmin, raw=True)
        elif name == "ts_rank":
            return r.apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1],
                          raw=False)
        return col


# ============================================================
# 3. MCTS 搜索 (Monte Carlo Tree Search)
# ============================================================

@dataclass
class MCTSNode:
    """MCTS 搜索树节点"""
    expr: str
    parent: Optional["MCTSNode"] = None
    children: List["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0
    
    @property
    def avg_reward(self) -> float:
        return self.total_reward / max(self.visits, 1)
    
    def ucb1(self, c: float = 1.414) -> float:
        """UCB1 探索-利用平衡"""
        if self.visits == 0:
            return float("inf")
        exploit = self.avg_reward
        explore = c * math.sqrt(math.log(self.parent.visits + 1) / self.visits)
        return exploit + explore


class AlphaMCTS:
    """
    基于 MCTS 的 Alpha 因子搜索器
    
    搜索流程:
    1. Selection: 用 UCB1 选择最有潜力的节点
    2. Expansion: 从文法中扩展新的因子变体
    3. Simulation: 计算因子 IC (信息系数) 评估质量
    4. Backpropagation: 将奖励传回根节点
    """
    
    def __init__(self, grammar: AlphaGrammar, engine: FactorEngine,
                 data: pd.DataFrame, returns: pd.Series):
        self.grammar = grammar
        self.engine = engine
        self.data = data
        self.returns = returns
        self.root = MCTSNode(expr="ROOT")
        self.root.visits = 1
        self.best_factors: List[Tuple[str, float]] = []
        self.rng = np.random.default_rng(42)
    
    def search(self, n_iterations: int = 500,
               top_k: int = 20) -> List[Tuple[str, float]]:
        """执行 MCTS 搜索, 返回 top-K 因子"""
        for i in range(n_iterations):
            # 1. Selection
            node = self._select(self.root)
            
            # 2. Expansion
            child = self._expand(node)
            
            # 3. Simulation
            reward = self._simulate(child.expr)
            
            # 4. Backpropagation
            self._backpropagate(child, reward)
            
            # 记录优秀因子
            if reward > 0.02:  # IC > 0.02 阈值
                self.best_factors.append((child.expr, reward))
            
            if (i + 1) % 100 == 0:
                top = sorted(self.best_factors, key=lambda x: -x[1])[:3]
                print(f"  [Iter {i+1}/{n_iterations}] "
                      f"Best IC: {top[0][1]:.4f} | {top[0][0][:60]}")
        
        # 去重排序
        seen = set()
        unique = []
        for expr, ic in sorted(self.best_factors, key=lambda x: -x[1]):
            if expr not in seen:
                seen.add(expr)
                unique.append((expr, ic))
        return unique[:top_k]
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """UCB1 选择"""
        while node.children:
            node = max(node.children, key=lambda c: c.ucb1())
        return node
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """文法扩展"""
        expr = self.grammar.sample_factor(self.rng, max_depth=3)
        child = MCTSNode(expr=expr, parent=node)
        node.children.append(child)
        return child
    
    def _simulate(self, expr: str) -> float:
        """
        模拟评估: 计算因子的 IC (Information Coefficient)
        IC = rank_corr(factor_values, forward_returns)
        """
        factor_vals = self.engine.compute(expr, self.data)
        if factor_vals.isna().all():
            return 0.0
        
        # Rank IC
        valid = factor_vals.dropna().index.intersection(self.returns.index)
        if len(valid) < 20:
            return 0.0
        
        ic = factor_vals.loc[valid].corr(
            self.returns.loc[valid], method="spearman"
        )
        return abs(ic) if not np.isnan(ic) else 0.0
    
    def _backpropagate(self, node: MCTSNode, reward: float):
        """反向传播奖励"""
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent


# ============================================================
# 4. 因子组合与评估
# ============================================================

class FactorEvaluator:
    """因子质量评估器"""
    
    @staticmethod
    def calc_ic_series(factor: pd.Series, returns: pd.Series,
                       method: str = "spearman") -> pd.Series:
        """计算滚动 IC 序列"""
        ic_list = []
        for date in factor.index:
            if date in returns.index:
                f = factor.loc[:date].tail(60)
                r = returns.loc[:date].tail(60)
                valid = f.dropna().index.intersection(r.dropna().index)
                if len(valid) > 10:
                    ic = f.loc[valid].corr(r.loc[valid], method=method)
                    ic_list.append({"date": date, "ic": ic})
        return pd.DataFrame(ic_list).set_index("date")["ic"]
    
    @staticmethod
    def factor_report(factor_name: str, factor_vals: pd.Series,
                      returns: pd.Series) -> Dict:
        """生成因子评估报告"""
        valid = factor_vals.dropna().index.intersection(returns.dropna().index)
        f, r = factor_vals.loc[valid], returns.loc[valid]
        
        ic = f.corr(r, method="spearman")
        ic_series = pd.Series([f.corr(r, method="spearman")
                               for f, r in zip(
                                   [f.iloc[max(0,i-60):i+1] for i in range(60, len(f))],
                                   [r.iloc[max(0,i-60):i+1] for i in range(60, len(r))]
                               )]).dropna()
        
        report = {
            "factor_name": factor_name,
            "IC_mean": ic,
            "IC_std": ic_series.std() if len(ic_series) > 0 else np.nan,
            "ICIR": ic / ic_series.std() if len(ic_series) > 0 and ic_series.std() > 0 else 0,
            "IC_positive_ratio": (ic_series > 0).mean() if len(ic_series) > 0 else 0,
            "coverage": factor_vals.notna().mean(),
            "turnover": factor_vals.diff().abs().mean() / factor_vals.std(),
        }
        return report


# ============================================================
# 5. 主程序: 端到端 Alpha 挖掘流程
# ============================================================

def generate_sample_data(n_days: int = 500, n_stocks: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
    """生成模拟股票数据用于演示"""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    
    # 模拟 OHLCV 数据
    close = 100 + np.cumsum(np.random.randn(n_days) * 0.5)
    high = close + np.abs(np.random.randn(n_days) * 0.3)
    low = close - np.abs(np.random.randn(n_days) * 0.3)
    open_ = close + np.random.randn(n_days) * 0.2
    volume = np.random.randint(1e6, 1e8, n_days).astype(float)
    
    data = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume, "vwap": (high + low + close) / 3,
        "returns": np.diff(close, prepend=close[0]) / close,
        "log_returns": np.diff(np.log(close), prepend=np.log(close[0])),
        "turnover": volume / 1e7,
        "amplitude": (high - low) / close,
    }, index=dates)
    
    # 前向收益 (5日)
    fwd_returns = data["close"].shift(-5) / data["close"] - 1
    
    return data, fwd_returns


def main():
    print("=" * 60)
    print("AlphaCFG: Grammar-Guided Alpha Factor Discovery")
    print("Paper: arXiv:2601.22119 (2026-01-15)")
    print("=" * 60)
    
    # 1. 生成数据
    print("\n[1/4] 准备数据...")
    data, returns = generate_sample_data()
    print(f"  数据维度: {data.shape[0]} days x {data.shape[1]} features")
    
    # 2. 初始化
    print("\n[2/4] 初始化 Alpha 文法和计算引擎...")
    grammar = AlphaGrammar()
    engine = FactorEngine()
    
    # 演示随机因子
    print("  随机因子示例:")
    for _ in range(5):
        f = grammar.sample_factor()
        print(f"    {f}")
    
    # 3. MCTS 搜索
    print("\n[3/4] MCTS 搜索 Alpha 因子...")
    mcts = AlphaMCTS(grammar, engine, data, returns)
    top_factors = mcts.search(n_iterations=500, top_k=20)
    
    # 4. 结果报告
    print("\n[4/4] Top-10 Alpha 因子:")
    print("-" * 60)
    evaluator = FactorEvaluator()
    for i, (expr, ic) in enumerate(top_factors[:10], 1):
        report = evaluator.factor_report(
            f"Factor_{i}",
            engine.compute(expr, data),
            returns
        )
        print(f"  #{i:2d} | IC={ic:.4f} | ICIR={report['ICIR']:.2f} | "
              f"{expr[:50]}")
    
    print("\n" + "=" * 60)
    print("搜索完成! 可进一步结合多因子模型做组合优化。")
    print("=" * 60)


if __name__ == "__main__":
    main()
