"""
量化数据加载工具
支持: yfinance, akshare, 本地CSV
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
import warnings

warnings.filterwarnings("ignore")


class QuantDataLoader:
    """统一数据加载接口"""
    
    @staticmethod
    def load_yfinance(tickers: List[str], start: str = "2020-01-01",
                      end: str = "2026-01-21",
                      interval: str = "1d") -> pd.DataFrame:
        """
        从 Yahoo Finance 加载数据
        需要: pip install yfinance
        """
        try:
            import yfinance as yf
            data = yf.download(tickers, start=start, end=end,
                              interval=interval, auto_adjust=True)
            return data
        except ImportError:
            print("[WARN] yfinance 未安装, 使用模拟数据")
            return QuantDataLoader._generate_mock(tickers, start, end)
    
    @staticmethod
    def load_akshare(symbol: str = "000300",
                     start: str = "20200101",
                     end: str = "20260121") -> pd.DataFrame:
        """
        从 AKShare 加载 A 股数据
        需要: pip install akshare
        """
        try:
            import akshare as ak
            df = ak.stock_zh_index_daily(symbol=f"sh{symbol}")
            df = df[(df.index >= start) & (df.index <= end)]
            return df
        except ImportError:
            print("[WARN] akshare 未安装, 使用模拟数据")
            return QuantDataLoader._generate_mock(
                [symbol], start[:4]+"-"+start[4:6]+"-"+start[6:], 
                end[:4]+"-"+end[4:6]+"-"+end[6:]
            )
    
    @staticmethod
    def load_csv(filepath: str, date_col: str = "date",
                 parse_dates: bool = True) -> pd.DataFrame:
        """加载本地 CSV"""
        df = pd.read_csv(filepath)
        if parse_dates and date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
        return df
    
    @staticmethod
    def _generate_mock(tickers: List[str], start: str,
                       end: str) -> pd.DataFrame:
        """生成模拟市场数据"""
        np.random.seed(42)
        dates = pd.date_range(start, end, freq="B")
        n = len(dates)
        
        data = {}
        for ticker in tickers:
            price = 100 * (1 + np.random.randn() * 0.3)
            prices = [price]
            for _ in range(1, n):
                r = np.random.randn() * 0.015
                prices.append(prices[-1] * (1 + r))
            data[ticker] = prices
        
        return pd.DataFrame(data, index=dates)


class FeatureBuilder:
    """量化特征构建器"""
    
    @staticmethod
    def add_technical_indicators(df: pd.DataFrame,
                                  price_col: str = "Close") -> pd.DataFrame:
        """添加技术指标"""
        result = df.copy()
        
        if price_col not in df.columns:
            if isinstance(df.columns, pd.MultiIndex):
                price_col = ("Close", df.columns.get_level_values(0)[0])
            else:
                return df
        
        close = df[price_col]
        
        # 移动平均
        for w in [5, 10, 20, 60]:
            result[f"SMA_{w}"] = close.rolling(w).mean()
            result[f"EMA_{w}"] = close.ewm(span=w).mean()
        
        # 波动率
        ret = close.pct_change()
        result["volatility_20d"] = ret.rolling(20).std() * np.sqrt(252)
        result["volatility_60d"] = ret.rolling(60).std() * np.sqrt(252)
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        result["RSI_14"] = 100 - 100 / (1 + rs)
        
        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        result["MACD"] = ema12 - ema26
        result["MACD_signal"] = result["MACD"].ewm(span=9).mean()
        result["MACD_hist"] = result["MACD"] - result["MACD_signal"]
        
        # Bollinger Bands
        result["BB_upper"] = result["SMA_20"] + 2 * result["SMA_20"].rolling(20).std()
        result["BB_lower"] = result["SMA_20"] - 2 * result["SMA_20"].rolling(20).std()
        result["BB_width"] = (result["BB_upper"] - result["BB_lower"]) / result["SMA_20"]
        
        # 动量
        result["momentum_5d"] = close / close.shift(5) - 1
        result["momentum_20d"] = close / close.shift(20) - 1
        
        return result
    
    @staticmethod
    def add_cross_sectional_features(
        returns_df: pd.DataFrame,
        windows: List[int] = [5, 10, 20, 60]
    ) -> pd.DataFrame:
        """添加截面特征 (多资产场景)"""
        result = returns_df.copy()
        
        for w in windows:
            rolling_mean = returns_df.rolling(w).mean()
            rolling_std = returns_df.rolling(w).std()
            
            # 截面排名
            result[f"rank_{w}d"] = rolling_mean.rank(axis=1, pct=True)
            # 截面 z-score
            result[f"zscore_{w}d"] = (
                (rolling_mean - rolling_mean.mean(axis=1).values.reshape(-1, 1)) /
                rolling_std.mean(axis=1).values.reshape(-1, 1)
            )
        
        return result
