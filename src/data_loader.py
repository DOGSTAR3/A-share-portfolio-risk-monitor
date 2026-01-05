#Tushare 数据加载、复权处理、收益率计算、3σ 异常值裁剪
# 作者：DogStar·Quant
# 时间：20260101
import logging
import pandas as pd
import numpy as np
import tushare as ts
from typing import Dict


class StockDataLoader:
    """股票数据加载器：复权股价、收益率计算、异常值处理"""
    def __init__(self, ts_token: str):
        ts.set_token(ts_token)
        self.pro = ts.pro_api()
        self.stock_data: Dict[str, pd.DataFrame] = {}

    def get_adj_stock_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        market_suffix = '.SH' if stock_code.startswith('6') else '.SZ'
        try:
            # 获取日线数据
            df = self.pro.daily(
                ts_code=f"{stock_code}{market_suffix}",
                start_date=start_date,
                end_date=end_date
            )
            if df.empty:
                logging.error(f"{stock_code}无日线数据返回")
                return pd.DataFrame()

            # 获取复权因子（免费版降级为原始价）
            try:
                adj_df = self.pro.adj_factor(
                    ts_code=f"{stock_code}{market_suffix}",
                    start_date=start_date,
                    end_date=end_date
                )
                adj_df['trade_date'] = pd.to_datetime(adj_df['trade_date'])
                adj_df = adj_df.sort_values('trade_date').set_index('trade_date')
            except Exception as e:
                logging.warning(f"{stock_code}复权因子获取失败：{e}")
                adj_df = pd.DataFrame()

            # 数据预处理
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date').set_index('trade_date')
            if not adj_df.empty:
                df = df.join(adj_df[['adj_factor']])
                latest_adj = df['adj_factor'].iloc[-1]
                df['close_adj'] = df['close'] * (df['adj_factor'] / latest_adj)
            else:
                df['close_adj'] = df['close']

            # 计算对数收益率（百分比）
            df['returns'] = np.log(df['close_adj'] / df['close_adj'].shift(1)) * 100
            df = df.dropna(subset=['returns'])

            # 3σ异常值处理
            mean_ret = df['returns'].mean()
            std_ret = df['returns'].std()
            df['returns'] = df['returns'].clip(
                lower=mean_ret - 3 * std_ret,
                upper=mean_ret + 3 * std_ret
            )

            self.stock_data[stock_code] = df
            logging.info(
                f"{stock_code}数据加载完成，有效收益率数据{len(df)}条"
            )
            return df
        except Exception as e:
            logging.error(f"{stock_code}数据加载失败：{e}")
            return pd.DataFrame()

    def get_portfolio_returns_matrix(self, codes: list) -> pd.DataFrame:
        """构建组合收益率矩阵（日期×个股）"""
        returns_list = []
        for code in codes:
            if code not in self.stock_data:
                continue  # 假设已通过get_adj_stock_data加载
            df = self.stock_data[code]
            if not df.empty:
                returns_list.append(df['returns'].rename(code))

        returns_matrix = pd.concat(returns_list, axis=1).dropna()
        logging.info(f"组合收益率矩阵构建完成，维度：{returns_matrix.shape}")
        return returns_matrix