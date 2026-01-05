#组合权重配置（等权 / 市值 / 行业中性）、组合收益率计算、风险贡献度分析
# 作者：DogStar·Quant
# 时间：20260101
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from numpy.typing import NDArray


class PortfolioManager:
    """组合管理：权重配置、收益率计算、风险贡献度、权重调整"""
    def __init__(self, data_loader, codes: List[str], names: List[str],
                 industries: List[str], weight_method: str, rolling_window: int):
        self.data_loader = data_loader
        self.codes = codes
        self.names = names
        self.industries = industries
        self.rolling_window = rolling_window
        self.returns_matrix = self.data_loader.get_portfolio_returns_matrix(codes)
        self.original_weights: NDArray[np.float64] = self._get_portfolio_weights(weight_method)
        self.adjusted_weights: Optional[NDArray[np.float64]] = None  # 调仓后权重
        self.portfolio_returns: pd.Series = self._calculate_portfolio_returns(self.original_weights)
        self.rolling_corr: pd.DataFrame = self._calculate_rolling_correlation()

    def _get_portfolio_weights(self, weight_method: str) -> NDArray[np.float64]:
        """获取初始权重（等权/市值/行业中性）"""
        n_assets = len(self.codes)
        if weight_method == 'equal':
            weights = np.array([1 / n_assets] * n_assets, dtype=np.float64)
        elif weight_method == 'market_cap':
            market_caps = []
            for code in self.codes:
                market_suffix = '.SH' if code.startswith('6') else '.SZ'
                try:
                    cap_df = self.data_loader.pro.daily_basic(
                        ts_code=f"{code}{market_suffix}",
                        trade_date=self.returns_matrix.index[-1].strftime('%Y%m%d')
                    )
                    market_caps.append(
                        cap_df['circ_mv'].iloc[0] if not cap_df.empty else 1
                    )
                except Exception as e:
                    logging.warning(f"{code}市值获取失败，默认权重：{e}")
                    market_caps.append(1)
            weights = np.array(market_caps, dtype=np.float64) / sum(market_caps)
        elif weight_method == 'industry_neutral':
            bench_weights = {
                '消费': 0.2, '新能源': 0.2, '金融': 0.2, '科技': 0.2, '周期': 0.2
            }
            weights = np.array(
                [bench_weights[ind] for ind in self.industries],
                dtype=np.float64
            )
        else:
            logging.warning(f"权重方式{weight_method}无效，使用等权")
            weights = np.array([1 / n_assets] * n_assets, dtype=np.float64)

        logging.info(
            f"组合权重配置完成（{weight_method}）：{[round(w, 4) for w in weights]}"
        )
        return weights

    def _calculate_portfolio_returns(self, weights: NDArray[np.float64]) -> pd.Series:
        """计算组合收益率（支持自定义权重）"""
        portfolio_returns = self.returns_matrix @ weights
        return portfolio_returns.rename('portfolio_returns')

    def _calculate_rolling_correlation(self) -> pd.DataFrame:
        """计算滚动相关性矩阵"""
        rolling_corr = self.returns_matrix.rolling(window=self.rolling_window).corr()
        logging.info(
            f"滚动相关性矩阵计算完成（窗口{self.rolling_window}天）"
        )
        return rolling_corr

    def calculate_risk_contribution(self, var: float) -> Dict[str, float]:
        """计算风险贡献度"""
        cov_matrix = self.returns_matrix.cov()
        weights = self.original_weights if self.adjusted_weights is None else self.adjusted_weights
        weights = weights.reshape(-1, 1)

        # 组合波动率
        portfolio_vol = np.sqrt(np.dot(np.dot(weights.T, cov_matrix), weights))
        marginal_risk = np.dot(cov_matrix, weights) / portfolio_vol
        risk_contrib = (weights * marginal_risk) * var / portfolio_vol

        # 转换为字典
        risk_contrib_dict = {
            code: float(rc) for code, rc in zip(self.codes, risk_contrib.flatten())
        }
        logging.info(
            f"组合风险贡献度计算完成：{[round(v, 4) for v in risk_contrib_dict.values()]}"
        )
        return risk_contrib_dict

    def update_weights(self, new_weights: NDArray[np.float64]) -> None:
        """更新组合权重"""
        self.adjusted_weights = new_weights
        self.portfolio_returns = self._calculate_portfolio_returns(new_weights)
        logging.info(f"组合权重已更新：{[round(w, 4) for w in new_weights]}")