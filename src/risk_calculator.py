#VaR/ES（历史 / 参数 / 蒙特卡洛）、GARCH (1,1) 波动率、压力测试
# 作者：DogStar·Quant
# 时间：20260101
import logging
import numpy as np
import pandas as pd
from scipy import stats
from arch import arch_model
from typing import Tuple, Dict, List


class RiskCalculator:
    """风险计量：VaR/ES计算、GARCH(1,1)波动率、压力测试"""
    def __init__(self, portfolio_manager):
        self.portfolio_manager = portfolio_manager
        self.portfolio_returns = portfolio_manager.portfolio_returns
        self.returns_matrix = portfolio_manager.returns_matrix
        self.risk_results: Dict[str, Dict[str, float]] = {}
        self.stress_results: Dict[str, float] = {}
        self.garch_vol: Dict[str, pd.Series] = self._calculate_garch_volatility()

    def _calculate_garch_volatility(self) -> Dict[str, pd.Series]:
        """GARCH(1,1)计算时变波动率"""
        garch_vol = {}
        for code in self.portfolio_manager.codes:
            returns = self.returns_matrix[code].dropna()
            if len(returns) < 100:
                logging.warning(f"{code}数据不足，跳过GARCH计算")
                garch_vol[code] = pd.Series()
                continue
            model = arch_model(returns, vol='GARCH', p=1, q=1, mean='Constant')
            try:
                results = model.fit(disp='off')
                garch_vol[code] = results.conditional_volatility
                logging.info(f"{code}GARCH波动率计算完成")
            except Exception as e:
                logging.error(f"{code}GARCH拟合失败：{e}")
                garch_vol[code] = pd.Series()
        return garch_vol

    @staticmethod
    def _calculate_var_es_single(returns: pd.Series, confidence_level: int,
                                 method: str = 'historical') -> Tuple[float, float]:
        """单置信度VaR/ES计算（改为静态方法，解决受保护成员警示）"""
        alpha = 1 - confidence_level / 100
        var, es = np.nan, np.nan

        if len(returns) < 100:
            logging.error("收益率数据不足，无法计算VaR/ES")
            return var, es

        try:
            if method == 'historical':
                var = np.percentile(returns, alpha * 100, method='nearest')
                es = returns[returns <= var].mean()
            elif method == 'parametric':
                mu = returns.mean()
                sigma = returns.std()
                z_score = stats.norm.ppf(alpha)
                var = mu + z_score * sigma
                es = returns[returns <= var].mean()
            elif method == 'monte_carlo':
                np.random.seed(42)
                mu = returns.mean()
                sigma = returns.std()
                mc_returns = np.random.normal(
                    loc=mu, scale=sigma, size=10000
                )  # 暂用固定值，可配置
                var = np.percentile(mc_returns, alpha * 100, method='nearest')
                es = mc_returns[mc_returns <= var].mean()

            logging.info(
                f"{method}法{confidence_level}%置信度：VaR={round(var, 4)}%，ES={round(es, 4)}%"
            )
            return var, es
        except Exception as e:
            logging.error(f"VaR/ES计算失败：{e}")
            return var, es

    def calculate_portfolio_var_es(self, confidence_levels: List[int],
                                   method: str = 'historical') -> Dict[str, Dict[str, float]]:
        """多置信度VaR/ES计算"""
        for cl in confidence_levels:
            var, es = self._calculate_var_es_single(self.portfolio_returns, cl, method)
            self.risk_results[str(cl)] = {'VaR': var, 'ES': es}
        logging.info(f"组合风险计量完成：{self.risk_results}")
        return self.risk_results

    def stress_test(self, stress_periods: List[str]) -> Dict[str, float]:
        """压力测试（2015股灾/2020疫情）"""
        stress_periods_map = {
            '2015_crash': ('20150601', '20150731'),
            '2020_pandemic': ('20200201', '20200331')
        }
        stress_results = {}

        for period in stress_periods:
            if period not in stress_periods_map:
                logging.warning(f"压力场景{period}无效")
                continue
            start, end = stress_periods_map[period]
            mask = (self.portfolio_returns.index >= start) & (self.portfolio_returns.index <= end)
            stress_returns = self.portfolio_returns[mask]
            if len(stress_returns) < 10:
                logging.warning(
                    f"{period}数据不足（仅{len(stress_returns)}条），跳过压力测试"
                )
                continue
            stress_var_99 = np.percentile(stress_returns, 1, method='nearest')
            stress_results[period] = stress_var_99
            logging.info(
                f"{period} 压力测试完成：99%置信度VaR={round(stress_var_99, 4)}%"
            )

        self.stress_results = stress_results
        return stress_results