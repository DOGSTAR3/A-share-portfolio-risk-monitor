# A股跨行业组合风控全流程系统
# 功能：数据加载→组合构建→风险计量→预警→调仓→回测→监管报告→可视化
# 作者：DogStar·Quant
# 时间：20260101
import logging
import numpy as np
import pandas as pd
import tushare as ts
import streamlit as st
import plotly.express as px
from scipy import stats
from arch import arch_model
from typing import Tuple, Dict, List, Optional
from numpy.typing import NDArray

# ===================== 1. 基础配置 =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)


# ===================== 2. 数据加载模块（StockDataLoader） =====================
class StockDataLoader:
    """股票数据加载器：复权股价、收益率计算、3σ异常值处理"""

    def __init__(self, ts_token: str):
        ts.set_token(ts_token)
        self.pro = ts.pro_api()
        self.stock_data: Dict[str, pd.DataFrame] = {}

    def get_adj_stock_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取复权后股票数据，计算收益率并处理异常值"""
        market_suffix = '.SH' if stock_code.startswith('6') else '.SZ'
        try:
            # 1. 获取日线数据
            df = self.pro.daily(
                ts_code=f"{stock_code}{market_suffix}",
                start_date=start_date,
                end_date=end_date
            )
            if df.empty:
                logging.error(f"{stock_code}无日线数据返回")
                return pd.DataFrame()

            # 2. 获取复权因子（降级处理）
            try:
                adj_df = self.pro.adj_factor(
                    ts_code=f"{stock_code}{market_suffix}",
                    start_date=start_date,
                    end_date=end_date
                )
                adj_df['trade_date'] = pd.to_datetime(adj_df['trade_date'])
                adj_df = adj_df.sort_values('trade_date').set_index('trade_date')
            except Exception as e:
                logging.warning(f"{stock_code}复权因子获取失败：{e}，使用原始收盘价")
                adj_df = pd.DataFrame()

            # 3. 数据预处理
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date').set_index('trade_date')
            if not adj_df.empty:
                df = df.join(adj_df[['adj_factor']])
                latest_adj = df['adj_factor'].iloc[-1]
                df['close_adj'] = df['close'] * (df['adj_factor'] / latest_adj)
            else:
                df['close_adj'] = df['close']

            # 4. 计算对数收益率（百分比）+ 3σ异常值裁剪
            df['returns'] = np.log(df['close_adj'] / df['close_adj'].shift(1)) * 100
            df = df.dropna(subset=['returns'])
            mean_ret = df['returns'].mean()
            std_ret = df['returns'].std()
            df['returns'] = df['returns'].clip(
                lower=mean_ret - 3 * std_ret,
                upper=mean_ret + 3 * std_ret
            )

            self.stock_data[stock_code] = df
            logging.info(f"{stock_code}数据加载完成，有效收益率数据{len(df)}条")
            return df
        except Exception as e:
            logging.error(f"{stock_code}数据加载失败：{e}")
            return pd.DataFrame()

    def get_portfolio_returns_matrix(self, codes: list) -> pd.DataFrame:
        """构建组合收益率矩阵（日期×个股）"""
        returns_list = []
        for code in codes:
            if code not in self.stock_data:
                logging.warning(f"{code}未加载数据，跳过")
                continue
            df = self.stock_data[code]
            if not df.empty:
                returns_list.append(df['returns'].rename(code))

        if not returns_list:
            logging.error("无有效收益率数据，无法构建组合矩阵")
            return pd.DataFrame()
        returns_matrix = pd.concat(returns_list, axis=1).dropna()
        logging.info(f"组合收益率矩阵构建完成，维度：{returns_matrix.shape}")
        return returns_matrix


# ===================== 3. 组合管理模块（PortfolioManager） =====================
class PortfolioManager:
    """组合管理：权重配置、收益率计算、风险贡献度、权重调整"""

    def __init__(self, data_loader: StockDataLoader, codes: List[str], names: List[str],
                 industries: List[str], weight_method: str, rolling_window: int):
        self.data_loader = data_loader
        self.codes = codes
        self.names = names
        self.industries = industries
        self.rolling_window = rolling_window
        # 核心数据
        self.returns_matrix = self.data_loader.get_portfolio_returns_matrix(codes)
        self.original_weights: NDArray[np.float64] = self._get_portfolio_weights(weight_method)
        self.adjusted_weights: Optional[NDArray[np.float64]] = None  # 调仓后权重
        self.portfolio_returns: pd.Series = self._calculate_portfolio_returns(self.original_weights)
        self.rolling_corr: pd.DataFrame = self._calculate_rolling_correlation()

    def _get_portfolio_weights(self, weight_method: str) -> NDArray[np.float64]:
        """获取初始权重（等权/市值/行业中性）"""
        n_assets = len(self.codes)
        if n_assets == 0:
            logging.error("无有效资产代码，返回空权重")
            return np.array([])

        # 1. 等权
        if weight_method == 'equal':
            weights = np.array([1 / n_assets] * n_assets, dtype=np.float64)
        # 2. 市值加权
        elif weight_method == 'market_cap':
            market_caps = []
            for code in self.codes:
                market_suffix = '.SH' if code.startswith('6') else '.SZ'
                try:
                    # 取最新交易日市值
                    trade_date = self.returns_matrix.index[-1].strftime(
                        '%Y%m%d') if not self.returns_matrix.empty else '20240101'
                    cap_df = self.data_loader.pro.daily_basic(
                        ts_code=f"{code}{market_suffix}",
                        trade_date=trade_date
                    )
                    market_caps.append(cap_df['circ_mv'].iloc[0] if not cap_df.empty else 1)
                except Exception as e:
                    logging.warning(f"{code}市值获取失败，默认权重1：{e}")
                    market_caps.append(1)
            weights = np.array(market_caps, dtype=np.float64) / sum(market_caps)
        # 3. 行业中性（修复KeyError问题）
        elif weight_method == 'industry_neutral':
            bench_weights = {
                '消费': 0.2, '新能源': 0.2, '金融': 0.2, '科技': 0.2, '周期': 0.2
            }
            weights = []
            for ind in self.industries:
                weights.append(bench_weights.get(ind, 0.2))  # 未知行业默认0.2
            weights = np.array(weights, dtype=np.float64)
        # 4. 默认等权
        else:
            logging.warning(f"权重方式{weight_method}无效，使用等权")
            weights = np.array([1 / n_assets] * n_assets, dtype=np.float64)

        logging.info(f"组合权重配置完成（{weight_method}）：{[round(w, 4) for w in weights]}")
        return weights

    def _calculate_portfolio_returns(self, weights: NDArray[np.float64]) -> pd.Series:
        """计算组合收益率（支持自定义权重）"""
        if self.returns_matrix.empty or len(weights) != self.returns_matrix.shape[1]:
            logging.error("收益率矩阵或权重异常，返回空Series")
            return pd.Series()
        portfolio_returns = self.returns_matrix @ weights
        return portfolio_returns.rename('portfolio_returns')

    def _calculate_rolling_correlation(self) -> pd.DataFrame:
        """计算滚动相关性矩阵"""
        if self.returns_matrix.empty:
            logging.error("收益率矩阵为空，无法计算滚动相关性")
            return pd.DataFrame()
        rolling_corr = self.returns_matrix.rolling(window=self.rolling_window).corr()
        logging.info(f"滚动相关性矩阵计算完成（窗口{self.rolling_window}天）")
        return rolling_corr

    def calculate_risk_contribution(self, var: float) -> Dict[str, float]:
        """计算风险贡献度"""
        if self.returns_matrix.empty:
            logging.error("收益率矩阵为空，无法计算风险贡献度")
            return {}
        cov_matrix = self.returns_matrix.cov()
        weights = self.original_weights if self.adjusted_weights is None else self.adjusted_weights
        weights = weights.reshape(-1, 1)

        # 组合波动率
        portfolio_vol = np.sqrt(np.dot(np.dot(weights.T, cov_matrix), weights))
        if portfolio_vol == 0:
            logging.error("组合波动率为0，无法计算风险贡献度")
            return {code: 0.0 for code in self.codes}
        marginal_risk = np.dot(cov_matrix, weights) / portfolio_vol
        risk_contrib = (weights * marginal_risk) * var / portfolio_vol

        # 转换为字典
        risk_contrib_dict = {
            code: float(rc) for code, rc in zip(self.codes, risk_contrib.flatten())
        }
        logging.info(f"组合风险贡献度计算完成：{[round(v, 4) for v in risk_contrib_dict.values()]}")
        return risk_contrib_dict

    def update_weights(self, new_weights: NDArray[np.float64]) -> None:
        """更新组合权重"""
        if len(new_weights) != len(self.codes):
            logging.error("新权重长度与资产数量不匹配，更新失败")
            return
        self.adjusted_weights = new_weights
        self.portfolio_returns = self._calculate_portfolio_returns(new_weights)
        logging.info(f"组合权重已更新：{[round(w, 4) for w in new_weights]}")


# ===================== 4. 风险计量模块（RiskCalculator） =====================
class RiskCalculator:
    """风险计量：VaR/ES计算、GARCH(1,1)波动率、压力测试"""

    def __init__(self, portfolio_manager: PortfolioManager):
        self.portfolio_manager = portfolio_manager
        self.portfolio_returns = portfolio_manager.portfolio_returns
        self.returns_matrix = portfolio_manager.returns_matrix
        self.risk_results: Dict[str, Dict[str, float]] = {}
        self.stress_results: Dict[str, float] = {}
        self.garch_vol: Dict[str, pd.Series] = self._calculate_garch_volatility()

    def _calculate_garch_volatility(self) -> Dict[str, pd.Series]:
        """GARCH(1,1)计算时变波动率"""
        garch_vol = {}
        if self.returns_matrix.empty:
            logging.error("收益率矩阵为空，无法计算GARCH波动率")
            return garch_vol
        for code in self.returns_matrix.columns:
            returns = self.returns_matrix[code].dropna()
            if len(returns) < 100:
                logging.warning(f"{code}数据不足（{len(returns)}条），跳过GARCH计算")
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
        """单置信度VaR/ES计算（修复参数法ES逻辑）"""
        alpha = 1 - confidence_level / 100
        var, es = np.nan, np.nan

        if len(returns) < 100:
            logging.error("收益率数据不足（<100条），无法计算VaR/ES")
            return var, es

        try:
            if method == 'historical':
                var = np.percentile(returns, alpha * 100, method='nearest')
                es = returns[returns <= var].mean()
            elif method == 'parametric':
                # 正态分布参数法（理论对齐）
                mu = returns.mean()
                sigma = returns.std()
                z_score = stats.norm.ppf(alpha)
                var = mu + z_score * sigma
                # 正态分布ES解析解：mu + sigma * stats.norm.pdf(z_score) / alpha
                es = mu + sigma * stats.norm.pdf(z_score) / alpha
            elif method == 'monte_carlo':
                np.random.seed(42)
                mu = returns.mean()
                sigma = returns.std()
                mc_returns = np.random.normal(loc=mu, scale=sigma, size=10000)
                var = np.percentile(mc_returns, alpha * 100, method='nearest')
                es = mc_returns[mc_returns <= var].mean()

            logging.info(f"{method}法{confidence_level}%置信度：VaR={round(var, 4)}%，ES={round(es, 4)}%")
            return var, es
        except Exception as e:
            logging.error(f"VaR/ES计算失败：{e}")
            return var, es

    def calculate_portfolio_var_es(self, confidence_levels: List[int],
                                   method: str = 'historical') -> Dict[str, Dict[str, float]]:
        """多置信度VaR/ES计算"""
        if self.portfolio_returns.empty:
            logging.error("组合收益率为空，无法计算VaR/ES")
            return self.risk_results
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

        if self.portfolio_returns.empty:
            logging.error("组合收益率为空，无法进行压力测试")
            return stress_results

        for period in stress_periods:
            if period not in stress_periods_map:
                logging.warning(f"压力场景{period}无效，可选：{list(stress_periods_map.keys())}")
                continue
            start, end = stress_periods_map[period]
            # 转换日期格式匹配
            mask = (self.portfolio_returns.index >= pd.to_datetime(start)) & (
                        self.portfolio_returns.index <= pd.to_datetime(end))
            stress_returns = self.portfolio_returns[mask]
            if len(stress_returns) < 10:
                logging.warning(f"{period}数据不足（仅{len(stress_returns)}条），跳过压力测试")
                continue
            stress_var_99 = np.percentile(stress_returns, 1, method='nearest')
            stress_results[period] = stress_var_99
            logging.info(f"{period} 压力测试完成：99%置信度VaR={round(stress_var_99, 4)}%")

        self.stress_results = stress_results
        return stress_results


# ===================== 5. 风险预警模块（RiskAlert） =====================
class RiskAlert:
    """风险预警：等级判断、原因分析、处置建议"""

    def __init__(self, risk_results: Dict[str, Dict[str, float]],
                 y95: float, o95: float, r95: float,
                 y99: float, o99: float, r99: float):
        self.risk_results = risk_results
        self.alert_level: str = "normal"
        self.alert_reason: List[str] = []
        self.alert_suggestion: str = ""
        # 预警阈值
        self.ALERT_Y95 = y95
        self.ALERT_O95 = o95
        self.ALERT_R95 = r95
        self.ALERT_Y99 = y99
        self.ALERT_O99 = o99
        self.ALERT_R99 = r99

    def evaluate_alert_level(self) -> Tuple[str, List[str], str]:
        """评估预警等级"""
        for cl in [95, 99]:  # 支持的置信度
            cl_str = str(cl)
            if cl_str not in self.risk_results:
                logging.warning(f"无{cl}%置信度风险结果，跳过该等级判断")
                continue
            var = self.risk_results[cl_str]['VaR']
            if np.isnan(var):
                logging.warning(f"{cl}%置信度VaR为NaN，跳过该等级判断")
                continue
            var_abs = abs(var)

            # 95%置信度预警
            if cl == 95:
                if var_abs > self.ALERT_R95:
                    self.alert_level = "red"
                    self.alert_reason.append(
                        f"95%VaR绝对值{round(var_abs, 2)}%＞{self.ALERT_R95}%（红色预警阈值）"
                    )
                elif var_abs > self.ALERT_O95:
                    if self.alert_level != "red":
                        self.alert_level = "orange"
                    self.alert_reason.append(
                        f"95%VaR绝对值{round(var_abs, 2)}%＞{self.ALERT_O95}%（橙色预警阈值）"
                    )
                elif var_abs > self.ALERT_Y95:
                    if self.alert_level not in ["red", "orange"]:
                        self.alert_level = "yellow"
                    self.alert_reason.append(
                        f"95%VaR绝对值{round(var_abs, 2)}%＞{self.ALERT_Y95}%（黄色预警阈值）"
                    )

            # 99%置信度预警
            elif cl == 99:
                if var_abs > self.ALERT_R99:
                    self.alert_level = "red"
                    self.alert_reason.append(
                        f"99%VaR绝对值{round(var_abs, 2)}%＞{self.ALERT_R99}%（红色预警阈值）"
                    )
                elif var_abs > self.ALERT_O99:
                    if self.alert_level != "red":
                        self.alert_level = "orange"
                    self.alert_reason.append(
                        f"99%VaR绝对值{round(var_abs, 2)}%＞{self.ALERT_O99}%（橙色预警阈值）"
                    )
                elif var_abs > self.ALERT_Y99:
                    if self.alert_level not in ["red", "orange"]:
                        self.alert_level = "yellow"
                    self.alert_reason.append(
                        f"99%VaR绝对值{round(var_abs, 2)}%＞{self.ALERT_Y99}%（黄色预警阈值）"
                    )

        # 处置建议
        if self.alert_level == "red":
            self.alert_suggestion = "【紧急处置】立即削减高风险标的权重50%，暂停新增该组合交易，启动极端风险应对预案"
        elif self.alert_level == "orange":
            self.alert_suggestion = "【重点关注】削减高风险标的权重50%，增加低风险标的配置，每日监控风险指标"
        elif self.alert_level == "yellow":
            self.alert_suggestion = "【常规关注】密切监控高风险标的波动，每周评估组合权重合理性"
        else:
            self.alert_suggestion = "【正常状态】组合风险指标在安全区间，按常规频率监控"

        logging.info(f"风险预警评估完成：等级={self.alert_level}，原因={self.alert_reason}")
        return self.alert_level, self.alert_reason, self.alert_suggestion


# ===================== 6. 风险处置模块（RiskDisposal） =====================
class RiskDisposal:
    """风险处置：识别高/低风险标的、自动调仓"""

    def __init__(self, portfolio_manager: PortfolioManager, alert_level: str, high_risk_cut: float):
        self.portfolio_manager = portfolio_manager
        self.alert_level = alert_level
        self.high_risk_codes: List[str] = []
        self.low_risk_codes: List[str] = []
        self.original_weights: NDArray[np.float64] = portfolio_manager.original_weights
        self.new_weights: Optional[NDArray[np.float64]] = None
        self.HIGH_RISK_CUT = high_risk_cut  # 高风险标的权重削减比例

    def identify_risk_level(self, risk_contrib: Dict[str, float]) -> None:
        """按风险贡献度识别高/低风险标的"""
        if not risk_contrib:
            logging.error("风险贡献度为空，无法识别风险等级")
            return
        contrib_vals = list(risk_contrib.values())
        median_contrib = np.median(contrib_vals)
        self.high_risk_codes = [
            code for code, val in risk_contrib.items() if abs(val) > abs(median_contrib)
        ]
        self.low_risk_codes = [
            code for code, val in risk_contrib.items() if abs(val) <= abs(median_contrib)
        ]
        logging.info(f"高风险标的：{self.high_risk_codes}，低风险标的：{self.low_risk_codes}")

    def adjust_weights(self) -> Optional[NDArray[np.float64]]:
        """根据预警等级调整权重"""
        # 仅红/橙色预警调仓
        if self.alert_level not in ["red", "orange"]:
            logging.info(f"预警等级{self.alert_level}，无需调仓")
            return None

        if len(self.original_weights) == 0:
            logging.error("原始权重为空，无法调仓")
            return None

        # 初始化新权重
        new_weights = self.original_weights.copy()
        code2idx = {code: idx for idx, code in enumerate(self.portfolio_manager.codes)}

        # 削减高风险标的权重
        total_cut = 0.0
        for code in self.high_risk_codes:
            if code not in code2idx:
                logging.warning(f"{code}不在组合中，跳过削减")
                continue
            idx = code2idx[code]
            cut_amount = float(new_weights[idx] * self.HIGH_RISK_CUT)
            new_weights[idx] -= cut_amount
            total_cut += cut_amount
            logging.info(
                f"削减{code}权重：{round(float(self.original_weights[idx]), 4)} → "
                f"{round(float(new_weights[idx]), 4)}（削减{round(cut_amount, 4)}）"
            )

        # 增加低风险标的权重（分配削减的总权重）
        if len(self.low_risk_codes) > 0:
            inc_per_low = total_cut / len(self.low_risk_codes)
            for code in self.low_risk_codes:
                if code not in code2idx:
                    logging.warning(f"{code}不在组合中，跳过增加")
                    continue
                idx = code2idx[code]
                new_weights[idx] += inc_per_low
                logging.info(
                    f"增加{code}权重：{round(float(self.original_weights[idx]), 4)} → "
                    f"{round(float(new_weights[idx]), 4)}（增加{round(inc_per_low, 4)}）"
                )
        else:
            logging.warning("无低风险标的，削减的权重将归一化分配")

        # 权重归一化（确保和为1）
        new_weights = new_weights / new_weights.sum()
        self.new_weights = new_weights

        # 输出调仓对比
        logging.info("=== 调仓前后权重对比 ===")
        for idx, code in enumerate(self.portfolio_manager.codes):
            logging.info(
                f"{code}：{round(float(self.original_weights[idx]), 4)} → {round(float(new_weights[idx]), 4)}"
            )

        return new_weights


# ===================== 7. VaR模型回测模块（ModelBacktest） =====================
class ModelBacktest:
    """VaR模型回测：返回检验、有效性验证"""

    def __init__(self, portfolio_manager: PortfolioManager, risk_calculator: RiskCalculator,
                 backtest_window: int, pass_threshold: float):
        self.portfolio_manager = portfolio_manager
        self.risk_calculator = risk_calculator
        self.portfolio_returns = portfolio_manager.portfolio_returns
        self.backtest_window = backtest_window  # 回测窗口大小
        self.pass_threshold = pass_threshold  # 合格超限率阈值
        self.backtest_results: Dict[str, float] = {}
        self.backtest_report: str = ""
        self.rolling_var = []
        self.rolling_returns = []

    def run_backtest(self, confidence_level: int = 95) -> Dict[str, float]:
        """运行VaR回测"""
        self.rolling_var = []
        self.rolling_returns = []
        returns = self.portfolio_returns.dropna()

        if len(returns) < self.backtest_window:
            logging.error(f"回测数据不足（仅{len(returns)}条，需{self.backtest_window}条）")
            self.backtest_results = {'exceed_times': 0, 'total_times': 0, 'exceed_rate': np.nan, 'pass': False}
            self._generate_backtest_report(confidence_level)
            return self.backtest_results

        # 滑动窗口计算VaR
        for i in range(self.backtest_window, len(returns)):
            window_returns = returns.iloc[i - self.backtest_window:i]
            var, _ = self.risk_calculator._calculate_var_es_single(
                window_returns, confidence_level, method='historical'
            )
            self.rolling_var.append(var)
            self.rolling_returns.append(returns.iloc[i])

        # 计算超限次数
        rolling_var = np.array(self.rolling_var)
        rolling_returns = np.array(self.rolling_returns)
        exceed_times = sum(rolling_returns < rolling_var)
        total_times = len(rolling_returns)
        exceed_rate = exceed_times / total_times if total_times > 0 else np.nan
        is_pass = exceed_rate <= self.pass_threshold if not np.isnan(exceed_rate) else False

        self.backtest_results = {
            'exceed_times': exceed_times,
            'total_times': total_times,
            'exceed_rate': exceed_rate,
            'pass': is_pass
        }

        # 生成回测报告
        self._generate_backtest_report(confidence_level)
        logging.info(
            f"VaR模型回测完成（{confidence_level}%置信度）："
            f"超限率={round(exceed_rate, 4) if not np.isnan(exceed_rate) else 'N/A'}，"
            f"是否通过={is_pass}"
        )
        return self.backtest_results

    def _generate_backtest_report(self, confidence_level: int) -> None:
        """生成回测报告"""
        if np.isnan(self.backtest_results['exceed_rate']):
            report = [
                "===== VaR模型回测报告 =====",
                f"置信度：{confidence_level}%",
                "回测结果：数据不足，无法验证",
                ""
            ]
            self.backtest_report = "\n".join(report)
        else:
            report = f"""===== VaR模型回测报告 =====
置信度：{confidence_level}%
回测窗口：{self.backtest_window}个交易日
实际超限次数：{self.backtest_results['exceed_times']}次
总测试次数：{self.backtest_results['total_times']}次
实际超限率：{round(self.backtest_results['exceed_rate'], 4)}
理论超限率：{round(1 - confidence_level / 100, 4)}
模型有效性：{'通过' if self.backtest_results['pass'] else '不通过'}
建议：{'模型有效，可继续使用' if self.backtest_results['pass'] else '模型失效，需重新校准（如调整GARCH参数）'}
"""
            self.backtest_report = report
        logging.info(f"回测报告生成完成：\n{self.backtest_report}")

    def export_backtest_plots(self, plot_dir: str = "reports/plots"):
        """导出回测可视化图表"""
        # 检查是否有回测数据
        if not self.rolling_var or not self.rolling_returns:
            logging.error("无回测数据，无法生成可视化图表（请先运行run_backtest方法）")
            return

        # 检查kaleido是否安装
        try:
            import kaleido
        except ImportError:
            logging.error("未检测到kaleido包，无法导出图片。请执行 'pip install --upgrade kaleido' 安装依赖")
            return

        # 创建plots目录
        import os
        os.makedirs(plot_dir, exist_ok=True)

        try:
            # 1. 滚动VaR曲线
            rolling_var = np.array(self.rolling_var)
            rolling_returns = np.array(self.rolling_returns)
            df_plot = pd.DataFrame({
                "日期": self.portfolio_returns.index[self.backtest_window:],
                "组合收益率": rolling_returns,
                "滚动VaR": rolling_var
            })
            # 优化图表样式：添加超限点标记、调整颜色
            fig1 = px.line(df_plot, x="日期", y=["组合收益率", "滚动VaR"],
                           title=f"滚动VaR曲线（{self.backtest_window}日窗口）",
                           color_discrete_map={"组合收益率": "#1f77b4", "滚动VaR": "#ff7f0e"})
            # 标记超限点
            df_plot["超限"] = df_plot["组合收益率"] < df_plot["滚动VaR"]
            fig1.add_scatter(x=df_plot[df_plot["超限"]]["日期"],
                             y=df_plot[df_plot["超限"]]["组合收益率"],
                             mode="markers", name="超限点", marker=dict(color="red", size=8))
            fig1.write_image(os.path.join(plot_dir, "rolling_var_curve.png"),
                             width=1200, height=600, scale=2)

            # 2. 超限次数分布
            exceed_flag = rolling_returns < rolling_var
            df_exceed = pd.DataFrame({"是否超限": ["超限" if x else "未超限" for x in exceed_flag]})
            df_exceed_count = df_exceed["是否超限"].value_counts().reset_index()
            df_exceed_count.columns = ["是否超限", "次数"]
            fig2 = px.pie(df_exceed_count, values="次数", names="是否超限",
                          title="超限次数分布",
                          color_discrete_map={"超限": "#d62728", "未超限": "#2ca02c"})
            fig2.update_traces(textposition='inside', textinfo='percent+label')
            fig2.write_image(os.path.join(plot_dir, "exceed_distribution.png"),
                             width=800, height=600, scale=2)

            logging.info(f"回测可视化图表导出完成，路径：{os.path.abspath(plot_dir)}/")
        except Exception as e:
            logging.error(f"图表导出失败：{str(e)}")


# ===================== 8. 监管报告模块（RegulatoryReportGenerator） =====================
class RegulatoryReportGenerator:
    """监管合规报告生成"""

    def __init__(self, risk_results: dict, reg_threshold: float, is_adjusted: bool = False):
        self.risk_results = risk_results
        self.reg_threshold = reg_threshold
        self.is_adjusted = is_adjusted

    def generate_reg_report(self) -> str:
        report = [
            "===== 组合风控监管合规报告 =====",
            f"报告类型：{'调仓后' if self.is_adjusted else '调仓前'}",
            "【监管依据】《证券公司风险控制指标管理办法》第12条：",
            "  证券公司应当采用多置信水平、多情景分析等方法计量市场风险，覆盖95%日常监控和99%极端风险场景；",
            "【监管依据】《证券公司风险控制指标管理办法》第15条：",
            "  单一投资组合的市场风险敞口不得超过证券公司净资本的10%；",
            "",
            "===== 风险指标与监管达标情况 ====="
        ]

        for cl in [95, 99]:
            cl_str = str(cl)
            if cl_str not in self.risk_results:
                report.append(f"{cl}%置信度VaR：无数据，无法验证")
                continue
            var = self.risk_results[cl_str]['VaR']
            if var is None or pd.isna(var):
                report.append(f"{cl}%置信度VaR：数据不足，无法验证")
                continue
            is_compliant = abs(var) <= self.reg_threshold
            report.append(f"{cl}%置信度VaR：{round(var, 4)}% → {'达标' if is_compliant else '超标'}")

        return "\n".join(report)


# ===================== 9. 可视化仪表盘模块（RiskDashboard） =====================
class RiskDashboard:
    """Streamlit可视化仪表盘（整合所有模块，启动入口）"""

    def __init__(self, portfolio_manager: PortfolioManager, risk_calculator: RiskCalculator,
                 reg_report: str, alert_info: Tuple[str, list, str],
                 backtest_report: str, disposal_result: Optional[np.ndarray] = None):
        self.portfolio_manager = portfolio_manager
        self.risk_calculator = risk_calculator
        self.reg_report = reg_report
        self.alert_level, self.alert_reason, self.alert_suggestion = alert_info
        self.backtest_report = backtest_report
        self.disposal_result = disposal_result

    def run_dashboard(self):
        st.set_page_config(page_title="A股组合风控监控", layout="wide")
        st.title("📊 A股跨行业组合风控实时监控系统（含预警-处置-回测）")

        # 预警等级展示
        alert_color = {"normal": "green", "yellow": "yellow", "orange": "orange", "red": "red"}
        st.markdown(
            f"### 🚨 风险预警等级：<span style='color:{alert_color[self.alert_level]};font-size:20px'>{self.alert_level.upper()}</span>",
            unsafe_allow_html=True
        )
        st.markdown(f"**预警原因**：{'; '.join(self.alert_reason) if self.alert_reason else '无'}")
        st.markdown(f"**处置建议**：{self.alert_suggestion}")
        st.divider()

        # 1. 组合基本信息
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📋 组合基本信息（调仓前）")
            portfolio_info = pd.DataFrame({
                '股票代码': self.portfolio_manager.codes,
                '股票名称': self.portfolio_manager.names,
                '行业': self.portfolio_manager.industries,
                '权重(%)': [round(w * 100, 2) for w in self.portfolio_manager.original_weights]
            })
            st.dataframe(portfolio_info, hide_index=True)

        with col2:
            if self.disposal_result is not None:
                st.subheader("📋 组合基本信息（调仓后）")
                portfolio_info_adjusted = pd.DataFrame({
                    '股票代码': self.portfolio_manager.codes,
                    '股票名称': self.portfolio_manager.names,
                    '行业': self.portfolio_manager.industries,
                    '权重(%)': [round(w * 100, 2) for w in self.disposal_result]
                })
                st.dataframe(portfolio_info_adjusted, hide_index=True)
            else:
                st.subheader("📋 调仓提示")
                st.info("当前预警等级无需调仓，展示调仓前权重")

        # 2. 核心风险指标
        st.subheader("🎯 核心风险指标")
        risk_df = pd.DataFrame({
            '置信度': [f"{cl}%" for cl in [95, 99]],
            'VaR(%)': [round(self.risk_calculator.risk_results[str(cl)]['VaR'], 4) if str(
                cl) in self.risk_calculator.risk_results else np.nan for cl in [95, 99]],
            'ES(%)': [round(self.risk_calculator.risk_results[str(cl)]['ES'], 4) if str(
                cl) in self.risk_calculator.risk_results else np.nan for cl in [95, 99]]
        })
        st.dataframe(risk_df, hide_index=True)

        # 3. 风险贡献度可视化
        st.subheader("🔥 个股风险贡献度（%）")
        var_95 = self.risk_calculator.risk_results.get('95', {}).get('VaR', 0)
        risk_contrib = self.portfolio_manager.calculate_risk_contribution(var_95)
        contrib_df = pd.DataFrame({
            '股票名称': self.portfolio_manager.names,
            '行业': self.portfolio_manager.industries,
            '风险贡献度(%)': [round(risk_contrib.get(code, 0), 4) for code in self.portfolio_manager.codes]
        })
        fig_contrib = px.bar(
            contrib_df,
            x='股票名称',
            y='风险贡献度(%)',
            color='行业',
            color_discrete_map={'消费': 'red', '新能源': 'green', '金融': 'blue', '科技': 'orange', '周期': 'purple'}
        )
        st.plotly_chart(fig_contrib, use_container_width=True)

        # 4. 模型回测报告
        st.subheader("📈 VaR模型回测报告")
        st.text(self.backtest_report)

        # 5. 监管合规报告
        st.subheader("📜 监管合规报告")
        st.text(self.reg_report)

        # 6. 压力测试结果
        st.subheader("⚠️ 压力测试结果（99%置信度VaR，%）")
        stress_results = self.risk_calculator.stress_results
        if stress_results:
            stress_df = pd.DataFrame({
                '压力场景': list(stress_results.keys()),
                '99%置信度VaR(%)': [round(v, 4) for v in stress_results.values()]
            })
            st.dataframe(stress_df, hide_index=True)
        else:
            st.warning("无有效压力测试数据（如场景时间范围无数据）")


# ===================== 10. 主执行函数 =====================
def main(config: Dict):
    """主执行函数：整合所有模块，完成全流程"""
    try:
        # Step 1: 数据加载
        logging.info("===== 步骤1：加载股票数据 =====")
        data_loader = StockDataLoader(config['TS_TOKEN'])
        for code in config['PORT_CODES']:
            data_loader.get_adj_stock_data(code, config['START_DATE'], config['END_DATE'])

        # Step 2: 构建投资组合
        logging.info("\n===== 步骤2：构建投资组合 =====")
        portfolio_manager = PortfolioManager(
            data_loader=data_loader,
            codes=config['PORT_CODES'],
            names=config['PORT_NAMES'],
            industries=config['PORT_INDUSTRIES'],
            weight_method=config['WEIGHT_METHOD'],
            rolling_window=config['ROLLING_WINDOW']
        )
        if portfolio_manager.portfolio_returns.empty:
            raise ValueError("组合收益率为空，无法继续后续流程")

        # Step 3: 风险计量
        logging.info("\n===== 步骤3：风险计量（VaR/ES/GARCH/压力测试） =====")
        risk_calculator = RiskCalculator(portfolio_manager)
        risk_results = risk_calculator.calculate_portfolio_var_es(
            confidence_levels=config['CONF_LEVELS'],
            method=config['RISK_METHOD']
        )
        risk_calculator.stress_test(config['STRESS_PERIODS'])

        # Step 4: VaR模型回测
        logging.info("\n===== 步骤4：VaR模型回测 =====")
        backtest = ModelBacktest(
            portfolio_manager=portfolio_manager,
            risk_calculator=risk_calculator,
            backtest_window=config['BACKTEST_WINDOW'],
            pass_threshold=config['PASS_THRESHOLD']
        )
        backtest.run_backtest(confidence_level=95)
        backtest.export_backtest_plots()  # 导出回测图表

        # Step 5: 风险预警
        logging.info("\n===== 步骤5：风险预警等级判断 =====")
        alert = RiskAlert(
            risk_results=risk_results,
            y95=config['ALERT_Y95'],
            o95=config['ALERT_O95'],
            r95=config['ALERT_R95'],
            y99=config['ALERT_Y99'],
            o99=config['ALERT_O99'],
            r99=config['ALERT_R99']
        )
        alert_level, alert_reason, alert_suggestion = alert.evaluate_alert_level()

        # Step 6: 风险处置（调仓）
        logging.info("\n===== 步骤6：风险处置（自动调仓） =====")
        disposal = RiskDisposal(
            portfolio_manager=portfolio_manager,
            alert_level=alert_level,
            high_risk_cut=config['HIGH_RISK_CUT']
        )
        # 计算风险贡献度，识别高/低风险标的
        var_95 = risk_results['95']['VaR'] if '95' in risk_results else 0
        risk_contrib = portfolio_manager.calculate_risk_contribution(var_95)
        disposal.identify_risk_level(risk_contrib)
        # 调仓并更新组合权重
        new_weights = disposal.adjust_weights()
        if new_weights is not None:
            portfolio_manager.update_weights(new_weights)
            # 调仓后重新计算风险指标
            risk_calculator_adjusted = RiskCalculator(portfolio_manager)
            risk_results_adjusted = risk_calculator_adjusted.calculate_portfolio_var_es(
                confidence_levels=config['CONF_LEVELS'],
                method=config['RISK_METHOD']
            )
            reg_generator = RegulatoryReportGenerator(
                risk_results=risk_results_adjusted,
                reg_threshold=config['REG_THRESHOLD'],
                is_adjusted=True
            )
        else:
            reg_generator = RegulatoryReportGenerator(
                risk_results=risk_results,
                reg_threshold=config['REG_THRESHOLD']
            )
        reg_report = reg_generator.generate_reg_report()

        # Step 7: 启动可视化仪表盘
        logging.info("\n===== 步骤7：启动风控可视化仪表盘 =====")
        dashboard = RiskDashboard(
            portfolio_manager=portfolio_manager,
            risk_calculator=risk_calculator,
            reg_report=reg_report,
            alert_info=(alert_level, alert_reason, alert_suggestion),
            backtest_report=backtest.backtest_report,
            disposal_result=new_weights
        )
        dashboard.run_dashboard()

    except Exception as e:
        logging.error(f"系统运行失败：{e}", exc_info=True)
        raise


# ===================== 运行入口 =====================
if __name__ == "__main__":
    # 配置字典（需根据实际情况修改）
    CONFIG = {
        # 数据加载相关
        "TS_TOKEN": "请替换为自己的Tushare token",  # 替换为自己的Tushare token
        "PORT_CODES": ["600519", "600036", "300750", "601668"],  # 茅台、招行、宁德时代、中国建筑
        "PORT_NAMES": ["贵州茅台", "招商银行", "宁德时代", "中国建筑"],
        "PORT_INDUSTRIES": ["消费", "金融", "新能源", "周期"],
        "START_DATE": "20200101",  # 数据起始日期
        "END_DATE": "20240101",  # 数据结束日期

        # 组合管理相关
        "WEIGHT_METHOD": "equal",  # 权重方法：equal/market_cap/industry_neutral
        "ROLLING_WINDOW": 60,  # 滚动窗口大小（交易日）

        # 风险计算相关
        "CONF_LEVELS": [95, 99],  # 置信水平
        "RISK_METHOD": "historical",  # VaR计算方法：historical/parametric/monte_carlo
        "STRESS_PERIODS": ["2015_crash", "2020_pandemic"],  # 压力测试场景

        # 回测相关
        "BACKTEST_WINDOW": 120,  # 回测窗口大小
        "PASS_THRESHOLD": 0.06,  # 合格超限率阈值（6%）

        # 预警阈值（不同置信度下的黄/橙/红预警阈值）
        "ALERT_Y95": 2.0, "ALERT_O95": 4.0, "ALERT_R95": 6.0,
        "ALERT_Y99": 3.0, "ALERT_O99": 5.0, "ALERT_R99": 7.0,

        # 风险处置相关
        "HIGH_RISK_CUT": 0.5,  # 高风险资产权重削减比例（50%）

        # 监管合规相关
        "REG_THRESHOLD": 7.0  # 监管阈值（VaR不得超过此值）
    }
    # 启动系统
    main(CONFIG)