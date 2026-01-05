# 滑动窗口回测、超限率计算、模型有效性验证
# 作者：DogStar·Quant
# 时间：20260101
import logging
import numpy as np
import pandas as pd
from typing import Dict
import os
import plotly.express as px

# 配置日志（方便查看执行过程）
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelBacktest:
    """VaR模型回测：返回检验、有效性验证"""

    def __init__(self, portfolio_manager, risk_calculator,
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
            logging.error(
                f"回测数据不足（仅{len(returns)}条，需{self.backtest_window}条）"
            )
            self.backtest_results = {'exceed_times': 0, 'total_times': 0, 'exceed_rate': np.nan, 'pass': False}
            return self.backtest_results

        # 滑动窗口计算VaR
        for i in range(self.backtest_window, len(returns)):
            window_returns = returns.iloc[i - self.backtest_window:i]
            var, _ = self.risk_calculator._calculate_var_es_single(
                window_returns, confidence_level
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

    def export_backtest_plots(self):
        """导出回测可视化图表（改用matplotlib，无kaleido依赖，Windows兼容）"""
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import logging
        import os

        # 解决中文显示和负号显示问题（Windows必备）
        plt.rcParams["font.sans-serif"] = ["SimHei"]  # 适配中文
        plt.rcParams["axes.unicode_minus"] = False  # 适配负号
        plt.rcParams["figure.dpi"] = 100  # 基础清晰度

        # 检查回测数据
        if not self.rolling_var or not self.rolling_returns:
            logging.error("无回测数据，无法生成可视化图表（请先运行run_backtest方法）")
            return

        # 创建保存目录
        plot_dir = os.path.join("reports", "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plot_dir_abs = os.path.abspath(plot_dir)
        logging.info(f"图表保存目录：{plot_dir_abs}")

        # 数据预处理
        rolling_var = np.array(self.rolling_var)
        rolling_returns = np.array(self.rolling_returns)
        dates = self.portfolio_returns.index[self.backtest_window:]  # 日期索引
        logging.info(f"绘图数据样例：\n日期：{dates[:5].tolist()}\n收益率：{rolling_returns[:5]}\nVaR：{rolling_var[:5]}")

        # ========== 1. 绘制滚动VaR曲线 ==========
        try:
            fig, ax = plt.subplots(figsize=(12, 6))  # 画布大小

            # 绘制收益率和VaR曲线
            ax.plot(dates, rolling_returns, color="#1f77b4", linewidth=1.2, label="组合收益率")
            ax.plot(dates, rolling_var, color="#ff7f0e", linewidth=1.2, label="滚动VaR（95%置信度）")

            # 标记超限点（收益率 < VaR 即为超限）
            exceed_mask = rolling_returns < rolling_var
            ax.scatter(dates[exceed_mask], rolling_returns[exceed_mask],
                       color="red", s=25, label="超限点", zorder=5)  # zorder让点在最上层

            # 图表样式优化
            ax.set_title(f"滚动VaR曲线（{self.backtest_window}日回测窗口）", fontsize=14, pad=15)
            ax.set_xlabel("日期", fontsize=12)
            ax.set_ylabel("收益率 / VaR", fontsize=12)
            ax.legend(loc="upper right", fontsize=10)
            ax.grid(alpha=0.3, linestyle="--")  # 透明网格
            plt.xticks(rotation=45)  # 日期旋转，避免重叠
            plt.tight_layout()  # 自动调整布局

            # 保存图片
            curve_path = os.path.join(plot_dir, "rolling_var_curve.png")
            plt.savefig(curve_path, bbox_inches="tight")  # bbox_inches避免内容被裁剪
            plt.close(fig)  # 释放内存
            logging.info(f"✅ 滚动VaR曲线已保存：{os.path.abspath(curve_path)}")
        except Exception as e:
            logging.error(f"❌ 滚动VaR曲线保存失败：{str(e)}", exc_info=True)

        # ========== 2. 绘制超限次数分布饼图 ==========
        try:
            # 统计超限/未超限次数
            exceed_count = exceed_mask.sum()
            normal_count = len(exceed_mask) - exceed_count
            logging.info(f"超限统计：超限{exceed_count}次，未超限{normal_count}次")

            # 绘制饼图
            fig, ax = plt.subplots(figsize=(8, 8))
            labels = ["超限", "未超限"]
            sizes = [exceed_count, normal_count]
            colors = ["#d62728", "#2ca02c"]
            explode = (0.05, 0)  # 让超限部分突出

            # 绘制饼图（显示百分比+数值）
            wedges, texts, autotexts = ax.pie(
                sizes, labels=labels, colors=colors, explode=explode,
                autopct=lambda p: f"{p:.1f}%\n({int(p / 100 * sum(sizes))}次)",
                startangle=90, textprops={"fontsize": 11}
            )
            ax.set_title("VaR超限次数分布", fontsize=14, pad=15)
            plt.tight_layout()

            # 保存图片
            pie_path = os.path.join(plot_dir, "exceed_distribution.png")
            plt.savefig(pie_path, bbox_inches="tight")
            plt.close(fig)
            logging.info(f"✅ 超限分布饼图已保存：{os.path.abspath(pie_path)}")
        except Exception as e:
            logging.error(f"❌ 超限分布饼图保存失败：{str(e)}", exc_info=True)

        logging.info("📊 所有图表导出流程执行完毕！")

# ---------------------- 关键：添加测试代码（实例化+调用方法） ----------------------
# 模拟PortfolioManager类（提供组合收益率数据）
class MockPortfolioManager:
    def __init__(self):
        # 生成模拟的组合收益率数据（1年交易日约250条，方便测试）
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='B')  # 工作日
        returns = np.random.normal(loc=0.0005, scale=0.02, size=len(dates))  # 模拟收益率
        self.portfolio_returns = pd.Series(returns, index=dates, name='portfolio_returns')

# 模拟RiskCalculator类（实现calculate_var_es_single方法）
class MockRiskCalculator:
    def _calculate_var_es_single(self, returns: pd.Series, confidence_level: int) -> tuple:
        """模拟计算VaR和ES（简单分位数法）"""
        var = np.percentile(returns, 100 - confidence_level)  # VaR：分位数
        es = returns[returns <= var].mean()  # ES：超限收益的均值
        return var, es

# 主执行逻辑
if __name__ == "__main__":
    # 1. 创建模拟的依赖实例
    portfolio_manager = MockPortfolioManager()
    risk_calculator = MockRiskCalculator()

    # 2. 实例化回测类
    backtest = ModelBacktest(
        portfolio_manager=portfolio_manager,
        risk_calculator=risk_calculator,
        backtest_window=60,  # 60日回测窗口
        pass_threshold=0.06  # 超限率阈值6%
    )

    # 3. 运行回测（必须先运行这个，才有数据生成图表）
    backtest.run_backtest(confidence_level=95)

    # 4. 导出可视化图表（核心：调用生成图表的方法）
    backtest.export_backtest_plots()

    # 打印回测报告
    print("\n" + backtest.backtest_report)