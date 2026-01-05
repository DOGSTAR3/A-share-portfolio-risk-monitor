#Streamlit 可视化仪表盘（整合所有模块，启动入口）
# 作者：DogStar·Quant
# 时间：20260101
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Tuple, Optional

class RiskDashboard:
    """Streamlit可视化仪表盘（整合所有模块，启动入口）"""

    def __init__(self, portfolio_manager, risk_calculator,
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
            'VaR(%)': [round(self.risk_calculator.risk_results[str(cl)]['VaR'], 4) for cl in [95, 99]],
            'ES(%)': [round(self.risk_calculator.risk_results[str(cl)]['ES'], 4) for cl in [95, 99]]
        })
        st.dataframe(risk_df, hide_index=True)

        # 3. 风险贡献度可视化
        st.subheader("🔥 个股风险贡献度（%）")
        var_95 = self.risk_calculator.risk_results['95']['VaR']
        risk_contrib = self.portfolio_manager.calculate_risk_contribution(var_95)
        contrib_df = pd.DataFrame({
            '股票名称': self.portfolio_manager.names,
            '行业': self.portfolio_manager.industries,
            '风险贡献度(%)': [round(risk_contrib[code], 4) for code in self.portfolio_manager.codes]
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
            st.warning("无有效压力测试数据（如2015股灾数据起始时间早于数据范围）")


# 监管报告生成器（附属功能）
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
                continue
            var = self.risk_results[cl_str]['VaR']
            if var is None or pd.isna(var):
                report.append(f"{cl}%置信度VaR：数据不足，无法验证")
                continue
            is_compliant = abs(var) <= self.reg_threshold
            report.append(f"{cl}%置信度VaR：{round(var, 4)}% → {'达标' if is_compliant else '超标'}")

        return "\n".join(report)


# 核心优化：适配新的配置加载逻辑，移除硬编码Token
def main(config):
    # 关键修复：所有内部模块导入添加 src. 前缀
    from data_loader import StockDataLoader
    from portfolio_manager import PortfolioManager
    from risk_calculator import RiskCalculator
    from alert_handler import RiskAlert, RiskDisposal
    from backtest import ModelBacktest
    import logging

    try:
        # 从统一配置字典读取TS_TOKEN（不再从ini读取）
        data_loader = StockDataLoader(config['TS_TOKEN'])
        for code in config['PORT_CODES']:
            data_loader.get_adj_stock_data(code, config['START_DATE'], config['END_DATE'])

        # 构建投资组合（全部使用config字典参数）
        portfolio_manager = PortfolioManager(
            data_loader=data_loader,
            codes=config['PORT_CODES'],
            names=config['PORT_NAMES'],
            industries=config['PORT_INDUSTRIES'],
            weight_method=config['WEIGHT_METHOD'],
            rolling_window=config['ROLLING_WINDOW']
        )

        # 风险计量
        risk_calculator = RiskCalculator(portfolio_manager)
        risk_results = risk_calculator.calculate_portfolio_var_es(
            confidence_levels=config['CONF_LEVELS'],
            method='historical'
        )
        risk_calculator.stress_test(config['STRESS_PERIODS'])

        # 模型回测
        backtest = ModelBacktest(
            portfolio_manager=portfolio_manager,
            risk_calculator=risk_calculator,
            backtest_window=config['BACKTEST_WINDOW'],
            pass_threshold=config['PASS_THRESHOLD']
        )
        backtest.run_backtest(confidence_level=95)

        # 风险预警
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

        # 风险处置
        disposal = RiskDisposal(
            portfolio_manager=portfolio_manager,
            alert_level=alert_level,
            high_risk_cut=config['HIGH_RISK_CUT']
        )
        risk_contrib = portfolio_manager.calculate_risk_contribution(risk_results['95']['VaR'])
        disposal.identify_risk_level(risk_contrib)
        new_weights = disposal.adjust_weights()
        portfolio_manager.update_weights(new_weights)

        # 生成监管报告
        if alert_level in ["red", "orange"]:
            risk_calculator_adjusted = RiskCalculator(portfolio_manager)
            risk_results_adjusted = risk_calculator_adjusted.calculate_portfolio_var_es(
                confidence_levels=config['CONF_LEVELS'],
                method='historical'
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

        # 启动仪表盘
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