# 三级预警规则、预警等级判断、自动调仓算法（权重调整）
# 作者：DogStar·Quant
# 时间：20260101
import logging
import numpy as np
from typing import List, Tuple, Dict, Optional
from numpy.typing import NDArray


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
                continue
            var = self.risk_results[cl_str]['VaR']
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


class RiskDisposal:
    """风险处置：识别高/低风险标的、自动调仓"""
    def __init__(self, portfolio_manager, alert_level: str, high_risk_cut: float):
        self.portfolio_manager = portfolio_manager
        self.alert_level = alert_level
        self.high_risk_codes: List[str] = []
        self.low_risk_codes: List[str] = []
        self.original_weights: NDArray[np.float64] = portfolio_manager.original_weights
        self.new_weights: Optional[NDArray[np.float64]] = None
        self.HIGH_RISK_CUT = high_risk_cut  # 高风险标的权重削减比例

    def identify_risk_level(self, risk_contrib: Dict[str, float]) -> None:
        """按风险贡献度识别高/低风险标的"""
        contrib_vals = list(risk_contrib.values())
        median_contrib = np.median(contrib_vals)
        self.high_risk_codes = [
            code for code, val in risk_contrib.items() if abs(val) > abs(median_contrib)
        ]
        self.low_risk_codes = [
            code for code, val in risk_contrib.items() if abs(val) <= abs(median_contrib)
        ]
        logging.info(f"高风险标的：{self.high_risk_codes}，低风险标的：{self.low_risk_codes}")

    def adjust_weights(self) -> NDArray[np.float64]:
        """根据预警等级调整权重"""
        if self.alert_level not in ["red", "orange"]:
            logging.info(f"预警等级{self.alert_level}，无需调仓")
            self.new_weights = self.original_weights
            return self.new_weights

        # 初始化新权重
        new_weights = self.original_weights.copy()
        code2idx = {code: idx for idx, code in enumerate(self.portfolio_manager.codes)}

        # 削减高风险标的权重
        for code in self.high_risk_codes:
            idx = code2idx[code]
            cut_amount = float(new_weights[idx] * self.HIGH_RISK_CUT)
            new_weights[idx] -= cut_amount
            logging.info(
                f"削减{code}权重：{round(float(self.original_weights[idx]), 4)} → "
                f"{round(float(new_weights[idx]), 4)}（削减{round(cut_amount, 4)}）"
            )

        # 增加低风险标的权重（分配削减的总权重）
        total_cut = sum([
            float(self.original_weights[code2idx[code]] * self.HIGH_RISK_CUT)
            for code in self.high_risk_codes
        ])
        inc_per_low = total_cut / len(self.low_risk_codes) if len(self.low_risk_codes) > 0 else 0
        for code in self.low_risk_codes:
            idx = code2idx[code]
            new_weights[idx] += inc_per_low
            logging.info(
                f"增加{code}权重：{round(float(self.original_weights[idx]), 4)} → "
                f"{round(float(new_weights[idx]), 4)}（增加{round(inc_per_low, 4)}）"
            )

        # 权重归一化
        new_weights = new_weights / new_weights.sum()
        self.new_weights = new_weights

        # 输出调仓对比
        logging.info("=== 调仓前后权重对比 ===")
        for idx, code in enumerate(self.portfolio_manager.codes):
            logging.info(
                f"{code}：{round(float(self.original_weights[idx]), 4)} → {round(float(new_weights[idx]), 4)}"
            )

        return new_weights