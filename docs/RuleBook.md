# 风控系统规则手册
## 1. 配置规则
### 1.1 Tushare配置
- Token获取：https://tushare.pro/register，实名认证后获取免费版Token；

### 1.2 参数调整
参数调整请在创建的config.ini中修改

### 数据接口-[Tushare]
token = #替换为自己的接口
timeout = 30 #请求时间

### 组合参数-[Portfolio]
- 股票代码codes：支持沪深A股代码，多个用英文逗号分隔；
- 日期格式start_date：YYYYMMDD，
- 结束日期end_date：需≤当前日期；
- 权重类型weight_type：equal（等权）/market_cap（市值加权）/industry_neutral（行业中性）。

### 风险计量与回测配置-[Risk]
定义计算 VaR（风险价值）的置信水平
- var_confidence ：
这里设置为 95% 和 99%。
程序会基于这两个置信度分别计算组合的风险值，
对应代码中CONF_LEVELS变量，用于RiskCalculator类的风险计量

指定 VaR 的计算方法
- var_method ：
historical（历史模拟法）、
parametric（参数法）、
monte_carlo（蒙特卡洛模拟法）。
程序会根据该配置调用对应的算法计算风险，
对应RiskCalculator.calculate_portfolio_var_es方法的method参数

定义 VaR 模型回测的滑动窗口大小（单位：交易日）
- backtest_window：
这里设置为 252 天（约 1 年），
回测时会用过去 252 天的收益率数据计算 VaR，并验证其有效性，
对应代码中BACKTEST_WINDOW变量，用于ModelBacktest类

定义回测中 “超限率” 的合格阈值（5%）
- exceed_rate_threshold：
若实际收益率低于 VaR 的次数占比（超限率）≤5%，则认为模型有效，
对应代码中PASS_THRESHOLD变量，用于判断回测是否通过

### 风险预警与处置配置-[Alert]
定义不同风险预警等级的触发阈值
- yellow_threshold = 2.0 #当 VaR 绝对值＞2.0% 时触发黄色预警
- orange_threshold = 3.5 #＞3.5% 时触发橙色预警
- red_threshold = 5.0    #5.0% 时触发红色预警
对应代码中ALERT_Y95等变量，用于RiskAlert类评估预警等级

定义高风险标的的权重削减比例（50%）
- high_risk_cut_ratio
定义高风险标的的权重削减比例（50%）。
当触发橙色或红色预警时，程序会自动削减高风险标的的权重（削减当前权重的 50%），
并将释放的权重分配给低风险标的
对应代码中HIGH_RISK_CUT变量，用于RiskDisposal类的调仓逻辑


## 2. 风控核心逻辑
### 2.1 VaR/ES计算规则
- 历史模拟法：排序收益率取分位数，无需分布假设；
- 参数法：假设收益率正态分布，基于均值+波动率计算；
- 蒙特卡洛法：随机模拟10000次收益率，取分位数。
### 2.2 预警阈值规则
- 黄色预警：VaR绝对值≥2.0%且＜3.5%，常规关注；
- 橙色预警：VaR绝对值≥3.5%且＜5.0%，建议调仓；
- 红色预警：VaR绝对值≥5.0%，强制调仓。

## 3. 调仓规则
### 3.1 高/低风险标的识别
- 高风险：风险贡献度＞所有标的中位数；
- 低风险：风险贡献度≤所有标的中位数。
### 3.2 权重调整公式
调整后高风险权重 = 原权重 × (1 - 削减比例)
低风险标的新增权重 = 总削减权重 × (低风险标的原权重 / 所有低风险标的原权重之和)

## 4. 回测标准
- 回测窗口：252天（1个交易日年）；
- 超限率阈值：≤5%，超过则模型需校准；
- 校准流程：调整GARCH(p,q)阶数→重新计算波动率→再次回测。

## 5. 合规依据
| 监管条款                | 系统对应逻辑                     |
|-------------------------|----------------------------------|
| 净资本比率≤10%          | 合规报告中校验组合风险敞口占比   |
| 单一标的持仓≤5%         | 调仓规则中限制单标的权重上限     |

# 3代码逻辑
- StockDataLoader加载数据→
- PortfolioManager构建组合（计算权重 / 收益率）→
- RiskCalculator计量风险（VaR/ES/ 压力测试）→
- RiskAlert预警→RiskDisposal调仓→
- ModelBacktest回测→
- RiskDashboard展示

