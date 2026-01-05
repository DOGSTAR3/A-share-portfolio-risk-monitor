# A股跨行业组合风控全流程系统
## 项目简介
基于Python实现A股组合风控全流程：数据加载→组合构建→风险计量（VaR/ES/GARCH）→风险预警→自动调仓→VaR回测→监管报告→Streamlit可视化。

## 技术栈
Python 3.8+、pandas、tushare、streamlit、plotly、arch、scipy

## 核心功能
1. 数据加载：复权股价计算、3σ异常值裁剪；
2. 风险计量：VaR/ES（历史法/参数法/蒙特卡洛）、GARCH(1,1)时变波动率；
3. 风险预警：多置信度阈值预警、自动调仓建议；
4. 回测验证：VaR模型返回检验、超限率分析；
5. 可视化：Streamlit仪表盘（风险贡献度、调仓前后对比）。

## 运行步骤
1. 安装依赖：`pip install -r requirements.txt`
2. 配置环境变量：复制.env.example为.env，填写Tushare Token；
3. 启动仪表盘：`streamlit run src/main.py`。

## 技术亮点
- 风险贡献度计算：基于协方差矩阵的边际风险分解；
- VaR模型回测：滑动窗口验证模型有效性，支持超限率分析；
- 行业中性权重：解决跨行业组合的风险敞口失衡问题。