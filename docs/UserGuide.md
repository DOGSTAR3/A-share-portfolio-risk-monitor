环境安装→配置修改→启动系统→解读结果」
# A 股组合风控系统 - 环境配置与依赖安装指南
本文档提供运行该风控系统所需的 Python 环境配置、软件包安装全流程命令，兼容 Windows/Linux/macOS 系统。
1. 环境配置（推荐使用虚拟环境）
# 1.1 创建 Python 虚拟环境
虚拟环境可避免包版本冲突，建议使用 Python 3.8 及以上版本，

1.1.1进入终端

进入终端按下 Win + R 组合键，弹出「运行」窗口；
输入 cmd 回车 → 打开命令提示符（CMD）；
输入 powershell 回车 → 打开 PowerShell；，

1.1.2创建名为portfolio_risk_env的虚拟环境，复制下面的代码

python -m venv portfolio_risk_env

# 1.2 激活虚拟环境
不同操作系统激活命令不同，激活后终端前缀会显示(portfolio_risk_env)：
bash运行指令

系统-Windows (打开方式-CMD终端):
portfolio_risk_env\Scripts\activate.bat

系统-Windows (打开方式-PowerShell)：
.\portfolio_risk_env\Scripts\Activate.ps1

系统-Linux/macOS：
source portfolio_risk_env/bin/activate

# 2. 安装所需软件包
# 2.1 手动逐个安装（激活虚拟环境后执行）
bash 运行指令

pip install pandas==2.1.4 numpy==1.26.2 scipy==1.11.4
pip install tushare==1.2.89
pip install arch==6.2.0
pip install streamlit==1.28.2
pip install plotly==5.17.0
pip configparser==5.3.0
pip python-dotenv==1.0.0
pip kaleido

当下载不畅时用：
国内镜像源加速
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pandas==2.1.4 numpy==1.26.2 scipy==1.11.4 tushare==1.2.89 arch==6.2.0 streamlit==1.28.2 plotly==5.17.0 configparser==5.3.0 python-dotenv==1.0.0 kaleido

# 2.2安装包功能解释：
pandas、numpy、scipy核心数据处理包，
TushareA股数据接口，
GARCH波动率模型包，
Streamlit可视化仪表盘，
Plotly交互式图表

# 2.3. 验证安装
安装完成后，执行以下命令检查关键包是否安装成功：
bash 运行
Linux/macOS
pip list | grep -E "pandas|numpy|tushare|arch|streamlit|plotly|configparser|python-dotenv|kaleido"

Windows
pip list | findstr /i "pandas numpy tushare arch streamlit plotly configparser python-dotenv kaleido"

若输出包含对应包名及版本号，说明安装成功。 

关键依赖包说明（注意版本，安装其他版本可能出现版本不兼容的报错，若出现可以针对性修改）
包名          	版本号	核心作用
pandas	        2.1.4	数据清洗、收益率计算、时间序列处理
numpy	        1.26.2	矩阵运算、风险贡献度 / 波动率计算
scipy	        1.11.4	统计分布（正态分布 z 分数）、分位数计算
tushare	        1.2.89	获取 A 股日线数据、复权因子、市值等核心数据
arch	        6.2.0	拟合 GARCH (1,1) 模型，计算时变波动率
streamlit	    1.28.2	搭建可视化仪表盘（系统启动入口）
plotly	        5.17.0	绘制风险贡献度等交互式图表
configparser	5.3.0	解析和操作 INI 格式的配置文件，支持配置项的读取、写入、修改与验证
python-dotenv	1.0.0	从.env 文件加载环境变量，隔离项目敏感配置（如密钥、数据库连接信息）
kaleido         0.2.1

# 3.启动风控系统

环境配置完成后，确保在项目根目录的虚拟环境内执行以下命令启动 Streamlit 仪表盘：
bash
运行

streamlit run src/dashboard.py

执行后终端会输出本地访问地址（如http://localhost:8501），浏览器打开即可使用。

# 总结
环境隔离：优先使用虚拟环境管理依赖，避免与系统 Python 环境冲突；
版本固定：指定包的具体版本，确保代码在不同环境下运行一致；
安装效率：国内用户建议使用清华镜像源加速包下载；
启动验证：安装完成后通过streamlit run dashboard.py验证环境是否配置成功。