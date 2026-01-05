# 工具函数（日志记录、数据格式转换、图表导出）
# 作者：DogStar·Quant
# 时间：20260101
import logging
import configparser
import os
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv  # 新增：引入python-dotenv

# --------------------------
# 关键修改1：初始化dotenv，加载.env文件
# --------------------------
# 优先加载.env文件，环境变量优先级高于.env内容
load_dotenv(override=True)

# 默认业务配置（仅保留非敏感配置，移除Tushare Token）
DEFAULT_BUSINESS_CONFIG: Dict[str, Dict[str, str]] = {
    'Portfolio': {
        'codes': '600519,300750,601318,000063,601668',
        'names': '贵州茅台,宁德时代,中国平安,中兴通讯,中国建筑',
        'industries': '消费,新能源,金融,科技,周期',
        'start_date': '20200101',
        'end_date': '20251231',
        'weight_method': 'equal'
    },
    'Calculation': {
        'confidence_levels': '95,99',
        'monte_carlo_times': '10000',
        'rolling_window': '60',
        'stress_periods': '2015_crash,2020_pandemic'
    },
    'Regulatory': {'net_capital_ratio': '10'},
    'Alert': {
        'yellow_95': '2.0',
        'orange_95': '3.0',
        'red_95': '4.0',
        'yellow_99': '3.5',
        'orange_99': '4.5',
        'red_99': '5.5'
    },
    'Disposal': {
        'high_risk_cut_ratio': '0.5',
        'low_risk_increase_ratio': '0.2'
    },
    'Backtest': {
        'backtest_window': '252',
        'pass_threshold': '0.05'
    }
}


# --------------------------
# 关键修改2：新增配置验证函数，提前暴露错误
# --------------------------
def validate_config(config: Dict[str, Any]) -> None:
    """验证配置合法性，避免运行时错误"""
    # 检查必填项
    required_keys = [
        'TS_TOKEN', 'PORT_CODES', 'START_DATE', 'END_DATE',
        'CONF_LEVELS', 'ROLLING_WINDOW', 'ALERT_Y95', 'BACKTEST_WINDOW'
    ]
    for key in required_keys:
        if key not in config or not config[key]:
            raise ValueError(f"配置项【{key}】缺失或为空，请检查.env/ini配置")

    # 检查类型/格式合法性
    if not isinstance(config['TS_TOKEN'], str) or len(config['TS_TOKEN']) < 10:
        raise ValueError("TS_TOKEN格式错误（需为有效Tushare Token，长度≥10）")
    if config['BACKTEST_WINDOW'] < 60:
        raise ValueError("回测窗口BACKTEST_WINDOW不能小于60个交易日")
    if not (0 < config['HIGH_RISK_CUT'] < 1):
        raise ValueError("高风险削减比例HIGH_RISK_CUT需在0-1之间（如0.5表示50%）")
    if len(config['PORT_CODES']) != len(config['PORT_NAMES']) or len(config['PORT_CODES']) != len(
            config['PORT_INDUSTRIES']):
        raise ValueError("股票代码、名称、行业列表长度必须一致")


# --------------------------
# 关键修改3：重构配置加载逻辑，分层管理.env+ini
# --------------------------
def load_config(ini_path: str = 'portfolio_risk_config.ini') -> Dict[str, Any]:
    """
    加载配置：
    - 敏感配置（TS_TOKEN）：来自.env/系统环境变量
    - 业务配置：来自ini文件 + 默认值
    - 加载优先级：系统环境变量 > .env > ini > 默认值
    """
    # 新增：确保配置文件所在目录存在（解决config/目录不存在导致的写入失败）
    ini_dir = os.path.dirname(ini_path)
    if ini_dir and not os.path.exists(ini_dir):
        os.makedirs(ini_dir, exist_ok=True)
        logging.info(f"配置目录 {ini_dir} 不存在，已自动创建")

    # 1. 从.env/环境变量读取敏感配置
    ts_token = os.getenv('TS_TOKEN', '')
    environment = os.getenv('ENVIRONMENT', 'dev')  # dev/prod
    log_level = os.getenv('LOG_LEVEL', 'INFO')

    # 2. 加载ini业务配置（使用configparser 5.3.0新特性）
    config = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation(),  # 支持配置插值
        strict=True,  # 严格模式，避免重复键/无效节
        empty_lines_in_values=False
    )

    # 读取或初始化ini文件
    if os.path.exists(ini_path):
        config.read(ini_path, encoding='utf-8')
        # 补充缺失的配置项
        for section, keys in DEFAULT_BUSINESS_CONFIG.items():
            if section not in config.sections():
                config[section] = keys
            else:
                for key, default_val in keys.items():
                    if key not in config[section]:
                        config[section][key] = default_val
        # 保存更新后的ini
        with open(ini_path, 'w', encoding='utf-8') as f:
            config.write(f)
    else:
        config.read_dict(DEFAULT_BUSINESS_CONFIG)
        with open(ini_path, 'w', encoding='utf-8') as f:
            config.write(f)

    # 3. 解析配置并转换类型
    parsed_config = {
        # 环境/敏感配置
        'TS_TOKEN': ts_token,
        'ENVIRONMENT': environment,
        'LOG_LEVEL': log_level,
        # 业务配置
        'PORT_CODES': [code.strip() for code in config['Portfolio']['codes'].split(',')],
        'PORT_NAMES': [name.strip() for name in config['Portfolio']['names'].split(',')],
        'PORT_INDUSTRIES': [ind.strip() for ind in config['Portfolio']['industries'].split(',')],
        'START_DATE': config['Portfolio']['start_date'],
        'END_DATE': config['Portfolio']['end_date'],
        'WEIGHT_METHOD': config['Portfolio']['weight_method'],
        'CONF_LEVELS': [int(cl) for cl in config['Calculation']['confidence_levels'].split(',')],
        'MC_TIMES': int(config['Calculation']['monte_carlo_times']),
        'ROLLING_WINDOW': int(config['Calculation']['rolling_window']),
        'STRESS_PERIODS': config['Calculation']['stress_periods'].split(','),
        'REG_THRESHOLD': float(config['Regulatory']['net_capital_ratio']),
        # 预警阈值
        'ALERT_Y95': float(config['Alert']['yellow_95']),
        'ALERT_O95': float(config['Alert']['orange_95']),
        'ALERT_R95': float(config['Alert']['red_95']),
        'ALERT_Y99': float(config['Alert']['yellow_99']),
        'ALERT_O99': float(config['Alert']['orange_99']),
        'ALERT_R99': float(config['Alert']['red_99']),
        # 处置参数
        'HIGH_RISK_CUT': float(config['Disposal']['high_risk_cut_ratio']),
        'LOW_RISK_INC': float(config['Disposal']['low_risk_increase_ratio']),
        # 回测参数
        'BACKTEST_WINDOW': int(config['Backtest']['backtest_window']),
        'PASS_THRESHOLD': float(config['Backtest']['pass_threshold'])
    }

    # 4. 配置验证
    validate_config(parsed_config)

    # 5. 环境差异化配置（生产环境收紧规则）
    if parsed_config['ENVIRONMENT'] == 'prod':
        parsed_config['ALERT_Y95'] = 1.5  # 黄色预警阈值从2%降至1.5%
        parsed_config['ALERT_O95'] = 2.5  # 橙色预警阈值从3%降至2.5%
        parsed_config['ALERT_R95'] = 3.5  # 红色预警阈值从4%降至3.5%
        parsed_config['BACKTEST_WINDOW'] = 504  # 回测窗口从252天增至504天
        parsed_config['LOG_LEVEL'] = 'WARNING'  # 生产环境日志级别降低

    return parsed_config


# --------------------------
# 关键修改4：动态日志配置（基于.env的LOG_LEVEL）
# --------------------------
def init_logger(config: Dict[str, Any]):
    """初始化日志系统，支持动态日志级别"""
    os.makedirs('logs', exist_ok=True)

    # 转换日志级别（字符串→logging常量）
    log_level = getattr(logging, config['LOG_LEVEL'].upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(
                f'logs/risk_system_{datetime.now().strftime("%Y%m%d")}.log',
                encoding='utf-8'
            ),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger('PortfolioRiskSystem')
    logger.info(f"日志系统初始化完成 | 运行环境：{config['ENVIRONMENT']} | 日志级别：{config['LOG_LEVEL']}")
    return logger


# 启动入口
if __name__ == "__main__":
    try:
        # 加载配置（自动整合.env和ini）
        config = load_config(ini_path='config/portfolio_risk_config.ini')
        # 初始化日志
        logger = init_logger(config)

        # 验证Token有效性
        if config['TS_TOKEN'] == '请替换为自己的Tushare Token' or not config['TS_TOKEN']:
            logger.error("❌ TS_TOKEN未配置！请在.env文件中设置有效Tushare Token")
        else:
            logger.info("✅ 配置加载完成，系统初始化成功")
            from dashboard import main as dashboard_main

            dashboard_main(config)
    except ValueError as e:
        logging.error(f"配置验证失败：{e}")
    except Exception as e:
        logging.error(f"系统启动失败：{e}", exc_info=True)