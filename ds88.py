"""
🤖 优化版加密货币交易机器人
包含多时间框架分析、精确风险控制、详细退出计划
"""

import os  # 操作系统接口模块，用于环境变量读取
import time  # 时间控制模块，用于延迟和计时
import schedule  # 任务调度模块，定时执行交易逻辑
from openai import OpenAI  # OpenAI客户端，连接DeepSeek API
import ccxt  # 加密货币交易所统一接口库
import pandas as pd  # 数据分析库，处理价格时间序列
from datetime import datetime, timedelta  # 日期时间处理
import json  # JSON数据序列化和反序列化
import re  # 正则表达式，用于字符串匹配
from dotenv import load_dotenv  # 环境变量加载器
import traceback  # 异常堆栈追踪
import logging  # 日志系统
from typing import Dict, Optional, List, Tuple  # 类型注解
from collections import defaultdict  # 默认字典，自动初始化
import sqlite3  # SQLite数据库，存储交易记录
import numpy as np  # 数值计算库

# ==================== 🎨 日志配置 ====================
logging.basicConfig(
    level=logging.INFO,  # 日志级别：信息级
    format='%(asctime)s - %(levelname)s - %(message)s',  # 时间-级别-消息格式
    handlers=[
        logging.FileHandler('enhanced_trading.log', encoding='utf-8'),  # 文件输出
        logging.StreamHandler()  # 控制台输出
    ]
)
logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

load_dotenv()  # 加载.env文件中的环境变量（API密钥等敏感信息）

# ==================== 🔧 初始化DeepSeek AI客户端 ====================
try:
    deepseek_client = OpenAI(
        api_key=os.getenv('DEEPSEEK_API_KEY'),  # 从环境变量获取API密钥
        base_url="https://api.deepseek.com",  # DeepSeek API服务地址
        timeout=30  # 请求超时30秒
    )
    logger.info("✅ DeepSeek AI客户端初始化成功")
except Exception as e:
    logger.error(f"❌ DeepSeek客户端初始化失败: {e}")
    deepseek_client = None  # 失败则设为None

# ==================== 💱 初始化OKX交易所连接 ====================
try:
    exchange = ccxt.okx({
        'options': {
            'defaultType': 'swap',  # 交易类型：永续合约
            'adjustForTimeDifference': True,  # 自动调整服务器时间差
            'recvWindow': 60000,  # 接收窗口60秒
        },
        'apiKey': os.getenv('OKX_API_KEY'),  # OKX API密钥
        'secret': os.getenv('OKX_SECRET'),  # OKX API密钥
        'password': os.getenv('OKX_PASSWORD'),  # OKX API密码
        'timeout': 30000,  # 超时30秒
        'enableRateLimit': True,  # 启用API速率限制保护
    })
    logger.info("✅ OKX交易所连接成功")
except Exception as e:
    logger.error(f"❌ OKX交易所初始化失败: {e}")
    exchange = None

# ==================== ⚙️ 交易配置参数 ====================
TRADE_CONFIG = {
    'target_coins': [  # 目标交易币种列表
        'BTC/USDT:USDT',  # 比特币永续合约
        'ETH/USDT:USDT',  # 以太坊永续合约
        'SOL/USDT:USDT',  # Solana永续合约
        'BNB/USDT:USDT',  # 币安币永续合约
        'XRP/USDT:USDT',  # 瑞波币永续合约
        'DOGE/USDT:USDT'  # 狗狗币永续合约
    ],
    'base_amount': 0.001,  # 基础交易数量（币的数量）
    'min_leverage': 10,  # 最小杠杆倍数（降低风险）
    'max_leverage': 20,  # 最大杠杆倍数（从20降至10）
    'max_margin_ratio': 0.9,  # 最大保证金使用比例50%（从90%降低）
    'timeframes': {  # 多时间框架配置
        'short': '3m',  # 短期：3分钟（快速反应）
        'medium': '15m',  # 中期：15分钟（主要交易周期）
        'long': '4h'  # 长期：4小时（趋势确认）
    },
    'test_mode': True,  # 测试模式开关（True=模拟，False=实盘）
    'data_points': {  # 各时间框架数据点数
        'short': 100,  # 3分钟K线获取100根
        'medium': 100,  # 15分钟K线获取100根
        'long': 50  # 4小时K线获取50根
    },
    'max_retries': 3,  # API调用失败最大重试次数
    'retry_delay': 2,  # 重试间隔秒数
    'risk_management': {  # 风险管理参数
        'max_daily_loss': 0.05,  # 最大日亏损5%（从5%降低）
        'max_single_loss': 0.03,  # 最大单笔亏损3%
        'max_consecutive_losses': 3,  # 最大连续亏损次数
        'min_confidence': 0.65,  # 最低信心度要求65%
        'risk_reward_ratio': 2.0,  # 风险回报比至少1:2
    },
    'monitoring': {  # 监控配置
        'profit_alert': 0.05,  # 盈利5%时告警
        'loss_alert': 0.05,  # 亏损5%时告警
        'update_interval': 180,  # 更新间隔3分钟
    }
}

# ==================== 📊 全局变量 ====================
price_history = defaultdict(lambda: defaultdict(list))  # 价格历史：{币种: {时间框架: [数据]}}
signal_history = defaultdict(list)  # 交易信号历史
current_positions = {}  # 当前所有持仓：{币种: 持仓信息}
daily_pnl = 0.0  # 当日盈亏
trade_count = 0  # 今日交易次数
consecutive_losses = 0  # 连续亏损次数
selected_coin = None  # 当前选中的交易币种
start_time = datetime.now()  # 机器人启动时间
invocation_count = 0  # 调用次数计数器

# ==================== 💾 数据库初始化 ====================
class TradeDatabase:
    """
    交易数据库类
    用于持久化存储所有交易记录和账户状态
    """
    def __init__(self, db_path='trades.db'):
        """初始化数据库连接"""
        self.conn = sqlite3.connect(db_path, check_same_thread=False)  # 允许多线程访问
        self.create_tables()  # 创建必要的数据表
        logger.info("💾 交易数据库初始化成功")
    
    def create_tables(self):
        """创建数据库表结构"""
        # 交易记录表
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL,
                exit_price REAL,
                quantity REAL,
                leverage INTEGER,
                pnl REAL,
                pnl_percent REAL,
                confidence REAL,
                reason TEXT,
                stop_loss REAL,
                take_profit REAL,
                liquidation_price REAL,
                status TEXT
            )
        ''')
        
        # 账户状态表
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS account_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                total_value REAL,
                available_cash REAL,
                total_return_percent REAL,
                sharpe_ratio REAL,
                trade_count INTEGER,
                win_count INTEGER,
                loss_count INTEGER
            )
        ''')
        
        self.conn.commit()  # 提交事务
    
    def log_trade(self, trade_data):
        """记录交易到数据库"""
        try:
            self.conn.execute('''
                INSERT INTO trades (timestamp, symbol, side, entry_price, exit_price, 
                                  quantity, leverage, pnl, pnl_percent, confidence, 
                                  reason, stop_loss, take_profit, liquidation_price, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                trade_data.get('symbol'),
                trade_data.get('side'),
                trade_data.get('entry_price'),
                trade_data.get('exit_price'),
                trade_data.get('quantity'),
                trade_data.get('leverage'),
                trade_data.get('pnl'),
                trade_data.get('pnl_percent'),
                trade_data.get('confidence'),
                trade_data.get('reason'),
                trade_data.get('stop_loss'),
                trade_data.get('take_profit'),
                trade_data.get('liquidation_price'),
                trade_data.get('status', 'open')
            ))
            self.conn.commit()
            logger.info(f"📝 交易记录已保存: {trade_data.get('symbol')}")
        except Exception as e:
            logger.error(f"❌ 保存交易记录失败: {e}")
    
    def get_statistics(self):
        """获取交易统计数据"""
        try:
            cursor = self.conn.execute('''
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses,
                    AVG(pnl) as avg_pnl,
                    MAX(pnl) as max_win,
                    MIN(pnl) as max_loss,
                    SUM(pnl) as total_pnl
                FROM trades
                WHERE status = 'closed'
            ''')
            stats = cursor.fetchone()
            return {
                'total_trades': stats[0] or 0,
                'wins': stats[1] or 0,
                'losses': stats[2] or 0,
                'win_rate': (stats[1] / stats[0] * 100) if stats[0] > 0 else 0,
                'avg_pnl': stats[3] or 0,
                'max_win': stats[4] or 0,
                'max_loss': stats[5] or 0,
                'total_pnl': stats[6] or 0
            }
        except Exception as e:
            logger.error(f"❌ 获取统计数据失败: {e}")
            return {}

# 初始化数据库实例
db = TradeDatabase()

# ==================== 🔄 安全API调用函数 ====================
def safe_api_call(func, *args, **kwargs):
    """
    安全的API调用封装，带重试机制
    防止因临时网络问题导致程序崩溃
    """
    for attempt in range(TRADE_CONFIG['max_retries']):  # 循环重试
        try:
            result = func(*args, **kwargs)  # 执行API调用
            return result
        except ccxt.NetworkError as e:  # 网络错误（临时性）
            logger.warning(f"🌐 网络错误 (尝试 {attempt + 1}/{TRADE_CONFIG['max_retries']}): {e}")
            if attempt < TRADE_CONFIG['max_retries'] - 1:
                time.sleep(TRADE_CONFIG['retry_delay'] * (attempt + 1))  # 指数退避
            else:
                raise e
        except ccxt.ExchangeError as e:  # 交易所错误（可能是参数问题）
            logger.error(f"💱 交易所错误: {e}")
            raise e
        except Exception as e:  # 其他未知错误
            logger.error(f"❓ 未知错误: {e}")
            raise e
    return None

# ==================== 🔌 交易所设置与验证 ====================
def setup_exchange():
    """
    设置交易所并验证连接
    检查API密钥是否有效，余额是否可读取
    """
    if not exchange:
        logger.error("❌ 交易所未初始化")
        return False
        
    try:
        # 获取账户余额以验证API连接
        balance = safe_api_call(exchange.fetch_balance)
        if balance:
            usdt_balance = balance['USDT']['free'] if 'USDT' in balance else 0
            logger.info(f"💰 当前USDT余额: {usdt_balance:.2f}")
            logger.info(f"📊 账户总价值: {balance.get('total', {}).get('USDT', 0):.2f} USDT")
            return True
        return False
        
    except Exception as e:
        logger.error(f"❌ 交易所设置失败: {e}")
        return False

# ==================== 📈 技术指标计算函数（增强版）====================
def calculate_technical_indicators(df):
    """
    计算全套技术指标
    包括趋势、动量、波动性、成交量指标
    """
    try:
        # 1. 📊 移动平均线系统
        periods = [5, 10, 20, 50, 100, 200]  # 多周期MA
        for period in periods:
            if len(df) >= period:  # 确保有足够数据
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()  # 简单移动平均
                df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()  # 指数移动平均

        # 2. 📉 MACD指标（Moving Average Convergence Divergence）
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()  # 快线
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()  # 慢线
        df['macd'] = df['ema_12'] - df['ema_26']  # MACD值（DIF）
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()  # 信号线（DEA）
        df['macd_histogram'] = df['macd'] - df['macd_signal']  # 柱状图（MACD-Signal）

        # 3. 💪 RSI指标（Relative Strength Index）
        for period in [7, 14]:  # 计算7期和14期RSI
            delta = df['close'].diff()  # 价格变化
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()  # 上涨平均
            loss = -delta.where(delta < 0, 0).rolling(window=period).mean()  # 下跌平均
            rs = gain / (loss + 1e-10)  # 相对强度（避免除零）
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))  # RSI公式

        # 4. 🎚️ 布林带（Bollinger Bands）
        df['bb_middle'] = df['close'].rolling(20).mean()  # 中轨=20日均线
        bb_std = df['close'].rolling(20).std()  # 标准差
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)  # 上轨=中轨+2倍标准差
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)  # 下轨=中轨-2倍标准差
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']  # 带宽（波动性）
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)  # 价格在带内位置

        # 5. 📊 成交量指标
        df['volume_sma_20'] = df['volume'].rolling(20).mean()  # 成交量20日均线
        df['volume_ratio'] = df['volume'] / (df['volume_sma_20'] + 1e-10)  # 成交量比率（放量/缩量）
        df['obv'] = (df['volume'] * ((df['close'] - df['close'].shift(1)) / df['close'].shift(1).abs())).cumsum()  # OBV能量潮

        # 6. 🎯 动量指标
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1  # 5期动量
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1  # 10期动量
        df['roc'] = ((df['close'] - df['close'].shift(12)) / df['close'].shift(12)) * 100  # 变动率

        # 7. 📏 波动性指标
        df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()  # 历史波动率
        
        # 8. 🔥 ATR（Average True Range）平均真实波幅
        high_low = df['high'] - df['low']  # 当日最高最低差
        high_close = abs(df['high'] - df['close'].shift())  # 当日最高与昨收差
        low_close = abs(df['low'] - df['close'].shift())  # 当日最低与昨收差
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)  # 真实波幅
        df['atr'] = tr.rolling(14).mean()  # 14期ATR
        df['atr_percent'] = df['atr'] / df['close'] * 100  # ATR百分比

        # 9. 📍 支撑阻力位
        df['resistance'] = df['high'].rolling(20).max()  # 20期最高价作为阻力
        df['support'] = df['low'].rolling(20).min()  # 20期最低价作为支撑

        # 10. 🌊 ADX（Average Directional Index）趋势强度
        plus_dm = df['high'].diff()  # +DM
        minus_dm = -df['low'].diff()  # -DM
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        tr_14 = tr.rolling(14).sum()
        plus_di = 100 * (plus_dm.rolling(14).sum() / tr_14)  # +DI
        minus_di = 100 * (minus_dm.rolling(14).sum() / tr_14)  # -DI
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df['adx'] = dx.rolling(14).mean()  # ADX

        # 填充NaN值
        df = df.bfill().ffill()
        
        return df
        
    except Exception as e:
        logger.error(f"❌ 技术指标计算失败: {e}")
        traceback.print_exc()
        return df

# ==================== 🎯 多时间框架数据获取 ====================
def get_multi_timeframe_data(symbol):
    """
    获取多时间框架的K线数据
    参考通义模型：同时分析3分钟、15分钟、4小时数据
    """
    if not exchange:
        logger.error("❌ 交易所未初始化")
        return None
    
    try:
        multi_tf_data = {}
        
        # 遍历所有配置的时间框架
        for tf_name, tf_value in TRADE_CONFIG['timeframes'].items():
            try:
                # 获取K线数据
                ohlcv = safe_api_call(
                    exchange.fetch_ohlcv,
                    symbol,
                    tf_value,
                    limit=TRADE_CONFIG['data_points'].get(tf_name, 100)
                )
                
                if not ohlcv:
                    logger.warning(f"⚠️ {symbol} {tf_value}时间框架数据获取失败")
                    continue
                
                # 转换为DataFrame
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # 计算技术指标
                df = calculate_technical_indicators(df)
                
                # 保存到字典
                multi_tf_data[tf_name] = {
                    'timeframe': tf_value,
                    'dataframe': df,
                    'current': df.iloc[-1].to_dict(),  # 最新数据
                    'previous': df.iloc[-2].to_dict() if len(df) > 1 else None  # 前一根K线
                }
                
                # 保存到历史记录
                price_history[symbol][tf_name].append(multi_tf_data[tf_name])
                if len(price_history[symbol][tf_name]) > 100:
                    price_history[symbol][tf_name].pop(0)
                
                logger.info(f"📊 {symbol} {tf_value} 数据获取成功 ({len(df)}根K线)")
                time.sleep(0.5)  # 避免API限流
                
            except Exception as e:
                logger.error(f"❌ 获取{symbol} {tf_value}数据失败: {e}")
                continue
        
        if not multi_tf_data:
            return None
        
        # 构建综合数据对象
        result = {
            'symbol': symbol,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'timeframes': multi_tf_data,
            'current_price': multi_tf_data['short']['current']['close'],  # 使用短期时间框架的当前价格
        }
        
        return result
        
    except Exception as e:
        logger.error(f"❌ 获取{symbol}多时间框架数据失败: {e}")
        traceback.print_exc()
        return None

# ==================== 📊 市场趋势分析（多时间框架）====================
def analyze_market_trend(multi_tf_data):
    """
    多时间框架趋势分析
    参考通义模型的多维度分析方法
    """
    try:
        trend_analysis = {}
        
        for tf_name, tf_data in multi_tf_data['timeframes'].items():
            current = tf_data['current']
            df = tf_data['dataframe']
            
            # 趋势判断
            price = current['close']
            sma_20 = current.get('sma_20', price)
            sma_50 = current.get('sma_50', price)
            ema_20 = current.get('ema_20', price)
            
            # MACD趋势
            macd = current.get('macd', 0)
            macd_signal = current.get('macd_signal', 0)
            macd_hist = current.get('macd_histogram', 0)
            
            # RSI
            rsi_7 = current.get('rsi_7', 50)
            rsi_14 = current.get('rsi_14', 50)
            
            # ADX趋势强度
            adx = current.get('adx', 0)
            
            # 综合趋势评分（-100到+100）
            trend_score = 0
            
            # 价格与均线关系（±30分）
            if price > sma_20: trend_score += 10
            if price > sma_50: trend_score += 10
            if price > ema_20: trend_score += 10
            
            if price < sma_20: trend_score -= 10
            if price < sma_50: trend_score -= 10
            if price < ema_20: trend_score -= 10
            
            # MACD（±20分）
            if macd > macd_signal: trend_score += 10
            if macd_hist > 0: trend_score += 10
            
            if macd < macd_signal: trend_score -= 10
            if macd_hist < 0: trend_score -= 10
            
            # RSI（±20分）
            if 40 <= rsi_14 <= 60:  # 中性区间
                trend_score += 10
            elif rsi_14 > 70:  # 超买
                trend_score -= 10
            elif rsi_14 < 30:  # 超卖
                trend_score += 10 # 超卖反而是买入机会
            
            # 趋势强度判断
            if adx > 25:
                trend_strength = "STRONG"
                if trend_score > 0:
                    trend_direction = "BULLISH"
                else:
                    trend_direction = "BEARISH"
            elif adx > 20:
                trend_strength = "MEDIUM"
                trend_direction = "NEUTRAL"
            else:
                trend_strength = "WEAK"
                trend_direction = "RANGING"
            
            trend_analysis[tf_name] = {
                'timeframe': tf_data['timeframe'],
                'trend_score': trend_score,
                'trend_strength': trend_strength,
                'trend_direction': trend_direction,
                'price': price,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'macd': macd,
                'macd_signal': macd_signal,
                'rsi_7': rsi_7,
                'rsi_14': rsi_14,
                'adx': adx,
                'volatility': current.get('volatility', 0),
                'atr_percent': current.get('atr_percent', 0)
            }
        
        # 计算多时间框架一致性
        scores = [ta['trend_score'] for ta in trend_analysis.values()]
        avg_score = np.mean(scores)
        consistency = 1 - (np.std(scores) / 100)  # 标准差越小，一致性越高
        
        # 判断整体趋势
        if avg_score > 30 and consistency > 0.7:
            overall_trend = "STRONG_BULLISH"
            overall_confidence = min(0.95, 0.6 + consistency * 0.35)
        elif avg_score > 15:
            overall_trend = "BULLISH"
            overall_confidence = min(0.85, 0.5 + consistency * 0.35)
        elif avg_score < -30 and consistency > 0.7:
            overall_trend = "STRONG_BEARISH"
            overall_confidence = min(0.95, 0.6 + consistency * 0.35)
        elif avg_score < -15:
            overall_trend = "BEARISH"
            overall_confidence = min(0.85, 0.5 + consistency * 0.35)
        else:
            overall_trend = "NEUTRAL"
            overall_confidence = 0.3
        
        return {
            'by_timeframe': trend_analysis,
            'overall_trend': overall_trend,
            'overall_confidence': overall_confidence,
            'avg_trend_score': avg_score,
            'consistency': consistency
        }
        
    except Exception as e:
        logger.error(f"❌ 趋势分析失败: {e}")
        traceback.print_exc()
        return {}

# ==================== 🎲 币种评分系统 ====================
def calculate_coin_score(multi_tf_data):
    """
    计算币种交易评分
    综合多个维度给出0-100分的评分
    """
    try:
        score = 0
        trend_analysis = analyze_market_trend(multi_tf_data)
        
        if not trend_analysis:
            return 0
        
        # 1. 趋势一致性得分（0-30分）
        consistency = trend_analysis['consistency']
        score += consistency * 30
        
        # 2. 趋势强度得分（0-25分）
        avg_score = trend_analysis['avg_trend_score']
        score += min(25, abs(avg_score) / 4)
        
        # 3. 成交量得分（0-15分）
        short_data = multi_tf_data['timeframes']['short']['current']
        volume_ratio = short_data.get('volume_ratio', 1)
        if volume_ratio > 1.5:
            score += 15
        elif volume_ratio > 1.2:
            score += 10
        elif volume_ratio > 0.8:
            score += 5
        
        # 4. 波动率得分（0-15分）
        volatility = short_data.get('volatility', 0)
        atr_percent = short_data.get('atr_percent', 0)
        if 0.01 <= volatility <= 0.05 and 0.5 <= atr_percent <= 2.0:
            score += 15  # 理想波动率
        elif volatility > 0.08 or atr_percent > 3.0:
            score -= 10  # 波动率过高扣分
        else:
            score += 5
        
        # 5. RSI得分（0-15分）
        rsi_14 = short_data.get('rsi_14', 50)
        if 45 <= rsi_14 <= 55:
            score += 15  # 中性最佳
        elif 35 <= rsi_14 <= 65:
            score += 10
        elif rsi_14 > 75 or rsi_14 < 25:
            score -= 5  # 极端值扣分
        
        logger.info(f"📊 {multi_tf_data['symbol']} 综合评分: {score:.1f}/100")
        return max(0, min(100, score))  # 限制在0-100
        
    except Exception as e:
        logger.error(f"❌ 计算评分失败: {e}")
        return 0

# ==================== 🏆 选择最佳币种 ====================
def select_best_coin():
    """
    从目标币种中选择评分最高的
    """
    global selected_coin
    
    try:
        coin_scores = {}
        
        logger.info("🔍 开始分析所有币种...")
        
        for symbol in TRADE_CONFIG['target_coins']:
            try:
                logger.info(f"📈 分析 {symbol}...")
                multi_tf_data = get_multi_timeframe_data(symbol)
                
                if multi_tf_data:
                    score = calculate_coin_score(multi_tf_data)
                    coin_scores[symbol] = {
                        'score': score,
                        'data': multi_tf_data
                    }
                
                time.sleep(1)  # 避免API限流
                
            except Exception as e:
                logger.error(f"❌ 分析{symbol}失败: {e}")
                continue
        
        if not coin_scores:
            logger.warning("⚠️ 没有找到可交易的币种")
            selected_coin = TRADE_CONFIG['target_coins'][0]
            return None
        
        # 按评分排序
        sorted_coins = sorted(coin_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        best_coin = sorted_coins[0]
        
        selected_coin = best_coin[0]
        logger.info(f"🏆 选择最佳币种: {selected_coin} (评分: {best_coin[1]['score']:.1f})")
        
        # 显示所有币种评分
        logger.info("📊 所有币种评分排名:")
        for i, (symbol, data) in enumerate(sorted_coins, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📍"
            logger.info(f"  {emoji} {symbol}: {data['score']:.1f}分")
        
        return best_coin[1]['data']
        
    except Exception as e:
        logger.error(f"❌ 选择币种失败: {e}")
        selected_coin = TRADE_CONFIG['target_coins'][0]
        return None

# ==================== ⚡ 动态杠杆计算 ====================
def calculate_dynamic_leverage(trend_analysis, current_price):
    """
    根据市场条件动态计算杠杆
    参考通义模型的风险控制
    """
    try:
        base_leverage = TRADE_CONFIG['min_leverage']
        max_leverage = TRADE_CONFIG['max_leverage']
        
        # 1. 趋势强度因子（0.3-1.0）
        confidence = trend_analysis.get('overall_confidence', 0.5)
        trend_factor = max(0.3, confidence)
        
        # 2. 波动率因子（0.4-1.0）
        short_tf = trend_analysis['by_timeframe']['short']
        volatility = short_tf.get('volatility', 0.02)
        atr_percent = short_tf.get('atr_percent', 1.0)
        
        if volatility < 0.02 and atr_percent < 1.0:
            vol_factor = 1.0  # 低波动率
        elif volatility < 0.04 and atr_percent < 1.5:
            vol_factor = 0.8
        elif volatility < 0.06 and atr_percent < 2.0:
            vol_factor = 0.6
        else:
            vol_factor = 0.4  # 高波动率降低杠杆
        
        # 3. RSI因子（0.5-1.0）
        rsi_14 = short_tf.get('rsi_14', 50)
        if 40 <= rsi_14 <= 60:
            rsi_factor = 1.0
        elif 30 <= rsi_14 <= 70:
            rsi_factor = 0.8
        else:
            rsi_factor = 0.5
        
        # 4. 连续亏损惩罚
        global consecutive_losses
        loss_penalty = max(0.3, 1 - (consecutive_losses * 0.2))
        
        # 计算最终杠杆
        leverage = base_leverage + (max_leverage - base_leverage) * trend_factor * vol_factor * rsi_factor * loss_penalty
        leverage = int(min(max(leverage, base_leverage), max_leverage))
        
        logger.info(f"⚡ 杠杆计算: {base_leverage}x 基础 * {trend_factor:.2f} 趋势 * {vol_factor:.2f} 波动 * {rsi_factor:.2f} RSI * {loss_penalty:.2f} 风控 = {leverage}x")
        
        return leverage
        
    except Exception as e:
        logger.error(f"❌ 杠杆计算失败: {e}")
        return TRADE_CONFIG['min_leverage']

# ==================== 💵 仓位大小计算 ====================
def calculate_position_size(price, confidence, leverage, available_balance, atr):
    """
    计算合理的仓位大小
    使用凯利公式的保守版本
    """
    try:
        # 1. 基于信心度的基础仓位
        base_position_usdt = available_balance * TRADE_CONFIG['max_margin_ratio']
        confidence_adjusted = base_position_usdt * confidence
        
        # 2. 基于ATR的风险调整
        # 止损距离 = 2倍ATR
        stop_distance = 2 * atr
        risk_per_unit = stop_distance
        
        # 最大风险金额 = 账户的1%
        max_risk_usdt = available_balance * TRADE_CONFIG['risk_management']['max_single_loss']
        
        # 3. 根据风险计算仓位
        max_position_by_risk = max_risk_usdt / risk_per_unit
        
        # 4. 取较小值
        position_usdt = min(confidence_adjusted, max_position_by_risk * price / leverage)
        
        # 5. 计算币的数量
        position_size = (position_usdt * leverage) / price
        
        # 6. 确保最小仓位
        min_size = TRADE_CONFIG['base_amount']
        position_size = max(position_size, min_size)
        
        logger.info(f"💵 仓位计算: 可用{available_balance:.2f} * 信心{confidence:.2f} * 杠杆{leverage}x = {position_size:.4f} (价值{position_usdt:.2f} USDT)")
        
        return position_size
        
    except Exception as e:
        logger.error(f"❌ 仓位计算失败: {e}")
        return TRADE_CONFIG['base_amount']

# ==================== 🎯 止损止盈计算 ====================
def calculate_stop_loss_take_profit(entry_price, side, atr, confidence):
    """
    计算止损和止盈价位
    参考通义模型的精确风险控制
    """
    try:
        # 止损距离 = 2倍ATR（动态）
        stop_distance = 2 * atr
        
        # 止盈距离 = 止损距离 * 风险回报比
        risk_reward = TRADE_CONFIG['risk_management']['risk_reward_ratio']
        take_profit_distance = stop_distance * risk_reward * (1 + confidence * 0.5)  # 高信心度更大目标
        
        if side == 'long':
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + take_profit_distance
        else:  # short
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - take_profit_distance
        
        logger.info(f"🎯 止损止盈: 入场{entry_price:.2f} -> 止损{stop_loss:.2f} ({stop_distance:.2f}) / 止盈{take_profit:.2f} ({take_profit_distance:.2f})")
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'stop_distance': stop_distance,
            'take_profit_distance': take_profit_distance
        }
        
    except Exception as e:
        logger.error(f"❌ 止损止盈计算失败: {e}")
        return {
            'stop_loss': entry_price * (0.98 if side == 'long' else 1.02),
            'take_profit': entry_price * (1.04 if side == 'long' else 0.96),
            'stop_distance': entry_price * 0.02,
            'take_profit_distance': entry_price * 0.04
        }

# ==================== 🤖 AI分析生成交易信号 ====================
def analyze_with_ai(multi_tf_data, trend_analysis, leverage):
    """
    使用DeepSeek AI分析市场
    参考通义模型的专业决策方式
    """
    if not deepseek_client:
        logger.warning("⚠️ AI客户端不可用")
        return create_fallback_signal(multi_tf_data)
    
    try:
        symbol = multi_tf_data['symbol']
        current_price = multi_tf_data['current_price']
        
        # 构建多时间框架数据文本
        tf_text = "【多时间框架分析】\n"
        for tf_name, tf_info in trend_analysis['by_timeframe'].items():
            tf_text += f"\n⏱️ {tf_info['timeframe']}时间框架:\n"
            tf_text += f"  - 当前价格: ${tf_info['price']:.2f}\n"
            tf_text += f"  - SMA20: ${tf_info['sma_20']:.2f}, SMA50: ${tf_info['sma_50']:.2f}\n"
            tf_text += f"  - MACD: {tf_info['macd']:.4f}, 信号线: {tf_info['macd_signal']:.4f}\n"
            tf_text += f"  - RSI(7): {tf_info['rsi_7']:.2f}, RSI(14): {tf_info['rsi_14']:.2f}\n"
            tf_text += f"  - ADX: {tf_info['adx']:.2f} ({tf_info['trend_strength']})\n"
            tf_text += f"  - 趋势评分: {tf_info['trend_score']} ({tf_info['trend_direction']})\n"
            tf_text += f"  - 波动率: {tf_info['volatility']:.2%}, ATR: {tf_info['atr_percent']:.2f}%\n"
        
        tf_text += f"\n🎯 综合判断:\n"
        tf_text += f"  - 整体趋势: {trend_analysis['overall_trend']}\n"
        tf_text += f"  - 信心度: {trend_analysis['overall_confidence']:.2%}\n"
        tf_text += f"  - 时间框架一致性: {trend_analysis['consistency']:.2%}\n"
        tf_text += f"  - 平均趋势得分: {trend_analysis['avg_trend_score']:.1f}\n"
        
        # 获取持仓信息
        current_pos = get_current_position(symbol)
        position_text = "无持仓"
        if current_pos:
            position_text = f"{current_pos['side']}仓 {current_pos['quantity']:.4f}, 入场{current_pos['entry_price']:.2f}, 未实现盈亏{current_pos['unrealized_pnl']:.2f} USDT"
        
        # 构建AI提示词（专业版）
        prompt = f"""你是一位经验丰富的量化交易分析师，请基于以下{symbol}的多时间框架数据进行专业分析：

{tf_text}

📊 当前市场状态:
- 当前价格: ${current_price:.2f}
- 分析时间: {multi_tf_data['timestamp']}
- 当前持仓: {position_text}
- 建议杠杆: {leverage}x
- 日盈亏: {daily_pnl:.2%}
- 今日交易: {trade_count}次
- 连续亏损: {consecutive_losses}次

🎯 分析要求:
1. 综合3分钟、15分钟、4小时三个时间框架的趋势
2. 评估多时间框架的一致性（一致性越高信号越可靠）
3. 给出BUY/SELL/HOLD信号（只有高确定性时才BUY/SELL）
4. 计算合理的止损止盈位（基于ATR）
5. 评估信号信心度（0.0-1.0，建议≥0.75才交易）
6. 说明详细理由（包括支撑阻力、成交量确认等）

⚠️ 风险控制原则:
- 最大单笔风险: {TRADE_CONFIG['risk_management']['max_single_loss']:.1%}
- 风险回报比要求: ≥{TRADE_CONFIG['risk_management']['risk_reward_ratio']}:1
- 低信心度时选择HOLD而非强行交易
- 多时间框架不一致时谨慎交易
- 极端RSI值(>80或<20)需要确认
- 高波动率时降低信心度

📋 输出格式（严格JSON）:
{{
    "signal": "BUY|SELL|HOLD",
    "confidence": 0.85,
    "entry_price": {current_price},
    "stop_loss": 具体价格,
    "take_profit": 具体价格,
    "invalidation_condition": "例如: 4小时收盘跌破105000",
    "reason": "详细分析理由",
    "risk_reward_ratio": 2.5,
    "timeframe_alignment": "HIGH|MEDIUM|LOW"
}}

请给出专业、理性的分析，优先保护资本。"""

        # 调用AI
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": "你是一位严格遵循风险管理的量化交易分析师。你基于数据和概率做决策，在不确定时选择观望，承认市场的不可预测性。你的首要目标是保护资本，其次才是盈利。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            timeout=30
        )
        
        result = response.choices[0].message.content
        logger.info(f"🤖 AI原始回复: {result[:200]}...")
        
        # 解析JSON
        start_idx = result.find('{')
        end_idx = result.rfind('}') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            json_str = result[start_idx:end_idx]
            signal_data = json.loads(json_str)
            
            # 验证必需字段
            required_fields = ['signal', 'confidence', 'stop_loss', 'take_profit', 'reason']
            if not all(field in signal_data for field in required_fields):
                logger.warning("⚠️ AI返回数据不完整，使用备用信号")
                return create_fallback_signal(multi_tf_data)
            
            # 确保数据类型正确
            signal_data['confidence'] = float(signal_data['confidence'])
            signal_data['stop_loss'] = float(signal_data['stop_loss'])
            signal_data['take_profit'] = float(signal_data['take_profit'])
            
            # 添加时间戳
            signal_data['timestamp'] = multi_tf_data['timestamp']
            signal_data['leverage'] = leverage
            signal_data['symbol'] = symbol
            
            # 保存到历史
            signal_history[symbol].append(signal_data)
            if len(signal_history[symbol]) > 50:
                signal_history[symbol].pop(0)
            
            logger.info(f"✅ AI分析完成: {signal_data['signal']} (信心度{signal_data['confidence']:.2%})")
            
            return signal_data
            
        else:
            logger.warning("⚠️ AI返回格式错误")
            return create_fallback_signal(multi_tf_data)
        
    except Exception as e:
        logger.error(f"❌ AI分析失败: {e}")
        traceback.print_exc()
        return create_fallback_signal(multi_tf_data)

# ==================== 🆘 备用信号生成 ====================
def create_fallback_signal(multi_tf_data):
    """
    当AI失败时的备用信号
    基于简单技术指标
    """
    try:
        short_data = multi_tf_data['timeframes']['short']['current']
        current_price = multi_tf_data['current_price']
        
        rsi_14 = short_data.get('rsi_14', 50)
        bb_position = short_data.get('bb_position', 0.5)
        macd = short_data.get('macd', 0)
        macd_signal = short_data.get('macd_signal', 0)
        
        # 简单规则
        if rsi_14 < 30 and bb_position < 0.2 and macd > macd_signal:
            signal = "BUY"
            reason = "备用信号: RSI超卖且MACD金叉"
        elif rsi_14 > 70 and bb_position > 0.8 and macd < macd_signal:
            signal = "SELL"
            reason = "备用信号: RSI超买且MACD死叉"
        else:
            signal = "HOLD"
            reason = "备用信号: 市场不明确，观望"
        
        return {
            "signal": signal,
            "confidence": 0.4,  # 备用信号信心度低
            "entry_price": current_price,
            "stop_loss": current_price * (0.98 if signal == "BUY" else 1.02),
            "take_profit": current_price * (1.04 if signal == "BUY" else 0.96),
            "reason": reason,
            "invalidation_condition": "价格突破关键支撑/阻力位",
            "is_fallback": True,
            "timeframe_alignment": "LOW"
        }
        
    except Exception as e:
        logger.error(f"❌ 备用信号生成失败: {e}")
        return {
            "signal": "HOLD",
            "confidence": 0.0,
            "reason": "系统错误，暂停交易"
        }

# ==================== 📍 获取当前持仓 ====================
def get_current_position(symbol=None):
    """
    获取指定币种的持仓
    如果symbol为None，返回所有持仓
    """
    if not exchange:
        return None
    
    try:
        positions = safe_api_call(exchange.fetch_positions)
        
        if not positions:
            return None
        
        result_positions = []
        
        for pos in positions:
            if pos['info']['instType'] == 'SWAP' and float(pos['contracts'] or 0) > 0:
                if symbol is None or pos['symbol'] == symbol:
                    position_info = {
                        'symbol': pos['symbol'],
                        'side': pos['side'],
                        'quantity': float(pos['contracts']),
                        'entry_price': float(pos['entryPrice']) if pos['entryPrice'] else 0,
                        'current_price': float(pos['markPrice']) if pos['markPrice'] else 0,
                        'unrealized_pnl': float(pos['unrealizedPnl']) if pos['unrealizedPnl'] else 0,
                        'leverage': float(pos['leverage']) if pos['leverage'] else 1,
                        'liquidation_price': float(pos['liquidationPrice']) if pos['liquidationPrice'] else 0,
                        'margin': float(pos['initialMargin']) if pos['initialMargin'] else 0
                    }
                    
                    if symbol:
                        return position_info
                    else:
                        result_positions.append(position_info)
        
        return result_positions if result_positions else None
        
    except Exception as e:
        logger.error(f"❌ 获取持仓失败: {e}")
        return None

# ==================== 🔴 平仓函数 ====================
def close_position(symbol, reason="手动平仓"):
    """
    平掉指定币种的持仓
    """
    try:
        position = get_current_position(symbol)
        
        if not position:
            logger.info(f"ℹ️ {symbol}无持仓需要平掉")
            return True
        
        # 平仓方向相反
        side = 'buy' if position['side'] == 'short' else 'sell'
        
        order = safe_api_call(
            exchange.create_market_order,
            symbol,
            side,
            position['quantity'],
            params={'reduceOnly': True}
        )
        
        logger.info(f"✅ {symbol}平仓成功: {position['side']} {position['quantity']:.4f} @ {position['current_price']:.2f}")
        logger.info(f"💰 平仓盈亏: {position['unrealized_pnl']:.2f} USDT")
        
        # 记录到数据库
        db.log_trade({
            'symbol': symbol,
            'side': position['side'],
            'entry_price': position['entry_price'],
            'exit_price': position['current_price'],
            'quantity': position['quantity'],
            'leverage': position['leverage'],
            'pnl': position['unrealized_pnl'],
            'pnl_percent': (position['unrealized_pnl'] / (position['entry_price'] * position['quantity'])) * 100,
            'confidence': 0,
            'reason': reason,
            'status': 'closed'
        })
        
        # 更新连续亏损计数
        global consecutive_losses
        if position['unrealized_pnl'] < 0:
            consecutive_losses += 1
        else:
            consecutive_losses = 0
        
        time.sleep(2)
        return True
        
    except Exception as e:
        logger.error(f"❌ {symbol}平仓失败: {e}")
        traceback.print_exc()
        return False

# ==================== 🚀 执行交易 ====================
def execute_trade(signal_data, multi_tf_data, leverage):
    """
    执行交易订单
    参考通义模型的精确执行
    """
    global trade_count
    
    if not exchange:
        logger.error("❌ 交易所未初始化")
        return False
    
    symbol = multi_tf_data['symbol']
    current_price = multi_tf_data['current_price']
    signal = signal_data['signal']
    confidence = signal_data['confidence']
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🎯 交易信号: {symbol} - {signal}")
    logger.info(f"💪 信心度: {confidence:.2%}")
    logger.info(f"📝 理由: {signal_data['reason']}")
    logger.info(f"⚡ 杠杆: {leverage}x")
    logger.info(f"🛑 止损: ${signal_data['stop_loss']:.2f}")
    logger.info(f"🎯 止盈: ${signal_data['take_profit']:.2f}")
    logger.info(f"❌ 失效条件: {signal_data.get('invalidation_condition', 'N/A')}")
    logger.info(f"{'='*60}\n")
    
    # 风险检查
    if not check_risk_management():
        logger.warning("⚠️ 风险检查未通过")
        return False
    
    # 信心度检查
    min_confidence = TRADE_CONFIG['risk_management']['min_confidence']
    if confidence < min_confidence:
        logger.warning(f"⚠️ 信心度{confidence:.2%}低于要求{min_confidence:.2%}，跳过交易")
        return False
    
    # HOLD信号
    if signal == "HOLD":
        logger.info("🤚 建议观望，不执行交易")
        return False
    
    # 测试模式
    if TRADE_CONFIG['test_mode']:
        logger.info("🧪 测试模式 - 仅模拟交易")
        return True
    
    try:
        # 获取账户余额
        balance = safe_api_call(exchange.fetch_balance)
        if not balance:
            logger.error("❌ 获取余额失败")
            return False
        
        available_usdt = balance['USDT']['free']
        logger.info(f"💰 可用余额: {available_usdt:.2f} USDT")
        
        # 计算仓位
        short_data = multi_tf_data['timeframes']['short']['current']
        atr = short_data.get('atr', current_price * 0.02)
        
        position_size = calculate_position_size(
            current_price,
            confidence,
            leverage,
            available_usdt,
            atr
        )
        
        # 设置杠杆
        try:
            safe_api_call(
                exchange.set_leverage,
                leverage,
                symbol,
                {'mgnMode': 'cross'}
            )
            logger.info(f"⚡ 杠杆设置成功: {leverage}x")
        except Exception as e:
            logger.warning(f"⚠️ 设置杠杆失败: {e}")
        
        # 检查是否有反向持仓需要平掉
        current_pos = get_current_position(symbol)
        if current_pos:
            if (signal == 'BUY' and current_pos['side'] == 'short') or \
               (signal == 'SELL' and current_pos['side'] == 'long'):
                logger.info(f"🔄 检测到反向持仓，先平仓...")
                close_position(symbol, "反向信号触发平仓")
                time.sleep(2)
            else:
                logger.info(f"ℹ️ 已有同向持仓，保持不变")
                return True
        
        # 执行开仓
        side = 'buy' if signal == 'BUY' else 'sell'
        
        logger.info(f"🚀 开始执行{side}订单...")
        
        order = safe_api_call(
            exchange.create_market_order,
            symbol,
            side,
            position_size
        )
        
        if not order:
            logger.error("❌ 订单执行失败")
            return False
        
        logger.info(f"✅ 订单执行成功!")
        logger.info(f"📊 订单ID: {order.get('id')}")
        logger.info(f"📦 数量: {position_size:.4f}")
        logger.info(f"💵 价格: ${current_price:.2f}")
        
        trade_count += 1
        
        # 设置止损止盈
        try:
            # 止损单
            sl_order = safe_api_call(
                exchange.create_order,
                symbol,
                'stop_market',
                'sell' if signal == 'BUY' else 'buy',
                position_size,
                None,
                {
                    'stopLossPrice': signal_data['stop_loss'],
                    'reduceOnly': True
                }
            )
            logger.info(f"🛑 止损单已设置: ${signal_data['stop_loss']:.2f}")
            
            # 止盈单
            tp_order = safe_api_call(
                exchange.create_order,
                symbol,
                'take_profit_market',
                'sell' if signal == 'BUY' else 'buy',
                position_size,
                None,
                {
                    'takeProfitPrice': signal_data['take_profit'],
                    'reduceOnly': True
                }
            )
            logger.info(f"🎯 止盈单已设置: ${signal_data['take_profit']:.2f}")
            
        except Exception as e:
            logger.warning(f"⚠️ 设置止损止盈失败: {e}")
        
        # 记录到数据库
        db.log_trade({
            'symbol': symbol,
            'side': signal,
            'entry_price': current_price,
            'quantity': position_size,
            'leverage': leverage,
            'confidence': confidence,
            'reason': signal_data['reason'],
            'stop_loss': signal_data['stop_loss'],
            'take_profit': signal_data['take_profit'],
            'liquidation_price': 0,  # 需要从持仓信息获取
            'status': 'open'
        })
        
        logger.info("✅ 交易执行完成!")
        return True
        
    except Exception as e:
        logger.error(f"❌ 交易执行失败: {e}")
        traceback.print_exc()
        return False

# ==================== 🛡️ 风险管理检查 ====================
def check_risk_management():
    """
    风险管理检查
    """
    global daily_pnl, trade_count, consecutive_losses
    
    # 1. 日亏损限制
    if daily_pnl < -TRADE_CONFIG['risk_management']['max_daily_loss']:
        logger.warning(f"⚠️ 已达到日亏损限制: {daily_pnl:.2%}")
        return False
    
    # 2. 交易次数限制
    if trade_count >= 96:  # 每15分钟一次，最多96次/天
        logger.warning(f"⚠️ 已达到日交易次数限制: {trade_count}次")
        return False
    
    # 3. 连续亏损限制
    max_consecutive_losses = TRADE_CONFIG['risk_management']['max_consecutive_losses']
    if consecutive_losses >= max_consecutive_losses:
        logger.warning(f"⚠️ 连续亏损{consecutive_losses}次，达到限制")
        return False
    
    return True

# ==================== 📊 监控持仓 ====================
def monitor_positions():
    """
    监控所有持仓
    检查止损止盈触发、发送告警
    """
    try:
        positions = get_current_position()
        
        if not positions:
            return
        
        if not isinstance(positions, list):
            positions = [positions]
        
        for pos in positions:
            symbol = pos['symbol']
            unrealized_pnl = pos['unrealized_pnl']
            entry_price = pos['entry_price']
            current_price = pos['current_price']
            
            pnl_percent = (unrealized_pnl / (entry_price * pos['quantity'])) * 100
            
            # 显示持仓信息
            logger.info(f"\n{'='*60}")
            logger.info(f"📊 持仓监控: {symbol}")
            logger.info(f"📍 方向: {pos['side']}")
            logger.info(f"📦 数量: {pos['quantity']:.4f}")
            logger.info(f"💵 入场价: ${entry_price:.2f}")
            logger.info(f"📈 当前价: ${current_price:.2f}")
            logger.info(f"💰 未实现盈亏: {unrealized_pnl:.2f} USDT ({pnl_percent:+.2f}%)")
            logger.info(f"⚡ 杠杆: {pos['leverage']}x")
            logger.info(f"💀 清算价: ${pos['liquidation_price']:.2f}")
            logger.info(f"{'='*60}\n")
            
            # 盈利告警
            profit_alert = TRADE_CONFIG['monitoring']['profit_alert']
            if pnl_percent > profit_alert * 100:
                logger.info(f"🎉 {symbol}盈利达到{pnl_percent:.2f}%!")
                # 可以在这里添加Telegram通知等
            
            # 亏损告警
            loss_alert = TRADE_CONFIG['monitoring']['loss_alert']
            if pnl_percent < -loss_alert * 100:
                logger.warning(f"⚠️ {symbol}亏损达到{pnl_percent:.2f}%!")
                # 可以在这里添加Telegram通知等
            
            # 接近清算价告警
            distance_to_liq = abs(current_price - pos['liquidation_price']) / current_price
            if distance_to_liq < 0.1:  # 距离清算价小于10%
                logger.error(f"🚨 {symbol}接近清算价! 当前{current_price:.2f}, 清算{pos['liquidation_price']:.2f}")
        
    except Exception as e:
        logger.error(f"❌ 监控持仓失败: {e}")

# ==================== 📈 显示账户统计 ====================
def display_account_stats():
    """
    显示账户统计信息
    参考通义模型的详细报告
    """
    try:
        # 获取余额
        balance = safe_api_call(exchange.fetch_balance)
        if not balance:
            return
        
        total_value = balance.get('total', {}).get('USDT', 0)
        available_cash = balance.get('free', {}).get('USDT', 0)
        
        # 获取交易统计
        stats = db.get_statistics()
        
        # 计算运行时长
        runtime = datetime.now() - start_time
        runtime_minutes = int(runtime.total_seconds() / 60)
        
        # 显示统计
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 账户统计报告")
        logger.info(f"{'='*70}")
        logger.info(f"⏱️  运行时长: {runtime_minutes}分钟 (调用{invocation_count}次)")
        logger.info(f"💰 账户总值: ${total_value:.2f} USDT")
        logger.info(f"💵 可用资金: ${available_cash:.2f} USDT")
        logger.info(f"📈 总收益率: {((total_value - 10000) / 10000 * 100):+.2f}%")  # 假设初始10000
        logger.info(f"📊 今日交易: {trade_count}次")
        logger.info(f"💹 日盈亏: {daily_pnl:+.2%}")
        logger.info(f"📉 连续亏损: {consecutive_losses}次")
        logger.info(f"")
        logger.info(f"📜 历史统计:")
        logger.info(f"  🔢 总交易: {stats.get('total_trades', 0)}笔")
        logger.info(f"  ✅ 盈利: {stats.get('wins', 0)}笔")
        logger.info(f"  ❌ 亏损: {stats.get('losses', 0)}笔")
        logger.info(f"  🎯 胜率: {stats.get('win_rate', 0):.1f}%")
        logger.info(f"  💰 累计盈亏: ${stats.get('total_pnl', 0):.2f}")
        logger.info(f"  📊 平均盈亏: ${stats.get('avg_pnl', 0):.2f}")
        logger.info(f"  🏆 最大单笔盈利: ${stats.get('max_win', 0):.2f}")
        logger.info(f"  💔 最大单笔亏损: ${stats.get('max_loss', 0):.2f}")
        logger.info(f"{'='*70}\n")
        
    except Exception as e:
        logger.error(f"❌ 显示统计失败: {e}")

# ==================== 🤖 主交易循环 ====================
def trading_bot():
    """
    主交易机器人函数
    每次执行完整的交易流程
    """
    global invocation_count
    invocation_count += 1
    
    logger.info(f"\n{'🚀'*30}")
    logger.info(f"🤖 开始第{invocation_count}次交易循环")
    logger.info(f"⏰ 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'🚀'*30}\n")
    
    try:
        # 1. 监控现有持仓
        logger.info("👀 监控现有持仓...")
        monitor_positions()
        
        # 2. 显示账户统计
        if invocation_count % 10 == 0:  # 每10次显示一次
            display_account_stats()
        
        # 3. 选择最佳币种
        logger.info("🔍 选择最佳交易币种...")
        best_coin_data = select_best_coin()
        
        if not best_coin_data:
            logger.warning("⚠️ 未找到合适的币种")
            return
        
        # 4. 分析趋势
        logger.info(f"📊 分析{selected_coin}市场趋势...")
        trend_analysis = analyze_market_trend(best_coin_data)
        
        if not trend_analysis:
            logger.warning("⚠️ 趋势分析失败")
            return
        
        # 5. 计算杠杆
        leverage = calculate_dynamic_leverage(
            trend_analysis,
            best_coin_data['current_price']
        )
        
        # 6. AI分析生成信号
        logger.info("🤖 AI分析生成交易信号...")
        signal_data = analyze_with_ai(best_coin_data, trend_analysis, leverage)
        
        if signal_data.get('is_fallback'):
            logger.warning("⚠️ 使用备用信号")
        
        # 7. 执行交易
        logger.info("🚀 执行交易决策...")
        execute_trade(signal_data, best_coin_data, leverage)
        
        logger.info(f"\n✅ 第{invocation_count}次交易循环完成\n")
        
    except Exception as e:
        logger.error(f"❌ 交易循环异常: {e}")
        traceback.print_exc()

# ==================== 🔄 重置每日统计 ====================
def reset_daily_stats():
    """
    每日0点重置统计数据
    """
    global daily_pnl, trade_count, consecutive_losses
    
    logger.info("\n🌅 新的一天开始!")
    logger.info("🔄 重置每日统计数据...")
    
    # 保存昨日统计
    logger.info(f"📊 昨日总结:")
    logger.info(f"  📈 交易次数: {trade_count}")
    logger.info(f"  💰 日盈亏: {daily_pnl:+.2%}")
    logger.info(f"  📉 连续亏损: {consecutive_losses}")
    
    # 重置
    daily_pnl = 0.0
    trade_count = 0
    # consecutive_losses不重置，跨天保持
    
    logger.info("✅ 统计数据已重置\n")

# ==================== 🎬 主函数 ====================
def main():
    """
    主函数
    初始化并启动交易机器人
    """
    print("\n" + "="*70)
    print("🤖 优化版加密货币交易机器人")
    print("🎯 参考通义模型优秀策略")
    print("="*70 + "\n")
    
    logger.info("🚀 交易机器人启动中...")
    logger.info(f"📅 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"🎯 目标币种: {', '.join([c.split('/')[0] for c in TRADE_CONFIG['target_coins']])}")
    logger.info(f"⏱️  时间框架: 3分钟/15分钟/4小时")
    logger.info(f"⚡ 杠杆范围: {TRADE_CONFIG['min_leverage']}-{TRADE_CONFIG['max_leverage']}x")
    logger.info(f"🛡️  最大保证金使用: {TRADE_CONFIG['max_margin_ratio']:.0%}")
    
    if TRADE_CONFIG['test_mode']:
        logger.info("⚠️  测试模式: 仅模拟，不实际下单")
    else:
        logger.info("🚨 实盘模式: 真实交易，请谨慎!")
    
    # 初始化交易所
    if not setup_exchange():
        logger.error("❌ 交易所初始化失败，程序退出")
        return
    
    # 设置定时任务
    main_timeframe = TRADE_CONFIG['timeframes']['medium']
    
    if main_timeframe == '15m':
        schedule.every(15).minutes.do(trading_bot)
        logger.info("⏰ 执行频率: 每15分钟")
    elif main_timeframe == '5m':
        schedule.every(5).minutes.do(trading_bot)
        logger.info("⏰ 执行频率: 每5分钟")
    elif main_timeframe == '1h':
        schedule.every().hour.at(":01").do(trading_bot)
        logger.info("⏰ 执行频率: 每小时")
    else:
        schedule.every(15).minutes.do(trading_bot)
        logger.info("⏰ 执行频率: 每15分钟（默认）")
    
    # 每日重置
    schedule.every().day.at("00:00").do(reset_daily_stats)
    
    # 立即执行一次
    logger.info("\n🎬 立即执行首次分析...\n")
    trading_bot()
    
    logger.info("\n✅ 交易机器人已启动，进入循环监控模式...\n")
    
    # 主循环
    while True:
        try:
            schedule.run_pending()
            time.sleep(1)
        except KeyboardInterrupt:
            logger.info("\n👋 用户中断程序")
            logger.info("📊 最终统计:")
            display_account_stats()
            logger.info("🛑 程序已停止")
            break
        except Exception as e:
            logger.error(f"❌ 主循环异常: {e}")
            traceback.print_exc()
            time.sleep(10)

# ==================== 🔧 辅助工具函数 ====================

def get_funding_rate(symbol):
    """
    获取资金费率
    参考通义模型监控funding rate
    """
    try:
        if not exchange:
            return None
        
        # 获取资金费率
        funding = safe_api_call(exchange.fetch_funding_rate, symbol)
        
        if funding:
            return {
                'rate': float(funding.get('fundingRate', 0)),
                'next_funding_time': funding.get('fundingTimestamp'),
                'mark_price': float(funding.get('markPrice', 0))
            }
        
        return None
        
    except Exception as e:
        logger.error(f"❌ 获取资金费率失败: {e}")
        return None


def get_open_interest(symbol):
    """
    获取持仓量数据
    参考通义模型监控open interest
    """
    try:
        if not exchange:
            return None
        
        # 注意：不是所有交易所都支持此API
        oi = safe_api_call(exchange.fetch_open_interest, symbol)
        
        if oi:
            return {
                'open_interest': float(oi.get('openInterest', 0)),
                'symbol': symbol,
                'timestamp': oi.get('timestamp')
            }
        
        return None
        
    except Exception as e:
        logger.debug(f"获取持仓量失败（某些交易所不支持）: {e}")
        return None


def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """
    计算夏普比率
    参考通义模型的Sharpe Ratio监控
    """
    try:
        if len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns)
        
        # 年化夏普比率（假设每日交易）
        annualized_sharpe = sharpe * np.sqrt(252)
        
        return annualized_sharpe
        
    except Exception as e:
        logger.error(f"❌ 计算夏普比率失败: {e}")
        return 0.0


def format_trade_report(position):
    """
    格式化交易报告
    生成类似通义模型的详细报告
    """
    try:
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║                      📊 持仓详细报告                          ║
╠══════════════════════════════════════════════════════════════╣
║ 🪙 交易对: {position['symbol']:<50} ║
║ 📍 方向: {position['side'].upper():<52} ║
║ 📦 数量: {position['quantity']:<52.4f} ║
║ 💵 入场价: ${position['entry_price']:<50.2f} ║
║ 📈 当前价: ${position['current_price']:<50.2f} ║
║ ⚡ 杠杆: {position['leverage']:<52}x ║
║ 💀 清算价: ${position['liquidation_price']:<50.2f} ║
╠══════════════════════════════════════════════════════════════╣
║ 💰 未实现盈亏: ${position['unrealized_pnl']:<44.2f} ║
║ 📊 盈亏比例: {(position['unrealized_pnl'] / (position['entry_price'] * position['quantity']) * 100):>52.2f}% ║
║ 💵 持仓价值: ${(position['current_price'] * position['quantity']):<46.2f} ║
║ 🛡️ 保证金: ${position['margin']:<50.2f} ║
╠══════════════════════════════════════════════════════════════╣
║ 📏 距清算价: {(abs(position['current_price'] - position['liquidation_price']) / position['current_price'] * 100):>50.2f}% ║
╚══════════════════════════════════════════════════════════════╝
        """
        return report
        
    except Exception as e:
        logger.error(f"❌ 生成报告失败: {e}")
        return "报告生成失败"


def calculate_win_rate_from_history():
    """
    从信号历史计算胜率
    """
    try:
        all_signals = []
        for symbol_signals in signal_history.values():
            all_signals.extend(symbol_signals)
        
        if len(all_signals) < 10:
            return 0.5  # 默认50%
        
        # 简单统计最近的信号准确性
        # 这里需要根据实际价格走势判断信号是否正确
        # 简化版：假设高信心度的信号更可能正确
        
        high_conf_signals = [s for s in all_signals[-30:] if s.get('confidence', 0) > 0.8]
        
        if not high_conf_signals:
            return 0.5
        
        # 这是一个简化的估算
        avg_confidence = np.mean([s.get('confidence', 0.5) for s in high_conf_signals])
        
        return avg_confidence
        
    except Exception as e:
        logger.error(f"❌ 计算胜率失败: {e}")
        return 0.5


def get_market_sentiment(multi_tf_data):
    """
    综合市场情绪分析
    结合多个指标判断市场情绪
    """
    try:
        sentiment_score = 0
        
        for tf_name, tf_data in multi_tf_data['timeframes'].items():
            current = tf_data['current']
            
            # RSI情绪
            rsi = current.get('rsi_14', 50)
            if rsi > 70:
                sentiment_score -= 2  # 超买，看跌
            elif rsi < 30:
                sentiment_score += 2  # 超卖，看涨
            
            # MACD情绪
            macd_hist = current.get('macd_histogram', 0)
            if macd_hist > 0:
                sentiment_score += 1
            else:
                sentiment_score -= 1
            
            # 布林带情绪
            bb_position = current.get('bb_position', 0.5)
            if bb_position > 0.8:
                sentiment_score -= 1
            elif bb_position < 0.2:
                sentiment_score += 1
        
        # 归一化到-10到10
        sentiment_score = max(-10, min(10, sentiment_score))
        
        if sentiment_score > 5:
            sentiment = "极度看涨"
            emoji = "🚀🚀🚀"
        elif sentiment_score > 2:
            sentiment = "看涨"
            emoji = "📈"
        elif sentiment_score < -5:
            sentiment = "极度看跌"
            emoji = "📉📉📉"
        elif sentiment_score < -2:
            sentiment = "看跌"
            emoji = "📉"
        else:
            sentiment = "中性"
            emoji = "😐"
        
        return {
            'score': sentiment_score,
            'sentiment': sentiment,
            'emoji': emoji
        }
        
    except Exception as e:
        logger.error(f"❌ 市场情绪分析失败: {e}")
        return {'score': 0, 'sentiment': '未知', 'emoji': '❓'}


def emergency_close_all():
    """
    紧急平掉所有持仓
    用于风险控制或程序停止时
    """
    try:
        logger.warning("🚨 执行紧急平仓操作...")
        
        positions = get_current_position()
        
        if not positions:
            logger.info("✅ 没有持仓需要平掉")
            return True
        
        if not isinstance(positions, list):
            positions = [positions]
        
        for pos in positions:
            symbol = pos['symbol']
            logger.info(f"🔴 平掉 {symbol} 持仓...")
            close_position(symbol, "紧急平仓")
            time.sleep(1)
        
        logger.info("✅ 所有持仓已平掉")
        return True
        
    except Exception as e:
        logger.error(f"❌ 紧急平仓失败: {e}")
        return False


def backup_database():
    """
    备份数据库
    """
    try:
        import shutil
        from datetime import datetime
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = f"trades_backup_{timestamp}.db"
        
        shutil.copy2('trades.db', backup_file)
        
        logger.info(f"💾 数据库已备份到: {backup_file}")
        return True
        
    except Exception as e:
        logger.error(f"❌ 数据库备份失败: {e}")
        return False


def health_check():
    """
    系统健康检查
    """
    try:
        checks = {
            'exchange': False,
            'deepseek': False,
            'database': False,
            'balance': False
        }
        
        # 检查交易所连接
        if exchange:
            try:
                balance = safe_api_call(exchange.fetch_balance)
                if balance:
                    checks['exchange'] = True
                    checks['balance'] = True
            except:
                pass
        
        # 检查AI客户端
        if deepseek_client:
            checks['deepseek'] = True
        
        # 检查数据库
        if db:
            try:
                db.get_statistics()
                checks['database'] = True
            except:
                pass
        
        all_ok = all(checks.values())
        
        logger.info(f"\n{'='*60}")
        logger.info("🏥 系统健康检查")
        logger.info(f"{'='*60}")
        logger.info(f"💱 交易所连接: {'✅' if checks['exchange'] else '❌'}")
        logger.info(f"🤖 AI客户端: {'✅' if checks['deepseek'] else '❌'}")
        logger.info(f"💾 数据库: {'✅' if checks['database'] else '❌'}")
        logger.info(f"💰 余额查询: {'✅' if checks['balance'] else '❌'}")
        logger.info(f"{'='*60}")
        logger.info(f"综合状态: {'✅ 健康' if all_ok else '⚠️ 异常'}\n")
        
        return all_ok
        
    except Exception as e:
        logger.error(f"❌ 健康检查失败: {e}")
        return False


# ==================== 🎮 命令行交互功能（可选）====================

def interactive_mode():
    """
    交互式命令模式
    允许手动控制机器人
    """
    print("\n" + "="*60)
    print("🎮 交互式命令模式")
    print("="*60)
    print("命令列表:")
    print("  status  - 显示当前状态")
    print("  stats   - 显示统计数据")
    print("  close   - 平掉所有持仓")
    print("  backup  - 备份数据库")
    print("  health  - 健康检查")
    print("  trade   - 立即执行一次交易循环")
    print("  quit    - 退出程序")
    print("="*60 + "\n")
    
    while True:
        try:
            cmd = input("请输入命令 > ").strip().lower()
            
            if cmd == "status":
                monitor_positions()
            elif cmd == "stats":
                display_account_stats()
            elif cmd == "close":
                emergency_close_all()
            elif cmd == "backup":
                backup_database()
            elif cmd == "health":
                health_check()
            elif cmd == "trade":
                trading_bot()
            elif cmd == "quit":
                print("👋 退出交互模式")
                break
            else:
                print("❓ 未知命令，请重新输入")
                
        except KeyboardInterrupt:
            print("\n👋 退出交互模式")
            break
        except Exception as e:
            print(f"❌ 命令执行失败: {e}")


# ==================== 📱 可选的Telegram通知（需要配置）====================

def send_telegram_notification(message):
    """
    发送Telegram通知
    需要配置TELEGRAM_BOT_TOKEN和TELEGRAM_CHAT_ID
    """
    try:
        import requests
        
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if not bot_token or not chat_id:
            return False
        
        url = f"https://api.telegram.org/bot8070179098:AAELUecfJYow2vgfGsOSs-jQ15EBX48Zr1o/sendMessage"
        
        data = {
            "chat_id": DeepS2088Bot,
            "text": message,
            "parse_mode": "HTML"
        }
        
        response = requests.post(url, data=data, timeout=10)
        
        return response.status_code == 200
        
    except Exception as e:
        logger.debug(f"Telegram通知发送失败: {e}")
        return False


# ==================== 🎯 启动选项 ====================

if __name__ == "__main__":
    import sys
    
    try:
        # 检查启动参数
        if len(sys.argv) > 1:
            if sys.argv[1] == "--interactive" or sys.argv[1] == "-i":
                # 交互模式
                health_check()
                interactive_mode()
            elif sys.argv[1] == "--test" or sys.argv[1] == "-t":
                # 测试模式（单次运行）
                logger.info("🧪 测试模式：单次运行")
                TRADE_CONFIG['test_mode'] = True
                setup_exchange()
                trading_bot()
                display_account_stats()
            elif sys.argv[1] == "--health":
                # 仅健康检查
                health_check()
            elif sys.argv[1] == "--backup":
                # 仅备份数据库
                backup_database()
            elif sys.argv[1] == "--help" or sys.argv[1] == "-h":
                # 帮助信息
                print("""
🤖 优化版加密货币交易机器人 - 使用帮助

用法: python optimized_trading_bot.py [选项]

选项:
  (无参数)        正常启动交易机器人（循环模式）
  -i, --interactive  启动交互式命令模式
  -t, --test      测试模式（单次运行，不循环）
  --health        执行系统健康检查
  --backup        备份交易数据库
  -h, --help      显示此帮助信息

示例:
  python optimized_trading_bot.py              # 正常启动
  python optimized_trading_bot.py -i           # 交互模式
  python optimized_trading_bot.py --test       # 测试运行
  python optimized_trading_bot.py --health     # 健康检查

配置文件: .env
必需环境变量:
  - DEEPSEEK_API_KEY
  - OKX_API_KEY
  - OKX_SECRET
  - OKX_PASSWORD

可选环境变量:
  - TELEGRAM_BOT_TOKEN
  - TELEGRAM_CHAT_ID

注意事项:
  ⚠️  请先在测试模式下运行充分验证
  ⚠️  建议从小资金开始（$100-500）
  ⚠️  严格遵守风险管理规则
  ⚠️  定期备份数据库

祝交易顺利！💰
                """)
            else:
                print(f"❓ 未知参数: {sys.argv[1]}")
                print("使用 --help 查看帮助信息")
        else:
            # 正常启动（循环模式）
            main()
            
    except Exception as e:
        logger.error(f"❌ 程序异常: {e}")
        traceback.print_exc()
    finally:
        logger.info("👋 程序已退出，感谢使用！")