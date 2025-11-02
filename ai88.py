#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🚀 DeepSeek增强版合约交易机器人 - 完整优化版
🎯 多策略决策引擎 + 动态仓位管理 + 严格风控
💰 目标:每日波段高胜率收益
"""

import os
import time
import schedule
import ccxt
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import sqlite3
from dotenv import load_dotenv
import traceback
from openai import OpenAI
import signal
import sys

# ==================== 🎨 日志配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deepseek_trading.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

# ==================== 🤖 DeepSeek AI客户端 ====================
try:
    deepseek_client = OpenAI(
        api_key=os.getenv('DEEPSEEK_API_KEY'),
        base_url="https://api.deepseek.com",
        timeout=30
    )
    logger.info("✅ DeepSeek AI客户端初始化成功")
except Exception as e:
    logger.error(f"❌ DeepSeek客户端初始化失败: {e}")
    deepseek_client = None

# ==================== 💱 交易所配置 ====================
EXCHANGE_CONFIG = {
    'okx': {
        'class': ccxt.okx,
        'config': {
            'apiKey': os.getenv('OKX_API_KEY'),
            'secret': os.getenv('OKX_SECRET'),
            'password': os.getenv('OKX_PASSWORD'),
            'enableRateLimit': True,
            'options': {'defaultType': 'swap'}
        }
    },
    'binance': {
        'class': ccxt.binance,
        'config': {
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_SECRET'),
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        }
    }
}

def initialize_exchange(exchange_name='okx'):
    """初始化交易所连接"""
    try:
        config = EXCHANGE_CONFIG.get(exchange_name)
        if not config:
            raise ValueError(f"不支持的交易所: {exchange_name}")
        
        exchange = config['class'](config['config'])
        exchange.load_markets()
        
        # 测试连接
        balance = exchange.fetch_balance()
        logger.info(f"✅ {exchange_name.upper()}交易所连接成功")
        logger.info(f"💰 账户余额: {balance['total']['USDT']:.2f} USDT")
        return exchange
    except Exception as e:
        logger.error(f"❌ {exchange_name.upper()}交易所初始化失败: {e}")
        return None

# 全局交易所实例
exchange = initialize_exchange('okx')

# ==================== ⚙️ 核心交易配置 ====================
TRADE_CONFIG = {
    # 🎯 目标交易对配置
    'target_symbols': [
        'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT',
        'BNB/USDT:USDT', 'XRP/USDT:USDT', 'DOGE/USDT:USDT'
    ],
    
    # 🎯 AI决策模式配置
    'ai_decision_mode': {
        'enabled': True,  # 🚀 开启AI决策主导模式，减少人为策略干扰
        'min_confidence': 0.51,  # AI信号最低信心度要求
        'override_strategy': True,  # AI信号覆盖策略信号
        'risk_adjustment': True  # AI根据市场状态动态调整风险
    },
    
    # 🎯 多策略配置 (可根据风险偏好调整)
    'strategies': {
        'trend_following': {
            'enabled': True,
            'weight': 0.4,  # ⚖️ 权重调整: 激进0.5, 平衡0.4, 保守0.3
            'timeframes': ['1h', '15m', '5m'],
            'indicators': ['ema_20', 'ema_50', 'macd', 'adx']
        },
        'mean_reversion': {
            'enabled': True,
            'weight': 0.3,  # ⚖️ 权重调整: 激进0.2, 平衡0.3, 保守0.4
            'timeframes': ['15m', '5m'],
            'indicators': ['rsi_14', 'bollinger_bands', 'stoch_rsi']
        },
        'breakout': {
            'enabled': True,
            'weight': 0.3,  # ⚖️ 权重调整: 激进0.3, 平衡0.3, 保守0.3
            'timeframes': ['4h', '1h'],
            'indicators': ['support_resistance', 'volume_profile']
        }
    },
    
    # 🛡️ 严格风险控制 (关键参数 - 根据风险偏好调整)
    'risk_management': {
        # 💰 亏损控制
        'max_daily_loss': 0.08,  # ⚠️ 激进0.08, 平衡0.05, 保守0.03
        'max_single_loss': 0.03,  # ⚠️ 激进0.03, 平衡0.02, 保守0.01
        
        # 📊 仓位控制
        'max_total_position': 0.85,  # ⚖️ 激进1.0, 平衡0.85, 保守0.7
        'max_single_position': 0.35,  # ⚖️ 激进0.35, 平衡0.25, 保守0.15
        'max_open_positions': 3,  # 📦 激进5, 平衡3, 保守2
        
        # 🎯 止损止盈
        'risk_reward_ratio': 1.98,  # 📈 激进2.0, 平衡2.5, 保守3.0
        'stop_loss_atr_multiple': 1.5,  # 🛑 激进2.0, 平衡1.5, 保守1.0
        'take_profit_atr_multiple': 3.0,  # 🎯 激进3.0, 平衡3.5, 保守4.0
        
        # 📈 移动止损
        'trailing_stop_enabled': True,
        'trailing_stop_activation': 0.02,  # 🎯 盈利激活点: 激进0.015, 平衡0.02, 保守0.025
        'trailing_stop_distance': 0.01  # 📏 跟踪距离: 激进0.015, 平衡0.01, 保守0.008
    },
    
    # ⚡ 动态杠杆配置
    'leverage': {
        'base_leverage': 5,  # ⚡ 基础杠杆: 激进8, 平衡5, 保守3
        'max_leverage': 20,  # ⚡ 最大杠杆: 激进25, 平衡20, 保守15
        'volatility_adjusted': True,
        'confidence_adjusted': True
    },
    
    # 📊 监控配置
    'monitoring': {
        'profit_alert': 0.03,  # 💰 激进0.02, 平衡0.03, 保守0.04
        'loss_alert': 0.015,   # 💸 激进0.02, 平衡0.015, 保守0.01
        'update_interval': 300,  # 5分钟
        'health_check_interval': 1800,  # 30分钟健康检查
        'stats_interval': 1800  # 30分钟输出统计
    },
    
    # 🧪 测试模式
    'test_mode': True
}

# ==================== 📊 全局状态管理 ====================
class TradingState:
    """交易状态管理器"""
    
    def __init__(self):
        self.daily_pnl = 0.0
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.consecutive_losses = 0
        self.positions = {}
        self.portfolio_value = 0.0
        self.initial_balance = 0.0
        self.signals_history = defaultdict(list)
        self.market_data = defaultdict(dict)
        
        # 策略表现追踪
        self.strategy_performance = {
            'trend_following': {'wins': 0, 'losses': 0, 'total_pnl': 0},
            'mean_reversion': {'wins': 0, 'losses': 0, 'total_pnl': 0},
            'breakout': {'wins': 0, 'losses': 0, 'total_pnl': 0}
        }
        
        # 时间追踪
        self.last_trade_time = None
        self.daily_reset_time = datetime.now().replace(hour=0, minute=0, second=0)
        self.last_stats_time = datetime.now()
        
        # 每日交易记录
        self.daily_trades = []
        self.closed_trades = []
    
    def reset_daily_stats(self):
        """重置每日统计"""
        if datetime.now().date() > self.daily_reset_time.date():
            logger.info("🔄 重置每日统计数据")
            self.daily_pnl = 0.0
            self.trade_count = 0
            self.win_count = 0
            self.loss_count = 0
            self.consecutive_losses = 0
            self.daily_trades = []
            self.daily_reset_time = datetime.now().replace(hour=0, minute=0, second=0)
    
    def should_print_stats(self):
        """判断是否应该输出统计"""
        return (datetime.now() - self.last_stats_time).seconds >= TRADE_CONFIG['monitoring']['stats_interval']
    
    def get_win_rate(self):
        """获取胜率"""
        total = self.win_count + self.loss_count
        return (self.win_count / total * 100) if total > 0 else 0
    
    def get_daily_summary(self):
        """获取当日汇总"""
        return {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'trades': self.trade_count,
            'wins': self.win_count,
            'losses': self.loss_count,
            'win_rate': self.get_win_rate(),
            'daily_pnl': self.daily_pnl,
            'daily_pnl_percent': (self.daily_pnl / self.initial_balance * 100) if self.initial_balance > 0 else 0,
            'open_positions': len(self.positions),
            'consecutive_losses': self.consecutive_losses
        }

# 初始化状态管理器
trading_state = TradingState()

# ==================== 💾 高级数据库管理 ====================
class AdvancedTradeDatabase:
    """增强版交易数据库"""
    
    def __init__(self, db_path='advanced_trades.db'):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()
        logger.info("💾 高级交易数据库初始化成功")
    
    def create_tables(self):
        """创建数据库表结构"""
        # 交易记录表
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                strategy TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL,
                exit_price REAL,
                quantity REAL,
                leverage INTEGER,
                pnl REAL,
                pnl_percent REAL,
                confidence REAL,
                stop_loss REAL,
                take_profit REAL,
                risk_reward_ratio REAL,
                market_condition TEXT,
                status TEXT,
                exit_reason TEXT,
                duration_seconds INTEGER
            )
        ''')
        
        # 策略表现表
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS strategy_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                strategy TEXT NOT NULL,
                trades INTEGER,
                wins INTEGER,
                losses INTEGER,
                win_rate REAL,
                total_pnl REAL,
                avg_pnl REAL,
                max_pnl REAL,
                min_pnl REAL
            )
        ''')
        
        # 市场状态表
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS market_conditions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                trend TEXT,
                volatility REAL,
                volume_ratio REAL,
                signal_strength REAL,
                rsi REAL,
                macd REAL,
                adx REAL
            )
        ''')
        
        # 每日统计表
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS daily_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL UNIQUE,
                total_trades INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER,
                win_rate REAL,
                total_pnl REAL,
                total_pnl_percent REAL,
                max_drawdown REAL,
                sharpe_ratio REAL,
                portfolio_value REAL
            )
        ''')
        
        self.conn.commit()
    
    def log_trade(self, trade_data):
        """记录交易到数据库"""
        try:
            self.conn.execute('''
                INSERT INTO trades (
                    timestamp, symbol, strategy, side, entry_price, exit_price,
                    quantity, leverage, pnl, pnl_percent, confidence,
                    stop_loss, take_profit, risk_reward_ratio, market_condition, 
                    status, exit_reason, duration_seconds
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                trade_data.get('symbol'),
                trade_data.get('strategy', 'unknown'),
                trade_data.get('side'),
                trade_data.get('entry_price'),
                trade_data.get('exit_price'),
                trade_data.get('quantity'),
                trade_data.get('leverage'),
                trade_data.get('pnl', 0),
                trade_data.get('pnl_percent', 0),
                trade_data.get('confidence', 0),
                trade_data.get('stop_loss'),
                trade_data.get('take_profit'),
                trade_data.get('risk_reward_ratio', 0),
                trade_data.get('market_condition', 'neutral'),
                trade_data.get('status', 'open'),
                trade_data.get('exit_reason', ''),
                trade_data.get('duration_seconds', 0)
            ))
            self.conn.commit()
            logger.info(f"📝 交易记录已保存: {trade_data.get('symbol')}")
        except Exception as e:
            logger.error(f"❌ 保存交易记录失败: {e}")
    
    def update_trade_exit(self, symbol, exit_data):
        """更新交易退出信息"""
        try:
            self.conn.execute('''
                UPDATE trades 
                SET exit_price = ?, pnl = ?, pnl_percent = ?, 
                    status = ?, exit_reason = ?, duration_seconds = ?
                WHERE symbol = ? AND status = 'open'
                ORDER BY id DESC LIMIT 1
            ''', (
                exit_data.get('exit_price'),
                exit_data.get('pnl'),
                exit_data.get('pnl_percent'),
                'closed',
                exit_data.get('exit_reason', 'manual'),
                exit_data.get('duration_seconds', 0),
                symbol
            ))
            self.conn.commit()
            logger.info(f"✅ 更新交易退出: {symbol}")
        except Exception as e:
            logger.error(f"❌ 更新交易退出失败: {e}")
    
    def get_statistics(self, days=30):
        """获取统计数据"""
        try:
            cursor = self.conn.execute('''
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses,
                    AVG(CASE WHEN pnl > 0 THEN 1.0 ELSE 0.0 END) * 100 as win_rate,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as avg_pnl,
                    MAX(pnl) as max_pnl,
                    MIN(pnl) as min_pnl
                FROM trades 
                WHERE timestamp >= datetime('now', '-' || ? || ' days')
                AND status = 'closed'
            ''', (days,))
            
            row = cursor.fetchone()
            return {
                'total_trades': row[0] or 0,
                'wins': row[1] or 0,
                'losses': row[2] or 0,
                'win_rate': row[3] or 0,
                'total_pnl': row[4] or 0,
                'avg_pnl': row[5] or 0,
                'max_pnl': row[6] or 0,
                'min_pnl': row[7] or 0
            }
        except Exception as e:
            logger.error(f"❌ 获取统计数据失败: {e}")
            return {}
    
    def save_daily_stats(self, stats):
        """保存每日统计"""
        try:
            self.conn.execute('''
                INSERT OR REPLACE INTO daily_stats (
                    date, total_trades, winning_trades, losing_trades,
                    win_rate, total_pnl, total_pnl_percent, max_drawdown,
                    sharpe_ratio, portfolio_value
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                stats['date'],
                stats['total_trades'],
                stats['winning_trades'],
                stats['losing_trades'],
                stats['win_rate'],
                stats['total_pnl'],
                stats['total_pnl_percent'],
                stats.get('max_drawdown', 0),
                stats.get('sharpe_ratio', 0),
                stats.get('portfolio_value', 0)
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"❌ 保存每日统计失败: {e}")

# 初始化数据库
db = AdvancedTradeDatabase()

# ==================== 🔧 工具函数 ====================
def safe_api_call(func, *args, **kwargs):
    """安全的API调用封装"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            logger.warning(f"🌐 网络错误 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise e
        except ccxt.ExchangeError as e:
            logger.error(f"💱 交易所错误: {e}")
            raise e
        except Exception as e:
            logger.error(f"❓ 未知错误: {e}")
            raise e
    return None

def calculate_technical_indicators(df):
    """计算全套技术指标"""
    try:
        # 移动平均线
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # 布林带
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * bb_std
        df['bb_lower'] = df['bb_middle'] - 2 * bb_std
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        df['atr_percent'] = df['atr'] / df['close']
        
        # 成交量指标
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-10)
        
        # ADX
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr_14 = tr.rolling(14).sum()
        plus_di = 100 * (plus_dm.rolling(14).sum() / (tr_14 + 1e-10))
        minus_di = 100 * (minus_dm.rolling(14).sum() / (tr_14 + 1e-10))
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df['adx'] = dx.rolling(14).mean()
        
        # 动量指标
        df['momentum'] = df['close'].pct_change(periods=10)
        df['roc'] = (df['close'] - df['close'].shift(12)) / df['close'].shift(12) * 100
        
        return df.bfill().ffill()
        
    except Exception as e:
        logger.error(f"❌ 技术指标计算失败: {e}")
        return df

# ==================== 📈 多策略决策引擎 ====================
class MultiStrategyEngine:
    """多策略决策引擎"""
    
    def __init__(self):
        self.strategies = TRADE_CONFIG['strategies']
    
    def trend_following_strategy(self, df, timeframe):
        """趋势跟踪策略"""
        try:
            current = df.iloc[-1]
            prev = df.iloc[-2]
            
            score = 0
            signals = []
            confidence = 0.5
            
            # EMA交叉信号
            if current['ema_20'] > current['ema_50']:
                if prev['ema_20'] <= prev['ema_50']:
                    score += 30
                    signals.append("EMA金叉")
                    confidence += 0.15
                else:
                    score += 15
                    signals.append("EMA多头排列")
                    confidence += 0.05
            elif current['ema_20'] < current['ema_50']:
                if prev['ema_20'] >= prev['ema_50']:
                    score -= 30
                    signals.append("EMA死叉")
                else:
                    score -= 15
                    signals.append("EMA空头排列")
            
            # MACD信号
            if current['macd'] > current['macd_signal']:
                if current['macd_histogram'] > prev['macd_histogram']:
                    score += 20
                    signals.append("MACD强势看涨")
                    confidence += 0.1
                else:
                    score += 10
                    signals.append("MACD看涨")
            elif current['macd'] < current['macd_signal']:
                score -= 15
                signals.append("MACD看跌")
            
            # ADX趋势强度
            if current['adx'] > 25:
                score += 15
                signals.append(f"强趋势(ADX:{current['adx']:.1f})")
                confidence += 0.1
            elif current['adx'] > 20:
                score += 8
                signals.append("中等趋势")
            
            # 价格位置
            if current['close'] > current['ema_20']:
                score += 10
                confidence += 0.05
            
            # 成交量确认
            if current['volume_ratio'] > 1.3:
                score += 15
                signals.append("放量确认")
                confidence += 0.1
            elif current['volume_ratio'] > 1.0:
                score += 5
            
            # RSI过滤
            if 60 < current['rsi_14'] < 70:
                signals.append("RSI健康")
            elif current['rsi_14'] > 70:
                score -= 15
                signals.append("RSI超买")
                confidence -= 0.1
            elif current['rsi_14'] < 30:
                score += 10
                signals.append("RSI超卖反弹")
            
            # 动量确认
            if current['momentum'] > 0.01:
                score += 5
                confidence += 0.05
            
            final_score = max(0, min(100, 50 + score))
            final_confidence = max(0.1, min(0.95, confidence))
            
            return {
                'score': final_score,
                'confidence': final_confidence,
                'signals': signals,
                'direction': 'long' if score >= 20 else 'short' if score <= -20 else 'neutral'
            }
            
        except Exception as e:
            logger.error(f"❌ 趋势策略分析失败: {e}")
            return {'score': 50, 'confidence': 0.3, 'signals': [], 'direction': 'neutral'}
    
    def mean_reversion_strategy(self, df, timeframe):
        """均值回归策略"""
        try:
            current = df.iloc[-1]
            
            score = 0
            signals = []
            confidence = 0.5
            
            # RSI超买超卖
            if current['rsi_14'] < 25:
                score += 35
                signals.append(f"RSI严重超卖({current['rsi_14']:.1f})")
                confidence += 0.2
            elif current['rsi_14'] < 30:
                score += 25
                signals.append("RSI超卖")
                confidence += 0.15
            elif current['rsi_14'] > 75:
                score -= 35
                signals.append(f"RSI严重超买({current['rsi_14']:.1f})")
            elif current['rsi_14'] > 70:
                score -= 25
                signals.append("RSI超买")
            
            # 布林带位置
            bb_pos = current['bb_position']
            if bb_pos < 0.1:
                score += 30
                signals.append("触及布林下轨")
                confidence += 0.15
            elif bb_pos < 0.2:
                score += 20
                signals.append("接近布林下轨")
                confidence += 0.1
            elif bb_pos > 0.9:
                score -= 30
                signals.append("触及布林上轨")
            elif bb_pos > 0.8:
                score -= 20
                signals.append("接近布林上轨")
            
            # 价格偏离均线
            price_vs_sma20 = (current['close'] - current['sma_20']) / current['sma_20']
            if price_vs_sma20 < -0.05:
                score += 25
                signals.append(f"价格低于均线{abs(price_vs_sma20)*100:.1f}%")
                confidence += 0.1
            elif price_vs_sma20 > 0.05:
                score -= 25
                signals.append(f"价格高于均线{price_vs_sma20*100:.1f}%")
            
            # 布林带宽度(波动率)
            if current['bb_width'] > 0.1:
                signals.append("高波动")
                confidence += 0.05
            
            final_score = max(0, min(100, 50 + score))
            final_confidence = max(0.1, min(0.95, confidence))
            
            return {
                'score': final_score,
                'confidence': final_confidence,
                'signals': signals,
                'direction': 'long' if score > 15 else 'short' if score < -15 else 'neutral'
            }
            
        except Exception as e:
            logger.error(f"❌ 均值回归策略分析失败: {e}")
            return {'score': 50, 'confidence': 0.3, 'signals': [], 'direction': 'neutral'}
    
    def breakout_strategy(self, df, timeframe):
        """突破策略"""
        try:
            current = df.iloc[-1]
            recent = df.tail(20)
            
            score = 0
            signals = []
            confidence = 0.5
            
            # 布林带突破
            if current['close'] > current['bb_upper']:
                score += 30
                signals.append("突破上轨")
                confidence += 0.15
            elif current['close'] < current['bb_lower']:
                score -= 30
                signals.append("突破下轨")
            
            # 成交量突破确认
            if current['volume_ratio'] > 2.0:
                score += 25
                signals.append("大幅放量")
                confidence += 0.15
            elif current['volume_ratio'] > 1.5:
                score += 15
                signals.append("放量突破")
                confidence += 0.1
            
            # ATR波动性
            if current['atr_percent'] > 0.03:
                score += 15
                signals.append("高波动环境")
                confidence += 0.05
            elif current['atr_percent'] > 0.02:
                score += 8
                signals.append("中等波动")
            
            # 价格创新高/新低
            if current['close'] >= recent['high'].max():
                score += 20
                signals.append("创20期新高")
                confidence += 0.1
            elif current['close'] <= recent['low'].min():
                score -= 20
                signals.append("创20期新低")
            
            # 连续上涨/下跌
            close_changes = df['close'].pct_change().tail(5)
            if (close_changes > 0).sum() >= 4:
                score += 10
                signals.append("连续上涨")
            elif (close_changes < 0).sum() >= 4:
                score -= 10
                signals.append("连续下跌")
            
            final_score = max(0, min(100, 50 + score))
            final_confidence = max(0.1, min(0.95, confidence))
            
            return {
                'score': final_score,
                'confidence': final_confidence,
                'signals': signals,
                'direction': 'long' if score > 15 else 'short' if score < -15 else 'neutral'
            }
            
        except Exception as e:
            logger.error(f"❌ 突破策略分析失败: {e}")
            return {'score': 50, 'confidence': 0.3, 'signals': [], 'direction': 'neutral'}
    
    def analyze_symbol(self, symbol, timeframe_data):
        """综合分析币种"""
        try:
            strategy_results = {}
            total_score = 0
            total_confidence = 0
            total_weight = 0
            
            for strategy_name, strategy_config in self.strategies.items():
                if not strategy_config['enabled']:
                    continue
                
                # 选择合适的时间框架
                best_tf_data = None
                for tf in strategy_config['timeframes']:
                    if tf in timeframe_data:
                        best_tf_data = timeframe_data[tf]
                        break
                
                if not best_tf_data or len(best_tf_data['df']) < 50:
                    continue
                
                # 执行策略分析
                if strategy_name == 'trend_following':
                    result = self.trend_following_strategy(best_tf_data['df'], best_tf_data['timeframe'])
                elif strategy_name == 'mean_reversion':
                    result = self.mean_reversion_strategy(best_tf_data['df'], best_tf_data['timeframe'])
                elif strategy_name == 'breakout':
                    result = self.breakout_strategy(best_tf_data['df'], best_tf_data['timeframe'])
                else:
                    continue
                
                strategy_results[strategy_name] = result
                
                # 加权计算总分
                weight = strategy_config['weight']
                total_score += result['score'] * weight
                total_confidence += result['confidence'] * weight
                total_weight += weight
            
            if total_weight == 0:
                return {'final_score': 0, 'final_confidence': 0, 'strategies': {}}
            
            final_score = total_score / total_weight
            final_confidence = total_confidence / total_weight
            
            return {
                'final_score': final_score,
                'final_confidence': final_confidence,
                'strategies': strategy_results,
                'recommended_direction': self._get_recommended_direction(strategy_results)
            }
            
        except Exception as e:
            logger.error(f"❌ 币种分析失败 {symbol}: {e}")
            return {'final_score': 0, 'final_confidence': 0, 'strategies': {}}
    
    def _get_recommended_direction(self, strategy_results):
        """根据策略结果推荐方向"""
        long_votes = 0
        short_votes = 0
        
        for strategy, result in strategy_results.items():
            if result['direction'] == 'long':
                long_votes += 1
            elif result['direction'] == 'short':
                short_votes += 1
        
        if long_votes > short_votes:
            return 'long'
        elif short_votes > long_votes:
            return 'short'
        else:
            return 'neutral'

# ==================== 🎯 DeepSeek AI决策引擎 ====================
class DeepSeekDecisionEngine:
    """DeepSeek AI决策引擎"""
    
    def __init__(self, client):
        self.client = client
    
    def generate_trading_signal(self, symbol, market_data, strategy_analysis):
        """生成AI交易信号"""
        if not self.client:
            logger.warning("⚠️ DeepSeek客户端未初始化，使用备用信号")
            return self._create_fallback_signal(market_data, strategy_analysis)
        
        try:
            prompt = self._build_ai_prompt(symbol, market_data, strategy_analysis)
            
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {
                        "role": "system",
                        "content": """你是一位急需数十万美元资金为女友治疗癌症的专业的量化交易员，擅长周期趋势分析加密货币合约交易。
                        请基于K线形态技术分析和市场数据给出理性的交易决策。
                        严格遵守风险管理和资金管理原则。
                        只在高胜率机会时才建议交易。"""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            return self._parse_ai_response(response.choices[0].message.content, market_data, strategy_analysis)
            
        except Exception as e:
            logger.error(f"❌ AI决策失败: {e}")
            return self._create_fallback_signal(market_data, strategy_analysis)
    
    def _build_ai_prompt(self, symbol, market_data, strategy_analysis):
        """构建AI提示词"""
        current_price = market_data['current_price']
        tf_data = list(market_data['timeframes'].values())[0]
        current = tf_data['current']
        
        # 🚀 AI决策模式增强提示
        ai_mode_note = ""
        if TRADE_CONFIG['ai_decision_mode']['enabled']:
            ai_mode_note = """
🚀 AI决策主导模式已启用:
- 您拥有最终决策权，策略分析仅供参考
- 请基于全面的市场分析做出独立判断
- 可以覆盖策略引擎的建议信号
- 重点关注风险回报比和资金安全
"""
        
        prompt = f"""
作为专业量化交易员，请分析以下{symbol}交易机会：

{ai_mode_note}

📊 市场数据：
- 当前价格: ${current_price:.4f}
- 24小时波动率: {current.get('atr_percent', 0)*100:.2f}%
- 成交量比率: {current.get('volume_ratio', 1):.2f}

🎯 多策略综合分析：
- 综合评分: {strategy_analysis.get('final_score', 0):.1f}/100
- 综合信心度: {strategy_analysis.get('final_confidence', 0):.1%}
- 推荐方向: {strategy_analysis.get('recommended_direction', 'neutral')}

策略详情:
{json.dumps(strategy_analysis.get('strategies', {}), indent=2, ensure_ascii=False)}

📈 技术指标状态：
- RSI(14): {current.get('rsi_14', 50):.1f}
- MACD: {current.get('macd', 0):.4f}
- MACD信号: {current.get('macd_signal', 0):.4f}
- ADX: {current.get('adx', 0):.1f}
- ATR: {current.get('atr', 0):.4f} ({current.get('atr_percent', 0)*100:.2f}%)
- 布林带位置: {current.get('bb_position', 0.5):.2%}
- EMA20: ${current.get('ema_20', 0):.4f}
- EMA50: ${current.get('ema_50', 0):.4f}

💰 账户状态：
- 今日盈亏: {trading_state.daily_pnl:+.2%}
- 今日交易: {trading_state.trade_count}笔
- 胜率: {trading_state.get_win_rate():.1f}%
- 连续亏损: {trading_state.consecutive_losses}次
- 当前持仓数: {len(trading_state.positions)}

⚠️ 风险控制要求：
- 单笔最大风险: {TRADE_CONFIG['risk_management']['max_single_loss']:.1%}
- 风险回报比要求: ≥{TRADE_CONFIG['risk_management']['risk_reward_ratio']}:1
- 最低信心度: {TRADE_CONFIG['ai_decision_mode']['min_confidence']:.2f}
- 杠杆范围: {TRADE_CONFIG['leverage']['base_leverage']}-{TRADE_CONFIG['leverage']['max_leverage']}x

请以JSON格式输出交易决策，包含以下字段：
{{
    "signal": "BUY|SELL|HOLD",
    "confidence": 0.00-1.00,
    "entry_price": {current_price},
    "stop_loss": 具体价格,
    "take_profit": 具体价格,
    "leverage": {TRADE_CONFIG['leverage']['base_leverage']}-{TRADE_CONFIG['leverage']['max_leverage']}整数,
    "position_size_percent": 0.05-0.25,
    "reason": "详细分析理由(100字以内)",
    "expected_risk_reward": 具体数值,
    "time_horizon": "SHORT|MEDIUM|LONG",
    "key_levels": {{"support": 价格, "resistance": 价格}}
}}

决策原则：
1. 只在信心度>{TRADE_CONFIG['ai_decision_mode']['min_confidence']}时才建议BUY/SELL
2. 基于技术分析建议合理的止损止盈价位，确保风险回报比≥{TRADE_CONFIG['risk_management']['risk_reward_ratio']}:1
3. 考虑当前账户状态，连续亏损时降低仓位
4. 高波动环境降低杠杆，低波动适当提高
5. 必须给出明确的入场、止损、止盈价格
6. 简要分析理由（考虑趋势连续性、支撑阻力、成交量等因素）
"""
        return prompt
    
    def _parse_ai_response(self, response_text, market_data, strategy_analysis):
        """解析AI响应"""
        try:
            # 提取JSON部分
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx == -1 or end_idx <= start_idx:
                raise ValueError("未找到有效的JSON响应")
            
            json_str = response_text[start_idx:end_idx]
            signal_data = json.loads(json_str)
            
            # 验证必需字段
            required_fields = ['signal', 'confidence', 'reason']
            if not all(field in signal_data for field in required_fields):
                raise ValueError("AI响应缺少必需字段")
            
            # 🚀 AI决策模式：覆盖策略分析结果
            if TRADE_CONFIG['ai_decision_mode']['override_strategy']:
                logger.info("🚀 AI决策模式：覆盖策略分析结果")
                # AI信号覆盖策略推荐方向
            
            # 设置默认值
            tf_data = list(market_data['timeframes'].values())[0]
            current = tf_data['current']
            
            signal_data.setdefault('entry_price', market_data['current_price'])
            signal_data.setdefault('leverage', 8)
            signal_data.setdefault('position_size_percent', 0.1)
            signal_data.setdefault('expected_risk_reward', 2.5)
            signal_data.setdefault('time_horizon', 'MEDIUM')
            
            # 计算止损止盈（如果未提供）
            if 'stop_loss' not in signal_data or 'take_profit' not in signal_data:
                stop_tp = self._calculate_stop_take_profit(
                    signal_data['entry_price'],
                    signal_data['signal'],
                    current['atr']
                )
                signal_data.update(stop_tp)
            
            # 添加策略分析信息
            signal_data['strategy_score'] = strategy_analysis.get('final_score', 0)
            signal_data['strategy_confidence'] = strategy_analysis.get('final_confidence', 0)
            
            return signal_data
            
        except Exception as e:
            logger.warning(f"⚠️ AI响应解析失败: {e}")
            return self._create_fallback_signal(market_data, strategy_analysis)
    
    def _calculate_stop_take_profit(self, entry_price, signal, atr):
        """计算止损止盈"""
        sl_multiple = TRADE_CONFIG['risk_management']['stop_loss_atr_multiple']
        tp_multiple = TRADE_CONFIG['risk_management']['take_profit_atr_multiple']
        
        if signal == 'BUY':
            stop_loss = entry_price - atr * sl_multiple
            take_profit = entry_price + atr * tp_multiple
        elif signal == 'SELL':
            stop_loss = entry_price + atr * sl_multiple
            take_profit = entry_price - atr * tp_multiple
        else:
            stop_loss = entry_price * 0.98
            take_profit = entry_price * 1.02
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }
    
    def _create_fallback_signal(self, market_data, strategy_analysis):
        """创建备用信号"""
        # 基于策略分析创建备用信号
        score = strategy_analysis.get('final_score', 0)
        confidence = strategy_analysis.get('final_confidence', 0)
        direction = strategy_analysis.get('recommended_direction', 'neutral')
        
        tf_data = list(market_data['timeframes'].values())[0]
        current = tf_data['current']
        
        signal = 'HOLD'
        if score > 70 and confidence > TRADE_CONFIG['ai_decision_mode']['min_confidence'] and direction == 'long':
            signal = 'BUY'
        elif score > 70 and confidence > TRADE_CONFIG['ai_decision_mode']['min_confidence'] and direction == 'short':
            signal = 'SELL'
        
        stop_tp = self._calculate_stop_take_profit(
            market_data['current_price'],
            signal,
            current['atr']
        )
        
        return {
            "signal": signal,
            "confidence": confidence,
            "entry_price": market_data['current_price'],
            "stop_loss": stop_tp['stop_loss'],
            "take_profit": stop_tp['take_profit'],
            "leverage": 8,
            "position_size_percent": 0.08,
            "reason": f"备用信号: 策略评分{score:.1f}, 方向{direction}",
            "expected_risk_reward": 2.5,
            "time_horizon": "MEDIUM",
            "is_fallback": True,
            "strategy_score": score,
            "strategy_confidence": confidence
        }

# ==================== 💼 投资组合管理器 ====================
class PortfolioManager:
    """投资组合管理器"""
    
    def __init__(self):
        self.max_total_position = TRADE_CONFIG['risk_management']['max_total_position']
        self.max_single_position = TRADE_CONFIG['risk_management']['max_single_position']
        self.max_open_positions = TRADE_CONFIG['risk_management']['max_open_positions']
    
    def calculate_position_size(self, symbol, signal_data, available_balance):
        """计算仓位大小"""
        try:
            # 基础仓位大小
            base_position_percent = signal_data['position_size_percent']
            
            # 根据信心度调整
            confidence = signal_data['confidence']
            confidence_factor = max(0.5, min(1.0, confidence))
            
            # 根据连续亏损调整
            loss_penalty = max(0.3, 1 - (trading_state.consecutive_losses * 0.15))
            
            # 根据策略评分调整
            score_factor = signal_data.get('strategy_score', 50) / 100
            score_factor = max(0.5, min(1.0, score_factor))
            
            # 计算最终仓位比例
            position_percent = base_position_percent * confidence_factor * loss_penalty * score_factor
            
            # 应用单币种限制
            position_percent = min(position_percent, self.max_single_position)
            
            # 应用总仓位限制
            current_total_position = self.get_current_total_position_ratio()
            available_total_position = self.max_total_position - current_total_position
            position_percent = min(position_percent, available_total_position)
            
            # 确保最小仓位
            if position_percent < 0.03:
                logger.warning(f"⚠️ 计算仓位过小: {position_percent:.2%}")
                return 0
            
            # 计算具体仓位大小
            position_value = available_balance * position_percent
            leverage = signal_data['leverage']
            quantity = (position_value * leverage) / signal_data['entry_price']
            
            logger.info(f"💼 仓位计算: {position_percent:.2%} × {leverage}x = {quantity:.6f} {symbol}")
            logger.info(f"   因子: 信心{confidence_factor:.2f} × 亏损{loss_penalty:.2f} × 评分{score_factor:.2f}")
            
            return quantity
            
        except Exception as e:
            logger.error(f"❌ 仓位计算失败: {e}")
            return 0
    
    def calculate_dynamic_leverage(self, signal_data, market_data):
        """计算动态杠杆"""
        try:
            base_leverage = TRADE_CONFIG['leverage']['base_leverage']
            max_leverage = TRADE_CONFIG['leverage']['max_leverage']
            
            # 基于波动率调整
            tf_data = list(market_data['timeframes'].values())[0]
            atr_percent = tf_data['current'].get('atr_percent', 0.02)
            
            if atr_percent > 0.05:  # 高波动
                volatility_factor = 0.6
            elif atr_percent > 0.03:  # 中等波动
                volatility_factor = 0.8
            else:  # 低波动
                volatility_factor = 1.0
            
            # 基于信心度调整
            confidence = signal_data['confidence']
            confidence_factor = 0.7 + (confidence - 0.6) * 0.75  # 0.6信心度对应0.7因子
            confidence_factor = max(0.5, min(1.2, confidence_factor))
            
            # 🚀 AI决策模式：风险调整
            if TRADE_CONFIG['ai_decision_mode']['risk_adjustment']:
                # AI模式下更激进的杠杆调整
                if confidence > 0.75:
                    confidence_factor *= 1.1
                elif confidence < 0.55:
                    confidence_factor *= 0.8
            
            # 计算最终杠杆
            leverage = base_leverage * volatility_factor * confidence_factor
            leverage = max(base_leverage, min(max_leverage, int(leverage)))
            
            logger.info(f"⚡ 动态杠杆: {leverage}x (波动{volatility_factor:.2f} × 信心{confidence_factor:.2f})")
            
            return leverage
            
        except Exception as e:
            logger.error(f"❌ 杠杆计算失败: {e}")
            return TRADE_CONFIG['leverage']['base_leverage']
    
    def get_current_total_position_ratio(self):
        """获取当前总仓位比例"""
        try:
            if not exchange:
                return 0
            
            balance = safe_api_call(exchange.fetch_balance)
            if not balance:
                return 0
            
            total_value = balance['total']['USDT']
            if total_value == 0:
                return 0
            
            positions = trading_state.positions.values()
            total_position_value = sum(
                abs(pos.get('quantity', 0) * pos.get('current_price', 0)) 
                for pos in positions
            )
            
            return total_position_value / total_value
            
        except Exception as e:
            logger.error(f"❌ 获取总仓位失败: {e}")
            return 0
    
    def should_open_position(self, symbol, proposed_position_value):
        """判断是否应该开新仓"""
        # 检查持仓数量限制
        if len(trading_state.positions) >= self.max_open_positions and symbol not in trading_state.positions:
            logger.warning(f"⚠️ 持仓数量限制: 已持有{len(trading_state.positions)}个币种")
            return False
        
        # 检查总仓位限制
        current_ratio = self.get_current_total_position_ratio()
        proposed_ratio = proposed_position_value / trading_state.portfolio_value if trading_state.portfolio_value > 0 else 0
        
        if current_ratio + proposed_ratio > self.max_total_position:
            logger.warning(f"⚠️ 总仓位限制: {current_ratio:.1%} + {proposed_ratio:.1%} > {self.max_total_position:.1%}")
            return False
        
        return True

# ==================== 📊 市场数据获取 ====================
class MarketDataProvider:
    """市场数据提供器"""
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 300  # 缓存5分钟
    
    def get_multi_timeframe_data(self, symbol, timeframes=None):
        """获取多时间框架数据"""
        if timeframes is None:
            timeframes = ['1h', '15m', '5m', '4h']
        
        try:
            result = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'timeframes': {},
                'current_price': 0
            }
            
            for tf in timeframes:
                try:
                    # 检查缓存
                    cache_key = f"{symbol}_{tf}"
                    if (cache_key in self.cache and 
                        (datetime.now() - self.cache[cache_key]['timestamp']).seconds < self.cache_ttl):
                        tf_data = self.cache[cache_key]['data']
                    else:
                        # 获取K线数据
                        ohlcv = safe_api_call(
                            exchange.fetch_ohlcv,
                            symbol,
                            tf,
                            limit=200  # 获取足够数据计算指标
                        )
                        
                        if not ohlcv or len(ohlcv) < 50:
                            continue
                        
                        # 转换为DataFrame
                        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        
                        # 计算技术指标
                        df = calculate_technical_indicators(df)
                        
                        tf_data = {
                            'timeframe': tf,
                            'df': df,
                            'current': df.iloc[-1].to_dict(),
                            'previous': df.iloc[-2].to_dict() if len(df) > 1 else None
                        }
                        
                        # 更新缓存
                        self.cache[cache_key] = {
                            'timestamp': datetime.now(),
                            'data': tf_data
                        }
                    
                    result['timeframes'][tf] = tf_data
                    result['current_price'] = tf_data['current']['close']
                    
                    time.sleep(0.1)  # 避免API限流
                    
                except Exception as e:
                    logger.error(f"❌ 获取{symbol} {tf}数据失败: {e}")
                    continue
            
            if not result['timeframes']:
                return None
                
            return result
            
        except Exception as e:
            logger.error(f"❌ 获取{symbol}市场数据失败: {e}")
            return None

# ==================== 🚀 交易执行引擎 ====================
class TradingExecutor:
    """交易执行引擎"""
    
    def __init__(self):
        self.portfolio_mgr = PortfolioManager()
    
    def execute_trade(self, symbol, signal_data, market_data):
        """执行交易"""
        if not exchange:
            logger.error("❌ 交易所未连接")
            return False
        
        try:
            # 风险检查
            if not self._pass_risk_checks(symbol, signal_data):
                return False
            
            # 获取账户余额
            balance = safe_api_call(exchange.fetch_balance)
            if not balance:
                return False
            
            available_balance = balance['free']['USDT']
            
            if available_balance < 10:
                logger.warning(f"⚠️ 可用余额不足: {available_balance:.2f} USDT")
                return False
            
            # 动态调整杠杆
            signal_data['leverage'] = self.portfolio_mgr.calculate_dynamic_leverage(signal_data, market_data)
            
            # 使用小资金优化器计算仓位
            position_value = available_balance * signal_data['position_size_percent']
            quantity = small_capital_optimizer.calculate_optimal_quantity(
                symbol,
                signal_data['entry_price'],
                position_value,
                signal_data['leverage']
            )
            
            if quantity <= 0:
                logger.warning("⚠️ 仓位计算为0，跳过交易")
                return False
            
            # 测试模式检查
            if TRADE_CONFIG['test_mode']:
                logger.info(f"🧪 测试模式 - 模拟交易 {symbol} {signal_data['signal']} {quantity:.6f}")
                self._log_simulated_trade(symbol, signal_data, quantity)
                return True
            
            # 设置杠杆
            try:
                exchange.set_leverage(signal_data['leverage'], symbol)
                logger.info(f"⚡ 设置杠杆: {signal_data['leverage']}x")
            except Exception as e:
                logger.warning(f"⚠️ 设置杠杆失败: {e}")
            
            # 执行交易
            side = 'buy' if signal_data['signal'] == 'BUY' else 'sell'
            
            order = safe_api_call(
                exchange.create_market_order,
                symbol,
                side,
                quantity
            )
            
            if order:
                logger.info(f"✅ 订单执行成功: {symbol} {side.upper()} {quantity:.6f}")
                
                # 记录交易
                self._log_trade_execution(symbol, signal_data, quantity, order)
                
                # 设置止损止盈
                self._set_stop_loss_take_profit(symbol, signal_data, quantity, side)
                
                trading_state.trade_count += 1
                trading_state.last_trade_time = datetime.now()
                
                return True
            else:
                logger.error("❌ 订单执行失败")
                return False
            
        except Exception as e:
            logger.error(f"❌ 交易执行失败: {e}")
            traceback.print_exc()
            return False
    
    def _pass_risk_checks(self, symbol, signal_data):
        """通过风险检查"""
        # 信心度检查
        min_confidence = TRADE_CONFIG['ai_decision_mode']['min_confidence']
        if signal_data['confidence'] < min_confidence:
            logger.warning(f"⚠️ 信心度不足: {signal_data['confidence']:.2%} < {min_confidence:.2%}")
            return False
        
        # 日亏损检查
        if trading_state.daily_pnl < -TRADE_CONFIG['risk_management']['max_daily_loss']:
            logger.warning(f"🚨 达到日亏损限制: {trading_state.daily_pnl:.2%}")
            return False
        
        # 连续亏损检查
        max_losses = 5
        if trading_state.consecutive_losses >= max_losses:
            logger.warning(f"🚨 连续亏损{trading_state.consecutive_losses}次，暂停交易")
            return False
        
        # 风险回报比检查
        entry = signal_data['entry_price']
        sl = signal_data['stop_loss']
        tp = signal_data['take_profit']
        
        if signal_data['signal'] == 'BUY':
            risk = entry - sl
            reward = tp - entry
        else:
            risk = sl - entry
            reward = entry - tp
        
        if risk <= 0:
            logger.warning("⚠️ 止损价格无效")
            return False
        
        rr_ratio = reward / risk
        min_rr_ratio = TRADE_CONFIG['risk_management']['risk_reward_ratio']
        if rr_ratio < min_rr_ratio:
            logger.warning(f"⚠️ 风险回报比不足: {rr_ratio:.2f} < {min_rr_ratio:.2f}")
            return False
        
        return True
    
    def _log_simulated_trade(self, symbol, signal_data, quantity):
        """记录模拟交易"""
        trade_record = {
            'symbol': symbol,
            'strategy': 'deepseek_ai',
            'side': signal_data['signal'],
            'entry_price': signal_data['entry_price'],
            'quantity': quantity,
            'leverage': signal_data['leverage'],
            'confidence': signal_data['confidence'],
            'stop_loss': signal_data['stop_loss'],
            'take_profit': signal_data['take_profit'],
            'risk_reward_ratio': signal_data.get('expected_risk_reward', 2.5),
            'market_condition': signal_data.get('time_horizon', 'MEDIUM'),
            'status': 'open'
        }
        
        db.log_trade(trade_record)
        trading_state.daily_trades.append(trade_record)
        
        message = f"""
🎯 模拟交易执行:
🪙 币种: {symbol}
📈 方向: {signal_data['signal']}
💵 价格: ${signal_data['entry_price']:.4f}
📦 数量: {quantity:.6f}
⚡ 杠杆: {signal_data['leverage']}x
💪 信心度: {signal_data['confidence']:.1%}
🛑 止损: ${signal_data['stop_loss']:.4f} ({((signal_data['stop_loss']/signal_data['entry_price']-1)*100):+.2f}%)
🎯 止盈: ${signal_data['take_profit']:.4f} ({((signal_data['take_profit']/signal_data['entry_price']-1)*100):+.2f}%)
📝 理由: {signal_data['reason']}
        """
        logger.info(message)
    
    def _log_trade_execution(self, symbol, signal_data, quantity, order):
        """记录交易执行"""
        trade_record = {
            'symbol': symbol,
            'strategy': 'deepseek_ai',
            'side': signal_data['signal'],
            'entry_price': signal_data['entry_price'],
            'quantity': quantity,
            'leverage': signal_data['leverage'],
            'confidence': signal_data['confidence'],
            'stop_loss': signal_data['stop_loss'],
            'take_profit': signal_data['take_profit'],
            'risk_reward_ratio': signal_data.get('expected_risk_reward', 2.5),
            'market_condition': signal_data.get('time_horizon', 'MEDIUM'),
            'status': 'open',
            'order_id': order.get('id', '')
        }
        
        db.log_trade(trade_record)
        trading_state.daily_trades.append(trade_record)
        
        # 添加到持仓
        trading_state.positions[symbol] = {
            'symbol': symbol,
            'side': signal_data['signal'],
            'entry_price': signal_data['entry_price'],
            'quantity': quantity,
            'leverage': signal_data['leverage'],
            'stop_loss': signal_data['stop_loss'],
            'take_profit': signal_data['take_profit'],
            'entry_time': datetime.now()
        }
        
        message = f"""
🎯 实盘交易执行:
🪙 币种: {symbol}
📈 方向: {signal_data['signal']}
💵 价格: ${signal_data['entry_price']:.4f}
📦 数量: {quantity:.6f}
⚡ 杠杆: {signal_data['leverage']}x
💪 信心度: {signal_data['confidence']:.1%}
🛑 止损: ${signal_data['stop_loss']:.4f}
🎯 止盈: ${signal_data['take_profit']:.4f}
📝 理由: {signal_data['reason']}
🆔 订单ID: {order.get('id', 'N/A')}
        """
        logger.info(message)
    
    def _set_stop_loss_take_profit(self, symbol, signal_data, quantity, side):
        """设置止损止盈订单"""
        try:
            # 止损订单
            sl_side = 'sell' if side == 'buy' else 'buy'
            sl_order = safe_api_call(
                exchange.create_order,
                symbol,
                'stop_market',
                sl_side,
                quantity,
                None,
                {
                    'stopPrice': signal_data['stop_loss'],
                    'reduceOnly': True
                }
            )
            if sl_order:
                logger.info(f"🛑 止损单设置成功: ${signal_data['stop_loss']:.4f}")
            
            # 止盈订单
            tp_order = safe_api_call(
                exchange.create_order,
                symbol,
                'take_profit_market',
                sl_side,
                quantity,
                None,
                {
                    'stopPrice': signal_data['take_profit'],
                    'reduceOnly': True
                }
            )
            if tp_order:
                logger.info(f"🎯 止盈单设置成功: ${signal_data['take_profit']:.4f}")
                
        except Exception as e:
            logger.warning(f"⚠️ 设置止损止盈失败: {e}")

# ==================== 🔄 持仓监控管理器 ====================
class PositionMonitor:
    """持仓监控管理器"""
    
    def __init__(self):
        self.alert_thresholds = {
            'profit': TRADE_CONFIG['monitoring']['profit_alert'],
            'loss': TRADE_CONFIG['monitoring']['loss_alert']
        }
    
    def monitor_all_positions(self):
        """监控所有持仓"""
        try:
            if not exchange:
                return
            
            positions = safe_api_call(exchange.fetch_positions)
            if not positions:
                return
            
            active_positions = []
            for pos in positions:
                if float(pos.get('contracts', 0)) > 0:
                    self._analyze_position(pos)
                    active_positions.append(pos['symbol'])
                    
                    # 检查并更新移动止损
                    if pos['symbol'] in trading_state.positions:
                        trailing_stop_manager.check_and_update_trailing_stop(
                            pos['symbol'], 
                            trading_state.positions[pos['symbol']]
                        )
            
            # 清理不存在的持仓
            for symbol in list(trading_state.positions.keys()):
                if symbol not in active_positions:
                    self._close_position_record(symbol)
            
            # 更新交易状态
            self._update_trading_state()
            
        except Exception as e:
            logger.error(f"❌ 监控持仓失败: {e}")
    
    def _analyze_position(self, position):
        """分析单个持仓"""
        try:
            symbol = position['symbol']
            entry_price = float(position.get('entryPrice', 0))
            current_price = float(position.get('markPrice', 0))
            quantity = float(position.get('contracts', 0))
            unrealized_pnl = float(position.get('unrealizedPnl', 0))
            
            if entry_price == 0:
                return
            
            # 计算盈亏百分比
            pnl_percent = unrealized_pnl / (entry_price * quantity) if (entry_price * quantity) > 0 else 0
            
            # 更新持仓信息
            if symbol in trading_state.positions:
                trading_state.positions[symbol].update({
                    'current_price': current_price,
                    'unrealized_pnl': unrealized_pnl,
                    'pnl_percent': pnl_percent,
                    'liquidation_price': float(position.get('liquidationPrice', 0))
                })
            else:
                trading_state.positions[symbol] = {
                    'symbol': symbol,
                    'side': position.get('side', 'long'),
                    'quantity': quantity,
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'unrealized_pnl': unrealized_pnl,
                    'pnl_percent': pnl_percent,
                    'leverage': float(position.get('leverage', 1)),
                    'liquidation_price': float(position.get('liquidationPrice', 0))
                }
            
            # 发送告警
            self._send_alerts(symbol, pnl_percent, unrealized_pnl)
            
            logger.info(f"📊 {symbol} {position.get('side', 'N/A')}: {pnl_percent:+.2%} (${unrealized_pnl:+.2f})")
            
        except Exception as e:
            logger.error(f"❌ 分析持仓失败: {e}")
    
    def _close_position_record(self, symbol):
        """关闭持仓记录"""
        try:
            if symbol not in trading_state.positions:
                return
            
            pos = trading_state.positions[symbol]
            pnl = pos.get('unrealized_pnl', 0)
            
            # 更新胜负统计
            if pnl > 0:
                trading_state.win_count += 1
                trading_state.consecutive_losses = 0
            else:
                trading_state.loss_count += 1
                trading_state.consecutive_losses += 1
            
            # 更新日盈亏
            trading_state.daily_pnl += pnl / trading_state.portfolio_value if trading_state.portfolio_value > 0 else 0
            
            # 记录到数据库
            exit_data = {
                'exit_price': pos.get('current_price', 0),
                'pnl': pnl,
                'pnl_percent': pos.get('pnl_percent', 0),
                'exit_reason': 'closed',
                'duration_seconds': (datetime.now() - pos.get('entry_time', datetime.now())).seconds if 'entry_time' in pos else 0
            }
            db.update_trade_exit(symbol, exit_data)
            
            # 移除持仓
            del trading_state.positions[symbol]
            
            logger.info(f"✅ 持仓已平仓: {symbol} 盈亏 ${pnl:+.2f}")
            
        except Exception as e:
            logger.error(f"❌ 关闭持仓记录失败: {e}")
    
    def _send_alerts(self, symbol, pnl_percent, unrealized_pnl):
        """发送盈亏告警"""
        try:
            # 盈利告警
            if pnl_percent >= self.alert_thresholds['profit']:
                logger.info(f"🎉 {symbol} 盈利告警: {pnl_percent:+.2%} (${unrealized_pnl:+.2f})")
            
            # 亏损告警
            elif pnl_percent <= -self.alert_thresholds['loss']:
                logger.warning(f"⚠️ {symbol} 亏损告警: {pnl_percent:+.2%} (${unrealized_pnl:+.2f})")
                
        except Exception as e:
            logger.error(f"❌ 发送告警失败: {e}")
    
    def _update_trading_state(self):
        """更新交易状态"""
        try:
            # 更新投资组合价值
            balance = safe_api_call(exchange.fetch_balance)
            if balance:
                trading_state.portfolio_value = balance['total']['USDT']
                if trading_state.initial_balance == 0:
                    trading_state.initial_balance = trading_state.portfolio_value
            
        except Exception as e:
            logger.error(f"❌ 更新交易状态失败: {e}")

# ==================== 📈 性能分析器 ====================
class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self):
        self.start_time = datetime.now()
    
    def print_realtime_stats(self):
        """实时输出统计信息"""
        try:
            stats = trading_state.get_daily_summary()
            hist_stats = db.get_statistics(days=7)
            
            print("\n" + "="*70)
            print(f"📊 实时交易统计 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*70)
            
            print(f"\n💰 账户状态:")
            print(f"  账户余额: ${trading_state.portfolio_value:.2f} USDT")
            print(f"  初始余额: ${trading_state.initial_balance:.2f} USDT")
            if trading_state.initial_balance > 0:
                total_return = (trading_state.portfolio_value - trading_state.initial_balance) / trading_state.initial_balance * 100
                print(f"  总回报率: {total_return:+.2f}%")
            
            print(f"\n📈 今日表现:")
            print(f"  交易次数: {stats['trades']}笔")
            print(f"  盈利次数: {stats['wins']}笔")
            print(f"  亏损次数: {stats['losses']}笔")
            print(f"  今日胜率: {stats['win_rate']:.1f}%")
            print(f"  今日盈亏: {stats['daily_pnl']:+.2%} (${stats['daily_pnl'] * trading_state.initial_balance:+.2f})")
            print(f"  连续亏损: {stats['consecutive_losses']}次")
            print(f"  持仓数量: {stats['open_positions']}个")
            
            print(f"\n📊 7日统计:")
            print(f"  总交易: {hist_stats.get('total_trades', 0)}笔")
            print(f"  历史胜率: {hist_stats.get('win_rate', 0):.1f}%")
            print(f"  总盈亏: ${hist_stats.get('total_pnl', 0):+.2f}")
            print(f"  平均盈亏: ${hist_stats.get('avg_pnl', 0):+.2f}")
            print(f"  最大盈利: ${hist_stats.get('max_pnl', 0):+.2f}")
            print(f"  最大亏损: ${hist_stats.get('min_pnl', 0):+.2f}")
            
            if trading_state.positions:
                print(f"\n🔄 当前持仓:")
                for symbol, pos in trading_state.positions.items():
                    print(f"  {symbol}:")
                    print(f"    方向: {pos.get('side', 'N/A')}")
                    print(f"    数量: {pos.get('quantity', 0):.6f}")
                    print(f"    入场: ${pos.get('entry_price', 0):.4f}")
                    print(f"    当前: ${pos.get('current_price', 0):.4f}")
                    print(f"    盈亏: {pos.get('pnl_percent', 0):+.2%} (${pos.get('unrealized_pnl', 0):+.2f})")
                    print(f"    杠杆: {pos.get('leverage', 0)}x")
            
            print("="*70 + "\n")
            
            # 更新最后统计时间
            trading_state.last_stats_time = datetime.now()
            
            # 保存到数据库
            daily_stats = {
                'date': stats['date'],
                'total_trades': stats['trades'],
                'winning_trades': stats['wins'],
                'losing_trades': stats['losses'],
                'win_rate': stats['win_rate'],
                'total_pnl': stats['daily_pnl'] * trading_state.initial_balance,
                'total_pnl_percent': stats['daily_pnl_percent'],
                'portfolio_value': trading_state.portfolio_value
            }
            db.save_daily_stats(daily_stats)
            
        except Exception as e:
            logger.error(f"❌ 输出统计失败: {e}")
    
    def generate_daily_report(self):
        """生成日报"""
        try:
            stats = trading_state.get_daily_summary()
            hist_stats = db.get_statistics(days=30)
            
            report = f"""
📊 DeepSeek交易机器人日报
⏰ 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*50}

🎯 今日表现:
• 交易次数: {stats['trades']}笔
• 盈利次数: {stats['wins']}笔
• 亏损次数: {stats['losses']}笔
• 今日胜率: {stats['win_rate']:.1f}%
• 日盈亏: {stats['daily_pnl']:+.2%}
• 连续亏损: {stats['consecutive_losses']}次
• 当前持仓: {stats['open_positions']}个

📈 30日统计:
• 总交易: {hist_stats.get('total_trades', 0)}笔
• 历史胜率: {hist_stats.get('win_rate', 0):.1f}%
• 累计盈亏: ${hist_stats.get('total_pnl', 0):.2f}
• 平均盈亏: ${hist_stats.get('avg_pnl', 0):.2f}

💰 账户状态:
• 投资组合价值: ${trading_state.portfolio_value:.2f}
• 初始余额: ${trading_state.initial_balance:.2f}
• 总回报率: {((trading_state.portfolio_value / trading_state.initial_balance - 1) * 100) if trading_state.initial_balance > 0 else 0:.2f}%

🚀 运行信息:
• 运行时长: {(datetime.now() - self.start_time).days}天
• 系统状态: {'✅ 健康' if self._health_check() else '⚠️ 异常'}
• AI决策模式: {'✅ 启用' if TRADE_CONFIG['ai_decision_mode']['enabled'] else '❌ 关闭'}
{'='*50}
            """
            
            logger.info(report)
            return report
            
        except Exception as e:
            logger.error(f"❌ 生成日报失败: {e}")
            return "报告生成失败"
    
    def _health_check(self):
        """健康检查"""
        checks = [
            exchange is not None,
            deepseek_client is not None,
            trading_state.portfolio_value > 0
        ]
        return all(checks)

# ==================== 📝 移动止损管理器 ====================
class TrailingStopManager:
    """移动止损管理器 - 自动上调止损锁定利润"""
    
    def __init__(self):
        self.trailing_enabled = TRADE_CONFIG['risk_management']['trailing_stop_enabled']
        self.activation_percent = TRADE_CONFIG['risk_management']['trailing_stop_activation']
        self.distance_percent = TRADE_CONFIG['risk_management']['trailing_stop_distance']
        self.tracked_positions = {}  # 记录已触发移动止损的持仓
    
    def check_and_update_trailing_stop(self, symbol, position_data):
        """检查并更新移动止损"""
        if not self.trailing_enabled:
            return
        
        try:
            entry_price = position_data.get('entry_price', 0)
            current_price = position_data.get('current_price', 0)
            side = position_data.get('side', 'long')
            pnl_percent = position_data.get('pnl_percent', 0)
            
            if entry_price == 0 or current_price == 0:
                return
            
            # 计算盈利百分比
            if side == 'long':
                profit_percent = (current_price - entry_price) / entry_price
            else:
                profit_percent = (entry_price - current_price) / entry_price
            
            # 检查是否达到激活条件
            if profit_percent >= self.activation_percent:
                # 计算新的止损位
                if side == 'long':
                    new_stop_loss = current_price * (1 - self.distance_percent)
                else:
                    new_stop_loss = current_price * (1 + self.distance_percent)
                
                # 获取当前止损位
                current_stop_loss = position_data.get('stop_loss', 0)
                
                # 只有新止损位更优时才更新
                should_update = False
                if side == 'long' and new_stop_loss > current_stop_loss:
                    should_update = True
                elif side == 'short' and new_stop_loss < current_stop_loss:
                    should_update = True
                
                if should_update:
                    self._update_stop_loss_order(symbol, new_stop_loss, position_data)
                    
                    # 记录移动止损状态
                    if symbol not in self.tracked_positions:
                        self.tracked_positions[symbol] = {
                            'initial_stop': current_stop_loss,
                            'highest_price': current_price,
                            'updates': 0
                        }
                    
                    self.tracked_positions[symbol]['highest_price'] = current_price
                    self.tracked_positions[symbol]['updates'] += 1
                    
                    logger.info(f"📈 {symbol} 移动止损已更新:")
                    logger.info(f"   当前价格: ${current_price:.4f}")
                    logger.info(f"   盈利: {profit_percent:+.2%}")
                    logger.info(f"   旧止损: ${current_stop_loss:.4f}")
                    logger.info(f"   新止损: ${new_stop_loss:.4f}")
                    logger.info(f"   锁定利润: {((new_stop_loss - entry_price) / entry_price):+.2%}")
                    
        except Exception as e:
            logger.error(f"❌ 更新移动止损失败: {e}")
    
    def _update_stop_loss_order(self, symbol, new_stop_loss, position_data):
        """更新止损订单"""
        try:
            if TRADE_CONFIG['test_mode']:
                logger.info(f"🧪 测试模式 - 模拟更新止损: {symbol} -> ${new_stop_loss:.4f}")
                position_data['stop_loss'] = new_stop_loss
                return
            
            # 取消旧的止损单
            try:
                open_orders = exchange.fetch_open_orders(symbol)
                for order in open_orders:
                    if order.get('type') == 'stop_market' or order.get('stopPrice'):
                        exchange.cancel_order(order['id'], symbol)
                        logger.info(f"✅ 已取消旧止损单: {order['id']}")
            except Exception as e:
                logger.warning(f"⚠️ 取消旧止损单失败: {e}")
            
            # 创建新的止损单
            side = 'sell' if position_data['side'] == 'long' else 'buy'
            quantity = position_data.get('quantity', 0)
            
            new_order = safe_api_call(
                exchange.create_order,
                symbol,
                'stop_market',
                side,
                quantity,
                None,
                {
                    'stopPrice': new_stop_loss,
                    'reduceOnly': True
                }
            )
            
            if new_order:
                position_data['stop_loss'] = new_stop_loss
                logger.info(f"✅ 新止损单已设置: ${new_stop_loss:.4f}")
            
        except Exception as e:
            logger.error(f"❌ 更新止损订单失败: {e}")

# ==================== 💰 小资金仓位优化器 ====================
class SmallCapitalPositionOptimizer:
    """小资金仓位优化器 - 支持100U操作不同价格币种"""
    
    def __init__(self):
        self.min_position_value = 5  # 最小开仓价值5 USDT
        self.max_position_value = 1000  # 最大单笔1000 USDT
    
    def calculate_optimal_quantity(self, symbol, entry_price, position_value, leverage):
        """计算最优数量 - 适配不同价格币种"""
        try:
            # 获取交易对精度信息
            market = exchange.market(symbol) if exchange else None
            if not market:
                logger.error(f"❌ 无法获取{symbol}市场信息")
                return 0
            
            # 精度信息
            amount_precision = market.get('precision', {}).get('amount', 8)
            min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)
            
            # 计算实际可用金额（考虑杠杆）
            actual_position_value = position_value * leverage
            
            # 计算数量
            quantity = actual_position_value / entry_price
            
            # 应用精度
            quantity = self._round_to_precision(quantity, amount_precision)
            
            # 检查最小数量
            if quantity < min_amount:
                logger.warning(f"⚠️ {symbol} 数量{quantity:.8f}小于最小值{min_amount}")
                return 0
            
            # 计算实际开仓价值
            actual_value = quantity * entry_price / leverage
            
            # 检查是否在合理范围
            if actual_value < self.min_position_value:
                logger.warning(f"⚠️ {symbol} 开仓价值${actual_value:.2f}低于最小值${self.min_position_value}")
                return 0
            
            if actual_value > self.max_position_value:
                logger.warning(f"⚠️ {symbol} 开仓价值${actual_value:.2f}超过最大值${self.max_position_value}")
                quantity = (self.max_position_value * leverage) / entry_price
                quantity = self._round_to_precision(quantity, amount_precision)
            
            logger.info(f"💰 {symbol} 仓位计算:")
            logger.info(f"   价格: ${entry_price:.8f}")
            logger.info(f"   数量: {quantity:.8f}")
            logger.info(f"   杠杆: {leverage}x")
            logger.info(f"   保证金: ${actual_value:.2f}")
            logger.info(f"   名义价值: ${quantity * entry_price:.2f}")
            
            return quantity
            
        except Exception as e:
            logger.error(f"❌ 计算仓位失败: {e}")
            return 0
    
    def _round_to_precision(self, value, precision):
        """按精度舍入"""
        if precision == 0:
            return int(value)
        return round(value, precision)
    
    def validate_multi_position_allocation(self, available_balance, planned_positions):
        """验证多仓位分配 - 确保100U能同时持有BTC和DOGE"""
        try:
            total_margin_required = 0
            
            for pos in planned_positions:
                symbol = pos['symbol']
                quantity = pos['quantity']
                entry_price = pos['entry_price']
                leverage = pos['leverage']
                
                # 计算所需保证金
                margin = (quantity * entry_price) / leverage
                total_margin_required += margin
                
                logger.info(f"📊 {symbol} 所需保证金: ${margin:.2f}")
            
            # 预留10%缓冲
            safety_margin = available_balance * 0.1
            
            if total_margin_required > (available_balance - safety_margin):
                logger.warning(f"⚠️ 资金不足:")
                logger.warning(f"   可用余额: ${available_balance:.2f}")
                logger.warning(f"   所需保证金: ${total_margin_required:.2f}")
                logger.warning(f"   安全缓冲: ${safety_margin:.2f}")
                return False
            
            logger.info(f"✅ 资金分配验证通过:")
            logger.info(f"   可用余额: ${available_balance:.2f}")
            logger.info(f"   已分配: ${total_margin_required:.2f}")
            logger.info(f"   剩余: ${available_balance - total_margin_required:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 验证资金分配失败: {e}")
            return False

# 全局实例
trailing_stop_manager = TrailingStopManager()
small_capital_optimizer = SmallCapitalPositionOptimizer()

# ==================== 🎮 主交易引擎 ====================
class DeepSeekTradingBot:
    """DeepSeek交易机器人主引擎"""
    
    def __init__(self):
        self.market_data_provider = MarketDataProvider()
        self.strategy_engine = MultiStrategyEngine()
        self.ai_engine = DeepSeekDecisionEngine(deepseek_client)
        self.trading_executor = TradingExecutor()
        self.position_monitor = PositionMonitor()
        self.performance_analyzer = PerformanceAnalyzer()
        
        self.cycle_count = 0
        self.last_analysis_time = None
        
        # 🚀 AI决策模式状态
        self.ai_mode_active = TRADE_CONFIG['ai_decision_mode']['enabled']
        
        logger.info("🚀 DeepSeek交易机器人初始化完成")
        if self.ai_mode_active:
            logger.info("🎯 AI决策主导模式已启用")
    
    def run_trading_cycle(self):
        """运行交易周期"""
        global exchange  # 🚨 修复：在函数顶部声明 global
        
        self.cycle_count += 1
        logger.info(f"\n{'🎯'*20}")
        logger.info(f"开始第{self.cycle_count}个交易周期")
        if self.ai_mode_active:
            logger.info("🚀 AI决策主导模式运行中")
        logger.info(f"{'🎯'*20}\n")
        
        try:
            # 检查交易所连接
            if not exchange:  # 🚨 现在这行在 global 声明之后
                logger.error("❌ 交易所连接丢失，尝试重连...")
                exchange = initialize_exchange('okx')
                if not exchange:
                    logger.error("❌ 重连失败，跳过本次周期")
                    return
            
            # 重置每日统计
            trading_state.reset_daily_stats()
            
            # 1. 监控当前持仓
            logger.info("👀 监控当前持仓...")
            self.position_monitor.monitor_all_positions()
            
            # 2. 输出实时统计
            if trading_state.should_print_stats():
                self.performance_analyzer.print_realtime_stats()
            
            # 3. 全局风险检查
            if not self._pass_global_risk_checks():
                logger.warning("⚠️ 全局风险检查未通过，跳过本次周期")
                return
            
            # 4. 分析所有目标币种
            logger.info("🔍 分析目标币种...")
            trading_opportunities = self._analyze_all_symbols()
            
            if not trading_opportunities:
                logger.info("💤 未发现交易机会")
                return
            
            # 5. 选择最佳交易机会
            best_opportunity = self._select_best_opportunity(trading_opportunities)
            
            if best_opportunity:
                # 6. 执行交易
                logger.info(f"🎯 执行最佳交易机会: {best_opportunity['symbol']}")
                success = self.trading_executor.execute_trade(
                    best_opportunity['symbol'],
                    best_opportunity['signal_data'],
                    best_opportunity['market_data']
                )
                
                if success:
                    logger.info(f"✅ 交易执行成功")
                else:
                    logger.warning("⚠️ 交易执行失败")
            
            self.last_analysis_time = datetime.now()
            logger.info(f"✅ 第{self.cycle_count}个交易周期完成")
            
        except Exception as e:
            logger.error(f"❌ 交易周期执行失败: {e}")
            logger.error(f"🔧 错误详情: {traceback.format_exc()}")
            
            # 紧急恢复机制
            self._emergency_recovery()
    
    def _analyze_all_symbols(self):
        """分析所有目标币种"""
        opportunities = []
        
        for symbol in TRADE_CONFIG['target_symbols']:
            try:
                logger.info(f"📈 分析 {symbol}...")
                
                # 获取市场数据
                market_data = self.market_data_provider.get_multi_timeframe_data(symbol)
                if not market_data or not market_data['timeframes']:
                    continue
                
                # 多策略分析
                strategy_analysis = self.strategy_engine.analyze_symbol(symbol, market_data['timeframes'])
                
                # AI决策
                signal_data = self.ai_engine.generate_trading_signal(symbol, market_data, strategy_analysis)
                
                # 🚀 AI决策模式：使用AI信心度作为主要过滤条件
                min_confidence = TRADE_CONFIG['ai_decision_mode']['min_confidence']
                if signal_data['confidence'] < min_confidence:
                    logger.info(f"⭐ {symbol} AI信心度不足: {signal_data['confidence']:.1%} < {min_confidence:.1%}")
                    continue
                
                # 只交易BUY/SELL信号
                if signal_data['signal'] == 'HOLD':
                    logger.info(f"⏸️  {symbol} 信号: HOLD")
                    continue
                
                # 计算综合评分
                composite_score = self._calculate_composite_score(strategy_analysis, signal_data)
                
                opportunity = {
                    'symbol': symbol,
                    'composite_score': composite_score,
                    'market_data': market_data,
                    'strategy_analysis': strategy_analysis,
                    'signal_data': signal_data
                }
                
                opportunities.append(opportunity)
                
                logger.info(f"✅ {symbol} 分析完成 - 评分: {composite_score:.1f} - 信号: {signal_data['signal']} ({signal_data['confidence']:.1%})")
                
                time.sleep(0.5)  # 避免API限流
                
            except Exception as e:
                logger.error(f"❌ 分析{symbol}失败: {e}")
                continue
        
        return opportunities
    
    def _calculate_composite_score(self, strategy_analysis, signal_data):
        """计算综合评分"""
        try:
            # 🚀 AI决策模式：AI信心度权重更高
            if self.ai_mode_active:
                # AI模式下，AI信心度占70%，策略评分占30%
                strategy_score = strategy_analysis.get('final_score', 0)
                ai_confidence = signal_data.get('confidence', 0)
                
                composite_score = (
                    strategy_score * 0.3 +          # 策略分析占30%
                    ai_confidence * 100 * 0.7       # AI信心度占70%
                )
            else:
                # 传统模式：平衡权重
                strategy_score = strategy_analysis.get('final_score', 0)
                ai_confidence = signal_data.get('confidence', 0)
                
                composite_score = (
                    strategy_score * 0.5 +          # 策略分析占50%
                    ai_confidence * 100 * 0.5       # AI信心度占50%
                )
            
            # 风险回报比加分
            rr_ratio = signal_data.get('expected_risk_reward', 0)
            min_rr_ratio = TRADE_CONFIG['risk_management']['risk_reward_ratio']
            rr_bonus = min(20, (rr_ratio - min_rr_ratio) * 10) if rr_ratio > min_rr_ratio else 0
            
            # 策略一致性加分
            strategies = strategy_analysis.get('strategies', {})
            directions = [s.get('direction') for s in strategies.values()]
            if directions.count(signal_data['signal'].lower()) >= 2:
                consistency_bonus = 10
            else:
                consistency_bonus = 0
            
            # 最终综合评分
            final_score = composite_score + rr_bonus + consistency_bonus
            
            return max(0, min(100, final_score))
            
        except Exception as e:
            logger.error(f"❌ 计算综合评分失败: {e}")
            return 0
    
    def _select_best_opportunity(self, opportunities):
        """选择最佳交易机会"""
        if not opportunities:
            return None
        
        # 按综合评分排序
        sorted_opportunities = sorted(opportunities, key=lambda x: x['composite_score'], reverse=True)
        
        best_opportunity = sorted_opportunities[0]
        
        # 显示排名
        logger.info("\n🏆 交易机会排名:")
        for i, opp in enumerate(sorted_opportunities[:5], 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📊"
            logger.info(f"  {emoji} {opp['symbol']}: {opp['composite_score']:.1f}分 - {opp['signal_data']['signal']} ({opp['signal_data']['confidence']:.1%})")
        
        # 🚀 AI决策模式：降低评分门槛
        if self.ai_mode_active:
            min_score = 70  # AI模式下门槛稍低
        else:
            min_score = 75  # 传统模式门槛较高
        
        # 检查是否达到最低分数要求
        if best_opportunity['composite_score'] < min_score:
            logger.info(f"💤 最佳机会评分{best_opportunity['composite_score']:.1f}低于{min_score}分，放弃交易")
            return None
        
        return best_opportunity
    
    def _pass_global_risk_checks(self):
        """全局风险检查"""
        # 日亏损检查
        if trading_state.daily_pnl < -TRADE_CONFIG['risk_management']['max_daily_loss']:
            logger.warning(f"🚨 达到日亏损限制: {trading_state.daily_pnl:.2%}")
            return False
        
        # 交易频率检查
        if (self.last_analysis_time and 
            (datetime.now() - self.last_analysis_time).seconds < 300):
            logger.info("⏰ 交易频率限制，等待...")
            return False
        
        # 持仓数量检查
        if len(trading_state.positions) >= TRADE_CONFIG['risk_management']['max_open_positions']:
            logger.info(f"📦 达到最大持仓数量: {len(trading_state.positions)}")
            return False
        
        return True
    
    def _emergency_recovery(self):
        """紧急恢复机制"""
        global exchange  # 🚨 修复：在函数顶部声明 global
        
        try:
            logger.warning("🔄 执行紧急恢复...")
            
            # 清理缓存
            if hasattr(self, 'market_data_provider'):
                self.market_data_provider.cache.clear()
            
            # 强制垃圾回收
            import gc
            gc.collect()
            
            # 重置交易所连接
            if exchange:  # 🚨 现在这行在 global 声明之后
                try:
                    exchange.close()
                except:
                    pass
            
            # 重新初始化交易所
            exchange = initialize_exchange('okx')
            
            logger.info("✅ 紧急恢复完成")
            
        except Exception as e:
            logger.error(f"❌ 紧急恢复失败: {e}")

# ==================== 🔄 定时任务管理 ====================
def setup_scheduling(bot):
    """设置定时任务"""
    
    # 主要交易周期 - 每15分钟
    schedule.every(15).minutes.do(bot.run_trading_cycle)
    logger.info("⏰ 设置交易周期: 每15分钟")
    
    # 持仓监控 - 每5分钟
    schedule.every(5).minutes.do(bot.position_monitor.monitor_all_positions)
    logger.info("👀 设置持仓监控: 每5分钟")
    
    # 实时统计 - 每30分钟
    schedule.every(30).minutes.do(bot.performance_analyzer.print_realtime_stats)
    logger.info("📊 设置实时统计: 每30分钟")
    
    # 日报生成 - 每天8点
    schedule.every().day.at("08:00").do(bot.performance_analyzer.generate_daily_report)
    logger.info("📈 设置日报生成: 每天08:00")
    
    # 健康检查 - 每10分钟
    schedule.every(10).minutes.do(system_health_check)
    logger.info("❤️ 设置健康检查: 每10分钟")

def system_health_check():
    """系统健康检查"""
    try:
        checks = []
        
        # 检查交易所连接
        if exchange:
            try:
                balance = safe_api_call(exchange.fetch_balance)
                checks.append(balance is not None)
            except:
                checks.append(False)
        else:
            checks.append(False)
        
        # 检查AI客户端
        checks.append(deepseek_client is not None)
        
        # 检查数据库连接
        checks.append(db.conn is not None)
        
        # 检查内存使用
        try:
            import psutil
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024
            checks.append(memory_usage < 500)  # 内存使用小于500MB
        except:
            checks.append(True)  # 如果无法检查内存，假设正常
        
        health_status = all(checks)
        
        if not health_status:
            logger.warning(f"⚠️ 系统健康检查失败: {checks}")
        else:
            logger.info("✅ 系统健康检查通过")
            
        return health_status
        
    except Exception as e:
        logger.error(f"❌ 健康检查异常: {e}")
        return False

# ==================== 🎬 主函数 ====================
def main():
    """主函数"""
    global exchange  # 🚨 修复：在函数顶部声明 global
    
    print("\n" + "="*70)
    print("🚀 DeepSeek AI合约交易机器人 - 完整优化版")
    print("🎯 多策略决策 + 动态风控 + 投资组合管理 + 实时统计")
    if TRADE_CONFIG['ai_decision_mode']['enabled']:
        print("🚀 AI决策主导模式已启用")
    print("="*70 + "\n")
    
    # 初始化交易所
    if not exchange:  # 🚨 现在这行在 global 声明之后
        logger.error("❌ 交易所初始化失败")
        return
    
    # 测试连接并初始化账户状态
    try:
        balance = safe_api_call(exchange.fetch_balance)
        if balance:
            trading_state.portfolio_value = balance['total']['USDT']
            trading_state.initial_balance = trading_state.portfolio_value
            logger.info(f"💰 账户余额: {trading_state.portfolio_value:.2f} USDT")
    except Exception as e:
        logger.error(f"❌ 账户连接测试失败: {e}")
        return
    
    # 创建交易机器人实例
    bot = DeepSeekTradingBot()
    
    # 设置定时任务
    setup_scheduling(bot)
    
    # 立即执行一次
    logger.info("\n🎬 立即执行首次分析...")
    bot.run_trading_cycle()
    
    logger.info("\n✅ 交易机器人已启动，进入监控模式...")
    logger.info("💡 使用 Ctrl+C 停止程序\n")
    
    # 主循环
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\n👋 用户中断程序")
        logger.info("📊 生成最终报告...")
        bot.performance_analyzer.generate_daily_report()
        logger.info("🛑 程序已安全停止")

def signal_handler(signum, frame):
    """信号处理器 - 用于PM2动态命令"""
    if signum == signal.SIGUSR1:
        logger.info("📋 收到SIGUSR1信号，生成日报...")
        analyzer = PerformanceAnalyzer()
        analyzer.generate_daily_report()
    elif signum == signal.SIGUSR2:
        logger.info("📊 收到SIGUSR2信号，输出统计...")
        analyzer = PerformanceAnalyzer()
        analyzer.print_realtime_stats()

# ==================== 🎯 命令行接口 ====================
if __name__ == "__main__":
    import sys
    
    # 注册信号处理器
    signal.signal(signal.SIGUSR1, signal_handler)
    signal.signal(signal.SIGUSR2, signal_handler)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            logger.info("🧪 测试模式启动")
            TRADE_CONFIG['test_mode'] = True
            main()
        elif sys.argv[1] == "--single":
            logger.info("🔍 单次分析模式")
            bot = DeepSeekTradingBot()
            bot.run_trading_cycle()
        elif sys.argv[1] == "--monitor":
            logger.info("👀 仅监控模式")
            monitor = PositionMonitor()
            monitor.monitor_all_positions()
        elif sys.argv[1] == "--stats":
            logger.info("📊 输出统计")
            analyzer = PerformanceAnalyzer()
            analyzer.print_realtime_stats()
        elif sys.argv[1] == "--report":
            logger.info("📈 生成报告")
            analyzer = PerformanceAnalyzer()
            analyzer.generate_daily_report()
        elif sys.argv[1] == "--ai-on":
            logger.info("🚀 启用AI决策模式")
            TRADE_CONFIG['ai_decision_mode']['enabled'] = True
            main()
        elif sys.argv[1] == "--ai-off":
            logger.info("🔧 关闭AI决策模式")
            TRADE_CONFIG['ai_decision_mode']['enabled'] = False
            main()
        elif sys.argv[1] == "--help":
            print("""
🤖 DeepSeek AI交易机器人 - 使用说明

命令:
  python bot888.py              # 正常启动（15分钟周期）
  python bot888.py --test       # 测试模式（不实际下单）
  python bot888.py --single     # 单次分析
  python bot888.py --monitor    # 仅监控持仓
  python bot888.py --stats      # 输出实时统计
  python bot888.py --report     # 生成日报
  python bot888.py --ai-on      # 启用AI决策模式
  python bot888.py --ai-off     # 关闭AI决策模式
  python bot888.py --help       # 显示帮助

PM2动态命令:
  pm2 sendSignal SIGUSR1 88-trader  # 生成日报
  pm2 sendSignal SIGUSR2 88-trader  # 输出统计

配置:
  请确保 .env 文件中包含以下配置:
  - DEEPSEEK_API_KEY
  - OKX_API_KEY / BINANCE_API_KEY
  - OKX_SECRET / BINANCE_SECRET
  - OKX_PASSWORD (仅OKX需要)

核心功能:
  ✅ 多策略决策引擎（趋势/均值回归/突破）
  ✅ DeepSeek AI智能决策（可开关）
  ✅ 动态杠杆和仓位管理
  ✅ 严格风险控制系统
  ✅ 实时监控和统计（每30分钟）
  ✅ 自动止损止盈 + 移动止损
  ✅ 连续亏损保护
  ✅ 完整的交易日志
  ✅ 小资金多币种支持
  ✅ 24小时不间断运行

风险提示:
  ⚠️  请在测试模式充分验证后再使用实盘
  ⚠️  建议从小资金开始
  ⚠️  严格监控机器人运行状态
  ⚠️  合约交易有爆仓风险，请谨慎操作
            """)
        else:
            print(f"❓ 未知参数: {sys.argv[1]}")
            print("使用 --help 查看帮助")
    else:
        # 正常启动
        main()