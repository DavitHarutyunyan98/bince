import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table, no_update
import plotly.graph_objs as go
from binance.client import Client
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import logging
import json
import io
import os
import random
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import re
import telegram
import asyncio
from dash.dependencies import Input, Output, State, ALL, MATCH
from strategy_utils import Backtester, STRATEGY_REGISTRY, CandlestickStrategy, SuperTrendStrategy
from stability_metrics import StabilityAnalyzer, integrate_stability_into_optimization
import optuna
import threading
import time

# --- Configuration ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Configuration ---
OPTIMIZATION_LOGS = []
PARTIAL_RESULTS_LIST = []  # To store results as they complete
LOG_FILENAME = None  # To store the current log filename
TELEGRAM_BOT_TOKEN = None
TELEGRAM_CHAT_ID = None
OPTIMIZATION_STOP_EVENT = threading.Event()
OPTIMIZATION_PAUSE_EVENT = threading.Event()  # For pausing optimization

# Robust annualization factors for Sharpe Ratio calculation across different timeframes
TIMEFRAME_TO_ANNUALIZATION_FACTOR = {
    '1m': np.sqrt(365 * 24 * 60), '5m': np.sqrt(365 * 24 * 12), '15m': np.sqrt(365 * 24 * 4),
    '30m': np.sqrt(365 * 24 * 2), '1h': np.sqrt(365 * 24), '4h': np.sqrt(365 * 6), '1d': np.sqrt(365)
}



def add_optimization_log(message):
    """Adds a timestamped message to the global log list and writes to a file."""
    global LOG_FILENAME
    timestamp = datetime.now().strftime('%H:%M:%S')
    log_entry = f"[{timestamp}] {message}"
    OPTIMIZATION_LOGS.append(log_entry)
    if LOG_FILENAME:
        try:
            with open(LOG_FILENAME, 'a', encoding='utf-8') as f:
                f.write(log_entry + '\n')
        except Exception as e:
            print(f"Error: Could not write to log file {LOG_FILENAME}: {e}")


def send_telegram_notification(message, files=None):
    """Sends a message and optional files to Telegram using asyncio."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        add_optimization_log(
            "Telegram credentials not configured. Skipping notification.")
        return

    async def main():
        try:
            bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
            await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
            if files:
                for file_info in files:
                    file_object = file_info.get('object')
                    filename = file_info.get('filename')
                    caption = file_info.get('caption', '')
                    if file_object and filename:
                        file_object.seek(0)
                        await bot.send_document(
                            chat_id=TELEGRAM_CHAT_ID,
                            document=file_object,
                            filename=filename,
                            caption=caption
                        )
            add_optimization_log("✅ Telegram notification sent successfully.")
        except Exception as e:
            add_optimization_log(
                f"!! Failed to send Telegram notification: {e}")

    try:
        asyncio.run(main())
    except RuntimeError:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())


# FIXED: Merged two conflicting classes into one unified FuturesTrader class
class FuturesTrader:
    def __init__(self, client, config):
        self.client = client
        self.config = config
        self.symbol = self.config.get('symbol', 'BTCUSDT').upper()
        self.data = pd.DataFrame()
        self.last_error = None
        self.data_cache = {}
        # Optimizer specific attributes
        self.initial_capital = 10000
        self.max_workers = os.cpu_count() or 4
        # Dynamic strategy selection based on parameters
        self.optimizer_strategy = CandlestickStrategy()  # Default strategy

    def __getstate__(self):
        """
        Make FuturesTrader picklable so per-symbol studies can run in a
        ProcessPoolExecutor. The Binance client is not picklable (and the
        worker processes don't need it — historical data is fetched up front
        and passed in), so we drop it. The data_cache can be large, so we drop
        it too; children only need their own pre-fetched DataFrame.
        """
        state = self.__dict__.copy()
        state["client"] = None
        state["data_cache"] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _get_strategy_for_params(self, selected_params):
        """Determine which strategy to use based on selected parameters."""
        from strategy_utils import SuperTrendStrategy
        
        # Check if ATR parameters are selected
        atr_params = {'atr_period', 'atr_multiplier'}
        candlestick_params = {'buy_signal_window', 'buy_pattern_lookback', 'sell_signal_window', 'sell_pattern_lookback'}
        
        selected_set = set(selected_params)
        
        if atr_params.intersection(selected_set):
            return SuperTrendStrategy()
        elif candlestick_params.intersection(selected_set):
            return CandlestickStrategy()
        else:
            return CandlestickStrategy()  # Default fallback

    def load_multi_pair_configs(self, config_path='trade_config.json'):
        """Load multiple trading pair configurations from JSON file."""
        try:
            with open(config_path, 'r') as f:
                configs = json.load(f)
            return [config for config in configs if config.get('enabled', True)]
        except Exception as e:
            logger.error(f"Error loading multi-pair configs: {e}")
            return []

    def save_multi_pair_configs(self, configs, config_path='trade_config.json'):
        """Save multiple trading pair configurations to JSON file."""
        try:
            with open(config_path, 'w') as f:
                json.dump(configs, f, indent=4)
            return True
        except Exception as e:
            logger.error(f"Error saving multi-pair configs: {e}")
            return False

    def launch_multi_pair_trading(self):
        """Launch the enhanced multi-pair trading bot."""
        try:
            from hybrid_trader import HybridTraderManager as TraderManager
            # Update the launch method to use correct class name
            manager = HybridTraderManager('trade_config.json', 'config.json')


            # Send notification about launch
            send_telegram_notification(
                "🚀 *Multi-Pair Trading Started from UI*\n\nBot launched from main interface.")

            # Start in a separate thread to avoid blocking UI
            import threading
            trading_thread = threading.Thread(
                target=manager.start, daemon=True)
            trading_thread.start()

            add_optimization_log(
                "✅ Multi-pair trading bot launched successfully")
            return True

        except Exception as e:
            add_optimization_log(f"❌ Failed to launch multi-pair trading: {e}")
            return False

    def export_best_results_to_config(self, results_df, top_n=5):
        """Export top N optimization results to trade_config.json for trading."""
        try:
            if results_df is None or results_df.empty:
                return False

            # Get the BEST result for EACH pair (not top N overall)
            best_per_pair = results_df.loc[results_df.groupby(
                'Trading_Pair')['Total_Return'].idxmax()]

            # Sort pairs by their best performance and take top N pairs
            best_pairs = best_per_pair.sort_values(
                'Total_Return', ascending=False).head(top_n)

            configs = []
            for _, row in best_pairs.iterrows():
                # Determine strategy type based on available columns
                if 'ATR_Period' in row and 'ATR_Multiplier' in row:
                    strategy_name = "ATR SuperTrend"
                    config = {
                        "enabled": True,
                        "strategy_name": strategy_name,
                        "symbol": row['Trading_Pair'],
                        "bar_length": row['Timeframe'],
                        "units_usdt": 50.0,
                        "leverage": 10,
                        "atr_period": int(row['ATR_Period']),
                        "atr_multiplier": float(row['ATR_Multiplier'])
                    }
                else:
                    strategy_name = "Candlestick Patterns"
                    config = {
                        "enabled": True,
                        "strategy_name": strategy_name,
                        "symbol": row['Trading_Pair'],
                        "bar_length": row['Timeframe'],
                        "units_usdt": 50.0,
                        "leverage": 10,
                        "buy_signal_window": int(row['Buy_Signal_Window']),
                        "buy_pattern_lookback": int(row['Buy_Pattern_Lookback']),
                        "sell_signal_window": int(row['Sell_Signal_Window']),
                        "sell_pattern_lookback": int(row['Sell_Pattern_Lookback'])
                    }
                configs.append(config)

            # Save to trade_config.json
            with open('trade_config.json', 'w') as f:
                json.dump(configs, f, indent=2)

            unique_pairs = len(best_pairs['Trading_Pair'].unique())
            add_optimization_log(
                f"✅ Exported best result for {unique_pairs} pairs to trade_config.json")
            for _, row in best_pairs.iterrows():
                add_optimization_log(
                    f"   {row['Trading_Pair']}: {row['Total_Return']:.2f}% return")
            return True

        except Exception as e:
            add_optimization_log(f"❌ Failed to export results: {e}")
            return False

    def export_best_results_to_config_enhanced(self, results_df, top_n=5, sort_by='Score',
                                               units_usdt=50.0, leverage=10, timeframe='15m'):
        """Enhanced export with custom sorting and configuration parameters."""
        try:
            if results_df is None or results_df.empty:
                return False

            # Get the BEST result for EACH pair based on the selected sorting column
            if sort_by in results_df.columns:
                # For each pair, get the row with the best value in the sort column
                best_per_pair = results_df.loc[results_df.groupby('Trading_Pair')[
                    sort_by].idxmax()]
            else:
                # Fallback to Score if column doesn't exist
                best_per_pair = results_df.loc[results_df.groupby('Trading_Pair')[
                    'Score'].idxmax()]

            # Sort pairs by their best performance and take top N pairs
            ascending = False  # Most metrics are better when higher
            if sort_by in ['Max_Drawdown']:  # These are better when lower
                ascending = True

            best_pairs = best_per_pair.sort_values(
                sort_by, ascending=ascending).head(top_n)

            configs = []
            for _, row in best_pairs.iterrows():
                # Determine strategy type based on available columns
                if 'ATR_Period' in row and 'ATR_Multiplier' in row:
                    strategy_name = "ATR SuperTrend"
                    config = {
                        "enabled": True,
                        "strategy_name": strategy_name,
                        "symbol": row['Trading_Pair'],
                        "bar_length": timeframe,  # Use the selected timeframe
                        "units_usdt": units_usdt,  # Use the selected trade size
                        "leverage": leverage,      # Use the selected leverage
                        "atr_period": int(row['ATR_Period']),
                        "atr_multiplier": float(row['ATR_Multiplier'])
                    }
                else:
                    strategy_name = "Candlestick Patterns"
                    config = {
                        "enabled": True,
                        "strategy_name": strategy_name,
                        "symbol": row['Trading_Pair'],
                        "bar_length": timeframe,  # Use the selected timeframe
                        "units_usdt": units_usdt,  # Use the selected trade size
                        "leverage": leverage,      # Use the selected leverage
                        "buy_signal_window": int(row['Buy_Signal_Window']),
                        "buy_pattern_lookback": int(row['Buy_Pattern_Lookback']),
                        "sell_signal_window": int(row['Sell_Signal_Window']),
                        "sell_pattern_lookback": int(row['Sell_Pattern_Lookback'])
                    }
                configs.append(config)

            # Save to trade_config.json
            with open('trade_config.json', 'w') as f:
                json.dump(configs, f, indent=2)

            unique_pairs = len(best_pairs['Trading_Pair'].unique())
            add_optimization_log(
                f"✅ Exported best result for {unique_pairs} pairs (sorted by {sort_by})")
            for _, row in best_pairs.iterrows():
                sort_value = row[sort_by]
                add_optimization_log(
                    f"   {row['Trading_Pair']}: {sort_value:.2f} ({sort_by})")

            return True

        except Exception as e:
            add_optimization_log(f"❌ Failed to export results: {e}")
            return False

    def get_historical_data_for_symbol(self, symbol, bar_length, start_date, end_date):
        """Fetches data for a specific symbol, used by optimizer."""
        cache_key = f"{symbol}-{bar_length}-{start_date}-{end_date}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        if self.client is None:
            return None
        try:
            df = self._get_data_by_date_range(
                symbol, bar_length, start_date, end_date)
            if df is not None and not df.empty:
                self.data_cache[cache_key] = df
            return df
        except Exception as e:
            logger.error(f"Failed to get data for {symbol}: {e}")
            return None

    def get_historical_data(self, symbol, bar_length, start_date=None, end_date=None):
        """Method for the manual backtester UI."""
        self.last_error = None
        self.symbol = symbol.upper()
        if self.client is None:
            self.last_error = "Binance client is not available."
            return None
        try:
            self.data = self._get_data_by_date_range(
                self.symbol, bar_length, start_date, end_date)
            if self.data is None or self.data.empty:
                self.last_error = f"Failed to fetch data for {self.symbol}."
            return self.data
        except Exception as e:
            self.last_error = f"An unexpected error occurred: {e}"
            return None

    def _get_data_by_date_range(self, symbol, bar_length, start_date_str, end_date_str):
        start_ms = int(datetime.strptime(
            start_date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp() * 1000)
        end_ms = int(datetime.strptime(
            end_date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp() * 1000)
        all_bars = []
        while start_ms < end_ms:
            bars = self.client.futures_historical_klines(symbol=symbol, interval=bar_length, start_str=str(start_ms),
                                                         limit=1500)
            if not bars:
                break
            all_bars.extend(bars)
            start_ms = bars[-1][0] + 1
        if not all_bars:
            return None
        df = self._process_bars_to_dataframe(all_bars)
        return df[~df.index.duplicated(keep='first')].sort_index()

    def _process_bars_to_dataframe(self, bars):
        df = pd.DataFrame(bars, columns=["Open Time", "Open", "High", "Low", "Close", "Volume", "Close Time",
                                         "Quote Asset Volume", "Number of Trades", "Taker Buy Base Asset Volume",
                                         "Taker Buy Quote Asset Volume", "Ignore"])
        df["Date"] = pd.to_datetime(df["Open Time"], unit="ms", utc=True)
        df.set_index("Date", inplace=True)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df[["Open", "High", "Low", "Close", "Volume"]]

    def get_all_usdt_futures_pairs(self):
        if not self.client:
            return []
        try:
            tickers = self.client.futures_ticker()
            return [{'symbol': t['symbol'], 'volume': float(t['quoteVolume']),
                     'volatility': abs(float(t['priceChangePercent']))}
                    for t in tickers if t['symbol'].endswith('USDT')]
        except Exception as e:
            self.last_error = f"Failed to fetch pairs: {e}"
            logger.error(f"Failed to fetch pairs from Binance: {e}")
            return []

    def _wilders_smoothing(self, series, length):
        """
        Calculate Wilder's Smoothing (RMA) - the correct method for ADX calculation.
        Formula: RMA(x, n) = ((prior_RMA * (n-1)) + current_x) / n
        
        Args:
            series: pandas Series to smooth
            length: smoothing period
        
        Returns:
            pandas Series with Wilder's smoothing applied
        """
        if series is None or series.empty or length <= 0:
            return series
        
        # Initialize result series
        result = pd.Series(index=series.index, dtype=float)
        
        # First value is simple average of first 'length' values
        first_valid_idx = length - 1
        if len(series) <= first_valid_idx:
            return pd.Series(0, index=series.index)
        
        # Calculate initial RMA (simple average of first 'length' values)
        result.iloc[first_valid_idx] = series.iloc[:length].mean()
        
        # Apply Wilder's smoothing formula for remaining values
        for i in range(first_valid_idx + 1, len(series)):
            prior_rma = result.iloc[i-1]
            current_value = series.iloc[i]
            result.iloc[i] = ((prior_rma * (length - 1)) + current_value) / length
        
        return result

    def calculate_adx(self, data, di_length=14, adx_smoothing=14):
        """
        Calculate ADX (Average Directional Index) using proper Wilder's Smoothing (RMA).
        This matches TradingView's ADX calculation exactly.
        
        Args:
            data: DataFrame with OHLC data
            di_length: Period for +DI/-DI calculation (default: 14)
            adx_smoothing: Period for final ADX smoothing (default: 14)
        
        Returns:
            DataFrame with ADX, +DI, -DI columns added
        """
        if data is None or data.empty or len(data) < max(di_length, adx_smoothing) + 1:
            return data
        
        df = data.copy()
        
        # Step 1: Calculate True Range (TR)
        df['prev_close'] = df['Close'].shift(1)
        df['tr1'] = abs(df['High'] - df['Low'])
        df['tr2'] = abs(df['High'] - df['prev_close']).fillna(0)
        df['tr3'] = abs(df['Low'] - df['prev_close']).fillna(0)
        df['TR'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate Directional Movement (+DM and -DM)
        df['high_diff'] = df['High'] - df['High'].shift(1)
        df['low_diff'] = df['Low'].shift(1) - df['Low']
        
        df['+DM'] = np.where(
            (df['high_diff'] > df['low_diff']) & (df['high_diff'] > 0),
            df['high_diff'], 0
        )
        df['-DM'] = np.where(
            (df['low_diff'] > df['high_diff']) & (df['low_diff'] > 0),
            df['low_diff'], 0
        )
        
        # Step 2: Smooth TR, +DM, -DM using Wilder's Smoothing over di_length
        df['ATR'] = self._wilders_smoothing(df['TR'], di_length)
        df['+DI_smooth'] = self._wilders_smoothing(df['+DM'], di_length)
        df['-DI_smooth'] = self._wilders_smoothing(df['-DM'], di_length)
        
        # Step 3: Calculate +DI and -DI
        df['+DI'] = np.where(df['ATR'] != 0, 100 * (df['+DI_smooth'] / df['ATR']), 0)
        df['-DI'] = np.where(df['ATR'] != 0, 100 * (df['-DI_smooth'] / df['ATR']), 0)
        
        # Step 4: Calculate DX = abs(+DI - -DI) / (+DI + -DI) * 100
        df['DI_sum'] = df['+DI'] + df['-DI']
        df['DI_diff'] = abs(df['+DI'] - df['-DI'])
        df['DX'] = np.where(df['DI_sum'] != 0, 100 * (df['DI_diff'] / df['DI_sum']), 0)
        
        # Step 5: Calculate ADX by smoothing DX using Wilder's Smoothing over adx_smoothing
        df['ADX'] = self._wilders_smoothing(df['DX'], adx_smoothing)
        
        # Clean up temporary columns
        cols_to_drop = ['prev_close', 'tr1', 'tr2', 'tr3', 'high_diff', 'low_diff', 
                       '+DI_smooth', '-DI_smooth', 'DI_sum', 'DI_diff', 'DX']
        df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        
        # Handle NaN values gracefully
        df['ADX'] = df['ADX'].fillna(0)
        df['+DI'] = df['+DI'].fillna(0)
        df['-DI'] = df['-DI'].fillna(0)
        
        return df

    def scan_market_adx(self, scan_limit=20, timeframe='1h', lookback_candles=200, 
                       di_length=14, adx_smoothing=14, strong_threshold=25, weak_threshold=20):
        """
        Scan top volume pairs for ADX trend strength analysis.
        
        Args:
            scan_limit: Number of top pairs to scan (configurable)
            timeframe: Timeframe for analysis (1m, 5m, 15m, 1h, 4h, 1d)
            lookback_candles: Number of candles to fetch for analysis
            di_length: Period for +DI/-DI calculation
            adx_smoothing: Period for final ADX smoothing
            strong_threshold: Threshold for strong trend classification (configurable)
            weak_threshold: Threshold for weak trend classification (configurable)
        
        Returns:
            List of dictionaries with ADX analysis results
        """
        if not self.client:
            return []
        
        try:
            # Get top pairs by volume
            all_pairs = self.get_all_usdt_futures_pairs()
            if not all_pairs:
                return []
            
            # Sort by volume and take top N
            top_pairs = sorted(all_pairs, key=lambda x: x['volume'], reverse=True)[:scan_limit]
            
            results = []
            
            # Calculate date range based on timeframe and lookback candles
            timeframe_minutes = {
                '1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440
            }
            
            minutes_needed = lookback_candles * timeframe_minutes.get(timeframe, 60)
            start_date = (datetime.now() - timedelta(minutes=minutes_needed * 2)).strftime('%Y-%m-%d')  # Extra buffer
            end_date = datetime.now().strftime('%Y-%m-%d')
            
            for pair_info in top_pairs:
                symbol = pair_info['symbol']
                try:
                    # Fetch historical data with specified timeframe
                    data = self.get_historical_data_for_symbol(symbol, timeframe, start_date, end_date)
                    if data is None or data.empty:
                        continue
                    
                    # Ensure we have enough data
                    if len(data) < max(di_length, adx_smoothing) + 10:
                        continue
                    
                    # Take only the last lookback_candles for analysis
                    data = data.tail(lookback_candles)
                    
                    # Calculate ADX with separate DI length and ADX smoothing
                    data_with_adx = self.calculate_adx(data, di_length=di_length, adx_smoothing=adx_smoothing)
                    
                    if data_with_adx.empty or 'ADX' not in data_with_adx.columns:
                        continue
                    
                    # Get latest values
                    latest_adx = data_with_adx['ADX'].iloc[-1]
                    current_price = data_with_adx['Close'].iloc[-1]
                    latest_plus_di = data_with_adx['+DI'].iloc[-1]
                    latest_minus_di = data_with_adx['-DI'].iloc[-1]
                    
                    # Skip if ADX is still 0 (not enough data for calculation)
                    if latest_adx == 0:
                        continue
                    
                    # Determine trend strength using configurable thresholds
                    if latest_adx >= strong_threshold:
                        trend_strength = "Strong"
                    elif latest_adx >= weak_threshold:
                        trend_strength = "Moderate"
                    else:
                        trend_strength = "Weak"
                    
                    # Determine trend direction
                    if latest_plus_di > latest_minus_di:
                        trend_direction = "↑"
                    else:
                        trend_direction = "↓"
                    
                    results.append({
                        'Symbol': symbol,
                        'Current Price': f"{current_price:.4f}",
                        'ADX Value': f"{latest_adx:.2f}",
                        'Trend Strength': f"{trend_strength} {trend_direction}",
                        'ADX_Numeric': latest_adx  # For sorting
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing {symbol} for ADX scan: {e}")
                    continue
            
            # Sort by ADX value (strongest trends first)
            results.sort(key=lambda x: x['ADX_Numeric'], reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in ADX market scan: {e}")
            return []

    # --- Optimizer Methods ---
    def _run_single_optimization_study(self, symbol, full_data, param_ranges, selected_params, timeframe, min_trades,
                                       n_trials, weights, is_start_date, is_end_date, oos1_start_date, oos1_end_date, oos2_start_date, oos2_end_date,
                                       stop_event, pause_event, strategy_name=None):
        """Runs an Optuna study for a single trading pair with IS/OOS and returns all trial results."""
        # Use specified strategy or fallback to dynamic selection
        if strategy_name:
            strategy_class = STRATEGY_REGISTRY.get(strategy_name)
            if strategy_class:
                strategy = strategy_class()
            else:
                strategy = self._get_strategy_for_params(selected_params)
        else:
            strategy = self._get_strategy_for_params(selected_params)
        default_params = strategy.get_parameters()

        # Strategy-First approach: Only keep parameters that actually exist in the strategy
        strategy_keys = set(default_params.keys())
        effective_params = [p for p in selected_params if p in strategy_keys]
        if len(effective_params) < len(selected_params):
            filtered_out = len(selected_params) - len(effective_params)
            add_optimization_log(f"-> {symbol}: Filtered out {filtered_out} invalid parameters for {strategy.__class__.__name__}")

        # Split data into In-Sample and dual Out-of-Sample periods
        is_data = full_data.loc[is_start_date:is_end_date]
        oos1_data = full_data.loc[oos1_start_date:oos1_end_date]
        oos2_data = full_data.loc[oos2_start_date:oos2_end_date]

        def run_backtest_and_get_metrics(data_df, params_dict):
            """Nested helper to run a backtest and return key metrics."""
            if data_df is None or data_df.empty:
                return {'Return': -1000, 'Trades': 0, 'Win_Rate': 0}

            df_with_signals = strategy.generate_signals(
                data_df.copy(), params_dict)
            backtester = Backtester(self.initial_capital)
            trades_df, portfolio_df = backtester.run_backtest(df_with_signals)

            if trades_df is None or portfolio_df is None or trades_df.empty:
                return {'Return': -1000, 'Trades': 0, 'Win_Rate': 0}

            total_return = (
                portfolio_df['Portfolio_Value'].iloc[-1] / self.initial_capital - 1) * 100
            win_rate = (len(trades_df[trades_df['PnL'] > 0]) /
                        len(trades_df) * 100) if not trades_df.empty else 0
            total_trades = len(trades_df)

            return {'Return': total_return, 'Trades': total_trades, 'Win_Rate': win_rate}

        def parse_range_for_optuna(range_str, is_int=True):
            if not range_str or not isinstance(range_str, str) or range_str.strip() == '':
                return None, None, None
            try:
                parts = [float(p.strip()) for p in range_str.split(',')]
                if len(parts) != 3:
                    return None, None, None
                low, high, step = parts

                if step <= 0:
                    return None, None, None
                if low > high:
                    return None, None, None

                return (int(low), int(high), int(step)) if is_int else (low, high, step)
            except:
                return None, None, None

        # Generate all unique parameter combinations upfront to avoid duplicates
        def generate_unique_combinations():
            """Generate all unique parameter combinations based on ranges."""
            combinations = []
            param_values = {}
            
            # Parse parameter ranges and generate value lists
            param_map = {
                'buy_signal_window': {'is_int': True}, 'buy_pattern_lookback': {'is_int': True},
                'sell_signal_window': {'is_int': True}, 'sell_pattern_lookback': {'is_int': True},
                'atr_period': {'is_int': True}, 'atr_multiplier': {'is_int': False}
            }
            
            # FIXED: Only process parameters that exist in the current strategy
            for p_name, p_config in param_map.items():
                if p_name not in default_params:
                    continue  # Skip parameters not in current strategy
                    
                if p_name in effective_params and param_ranges.get(p_name):
                    low, high, step = parse_range_for_optuna(
                        param_ranges[p_name], is_int=p_config['is_int'])
                    if low is not None:
                        if p_config['is_int']:
                            param_values[p_name] = list(range(int(low), int(high) + 1, int(step)))
                        else:
                            param_values[p_name] = [round(v, 4) for v in np.arange(
                                low, high + step / 2, step)]
                    else:
                        # Use default value if range parsing failed
                        param_values[p_name] = [default_params[p_name]['default']]
                else:
                    # Use default value for parameters not being optimized
                    param_values[p_name] = [default_params[p_name]['default']]
            
            # FIXED: Ensure we have at least one parameter to optimize
            if not param_values:
                add_optimization_log(f"-> {symbol}: No valid parameters found for optimization")
                return []
            
            # Generate all combinations
            import itertools
            param_names = list(param_values.keys())
            param_lists = [param_values[name] for name in param_names]
            
            # FIXED: Handle empty param_lists
            if not param_lists:
                return []
            
            for combination in itertools.product(*param_lists):
                param_dict = dict(zip(param_names, combination))
                combinations.append(param_dict)
            
            # Shuffle combinations for better distribution
            random.shuffle(combinations)
            add_optimization_log(f"-> {symbol}: Generated {len(combinations)} combinations for strategy {strategy.__class__.__name__}")
            return combinations

        # Track tested combinations to avoid duplicates
        tested_combinations = set()
        unique_combinations = generate_unique_combinations()
        
        add_optimization_log(f"-> {symbol}: Generated {len(unique_combinations)} unique parameter combinations")
        
        # Limit trials to available combinations or requested trials, whichever is smaller
        effective_trials = min(n_trials, len(unique_combinations))
        if effective_trials < n_trials:
            add_optimization_log(f"-> {symbol}: Limited to {effective_trials} trials (all unique combinations)")

        def objective(trial):
            # Heartbeat logging to verify loop is running
            print(f"DEBUG: {symbol} processing trial {trial.number}")
            
            if stop_event.is_set():
                raise optuna.exceptions.TrialPruned(
                    "Optimization stopped by user.")

            if pause_event.is_set():
                add_optimization_log(
                    f"-> {symbol}: Study paused. Waiting for continue signal...")
                while pause_event.is_set():
                    if stop_event.is_set():
                        raise optuna.exceptions.TrialPruned(
                            "Optimization stopped while paused.")
                    time.sleep(1)
                add_optimization_log(f"-> {symbol}: Study resumed.")

            # Use pre-generated unique combinations to avoid duplicates
            trial_index = trial.number
            if trial_index < len(unique_combinations):
                params = unique_combinations[trial_index].copy()
            else:
                # Fallback to random selection if we somehow exceed combinations
                params = random.choice(unique_combinations).copy()
            
            # Create a hashable key for this combination
            param_key = tuple(sorted(params.items()))
            
            # Skip if already tested (extra safety check)
            if param_key in tested_combinations:
                add_optimization_log(f"-> {symbol}: Skipping duplicate combination")
                return -1000.0
            
            tested_combinations.add(param_key)
            
            # Suggest parameters to Optuna (for compatibility)
            param_map = {
                'buy_signal_window': {'is_int': True}, 'buy_pattern_lookback': {'is_int': True},
                'sell_signal_window': {'is_int': True}, 'sell_pattern_lookback': {'is_int': True},
                'atr_period': {'is_int': True}, 'atr_multiplier': {'is_int': False}
            }
            for p_name, p_config in param_map.items():
                # SAFETY CHECK: Skip parameters not in current strategy
                if p_name not in default_params:
                    continue
                if p_name in selected_params and param_ranges.get(p_name):
                    low, high, step = parse_range_for_optuna(
                        param_ranges[p_name], is_int=p_config['is_int'])
                    if low is not None:
                        if p_config['is_int']:
                            trial.suggest_int(p_name, low, high, step=step)
                        else:
                            choices = [round(v, 4) for v in np.arange(
                                low, high + step / 2, step)]
                            trial.suggest_categorical(p_name, choices)
                
                # Set the actual parameter value from our pre-generated combination
                trial.set_user_attr(f'param_{p_name}', params[p_name])

            try:
                # --- IN-SAMPLE BACKTEST (for optimization score) ---
                df_with_signals_is = strategy.generate_signals(
                    is_data.copy(), params)
                backtester_is = Backtester(self.initial_capital)
                trades_df_is, portfolio_df_is = backtester_is.run_backtest(
                    df_with_signals_is)

                if trades_df_is is None or portfolio_df_is is None or len(trades_df_is) < min_trades:
                    return -1000.0

                # --- Calculate IS Metrics ---
                total_return_is = (
                    portfolio_df_is['Portfolio_Value'].iloc[-1] / self.initial_capital - 1) * 100
                win_rate_is = (len(trades_df_is[trades_df_is['PnL'] > 0]) / len(
                    trades_df_is) * 100) if not trades_df_is.empty else 0
                total_trades_is = len(trades_df_is)
                portfolio_values = portfolio_df_is['Portfolio_Value'].values
                running_max = np.maximum.accumulate(portfolio_values)
                drawdown = (running_max - portfolio_values) / running_max * 100
                max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
                returns = portfolio_df_is['Portfolio_Value'].pct_change(
                ).dropna()
                annualization_factor = TIMEFRAME_TO_ANNUALIZATION_FACTOR.get(
                    timeframe, np.sqrt(365))
                sharpe_ratio = (returns.mean() / returns.std() *
                                annualization_factor) if returns.std() > 0 else 0
                gross_profit = trades_df_is[trades_df_is['PnL'] > 0]['PnL'].sum(
                )
                gross_loss = abs(
                    trades_df_is[trades_df_is['PnL'] < 0]['PnL'].sum())
                profit_factor = gross_profit / \
                    gross_loss if gross_loss > 0 else float('inf')

                # --- Calculate IS Score ---
                total_weight = weights['total_return'] + \
                    weights['win_rate'] + weights['total_trades']
                if total_weight == 0:
                    total_weight = 1
                w_return = weights['total_return'] / total_weight
                w_winrate = weights['win_rate'] / total_weight
                w_trades = weights['total_trades'] / total_weight
                capped_trades = min(total_trades_is, 100)
                score = (total_return_is * w_return) + \
                    (win_rate_is * w_winrate) + (capped_trades * w_trades)

                # --- OUT-OF-SAMPLE BACKTESTS (for reporting) ---
                oos1_results = run_backtest_and_get_metrics(oos1_data, params)
                oos2_results = run_backtest_and_get_metrics(oos2_data, params)
                
                # Calculate combined OOS metrics
                total_oos_return = (oos1_results['Return'] + oos2_results['Return']) / 2  # Average return
                total_oos_trades = oos1_results['Trades'] + oos2_results['Trades']
                total_oos_win_rate = (oos1_results['Win_Rate'] + oos2_results['Win_Rate']) / 2  # Average win rate

                # Calculate additional priority metrics
                # Risk-adjusted return (Calmar ratio)
                calmar_ratio = total_return_is / max_drawdown if max_drawdown > 0 else 0
                
                # Consistency score (lower volatility of returns is better)
                if len(returns) > 1:
                    return_volatility = returns.std() * 100
                    consistency_score = max(0, 100 - return_volatility)  # Higher is better
                else:
                    consistency_score = 0
                
                # OOS vs IS performance ratio (closer to 1 is better)
                oos1_is_ratio = abs(oos1_results['Return'] / total_return_is) if total_return_is != 0 else 0
                oos2_is_ratio = abs(oos2_results['Return'] / total_return_is) if total_return_is != 0 else 0
                avg_oos_is_ratio = (oos1_is_ratio + oos2_is_ratio) / 2
                robustness_score = max(0, 100 - abs(100 * (1 - avg_oos_is_ratio)))  # Higher is better
                
                # Calculate individual OOS scores using same weighting as IS
                oos1_score = (oos1_results['Return'] * w_return) + (oos1_results['Win_Rate'] * w_winrate) + (min(oos1_results['Trades'], 100) * w_trades)
                oos2_score = (oos2_results['Return'] * w_return) + (oos2_results['Win_Rate'] * w_winrate) + (min(oos2_results['Trades'], 100) * w_trades)
                
                # Calculate Composite Score: IS (60%) + OOS1 (20%) + OOS2 (20%)
                composite_score = (score * 0.6) + (oos1_score * 0.2) + (oos2_score * 0.2)

                # Calculate trade balance metrics
                if not trades_df_is.empty and 'Position' in trades_df_is.columns:
                    long_trades = len(trades_df_is[trades_df_is['Position'] == 'Long'])
                    short_trades = len(trades_df_is[trades_df_is['Position'] == 'Short'])
                else:
                    # Fallback: analyze signals to determine trade direction
                    if 'position' in df_with_signals_is.columns:
                        # Count actual signal transitions (entry points)
                        position_changes = df_with_signals_is['position'].diff().fillna(0)
                        long_trades = len(position_changes[position_changes == 1])  # Transitions to long
                        short_trades = len(position_changes[position_changes == -1])  # Transitions to short
                    else:
                        long_trades = total_trades_is // 2  # Fallback assumption
                        short_trades = total_trades_is - long_trades

                # Calculate trade balance ratio (higher value means more imbalanced)
                if long_trades > 0 and short_trades > 0:
                    trade_balance_ratio = max(long_trades / short_trades, short_trades / long_trades)
                elif long_trades > 0 or short_trades > 0:
                    trade_balance_ratio = float('inf')  # Completely one-sided
                else:
                    trade_balance_ratio = 1.0  # No trades
                
                # Trade balance score (100 = perfectly balanced, 0 = completely one-sided)
                if trade_balance_ratio == float('inf'):
                    trade_balance_score = 0
                else:
                    trade_balance_score = max(0, 100 - (trade_balance_ratio - 1) * 50)  # Higher is better

                # Calculate trade difference for manual filtering
                trade_difference = abs(long_trades - short_trades)

                # Calculate detailed profitability metrics
                if not trades_df_is.empty and 'Position' in trades_df_is.columns and 'PnL' in trades_df_is.columns:
                    # Overall profitable trades
                    profitable_trades = trades_df_is[trades_df_is['PnL'] > 0]
                    unprofitable_trades = trades_df_is[trades_df_is['PnL'] <= 0]
                    
                    profitable_trade_count = len(profitable_trades)
                    unprofitable_trade_count = len(unprofitable_trades)
                    
                    # Profitable long trades
                    profitable_long_trades = profitable_trades[profitable_trades['Position'] == 'Long']
                    profitable_long_count = len(profitable_long_trades)
                    avg_profitable_long = profitable_long_trades['PnL'].mean() if profitable_long_count > 0 else 0
                    
                    # Profitable short trades
                    profitable_short_trades = profitable_trades[profitable_trades['Position'] == 'Short']
                    profitable_short_count = len(profitable_short_trades)
                    avg_profitable_short = profitable_short_trades['PnL'].mean() if profitable_short_count > 0 else 0
                    
                    # Unprofitable long trades
                    unprofitable_long_trades = unprofitable_trades[unprofitable_trades['Position'] == 'Long']
                    unprofitable_long_count = len(unprofitable_long_trades)
                    avg_unprofitable_long = unprofitable_long_trades['PnL'].mean() if unprofitable_long_count > 0 else 0
                    
                    # Unprofitable short trades
                    unprofitable_short_trades = unprofitable_trades[unprofitable_trades['Position'] == 'Short']
                    unprofitable_short_count = len(unprofitable_short_trades)
                    avg_unprofitable_short = unprofitable_short_trades['PnL'].mean() if unprofitable_short_count > 0 else 0
                    
                    # Overall averages
                    avg_profitable_trade = profitable_trades['PnL'].mean() if profitable_trade_count > 0 else 0
                    avg_unprofitable_trade = unprofitable_trades['PnL'].mean() if unprofitable_trade_count > 0 else 0
                    
                else:
                    # Fallback values when detailed trade data is not available
                    profitable_trade_count = int(total_trades_is * win_rate_is / 100) if win_rate_is > 0 else 0
                    unprofitable_trade_count = total_trades_is - profitable_trade_count
                    
                    # Estimate based on overall profit factor and trade distribution
                    if profit_factor > 0 and profit_factor != float('inf'):
                        avg_profitable_trade = gross_profit / profitable_trade_count if profitable_trade_count > 0 else 0
                        avg_unprofitable_trade = -gross_loss / unprofitable_trade_count if unprofitable_trade_count > 0 else 0
                    else:
                        avg_profitable_trade = 0
                        avg_unprofitable_trade = 0
                    
                    # Estimate long/short distribution (assume proportional to total long/short trades)
                    if long_trades > 0 and short_trades > 0:
                        long_ratio = long_trades / (long_trades + short_trades)
                        short_ratio = short_trades / (long_trades + short_trades)
                        
                        profitable_long_count = int(profitable_trade_count * long_ratio)
                        profitable_short_count = profitable_trade_count - profitable_long_count
                        unprofitable_long_count = int(unprofitable_trade_count * long_ratio)
                        unprofitable_short_count = unprofitable_trade_count - unprofitable_long_count
                        
                        avg_profitable_long = avg_profitable_trade
                        avg_profitable_short = avg_profitable_trade
                        avg_unprofitable_long = avg_unprofitable_trade
                        avg_unprofitable_short = avg_unprofitable_trade
                    else:
                        profitable_long_count = profitable_trade_count if long_trades > short_trades else 0
                        profitable_short_count = profitable_trade_count if short_trades > long_trades else 0
                        unprofitable_long_count = unprofitable_trade_count if long_trades > short_trades else 0
                        unprofitable_short_count = unprofitable_trade_count if short_trades > long_trades else 0
                        
                        avg_profitable_long = avg_profitable_trade if long_trades > short_trades else 0
                        avg_profitable_short = avg_profitable_trade if short_trades > long_trades else 0
                        avg_unprofitable_long = avg_unprofitable_trade if long_trades > short_trades else 0
                        avg_unprofitable_short = avg_unprofitable_trade if short_trades > long_trades else 0

                # --- Store All Results ---
                trial.set_user_attr('results', {
                    'Total_Return': total_return_is, 'Total_Trades': total_trades_is, 'Win_Rate': win_rate_is,
                    'Max_Drawdown': max_drawdown, 'Sharpe_Ratio': sharpe_ratio, 'Profit_Factor': profit_factor,
                    'Score': score, 'Calmar_Ratio': calmar_ratio, 'Consistency_Score': consistency_score,
                    'Robustness_Score': robustness_score, 'Long_Trades': long_trades, 'Short_Trades': short_trades,
                    'Trade_Balance_Ratio': trade_balance_ratio, 'Trade_Balance_Score': trade_balance_score,
                    'Trade_Difference': trade_difference,
                    'Profitable_Trades': profitable_trade_count, 'Unprofitable_Trades': unprofitable_trade_count,
                    'Profitable_Long_Count': profitable_long_count, 'Profitable_Short_Count': profitable_short_count,
                    'Unprofitable_Long_Count': unprofitable_long_count, 'Unprofitable_Short_Count': unprofitable_short_count,
                    'Avg_Profitable_Trade': round(avg_profitable_trade, 2), 'Avg_Unprofitable_Trade': round(avg_unprofitable_trade, 2),
                    'Avg_Profitable_Long': round(avg_profitable_long, 2), 'Avg_Profitable_Short': round(avg_profitable_short, 2),
                    'Avg_Unprofitable_Long': round(avg_unprofitable_long, 2), 'Avg_Unprofitable_Short': round(avg_unprofitable_short, 2),
                    'OOS1_Return': oos1_results['Return'],
                    'OOS1_Trades': oos1_results['Trades'],
                    'OOS1_Win_Rate': oos1_results['Win_Rate'],
                    'OOS2_Return': oos2_results['Return'],
                    'OOS2_Trades': oos2_results['Trades'],
                    'OOS2_Win_Rate': oos2_results['Win_Rate'],
                    'Total_OOS_Return': total_oos_return,
                    'Composite_Score': composite_score,
                    'params': params
                })
                return score
            except Exception as e:
                add_optimization_log(f"!! Trial ERROR for {symbol}: {e}")
                return -1000.0

        def log_trial_progress(study, trial):
            # Log every 5th trial, on completion, or if it's a new best score
            is_new_best = hasattr(study, 'best_trial') and study.best_trial == trial
            should_log = ((trial.number + 1) % 5 == 0 or 
                         trial.state == optuna.trial.TrialState.COMPLETE or 
                         is_new_best)
            
            if not should_log:
                return
                
            progress_pct = ((trial.number + 1) / effective_trials) * 100
            message = f"-> {symbol}: Trial {trial.number + 1}/{effective_trials} ({progress_pct:.1f}%)"

            if trial.state == optuna.trial.TrialState.COMPLETE and 'results' in trial.user_attrs:
                results = trial.user_attrs['results']
                params = results.get('params', {})
                ret = results.get('Total_Return', 0)
                wr = results.get('Win_Rate', 0)
                trades = results.get('Total_Trades', 0)
                oos1_ret = results.get('OOS1_Return', 0)
                oos2_ret = results.get('OOS2_Return', 0)

                # Add relevant parameters to log based on strategy
                param_parts = []
                if strategy_name:
                    strategy_class = STRATEGY_REGISTRY.get(strategy_name)
                    if strategy_class:
                        strategy_params = strategy_class.get_parameters()
                        for param_key in strategy_params.keys():
                            if param_key in params:
                                param_value = params[param_key]
                                # Create short abbreviations for common parameters
                                param_abbrev = {
                                    'buy_signal_window': 'BW',
                                    'buy_pattern_lookback': 'BL', 
                                    'sell_signal_window': 'SW',
                                    'sell_pattern_lookback': 'SL',
                                    'atr_period': 'ATR_P',
                                    'atr_multiplier': 'ATR_M',
                                    'rsi_period': 'RSI_P',
                                    'oversold_threshold': 'OS',
                                    'overbought_threshold': 'OB'
                                }.get(param_key, param_key.upper()[:4])
                                param_parts.append(f"{param_abbrev}{param_value}")
                else:
                    # Fallback to old behavior if no strategy specified
                    bw = params.get('buy_signal_window', 'N/A')
                    bl = params.get('buy_pattern_lookback', 'N/A')
                    sw = params.get('sell_signal_window', 'N/A')
                    sl = params.get('sell_pattern_lookback', 'N/A')
                    param_parts = [f"BW{bw}", f"BL{bl}", f"SW{sw}", f"SL{sl}"]

                if param_parts:
                    message += f" | Params: {'/'.join(param_parts)}"
                message += f" | Ret: {ret:.2f}% | WR: {wr:.1f}% | Trades: {trades} | OOS1: {oos1_ret:.2f}% | OOS2: {oos2_ret:.2f}%"
            elif trial.state != optuna.trial.TrialState.RUNNING:
                message += f" ({trial.state.name})"
            
            add_optimization_log(message)

        add_optimization_log(
            f"🚀 Starting optimization for {symbol} ({effective_trials} unique trials)...")
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Use RandomSampler to ensure no duplicates when combined with our pre-generated combinations
        sampler = optuna.samplers.RandomSampler(seed=42)
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=10)

        study = optuna.create_study(
            direction='maximize', sampler=sampler, pruner=pruner)
        study.optimize(objective, n_trials=effective_trials,
                       callbacks=[log_trial_progress])
        
        all_trials_results = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE and 'results' in trial.user_attrs:
                results_dict = trial.user_attrs['results']
                params = results_dict.pop('params')
                # Format all params to Title Case (e.g., 'atr_period' -> 'ATR_Period')
                formatted_params = {k.replace('_', ' ').title().replace(' ', '_'): v for k, v in params.items()}
                trial_result = {
                    'Trading_Pair': symbol, 'Timeframe': timeframe,
                    # Dynamically unpack ALL parameters here
                    **formatted_params,
                    # Then include the metrics
                    **results_dict
                }
                all_trials_results.append(trial_result)
        
        if not all_trials_results:
            add_optimization_log(f"-> No valid trials completed for {symbol}.")
            return pd.DataFrame()
        
        best_trial = study.best_trial
        best_return = best_trial.user_attrs.get(
            'results', {}).get('Total_Return', 0)
        unique_tested = len(tested_combinations)
        add_optimization_log(
            f"-> Best for {symbol}: Score={best_trial.value:.2f}, Return={best_return:.2f}% ({unique_tested} unique combinations tested)")
        return pd.DataFrame(all_trials_results)

    def _run_comprehensive_optimization_study(self, symbol, full_data, param_ranges, selected_params, timeframe, min_trades,
                                             n_trials, weights, is_start_date, is_end_date, oos1_start_date, oos1_end_date, oos2_start_date, oos2_end_date,
                                             stop_event, pause_event, strategy_name=None):
        """Runs comprehensive optimization testing ALL possible combinations."""
        add_optimization_log(f"🔍 Running COMPREHENSIVE optimization for {symbol}")
        
        # Use the existing method but with GridSampler for exhaustive search
        return self._run_single_optimization_study(
            symbol, full_data, param_ranges, selected_params, timeframe, min_trades,
            n_trials, weights, is_start_date, is_end_date, oos1_start_date, oos1_end_date, oos2_start_date, oos2_end_date,
            stop_event, pause_event, strategy_name
        )

    def _run_smart_optimization_study(self, symbol, full_data, param_ranges, selected_params, timeframe, min_trades,
                                      n_trials, weights, is_start_date, is_end_date, oos1_start_date, oos1_end_date, oos2_start_date, oos2_end_date,
                                      stop_event, pause_event, strategy_name=None):
        """Runs smart Bayesian optimization using TPE sampler."""
        add_optimization_log(f"🧠 Running SMART optimization for {symbol}")
        
        # Use specified strategy or fallback to dynamic selection
        if strategy_name:
            strategy_class = STRATEGY_REGISTRY.get(strategy_name)
            if strategy_class:
                strategy = strategy_class()
            else:
                strategy = self._get_strategy_for_params(selected_params)
        else:
            strategy = self._get_strategy_for_params(selected_params)
        default_params = strategy.get_parameters()

        # Strategy-First approach: Only keep parameters that actually exist in the strategy
        strategy_keys = set(default_params.keys())
        effective_params = [p for p in selected_params if p in strategy_keys]
        if len(effective_params) < len(selected_params):
            filtered_out = len(selected_params) - len(effective_params)
            add_optimization_log(f"-> {symbol}: Filtered out {filtered_out} invalid parameters for {strategy.__class__.__name__}")

        is_data = full_data.loc[is_start_date:is_end_date]
        oos1_data = full_data.loc[oos1_start_date:oos1_end_date]
        oos2_data = full_data.loc[oos2_start_date:oos2_end_date]

        def run_backtest_and_get_metrics(data_df, params_dict):
            if data_df is None or data_df.empty:
                return {'Return': -1000, 'Trades': 0, 'Win_Rate': 0}

            df_with_signals = strategy.generate_signals(data_df.copy(), params_dict)
            backtester = Backtester(self.initial_capital)
            trades_df, portfolio_df = backtester.run_backtest(df_with_signals)

            if trades_df is None or portfolio_df is None or trades_df.empty:
                return {'Return': -1000, 'Trades': 0, 'Win_Rate': 0}

            total_return = (portfolio_df['Portfolio_Value'].iloc[-1] / self.initial_capital - 1) * 100
            win_rate = (len(trades_df[trades_df['PnL'] > 0]) / len(trades_df) * 100) if not trades_df.empty else 0
            total_trades = len(trades_df)
            return {'Return': total_return, 'Trades': total_trades, 'Win_Rate': win_rate}

        def parse_range_for_optuna(range_str, is_int=True):
            if not range_str or not isinstance(range_str, str) or range_str.strip() == '':
                return None, None, None
            try:
                parts = [float(p.strip()) for p in range_str.split(',')]
                if len(parts) != 3:
                    return None, None, None
                low, high, step = parts
                if step <= 0 or low > high:
                    return None, None, None
                return (int(low), int(high), int(step)) if is_int else (low, high, step)
            except:
                return None, None, None

        # Track tested combinations to avoid duplicates in smart sampling
        tested_combinations = set()
        add_optimization_log(f"-> {symbol}: Smart sampling with duplicate prevention enabled")

        def objective(trial):
            # Heartbeat logging to verify loop is running
            print(f"DEBUG: {symbol} processing trial {trial.number}")
            
            if stop_event.is_set():
                raise optuna.exceptions.TrialPruned("Optimization stopped by user.")

            if pause_event.is_set():
                add_optimization_log(f"-> {symbol}: Study paused. Waiting for continue signal...")
                while pause_event.is_set():
                    if stop_event.is_set():
                        raise optuna.exceptions.TrialPruned("Optimization stopped while paused.")
                    time.sleep(1)
                add_optimization_log(f"-> {symbol}: Study resumed.")

            params = {}
            param_map = {
                'buy_signal_window': {'is_int': True}, 'buy_pattern_lookback': {'is_int': True},
                'sell_signal_window': {'is_int': True}, 'sell_pattern_lookback': {'is_int': True},
                'atr_period': {'is_int': True}, 'atr_multiplier': {'is_int': False}
            }
            
            for p_name, p_config in param_map.items():
                if p_name not in default_params:
                    continue
                if p_name in effective_params and param_ranges.get(p_name):
                    low, high, step = parse_range_for_optuna(param_ranges[p_name], is_int=p_config['is_int'])
                    if low is not None:
                        if p_config['is_int']:
                            params[p_name] = trial.suggest_int(p_name, low, high, step=step)
                        else:
                            choices = [round(v, 4) for v in np.arange(low, high + step / 2, step)]
                            params[p_name] = trial.suggest_categorical(p_name, choices)
                    else:
                        params[p_name] = default_params[p_name]['default']
                else:
                    params[p_name] = default_params[p_name]['default']

            # Check for duplicate combinations in smart sampling
            param_key = tuple(sorted(params.items()))
            if param_key in tested_combinations:
                add_optimization_log(f"-> {symbol}: Skipping duplicate combination in smart sampling")
                return -1000.0  # Return poor score to discourage this combination
            
            tested_combinations.add(param_key)

            try:
                # IN-SAMPLE BACKTEST
                df_with_signals_is = strategy.generate_signals(is_data.copy(), params)
                backtester_is = Backtester(self.initial_capital)
                trades_df_is, portfolio_df_is = backtester_is.run_backtest(df_with_signals_is)

                if trades_df_is is None or portfolio_df_is is None or len(trades_df_is) < min_trades:
                    return -1000.0

                # Calculate metrics
                total_return_is = (portfolio_df_is['Portfolio_Value'].iloc[-1] / self.initial_capital - 1) * 100
                win_rate_is = (len(trades_df_is[trades_df_is['PnL'] > 0]) / len(trades_df_is) * 100) if not trades_df_is.empty else 0
                total_trades_is = len(trades_df_is)
                
                portfolio_values = portfolio_df_is['Portfolio_Value'].values
                running_max = np.maximum.accumulate(portfolio_values)
                drawdown = (running_max - portfolio_values) / running_max * 100
                max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
                
                returns = portfolio_df_is['Portfolio_Value'].pct_change().dropna()
                annualization_factor = TIMEFRAME_TO_ANNUALIZATION_FACTOR.get(timeframe, np.sqrt(365))
                sharpe_ratio = (returns.mean() / returns.std() * annualization_factor) if returns.std() > 0 else 0
                
                gross_profit = trades_df_is[trades_df_is['PnL'] > 0]['PnL'].sum()
                gross_loss = abs(trades_df_is[trades_df_is['PnL'] < 0]['PnL'].sum())
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

                # Calculate score
                total_weight = weights['total_return'] + weights['win_rate'] + weights['total_trades']
                if total_weight == 0:
                    total_weight = 1
                w_return = weights['total_return'] / total_weight
                w_winrate = weights['win_rate'] / total_weight
                w_trades = weights['total_trades'] / total_weight
                capped_trades = min(total_trades_is, 100)
                score = (total_return_is * w_return) + (win_rate_is * w_winrate) + (capped_trades * w_trades)

                # OUT-OF-SAMPLE BACKTESTS
                oos1_results = run_backtest_and_get_metrics(oos1_data, params)
                oos2_results = run_backtest_and_get_metrics(oos2_data, params)
                
                # Calculate combined OOS metrics
                total_oos_return = (oos1_results['Return'] + oos2_results['Return']) / 2
                total_oos_trades = oos1_results['Trades'] + oos2_results['Trades']
                total_oos_win_rate = (oos1_results['Win_Rate'] + oos2_results['Win_Rate']) / 2

                # Additional metrics
                calmar_ratio = total_return_is / max_drawdown if max_drawdown > 0 else 0
                if len(returns) > 1:
                    return_volatility = returns.std() * 100
                    consistency_score = max(0, 100 - return_volatility)
                else:
                    consistency_score = 0
                
                oos_is_ratio = abs(total_oos_return / total_return_is) if total_return_is != 0 else 0
                robustness_score = max(0, 100 - abs(100 * (1 - oos_is_ratio)))

                # Calculate trade balance metrics
                if not trades_df_is.empty and 'Position' in trades_df_is.columns:
                    long_trades = len(trades_df_is[trades_df_is['Position'] == 'Long'])
                    short_trades = len(trades_df_is[trades_df_is['Position'] == 'Short'])
                else:
                    # Fallback: analyze signals to determine trade direction
                    if 'position' in df_with_signals_is.columns:
                        # Count actual signal transitions (entry points)
                        position_changes = df_with_signals_is['position'].diff().fillna(0)
                        long_trades = len(position_changes[position_changes == 1])  # Transitions to long
                        short_trades = len(position_changes[position_changes == -1])  # Transitions to short
                    else:
                        long_trades = total_trades_is // 2  # Fallback assumption
                        short_trades = total_trades_is - long_trades

                # Calculate trade balance ratio and score
                if long_trades > 0 and short_trades > 0:
                    trade_balance_ratio = max(long_trades / short_trades, short_trades / long_trades)
                elif long_trades > 0 or short_trades > 0:
                    trade_balance_ratio = float('inf')
                else:
                    trade_balance_ratio = 1.0

                if trade_balance_ratio == float('inf'):
                    trade_balance_score = 0
                else:
                    trade_balance_score = max(0, 100 - (trade_balance_ratio - 1) * 50)

                # Calculate trade difference for manual filtering
                trade_difference = abs(long_trades - short_trades)

                # Calculate detailed profitability metrics
                if not trades_df_is.empty and 'Position' in trades_df_is.columns and 'PnL' in trades_df_is.columns:
                    # Overall profitable trades
                    profitable_trades = trades_df_is[trades_df_is['PnL'] > 0]
                    unprofitable_trades = trades_df_is[trades_df_is['PnL'] <= 0]
                    
                    profitable_trade_count = len(profitable_trades)
                    unprofitable_trade_count = len(unprofitable_trades)
                    
                    # Profitable long trades
                    profitable_long_trades = profitable_trades[profitable_trades['Position'] == 'Long']
                    profitable_long_count = len(profitable_long_trades)
                    avg_profitable_long = profitable_long_trades['PnL'].mean() if profitable_long_count > 0 else 0
                    
                    # Profitable short trades
                    profitable_short_trades = profitable_trades[profitable_trades['Position'] == 'Short']
                    profitable_short_count = len(profitable_short_trades)
                    avg_profitable_short = profitable_short_trades['PnL'].mean() if profitable_short_count > 0 else 0
                    
                    # Unprofitable long trades
                    unprofitable_long_trades = unprofitable_trades[unprofitable_trades['Position'] == 'Long']
                    unprofitable_long_count = len(unprofitable_long_trades)
                    avg_unprofitable_long = unprofitable_long_trades['PnL'].mean() if unprofitable_long_count > 0 else 0
                    
                    # Unprofitable short trades
                    unprofitable_short_trades = unprofitable_trades[unprofitable_trades['Position'] == 'Short']
                    unprofitable_short_count = len(unprofitable_short_trades)
                    avg_unprofitable_short = unprofitable_short_trades['PnL'].mean() if unprofitable_short_count > 0 else 0
                    
                    # Overall averages
                    avg_profitable_trade = profitable_trades['PnL'].mean() if profitable_trade_count > 0 else 0
                    avg_unprofitable_trade = unprofitable_trades['PnL'].mean() if unprofitable_trade_count > 0 else 0
                    
                else:
                    # Fallback values when detailed trade data is not available
                    profitable_trade_count = int(total_trades_is * win_rate_is / 100) if win_rate_is > 0 else 0
                    unprofitable_trade_count = total_trades_is - profitable_trade_count
                    
                    # Estimate based on overall profit factor and trade distribution
                    if profit_factor > 0 and profit_factor != float('inf'):
                        avg_profitable_trade = gross_profit / profitable_trade_count if profitable_trade_count > 0 else 0
                        avg_unprofitable_trade = -gross_loss / unprofitable_trade_count if unprofitable_trade_count > 0 else 0
                    else:
                        avg_profitable_trade = 0
                        avg_unprofitable_trade = 0
                    
                    # Estimate long/short distribution (assume proportional to total long/short trades)
                    if long_trades > 0 and short_trades > 0:
                        long_ratio = long_trades / (long_trades + short_trades)
                        short_ratio = short_trades / (long_trades + short_trades)
                        
                        profitable_long_count = int(profitable_trade_count * long_ratio)
                        profitable_short_count = profitable_trade_count - profitable_long_count
                        unprofitable_long_count = int(unprofitable_trade_count * long_ratio)
                        unprofitable_short_count = unprofitable_trade_count - unprofitable_long_count
                        
                        avg_profitable_long = avg_profitable_trade
                        avg_profitable_short = avg_profitable_trade
                        avg_unprofitable_long = avg_unprofitable_trade
                        avg_unprofitable_short = avg_unprofitable_trade
                    else:
                        profitable_long_count = profitable_trade_count if long_trades > short_trades else 0
                        profitable_short_count = profitable_trade_count if short_trades > long_trades else 0
                        unprofitable_long_count = unprofitable_trade_count if long_trades > short_trades else 0
                        unprofitable_short_count = unprofitable_trade_count if short_trades > long_trades else 0
                        
                        avg_profitable_long = avg_profitable_trade if long_trades > short_trades else 0
                        avg_profitable_short = avg_profitable_trade if short_trades > long_trades else 0
                        avg_unprofitable_long = avg_unprofitable_trade if long_trades > short_trades else 0
                        avg_unprofitable_short = avg_unprofitable_trade if short_trades > long_trades else 0

                # Calculate composite score and additional metrics
                oos1_score = (oos1_results['Return'] * w_return) + (oos1_results['Win_Rate'] * w_winrate) + (min(oos1_results['Trades'], 100) * w_trades)
                oos2_score = (oos2_results['Return'] * w_return) + (oos2_results['Win_Rate'] * w_winrate) + (min(oos2_results['Trades'], 100) * w_trades)
                composite_score = (score * 0.6) + (oos1_score * 0.2) + (oos2_score * 0.2)
                
                trial.set_user_attr('results', {
                    'Total_Return': total_return_is, 'Total_Trades': total_trades_is, 'Win_Rate': win_rate_is,
                    'Max_Drawdown': max_drawdown, 'Sharpe_Ratio': sharpe_ratio, 'Profit_Factor': profit_factor,
                    'Score': score, 'Calmar_Ratio': calmar_ratio, 'Consistency_Score': consistency_score,
                    'Robustness_Score': robustness_score, 'Long_Trades': long_trades, 'Short_Trades': short_trades,
                    'Trade_Balance_Ratio': trade_balance_ratio, 'Trade_Balance_Score': trade_balance_score,
                    'Trade_Difference': trade_difference,
                    'Profitable_Trades': profitable_trade_count, 'Unprofitable_Trades': unprofitable_trade_count,
                    'Profitable_Long_Count': profitable_long_count, 'Profitable_Short_Count': profitable_short_count,
                    'Unprofitable_Long_Count': unprofitable_long_count, 'Unprofitable_Short_Count': unprofitable_short_count,
                    'Avg_Profitable_Trade': round(avg_profitable_trade, 2), 'Avg_Unprofitable_Trade': round(avg_unprofitable_trade, 2),
                    'Avg_Profitable_Long': round(avg_profitable_long, 2), 'Avg_Profitable_Short': round(avg_profitable_short, 2),
                    'Avg_Unprofitable_Long': round(avg_unprofitable_long, 2), 'Avg_Unprofitable_Short': round(avg_unprofitable_short, 2),
                    'OOS1_Return': oos1_results['Return'], 'OOS1_Trades': oos1_results['Trades'], 'OOS1_Win_Rate': oos1_results['Win_Rate'],
                    'OOS2_Return': oos2_results['Return'], 'OOS2_Trades': oos2_results['Trades'], 'OOS2_Win_Rate': oos2_results['Win_Rate'],
                    'Total_OOS_Return': total_oos_return, 'Composite_Score': composite_score, 'params': params
                })
                return score
            except Exception as e:
                add_optimization_log(f"!! Trial ERROR for {symbol}: {e}")
                return -1000.0

        def log_trial_progress(study, trial):
            # Log every 5th trial, on completion, or if it's a new best score
            is_new_best = hasattr(study, 'best_trial') and study.best_trial == trial
            should_log = ((trial.number + 1) % 5 == 0 or 
                         trial.state == optuna.trial.TrialState.COMPLETE or 
                         is_new_best)
            
            if not should_log:
                return
                
            progress_pct = ((trial.number + 1) / n_trials) * 100
            message = f"-> {symbol}: Trial {trial.number + 1}/{n_trials} ({progress_pct:.1f}%)"

            if trial.state == optuna.trial.TrialState.COMPLETE and 'results' in trial.user_attrs:
                results = trial.user_attrs['results']
                params = results.get('params', {})
                ret = results.get('Total_Return', 0)
                wr = results.get('Win_Rate', 0)
                trades = results.get('Total_Trades', 0)
                oos1_ret = results.get('OOS1_Return', 0)
                oos2_ret = results.get('OOS2_Return', 0)

                # Add relevant parameters to log based on strategy
                param_parts = []
                if strategy_name:
                    strategy_class = STRATEGY_REGISTRY.get(strategy_name)
                    if strategy_class:
                        strategy_params = strategy_class.get_parameters()
                        for param_key in strategy_params.keys():
                            if param_key in params:
                                param_value = params[param_key]
                                # Create short abbreviations for common parameters
                                param_abbrev = {
                                    'buy_signal_window': 'BW',
                                    'buy_pattern_lookback': 'BL', 
                                    'sell_signal_window': 'SW',
                                    'sell_pattern_lookback': 'SL',
                                    'atr_period': 'ATR_P',
                                    'atr_multiplier': 'ATR_M',
                                    'rsi_period': 'RSI_P',
                                    'oversold_threshold': 'OS',
                                    'overbought_threshold': 'OB'
                                }.get(param_key, param_key.upper()[:4])
                                param_parts.append(f"{param_abbrev}{param_value}")
                else:
                    # Fallback to old behavior if no strategy specified
                    bw = params.get('buy_signal_window', 'N/A')
                    bl = params.get('buy_pattern_lookback', 'N/A')
                    sw = params.get('sell_signal_window', 'N/A')
                    sl = params.get('sell_pattern_lookback', 'N/A')
                    param_parts = [f"BW{bw}", f"BL{bl}", f"SW{sw}", f"SL{sl}"]

                if param_parts:
                    message += f" | Params: {'/'.join(param_parts)}"
                message += f" | Ret: {ret:.2f}% | WR: {wr:.1f}% | Trades: {trades} | OOS1: {oos1_ret:.2f}% | OOS2: {oos2_ret:.2f}%"
            elif trial.state != optuna.trial.TrialState.RUNNING:
                message += f" ({trial.state.name})"
            
            add_optimization_log(message)

        # Use TPE sampler for intelligent exploration
        sampler = optuna.samplers.TPESampler(n_startup_trials=min(20, n_trials//3))
        pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=15)

        study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
        study.optimize(objective, n_trials=n_trials, callbacks=[log_trial_progress])
        
        all_trials_results = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE and 'results' in trial.user_attrs:
                results_dict = trial.user_attrs['results']
                params = results_dict.pop('params')
                # Format all params to Title Case (e.g., 'atr_period' -> 'ATR_Period')
                formatted_params = {k.replace('_', ' ').title().replace(' ', '_'): v for k, v in params.items()}
                trial_result = {
                    'Trading_Pair': symbol, 'Timeframe': timeframe,
                    # Dynamically unpack ALL parameters here
                    **formatted_params,
                    # Then include the metrics
                    **results_dict
                }
                all_trials_results.append(trial_result)
        
        if not all_trials_results:
            add_optimization_log(f"-> No valid trials completed for {symbol}.")
            return pd.DataFrame()
        
        best_trial = study.best_trial
        best_return = best_trial.user_attrs.get('results', {}).get('Total_Return', 0)
        add_optimization_log(f"-> Best for {symbol}: Score={best_trial.value:.2f}, Return={best_return:.2f}% (Smart sampling)")
        return pd.DataFrame(all_trials_results)

    def optimize_trading_pairs(self, trading_pairs, param_ranges, selected_params, is_start_date, is_end_date,
                               oos1_start_date, oos1_end_date, oos2_start_date, oos2_end_date, timeframe, min_trades, n_trials, weights, min_candles,
                               stop_event, pause_event, optimization_mode='efficient', strategy_name=None,
                               executor_type='process'):
        pair_data = {}
        add_optimization_log(
            f"Fetching full-range data for {len(trading_pairs)} pairs...")
        for symbol in trading_pairs:
            if stop_event.is_set():
                add_optimization_log("Stop requested during data fetching.")
                return pd.DataFrame()
            # Fetch data covering IS and both OOS periods
            data = self.get_historical_data_for_symbol(
                symbol, timeframe, is_start_date, oos2_end_date)
            # This check is now mostly redundant due to pre-flight check, but serves as a final safeguard
            if data is not None and len(data) >= min_candles:
                pair_data[symbol] = data

        if not pair_data:
            add_optimization_log(
                "ERROR: No data could be fetched for the validated pairs. This may be a network issue.")
            return pd.DataFrame()

        all_results_dfs = []
        add_optimization_log(
            f"Starting {optimization_mode.upper()} optimization with {self.max_workers} workers...")
        
        # Choose optimization method based on mode
        if optimization_mode == 'comprehensive':
            optimization_method = self._run_comprehensive_optimization_study
        elif optimization_mode == 'smart':
            optimization_method = self._run_smart_optimization_study
        else:  # efficient (default)
            optimization_method = self._run_single_optimization_study
            
        # Per-symbol studies are CPU-bound, so a ProcessPoolExecutor uses all
        # cores for a real speedup. This is safe now that stop/pause events are
        # _RedisEvent (picklable, cross-process) rather than threading.Event,
        # which was the original cause of the multiprocessing deadlock. The
        # 'thread' option remains as a fallback for environments where forking
        # misbehaves (e.g. some Windows/Numba combinations).
        if executor_type == 'process':
            ExecutorCls = ProcessPoolExecutor
            add_optimization_log(
                f"Using ProcessPoolExecutor ({self.max_workers} cores) for parallel optimization.")
        else:
            ExecutorCls = ThreadPoolExecutor
            add_optimization_log(
                f"Using ThreadPoolExecutor ({self.max_workers} workers) for parallel optimization.")

        with ExecutorCls(max_workers=self.max_workers) as executor:
            futures = {executor.submit(optimization_method, symbol, data, param_ranges,
                                       selected_params, timeframe, min_trades, n_trials, weights,
                                       is_start_date, is_end_date, oos1_start_date, oos1_end_date, oos2_start_date, oos2_end_date,
                                       stop_event, pause_event, strategy_name): symbol
                       for symbol, data in pair_data.items()}
            for i, future in enumerate(as_completed(futures), 1):
                symbol = futures[future]
                try:
                    if stop_event.is_set():
                        add_optimization_log(
                            f"Skipping results for {symbol} due to stop request.")
                        continue
                    result_df = future.result()
                    if not result_df.empty:
                        all_results_dfs.append(result_df)
                        PARTIAL_RESULTS_LIST.append(result_df)
                    overall_progress = (i / len(pair_data)) * 100
                    add_optimization_log(
                        f"✅ ({i}/{len(pair_data)}) {symbol} completed - Overall Progress: {overall_progress:.1f}%")
                except Exception as e:
                    add_optimization_log(
                        f"!! An entire study for {symbol} failed: {e}")
        if not all_results_dfs:
            return pd.DataFrame()
        return pd.concat(all_results_dfs, ignore_index=True)

    def optimize_trading_pairs_with_stability(self, trading_pairs, param_ranges, selected_params, is_start_date, is_end_date,
                                            oos1_start_date, oos1_end_date, oos2_start_date, oos2_end_date, timeframe, min_trades, n_trials, weights, min_candles,
                                            stop_event, pause_event, stability_weight=0.5, optimization_mode='efficient', strategy_name=None):
        """Enhanced optimization that prioritizes stability over raw returns."""
        
        add_optimization_log("🎯 Starting STABILITY-BASED optimization...")
        
        # First run the standard optimization
        results_df = self.optimize_trading_pairs(
            trading_pairs, param_ranges, selected_params, is_start_date, is_end_date,
            oos1_start_date, oos1_end_date, oos2_start_date, oos2_end_date, timeframe, min_trades, n_trials, weights, 
            min_candles, stop_event, pause_event, optimization_mode, strategy_name
        )
        
        if results_df.empty:
            return results_df
        
        # Now enhance with stability analysis
        add_optimization_log("📊 Calculating stability metrics for all results...")
        
        # Group results by pair and calculate stability for best result per pair
        stability_analyzer = StabilityAnalyzer()
        pair_stability_data = {}
        
        for pair in results_df['Trading_Pair'].unique():
            if stop_event.is_set():
                break
                
            pair_results = results_df[results_df['Trading_Pair'] == pair]
            best_result = pair_results.loc[pair_results['Score'].idxmax()]
            
            # Get the portfolio data for this best result by re-running backtest
            try:
                # Fetch data for this pair
                full_data = self.get_historical_data_for_symbol(
                    pair, timeframe, is_start_date, oos2_end_date
                )
                
                if full_data is None or full_data.empty:
                    continue
                
                # Split into IS and dual OOS
                is_data = full_data.loc[is_start_date:is_end_date]
                oos1_data = full_data.loc[oos1_start_date:oos1_end_date]
                oos2_data = full_data.loc[oos2_start_date:oos2_end_date]
                oos_data = full_data.loc[oos_start_date:oos_end_date]
                
                # Extract parameters from best result (dynamic based on strategy)
                params = {}
                
                # Add candlestick parameters if they exist
                if 'Buy_Signal_Window' in best_result:
                    params['buy_signal_window'] = int(best_result['Buy_Signal_Window'])
                if 'Buy_Pattern_Lookback' in best_result:
                    params['buy_pattern_lookback'] = int(best_result['Buy_Pattern_Lookback'])
                if 'Sell_Signal_Window' in best_result:
                    params['sell_signal_window'] = int(best_result['Sell_Signal_Window'])
                if 'Sell_Pattern_Lookback' in best_result:
                    params['sell_pattern_lookback'] = int(best_result['Sell_Pattern_Lookback'])
                
                # Add SuperTrend parameters if they exist
                if 'ATR_Period' in best_result:
                    params['atr_period'] = int(best_result['ATR_Period'])
                if 'ATR_Multiplier' in best_result:
                    params['atr_multiplier'] = float(best_result['ATR_Multiplier'])
                
                # Run backtests to get portfolio values
                def get_portfolio_values(data, params):
                    if strategy_name:
                        strategy_class = STRATEGY_REGISTRY.get(strategy_name)
                        if strategy_class:
                            strategy = strategy_class()
                        else:
                            strategy = self._get_strategy_for_params(selected_params)
                    else:
                        strategy = self._get_strategy_for_params(selected_params)
                    df_with_signals = strategy.generate_signals(data.copy(), params)
                    backtester = Backtester(self.initial_capital)
                    trades_df, portfolio_df = backtester.run_backtest(df_with_signals)
                    
                    if portfolio_df is not None and not portfolio_df.empty:
                        return portfolio_df['Portfolio_Value']
                    return None
                
                is_portfolio_values = get_portfolio_values(is_data, params)
                oos1_portfolio_values = get_portfolio_values(oos1_data, params)
                oos2_portfolio_values = get_portfolio_values(oos2_data, params)
                
                if is_portfolio_values is not None and oos1_portfolio_values is not None and oos2_portfolio_values is not None:
                    # Combine OOS1 and OOS2 portfolio values for stability analysis
                    combined_oos_values = pd.concat([oos1_portfolio_values, oos2_portfolio_values])
                    pair_stability_data[pair] = {
                        'is_portfolio_values': is_portfolio_values,
                        'oos_portfolio_values': combined_oos_values
                    }
                    
            except Exception as e:
                add_optimization_log(f"⚠️ Error calculating stability for {pair}: {e}")
                continue
        
        if not pair_stability_data:
            add_optimization_log("❌ No stability data could be calculated")
            return results_df
        
        # Calculate stability rankings
        add_optimization_log(f"🔍 Analyzing stability for {len(pair_stability_data)} pairs...")
        stability_rankings = stability_analyzer.rank_pairs_by_stability(pair_stability_data)
        
        # Create stability lookup
        stability_lookup = {r['pair']: r for r in stability_rankings}
        
        # Add stability metrics to results DataFrame
        results_df['Stability_Score'] = results_df.apply(
            lambda row: stability_lookup.get(row['Trading_Pair'], {}).get('combined_stability', -100), 
            axis=1
        )
        
        results_df['Weighted_Return'] = results_df.apply(
            lambda row: stability_lookup.get(row['Trading_Pair'], {}).get('weighted_return', row['Total_Return']), 
            axis=1
        )
        
        results_df['IS_Max_Drawdown_Stability'] = results_df.apply(
            lambda row: stability_lookup.get(row['Trading_Pair'], {}).get('is_max_drawdown', 100), 
            axis=1
        )
        
        results_df['OOS_Max_Drawdown_Stability'] = results_df.apply(
            lambda row: stability_lookup.get(row['Trading_Pair'], {}).get('oos_max_drawdown', 100), 
            axis=1
        )
        
        # Calculate combined stability-performance score
        results_df['Stability_Performance_Score'] = (
            results_df['Score'] * (1 - stability_weight) + 
            results_df['Stability_Score'] * stability_weight
        )
        
        # Sort by stability-performance score
        results_df = results_df.sort_values('Stability_Performance_Score', ascending=False)
        
        # Generate stability report
        report = stability_analyzer.create_stability_report(stability_rankings, top_n=10)
        add_optimization_log("📈 STABILITY ANALYSIS COMPLETE")
        add_optimization_log(report)
        
        # Log top stable pairs
        top_stable = results_df.head(5)
        add_optimization_log("🏆 TOP 5 MOST STABLE PAIRS:")
        for _, row in top_stable.iterrows():
            add_optimization_log(
                f"   {row['Trading_Pair']}: Stability={row['Stability_Score']:.2f}, "
                f"Return={row['Total_Return']:.1f}%, Combined={row['Stability_Performance_Score']:.2f}"
            )
        
        return results_df

    def export_stable_pairs_to_config(self, results_df, top_n=5, stability_threshold=-10.0, 
                                    units_usdt=50.0, leverage=10, timeframe='15m'):
        """Export top stable pairs to trading configuration."""
        try:
            if results_df is None or results_df.empty:
                return False
            
            # Filter by stability threshold first
            stable_results = results_df[results_df['Stability_Score'] >= stability_threshold]
            
            if stable_results.empty:
                add_optimization_log(f"❌ No pairs meet stability threshold of {stability_threshold}")
                return False
            
            # Get best result per pair based on stability-performance score
            best_per_pair = stable_results.loc[stable_results.groupby('Trading_Pair')['Stability_Performance_Score'].idxmax()]
            
            # Take top N most stable pairs
            top_stable_pairs = best_per_pair.head(top_n)
            
            configs = []
            for _, row in top_stable_pairs.iterrows():
                config = {
                    "enabled": True,
                    "strategy_name": "Candlestick Patterns",
                    "symbol": row['Trading_Pair'],
                    "bar_length": timeframe,
                    "units_usdt": units_usdt,
                    "leverage": leverage,
                    "buy_signal_window": int(row['Buy_Signal_Window']),
                    "buy_pattern_lookback": int(row['Buy_Pattern_Lookback']),
                    "sell_signal_window": int(row['Sell_Signal_Window']),
                    "sell_pattern_lookback": int(row['Sell_Pattern_Lookback'])
                }
                configs.append(config)
            
            # Save to trade_config.json
            with open('trade_config.json', 'w') as f:
                json.dump(configs, f, indent=2)
            
            add_optimization_log(f"✅ Exported {len(configs)} STABLE pairs to trade_config.json")
            add_optimization_log("📊 STABLE PAIRS EXPORTED:")
            for _, row in top_stable_pairs.iterrows():
                add_optimization_log(
                    f"   {row['Trading_Pair']}: Stability={row['Stability_Score']:.2f}, "
                    f"Return={row['Total_Return']:.1f}%, Weighted={row['Weighted_Return']:.1f}%"
                )
            
            return True
            
        except Exception as e:
            add_optimization_log(f"❌ Failed to export stable pairs: {e}")
            return False


# --- Dash Application Setup ---
app = dash.Dash(__name__,
                external_stylesheets=[
                    'https://codepen.io/chriddyp/pen/bWLwgP.css',
                    '/assets/style.css',
                    '/assets/dark-theme.css'
                ],
                external_scripts=[
                    '/assets/force-dark-theme.js', '/assets/scripts.js'],
                suppress_callback_exceptions=True,
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1.0"}])
trader = None


# --- UI Helper Functions ---
def create_collapsible_container(title, component_id, children, export_info=None):
    header_children = [html.H3(title, style={'textAlign': 'center', 'flex': '1', 'margin': '0'}),
                       html.Button("Hide", id={'type': 'hide-show-btn', 'index': component_id}, n_clicks=0,
                                   className='small-button')]
    if export_info:
        header_children.insert(1, html.Button("Export to XLSX", id=export_info['button_id'], n_clicks=0,
                                              className='small-button'))
    return html.Div([html.Div(header_children, className='collapsible-header'),
                     html.Div(id={'type': 'collapsible-body', 'index': component_id}, children=children,
                              className='collapsible-body')])


def generate_filename(symbol, start_date, end_date, params):
    param_str = "_".join(
        [f"{k.replace('_', '')}{v}" for k, v in params.items()])
    clean_param_str = re.sub(r'[^a-zA-Z0-9_-]', '', param_str)
    date_str = datetime.now().strftime('%Y%m%d')
    return f"{date_str}_{symbol}_{start_date}_to_{end_date}_{clean_param_str}.xlsx"


# --- UI Builder Functions ---
def date_range_inputs(base_id, start_default, end_default):
    """Native HTML date range (two <input type=date>) styled by .custom-input.
    Produces components with ids '{base_id}-start' and '{base_id}-end',
    whose 'value' is a 'YYYY-MM-DD' string."""
    return html.Div([
        html.Input(id=f'{base_id}-start', type='date', value=str(start_default),
                   className='custom-input', style={'colorScheme': 'dark'}),
        html.Span('→', style={'margin': '0 8px'}),
        html.Input(id=f'{base_id}-end', type='date', value=str(end_default),
                   className='custom-input', style={'colorScheme': 'dark'}),
    ], style={'display': 'flex', 'alignItems': 'center'})


def build_config_panel():
    today = datetime.now(timezone.utc)
    sixty_days_ago = today - timedelta(days=60)
    post_exit_options = [{'label': 'Immediate', 'value': 'immediate'},
                         {'label': 'Next Candle', 'value': 'wait_next_candle'},
                         {'label': 'Signal Flip', 'value': 'wait_signal_flip'}]

    return html.Div([
        create_collapsible_container("Manual Backtester", "manual-panel", [
            html.Div([
                html.H4("Data Settings"),
                html.Div([
                    html.Div([html.Label('Trading Pair (e.g., BTCUSDT):'),
                              dcc.Input(id='symbol-input', value='BTCUSDT', type='text', className='custom-input')],
                             className='flex-item'),
                    html.Div([
                        html.Label('Timeframe:'),
                        dcc.Dropdown(
                            id='timeframe-input',
                            options=[{'label': t, 'value': t}
                                     for t in ['1m', '5m', '15m', '1h', '4h', '1d']],
                            value='1h',
                            clearable=False,
                            className='custom-input'
                        )
                    ], className='flex-item'),
                    html.Div([html.Label('Date Range:'),
                              date_range_inputs('date-range', sixty_days_ago.date(),
                                                today.date())], className='flex-item'),
                ], className='flex-container'),
            ], className='control-panel-group'),
            html.Div([
                html.H4("Strategy Parameters"),
                html.Label("Select Strategy:"),
                dcc.Dropdown(id='strategy-selector-dropdown',
                             options=[{'label': name, 'value': name}
                                      for name in STRATEGY_REGISTRY.keys()],
                             value=list(STRATEGY_REGISTRY.keys())[0], clearable=False, className='custom-input',
                             style={'backgroundColor': '#2c2c2c', 'color': '#f0f0f0'}),
                html.Div(id='strategy-params-container',
                         style={'marginTop': '15px'}),
            ], className='control-panel-group'),
            html.Div([
                html.H4("Backtest Settings"),
                html.Div([
                    html.Div([html.Label('Initial Capital:'),
                              dcc.Input(id='capital-input', value=10000, type='number', min=1,
                                        className='custom-input')], className='flex-item'),
                    # SL/TP removed - using signal flip only
                ], className='flex-container')
            ], className='control-panel-group'),
            html.Div([
                dcc.Checklist(id='live-update-checklist',
                              options=[{'label': 'Enable Auto-Refresh', 'value': 'ENABLED'}], value=[]),
                html.Label('Update every (sec):'),
                dcc.Input(id='live-update-frequency-input', value=30, type='number', min=5, style={'width': '80px'},
                          className='custom-input'),
            ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'gap': '10px'}),
            html.Div([
                html.Button('Refresh Data & Apply', id='refresh-button',
                            n_clicks=0, className='custom-button'),
                html.Button('Run Backtest', id='backtest-button',
                            n_clicks=0, className='custom-button'),
            ], style={'display': 'flex', 'justifyContent': 'center', 'gap': '20px', 'marginTop': '10px'}),
        ])
    ], className="control-panel")


def build_live_config_panel():
    return html.Div([
        create_collapsible_container("Live Trading Config", "live-config-panel", [
            html.Div([html.Label('Live Start Date:'),
                      html.Input(id='live-start-date-picker', type='date',
                                 value=str(datetime.now().date()),
                                 className='custom-input', style={'colorScheme': 'dark'})],
                     className='flex-item', style={'marginBottom': '15px'}),
            html.Div(
                [html.Label('Trade Size (USDT):'),
                 dcc.Input(id='units-usdt-input', value=10.0, type='number', min=0, className='custom-input')],
                className='flex-item', style={'marginBottom': '15px'}),
            html.Div([html.Label('Leverage:'),
                      dcc.Input(id='leverage-input', value=5, type='number', min=1, className='custom-input')],
                     className='flex-item', style={'marginBottom': '15px'}),
            html.Hr(),
            html.Label("Load Optimized Parameters (Optional)"),
            dcc.Dropdown(id='live-opt-pair-dropdown', placeholder="1. Select Optimized Pair...",
                         className='custom-input'),
            dcc.Dropdown(id='live-opt-params-dropdown', placeholder="2. Select Parameters for Pair...",
                         className='custom-input', style={'marginTop': '10px'}),
            html.Div([
                html.Button('Apply to Manual', id='apply-to-manual-btn', n_clicks=0,
                            className='custom-button', style={'backgroundColor': '#ffc107', 'color': 'black'}),
                html.Button('Save for Live Trading', id='save-config-button', n_clicks=0,
                            className='custom-button', style={'backgroundColor': '#28a745'}),
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginTop': '15px'}),
            html.P(id='save-config-confirmation',
                   style={'textAlign': 'center', 'marginTop': '10px', 'color': 'lime', 'minHeight': '20px'})
        ])
    ], className="control-panel")


def build_optimizer_panel():
    today = datetime.now(timezone.utc)
    fifteen_days_ago = today - timedelta(days=15)
    sixty_days_ago = today - timedelta(days=60)
    post_exit_options = [{'label': 'Immediate', 'value': 'immediate'},
                         {'label': 'Next Candle', 'value': 'wait_next_candle'},
                         {'label': 'Signal Flip', 'value': 'wait_signal_flip'}]
    return html.Div([
        create_collapsible_container("Multi-Pair Strategy Optimizer", "optimizer-panel", [
            html.Div([
                html.H4("Pair Selection"),
                dcc.Loading(id="loading-pair-selection", children=[
                    dcc.Dropdown(
                        id='manual-pair-dropdown', multi=True, className='custom-input',
                        placeholder="Select pairs manually or use Fill buttons below..."
                    ),
                    html.Div(id='pair-loading-error-message',
                             style={'color': '#dc3545', 'textAlign': 'center', 'marginTop': '10px'}),
                    html.Div([
                        dcc.Input(id='top-n-fill-input', type='number', value=5, min=1, style={'width': '60px'},
                                  className='custom-input'),
                        html.Button("Fill by Volume", id="fill-volume-btn",
                                    n_clicks=0, className='small-button'),
                        html.Button("Fill by Volatility", id="fill-volatility-btn", n_clicks=0,
                                    className='small-button')
                    ], style={'display': 'flex', 'gap': '10px', 'marginTop': '10px', 'alignItems': 'center'})
                ])
            ], className='control-panel-group'),
            html.Div([html.H4("Parameter Ranges (Multi-Strategy)"),
                      html.Div([
                          html.Label("Select Parameters to Optimize:"),
                          dcc.Checklist(
                              id='param-selection-checklist',
                              options=[
                                  {'label': 'Buy Signal Window',
                                      'value': 'buy_signal_window'},
                                  {'label': 'Buy Pattern Lookback',
                                      'value': 'buy_pattern_lookback'},
                                  {'label': 'Sell Signal Window',
                                      'value': 'sell_signal_window'},
                                  {'label': 'Sell Pattern Lookback',
                                      'value': 'sell_pattern_lookback'},
                                  {'label': 'ATR Period',
                                      'value': 'atr_period'},
                                  {'label': 'ATR Multiplier',
                                      'value': 'atr_multiplier'},
                                  # SL/TP removed
                              ],
                              value=['buy_signal_window', 'buy_pattern_lookback', 'sell_signal_window',
                                     'sell_pattern_lookback'],
                              inline=True,
                              className='responsive-checklist',
                              style={'marginBottom': '15px'}
                          )
                      ]),
                      html.Div(
                          [html.Button("Fast Scan", id="preset-fast-btn", className='small-button'),
                           html.Button(
                               "Normal Scan", id="preset-normal-btn", className='small-button'),
                           html.Button("Deep Scan", id="preset-deep-btn", className='small-button')],
                          style={'display': 'flex', 'justifyContent': 'center', 'gap': '10px', 'marginBottom': '15px',
                                 'flexWrap': 'wrap'}), html.Label("Ranges (min, max, step)"),
                      # Candlestick Parameters Container
                      html.Div([
                          dcc.Input(id='buy-window-range', placeholder='Buy Window (e.g. 5,15,2)',
                                    className='custom-input'),
                          dcc.Input(id='buy-lookback-range', placeholder='Buy Lookback (e.g. 2,5,1)',
                                    className='custom-input'),
                          dcc.Input(id='sell-window-range', placeholder='Sell Window (e.g. 5,15,2)',
                                    className='custom-input'),
                          dcc.Input(id='sell-lookback-range', placeholder='Sell Lookback (e.g. 2,5,1)',
                                    className='custom-input')],
                               id='candlestick-params-container', className='responsive-grid'),
                      # ATR SuperTrend Parameters Container
                      html.Div([
                          dcc.Input(id='atr-period-range', placeholder='ATR Period (e.g. 7,14,1)',
                                    className='custom-input'),
                          dcc.Input(id='atr-multiplier-range', placeholder='ATR Mult (e.g. 1.0,4.0,0.5)',
                                    className='custom-input')],
                               id='atr-params-container', className='responsive-grid', style={'display': 'none'}),
                      # SL/TP inputs removed
                      ], className='control-panel-group'),
            # Post-exit actions removed (no SL/TP)
            html.Div([html.H4("Run Settings"), html.Div([html.Div(
                [html.Label("Strategy to Optimize"), dcc.Dropdown(id='optimizer-strategy-selector',
                                                                 options=[{'label': k, 'value': k} for k in STRATEGY_REGISTRY.keys()],
                                                                 value=list(STRATEGY_REGISTRY.keys())[0],
                                                                 className='custom-input', clearable=False)],
                className='flex-item'),
                html.Div(
                [html.Label("Timeframe"), dcc.Dropdown(id='opt-timeframe-dropdown',
                                                       options=[{'label': t, 'value': t} for t in
                                                                ['5m', '15m', '30m', '1h', '4h']], value='1h',
                                                       className='custom-input', clearable=False)],
                className='flex-item'),
                html.Div([html.Label("In-Sample (IS) Date Range"),
                          date_range_inputs('is-date', sixty_days_ago.date(), fifteen_days_ago.date())],
                         className='flex-item'),
                html.Div([html.Label("Out-of-Sample 1 (OOS1) Date Range"),
                          date_range_inputs('oos1-date', fifteen_days_ago.date(), (today - timedelta(days=7)).date())],
                         className='flex-item'),
                html.Div([html.Label("Out-of-Sample 2 (OOS2) Date Range"),
                          date_range_inputs('oos2-date', (today - timedelta(days=7)).date(), today.date())],
                         className='flex-item'),
                html.Div(
                    [html.Label("Number of Trials"),
                     dcc.Input(id='max-combinations-input', type='number', value=100, className='custom-input')],
                    className='flex-item'), html.Div(
                [html.Label("Min Trades"),
                 dcc.Input(id='min-trades-input', type='number', value=5, className='custom-input')],
                className='flex-item'),
                html.Div(
                    [html.Label("Min Candles"),
                     dcc.Input(id='min-candles-input', type='number', value=500, min=50, className='custom-input')],
                    className='flex-item')
            ], className='flex-container')], className='control-panel-group'),
            html.Div([
                html.H4("Optimization Strategy"),
                html.Div([
                    html.Label("Optimization Mode:"),
                    dcc.Dropdown(
                        id='optimization-mode-dropdown',
                        options=[
                            {'label': 'Comprehensive (All Combinations)', 'value': 'comprehensive'},
                            {'label': 'Efficient (No Duplicates)', 'value': 'efficient'},
                            {'label': 'Smart Sampling (Bayesian)', 'value': 'smart'}
                        ],
                        value='efficient',
                        clearable=False,
                        className='custom-input'
                    )
                ], className='flex-item', style={'marginBottom': '15px'}),
                html.P("• Comprehensive: Tests all possible combinations (may be slow)", 
                       style={'fontSize': '12px', 'color': '#888', 'margin': '0'}),
                html.P("• Efficient: Avoids duplicate combinations (recommended)", 
                       style={'fontSize': '12px', 'color': '#888', 'margin': '0'}),
                html.P("• Smart Sampling: Uses AI to focus on promising areas", 
                       style={'fontSize': '12px', 'color': '#888', 'margin': '0 0 15px 0'}),
            ], className='control-panel-group'),
            html.Div([
                html.H4("Optimization Goal (Weights)"),
                html.Div([
                    html.Div([html.Label("Total Return % Weight:"),
                              dcc.Input(id='weight-return-input', type='number', value=50, min=0,
                                        className='custom-input')], className='flex-item'),
                    html.Div([html.Label("Win Rate % Weight:"),
                              dcc.Input(id='weight-winrate-input', type='number', value=30, min=0,
                                        className='custom-input')], className='flex-item'),
                    html.Div([html.Label("Total Trades Weight:"),
                              dcc.Input(id='weight-trades-input', type='number', value=20, min=0,
                                        className='custom-input')], className='flex-item'),
                ], className='flex-container'),
                html.Div([
                    html.Div([
                        dcc.Checklist(
                            id='stability-optimization-checkbox',
                            options=[{'label': ' Enable Stability-Based Optimization (prioritizes consistent growth)', 'value': 'enabled'}],
                            value=['enabled'],
                            className='custom-checklist'
                        )
                    ], className='flex-item'),
                    html.Div([
                        html.Label("Stability Weight (0-1):"),
                        dcc.Input(id='stability-weight-input', type='number', value=0.7, min=0, max=1, step=0.1,
                                  className='custom-input', disabled=False)
                    ], className='flex-item', style={'marginLeft': '20px'})
                ], className='flex-container', style={'alignItems': 'center'}),
                html.Hr(style={'margin': '15px 0'}),
                html.H4("Trade Balance Filter", style={'textAlign': 'center'}),
                html.Div([
                    html.Div([
                        dcc.Checklist(
                            id='trade-balance-filter-checkbox',
                            options=[{'label': ' Enable Trade Balance Filter (balanced long/short trades)', 'value': 'enabled'}],
                            value=[],
                            className='custom-checklist'
                        )
                    ], className='flex-item'),
                    html.Div([
                        html.Label("Max Long/Short Ratio:"),
                        dcc.Input(id='max-trade-ratio-input', type='number', value=3.0, min=1.0, max=10.0, step=0.1,
                                  className='custom-input', disabled=True)
                    ], className='flex-item', style={'marginLeft': '20px'})
                ], className='flex-container', style={'alignItems': 'center'}),
                html.P("• Filters out strategies heavily biased toward long or short trades", 
                       style={'fontSize': '12px', 'color': '#888', 'margin': '5px 0 0 0', 'textAlign': 'center'}),
                html.P("• Ratio 2.0 means max 2:1 long:short or short:long trade count", 
                       style={'fontSize': '12px', 'color': '#888', 'margin': '0 0 15px 0', 'textAlign': 'center'})
            ], className='control-panel-group'),
            html.Div([
                html.Div([
                    html.Button("Start Optimization", id='start-opt-button',
                                n_clicks=0, className='custom-button'),
                    html.Button("Stop Optimization", id='stop-opt-button', n_clicks=0, className='custom-button',
                                style={'backgroundColor': '#dc3545'}, disabled=True),
                    html.Button("Pause", id='pause-opt-button', n_clicks=0, className='custom-button',
                                style={'backgroundColor': '#ffc107', 'color': 'black'}, disabled=True),
                    html.Button("Continue", id='continue-opt-button', n_clicks=0, className='custom-button',
                                style={'backgroundColor': '#28a745'}, disabled=True),
                    html.Button("Download Partial Results", id='download-partial-results-btn', n_clicks=0,
                                className='small-button', disabled=True),
                ], style={'display': 'flex', 'justifyContent': 'center', 'gap': '10px', 'flexWrap': 'wrap'}),
                html.Div(
                    html.Button("Clear Log", id='clear-log-button', n_clicks=0, className='small-button',
                                style={'marginTop': '10px'}),
                    style={'textAlign': 'center'}
                )
            ], style={'marginTop': '20px'})
        ])
    ], className="control-panel")


def build_refine_panel():
    """Builds the UI for funneling/refining optimization pairs."""
    return html.Div([
        create_collapsible_container("Refine & Re-run", "refine-panel", [
            html.P("Use results from the last optimization to select the top pairs for the next run.",
                   style={'textAlign': 'center'}),
            html.Div([
                html.Div([
                    html.Label("Sort Top Pairs By:"),
                    dcc.Dropdown(
                        id='refine-sort-by-dropdown',
                        options=[
                            {'label': 'Score (Weighted)', 'value': 'Score'},
                            {'label': 'Total Return %', 'value': 'Total_Return'},
                            {'label': 'Win Rate %', 'value': 'Win_Rate'},
                            {'label': 'Total Trades', 'value': 'Total_Trades'},
                            {'label': 'Profit Factor', 'value': 'Profit_Factor'},
                            {'label': 'Sharpe Ratio', 'value': 'Sharpe_Ratio'},
                            {'label': 'Calmar Ratio (Risk-Adjusted)', 'value': 'Calmar_Ratio'},
                            {'label': 'Consistency Score', 'value': 'Consistency_Score'},
                            {'label': 'Robustness Score (IS/OOS)', 'value': 'Robustness_Score'},
                            {'label': 'Trade Balance Score', 'value': 'Trade_Balance_Score'},
                            {'label': 'Trade Balance Ratio (Lower Better)', 'value': 'Trade_Balance_Ratio'},
                            {'label': 'Trade Difference (Lower Better)', 'value': 'Trade_Difference'},
                            {'label': 'Profitable Trades Count', 'value': 'Profitable_Trades'},
                            {'label': 'Avg Profitable Trade', 'value': 'Avg_Profitable_Trade'},
                            {'label': 'Avg Profitable Long', 'value': 'Avg_Profitable_Long'},
                            {'label': 'Avg Profitable Short', 'value': 'Avg_Profitable_Short'},
                            {'label': 'Avg Unprofitable Trade (Higher Better)', 'value': 'Avg_Unprofitable_Trade'},
                            {'label': 'OOS1 Return %', 'value': 'OOS1_Return'},
                            {'label': 'OOS2 Return %', 'value': 'OOS2_Return'},
                            {'label': 'Total OOS Return %', 'value': 'Total_OOS_Return'},
                            {'label': 'Composite Score', 'value': 'Composite_Score'},
                            {'label': 'Max Drawdown (Lower Better)', 'value': 'Max_Drawdown'},
                        ],
                        value='Score',
                        clearable=False,
                        className='custom-input'
                    )
                ], className='flex-item'),
                html.Div([
                    html.Label("Best Results Priority:"),
                    dcc.Dropdown(
                        id='best-results-priority-dropdown',
                        options=[
                            {'label': 'Score (Weighted)', 'value': 'Score'},
                            {'label': 'Total Return %', 'value': 'Total_Return'},
                            {'label': 'Calmar Ratio', 'value': 'Calmar_Ratio'},
                            {'label': 'Robustness Score', 'value': 'Robustness_Score'},
                            {'label': 'Trade Balance Score', 'value': 'Trade_Balance_Score'},
                            {'label': 'Avg Profitable Trade', 'value': 'Avg_Profitable_Trade'},
                            {'label': 'Profitable Trades Count', 'value': 'Profitable_Trades'},
                            {'label': 'Stability Performance Score', 'value': 'Stability_Performance_Score'},
                        ],
                        value='Score',
                        clearable=False,
                        className='custom-input'
                    )
                ], className='flex-item'),
                html.Div([
                    html.Label("Number of Top Pairs to Keep:"),
                    dcc.Input(id='refine-top-n-input', type='number',
                              value=10, min=1, className='custom-input')
                ], className='flex-item'),
            ], className='flex-container', style={'alignItems': 'flex-end'}),
            html.Hr(style={'margin': '20px 0'}),
            html.H4("Export Configuration", style={'textAlign': 'center'}),
            html.Div([
                html.Div([
                    html.Label("Export Count:"),
                    dcc.Input(id='export-count-input', type='number', value=5, min=1, max=20,
                              className='custom-input', style={'width': '80px'})
                ], className='flex-item'),
                html.Div([
                    html.Label("Trade Size (USDT):"),
                    dcc.Input(id='export-units-input', type='number', value=50.0, min=1,
                              className='custom-input', style={'width': '100px'})
                ], className='flex-item'),
                html.Div([
                    html.Label("Leverage:"),
                    dcc.Input(id='export-leverage-input', type='number', value=5, min=1, max=125,
                              className='custom-input', style={'width': '80px'})
                ], className='flex-item'),
                html.Div([
                    html.Label("Timeframe:"),
                    dcc.Dropdown(id='export-timeframe-dropdown',
                                 options=[{'label': t, 'value': t} for t in [
                                     '1m', '5m', '15m', '30m', '1h', '4h', '1d']],
                                 value='15m', clearable=False, className='custom-input')
                ], className='flex-item'),
            ], className='flex-container', style={'alignItems': 'flex-end', 'marginBottom': '15px'}),
            html.P("Export will use the sorting method above and take the best result for each selected pair.",
                   style={'textAlign': 'center', 'color': '#888', 'fontSize': '14px', 'marginBottom': '15px'}),
            html.Div([
                html.Button("Use Top Pairs for Next Run", id='refine-pairs-btn', n_clicks=0, className='custom-button',
                            style={'marginTop': '10px', 'marginRight': '10px'}),
                html.Button("Export to trade_config.json", id='export-to-config-btn', n_clicks=0,
                            className='custom-button', style={'marginTop': '10px', 'backgroundColor': '#28a745'})
            ], style={'textAlign': 'center'})
        ])
    ])


# --- App Layout ---
app.layout = html.Div(style={'backgroundColor': '#111111', 'color': '#FFFFFF', 'padding': '10px'}, children=[
    dcc.Store(id='trades-data-store'),
    dcc.Store(id='all-pairs-store'),
    dcc.Store(id='opt-results-store'),
    dcc.Store(id='all-trials-store'),
    dcc.Store(id='optimized-pairs-store'),
    dcc.Store(id='detailed-trades-store'),
    dcc.Store(id='applied-params-store'),
    # idle, running, paused, stopped, finished
    dcc.Store(id='opt-status-store', data='idle'),
    dcc.Store(id='optimization-settings-store'),
    dcc.Interval(id='optimization-trigger-interval',
                 interval=500, n_intervals=0, max_intervals=0),

    # Hidden no-op outputs for clientside "scroll to results" callbacks.
    html.Div(id='scroll-backtest-dummy', style={'display': 'none'}),
    html.Div(id='scroll-optimizer-dummy', style={'display': 'none'}),

    html.Div([
        html.H2("Futures Dashboard", style={'margin': '0', 'flex': '1'}),
        html.A('Manual Backtester', href='#manual-section', className='nav-link'),
        html.A('Optimizer', href='#optimizer-section', className='nav-link'),
        html.A('Backtest Results', href='#manual-results-section',
               className='nav-link'),
        html.A('Optimizer Results', href='#optimizer-results-section',
               className='nav-link'),
    ], className='nav-bar'),

    html.H1("Cryptocurrency Futures Trading Dashboard", style={
            'textAlign': 'center', 'marginTop': '70px'}),

    html.Div([
        html.Div(build_config_panel(), id='manual-section'),
        html.Div(build_live_config_panel(), id='live-config-section'),
        html.Div(build_optimizer_panel(), id='optimizer-section'),
    ], className='main-container'),
    html.Hr(style={'borderColor': '#555', 'marginTop': '30px'}),

    html.Div(id='manual-results-section', children=[
        
        create_collapsible_container("Price Chart", "crypto-candlestick-graph",
                                     dcc.Graph(id='crypto-candlestick-graph', style={'height': '60vh'})),
        html.H4(id='backtest-summary',
                style={'textAlign': 'center', 'color': '#4CAF50', 'margin': '20px'}),
        create_collapsible_container("Portfolio Value", "portfolio-graph",
                                     dcc.Graph(id='portfolio-graph', style={'height': '40vh'})),
        create_collapsible_container("Trades Log", "trades-table",
                                     html.Div(dash_table.DataTable(
                                         id='trades-table',
                                         style_cell={'backgroundColor': '#2c2c2c', 'color': '#f0f0f0',
                                                     'border': '1px solid #444'},
                                         style_header={'backgroundColor': '#00BFFF', 'color': '#000000',
                                                       'fontWeight': 'bold'},
                                         style_data={
                                             'backgroundColor': '#2c2c2c', 'color': '#f0f0f0'}
                                     ), className='dash-table-container'),
                                     export_info={'button_id': 'export-manual-trades-btn'}),
    ]),

    html.Div(id='optimizer-results-section', children=[
        dcc.Loading(id="loading-opt", children=[html.Div(
            id='opt-progress-output', style={'textAlign': 'center'})]),
        create_collapsible_container("Optimization Log", "opt-log", dcc.Textarea(id='opt-log-textarea', readOnly=True,
                                                                                 style={'width': '100%',
                                                                                        'height': '400px',
                                                                                        'backgroundColor': '#222',
                                                                                        'color': 'lightgray'})),
        create_collapsible_container("Optimization Results (Best per Pair)", "opt-results-table",
                                     html.Div(
                                         dash_table.DataTable(
                                             id='opt-results-table',
                                             sort_action='native', page_size=15, filter_action='native',
                                             style_cell={'backgroundColor': '#2c2c2c', 'color': '#f0f0f0',
                                                         'border': '1px solid #444'},
                                             style_header={'backgroundColor': '#00BFFF', 'color': '#000000',
                                                           'fontWeight': 'bold'},
                                             style_data={'backgroundColor': '#2c2c2c', 'color': '#f0f0f0'},
                                             style_data_conditional=[
                                                 # Color coding for filtering
                                                 {'if': {'filter_query': '{Total_Return} > 10'}, 'backgroundColor': '#1e5631', 'color': '#4caf50'},
                                                 {'if': {'filter_query': '{Total_Return} < 0'}, 'backgroundColor': '#5c1e1e', 'color': '#f44336'},
                                                 {'if': {'filter_query': '{Win_Rate} > 70'}, 'backgroundColor': '#1e4d5c', 'color': '#4fc3f7'},
                                                 {'if': {'filter_query': '{Trade_Balance_Score} > 80'}, 'backgroundColor': '#2e4d1e', 'color': '#8bc34a'},
                                                 {'if': {'filter_query': '{Avg_Profitable_Trade} > 50'}, 'backgroundColor': '#1e4d2e', 'color': '#66bb6a'},
                                                 {'if': {'filter_query': '{Profitable_Trades} > 20'}, 'backgroundColor': '#2e1e4d', 'color': '#ab47bc'},
                                             ]
                                         ),
                                         className='dash-table-container'),
                                     export_info={'button_id': 'export-opt-results-btn'}),
        create_collapsible_container("All Optimization Trials", "opt-all-trials-table",
                                     html.Div(
                                         dash_table.DataTable(
                                             id='opt-all-trials-table',
                                             sort_action='native', page_size=20, filter_action='native',
                                             style_cell={'backgroundColor': '#2c2c2c', 'color': '#f0f0f0',
                                                         'border': '1px solid #444'},
                                             style_header={'backgroundColor': '#FF6B35', 'color': '#000000',
                                                           'fontWeight': 'bold'},
                                             style_data={'backgroundColor': '#2c2c2c', 'color': '#f0f0f0'},
                                             style_data_conditional=[
                                                 # Enhanced color coding for all trials
                                                 {'if': {'filter_query': '{Total_Return} > 15'}, 'backgroundColor': '#1e5631', 'color': '#4caf50'},
                                                 {'if': {'filter_query': '{Total_Return} > 5 && {Total_Return} <= 15'}, 'backgroundColor': '#2d4a1e', 'color': '#81c784'},
                                                 {'if': {'filter_query': '{Total_Return} < -5'}, 'backgroundColor': '#5c1e1e', 'color': '#f44336'},
                                                 {'if': {'filter_query': '{Total_Return} >= -5 && {Total_Return} < 0'}, 'backgroundColor': '#4a1e1e', 'color': '#ef5350'},
                                                 {'if': {'filter_query': '{Win_Rate} > 80'}, 'backgroundColor': '#1e4d5c', 'color': '#4fc3f7'},
                                                 {'if': {'filter_query': '{Win_Rate} > 60 && {Win_Rate} <= 80'}, 'backgroundColor': '#1e3d4c', 'color': '#64b5f6'},
                                                 {'if': {'filter_query': '{Trade_Balance_Score} > 90'}, 'backgroundColor': '#2e4d1e', 'color': '#8bc34a'},
                                                 {'if': {'filter_query': '{Trade_Balance_Score} > 70 && {Trade_Balance_Score} <= 90'}, 'backgroundColor': '#2e3d1e', 'color': '#aed581'},
                                                 {'if': {'filter_query': '{Max_Drawdown} < 5'}, 'backgroundColor': '#4d2e1e', 'color': '#ffb74d'},
                                                 {'if': {'filter_query': '{Sharpe_Ratio} > 2'}, 'backgroundColor': '#4d1e4d', 'color': '#ba68c8'},
                                                 {'if': {'filter_query': '{Avg_Profitable_Trade} > 100'}, 'backgroundColor': '#1e4d2e', 'color': '#66bb6a'},
                                                 {'if': {'filter_query': '{Avg_Profitable_Trade} > 50 && {Avg_Profitable_Trade} <= 100'}, 'backgroundColor': '#1e3d2e', 'color': '#81c784'},
                                                 {'if': {'filter_query': '{Profitable_Trades} > 30'}, 'backgroundColor': '#2e1e4d', 'color': '#ab47bc'},
                                                 {'if': {'filter_query': '{Profitable_Trades} > 15 && {Profitable_Trades} <= 30'}, 'backgroundColor': '#2e1e3d', 'color': '#ba68c8'},
                                                 {'if': {'filter_query': '{Avg_Unprofitable_Trade} > -20'}, 'backgroundColor': '#4d4d1e', 'color': '#d4e157'},
                                                 {'if': {'filter_query': '{Avg_Unprofitable_Trade} > -50 && {Avg_Unprofitable_Trade} <= -20'}, 'backgroundColor': '#3d3d1e', 'color': '#cddc39'},
                                             ]
                                         ),
                                         className='dash-table-container'),
                                     export_info={'button_id': 'export-all-trials-btn'}),
        build_refine_panel(),
    ]),
    dcc.Interval(id='log-update-interval', interval=1000, n_intervals=0),
    dcc.Interval(id='live-update-interval', interval=30 *
                 1000, n_intervals=0, disabled=True),
    html.Div(id='dummy-autoscroll-output', style={'display': 'none'}),
    html.Div(id='dummy-feedback-output-regular', style={'display': 'none'}),
    html.Div(id='dummy-feedback-output-pattern', style={'display': 'none'}),
    dcc.Download(id="download-manual-xlsx"),
    dcc.Download(id="download-opt-xlsx"),
    dcc.Download(id="download-all-trials-xlsx"),
    dcc.Download(id="download-partial-opt-xlsx")
])


# --- Generic Callbacks ---
@app.callback(
    Output({'type': 'collapsible-body', 'index': MATCH}, 'style'),
    Output({'type': 'hide-show-btn', 'index': MATCH}, 'children'),
    Input({'type': 'hide-show-btn', 'index': MATCH}, 'n_clicks'),
    prevent_initial_call=True
)
def toggle_collapsible_body(n_clicks):
    if n_clicks and n_clicks % 2 == 1:
        return {'display': 'none'}, "Show"
    return {'display': 'block'}, "Hide"


@app.callback(
    [Output('live-update-interval', 'disabled'),
     Output('live-update-interval', 'interval')],
    [Input('live-update-checklist', 'value'),
     Input('live-update-frequency-input', 'value')]
)
def control_live_update_interval(checklist_value, frequency):
    is_disabled = 'ENABLED' not in checklist_value
    interval_ms = int(frequency) * 1000 if frequency else 30000
    return is_disabled, interval_ms


# --- Auto-scroll to results after an action (runs in the browser) ---
app.clientside_callback(
    """
    function(n) {
        if (n) {
            const el = document.getElementById('manual-results-section');
            if (el) { el.scrollIntoView({behavior: 'smooth', block: 'start'}); }
        }
        return '';
    }
    """,
    Output('scroll-backtest-dummy', 'children'),
    Input('backtest-button', 'n_clicks'),
    prevent_initial_call=True,
)

app.clientside_callback(
    """
    function(n) {
        if (n) {
            const el = document.getElementById('optimizer-results-section');
            if (el) { el.scrollIntoView({behavior: 'smooth', block: 'start'}); }
        }
        return '';
    }
    """,
    Output('scroll-optimizer-dummy', 'children'),
    Input('start-opt-button', 'n_clicks'),
    prevent_initial_call=True,
)


# --- Manual Backtester Callbacks ---
@app.callback(
    Output('strategy-params-container', 'children'),
    Input('strategy-selector-dropdown', 'value'),
    State('applied-params-store', 'data')
)
def update_strategy_parameters_ui(strategy_name, applied_params):
    if not strategy_name:
        return []
    strategy_class = STRATEGY_REGISTRY.get(strategy_name)
    if not strategy_class:
        return []
    params = strategy_class.get_parameters()
    ui_elements = []
    applied_params = applied_params if isinstance(applied_params, dict) else {}
    for param_name, config in params.items():
        default_value = applied_params.get(param_name, config['default'])
        ui_elements.append(html.Div([
            html.Label(f"{param_name.replace('_', ' ').title()}:"),
            dcc.Input(id={'type': 'strategy-param-input', 'param': param_name}, type=config['type'],
                      value=default_value, step=config.get('step', 1), min=0 if config['type'] == 'number' else None,
                      className='custom-input')
        ], className='flex-item'))
    return [html.Div(ui_elements, className='flex-container')]


@app.callback(
    Output('crypto-candlestick-graph', 'figure'),
    [Input('refresh-button', 'n_clicks'),
     Input('live-update-interval', 'n_intervals')],
    [State('symbol-input', 'value'),
     State('timeframe-input', 'value'),
     State('date-range-start', 'value'), State('date-range-end', 'value'),
     State('strategy-selector-dropdown',
           'value'), State({'type': 'strategy-param-input', 'param': ALL}, 'value'),
     State({'type': 'strategy-param-input', 'param': ALL}, 'id')]
)
def update_main_chart(n_clicks, n_intervals, symbol, timeframe, start_date, end_date, strategy_name, param_values,
                      param_ids):
    if not all([symbol, strategy_name, trader]):
        return go.Figure()
    trader.get_historical_data(symbol, timeframe, start_date, end_date)
    if trader.data is None or trader.data.empty:
        return go.Figure().update_layout(title="No data loaded.",
                                         template='plotly_dark')
    strategy_class = STRATEGY_REGISTRY.get(strategy_name)
    if not strategy_class:
        return go.Figure()
    params = {p['param']: v for p, v in zip(param_ids, param_values)}
    strategy_instance = strategy_class()
    df = strategy_instance.generate_signals(trader.data.copy(), params)
    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                                         name='Candlestick')])
    if 'position' in df.columns:
        buy_signals = df[df['position'] == 1]
        sell_signals = df[df['position'] == -1]
        fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Low'], mode='markers',
                                 marker=dict(color='cyan', size=10, symbol='triangle-up'), name='Buy Signal'))
        fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['High'], mode='markers',
                                 marker=dict(color='yellow', size=10, symbol='triangle-down'), name='Sell Signal'))

    # Add Three White Soldiers and Three Black Crows pattern visualization
    if 'ThreeWhiteSoldiers' in df.columns and 'ThreeBlackCrows' in df.columns:
        # Three White Soldiers patterns (bullish) - show on candle close price
        white_soldiers = df[df['ThreeWhiteSoldiers'] == 1]
        if not white_soldiers.empty:
            fig.add_trace(go.Scatter(x=white_soldiers.index, y=white_soldiers['Close'], mode='markers',
                                     marker=dict(color='lime', size=13, symbol='circle'), name='Three White Soldiers'))

        # Three Black Crows patterns (bearish) - show on candle close price
        black_crows = df[df['ThreeBlackCrows'] == 1]
        if not black_crows.empty:
            fig.add_trace(go.Scatter(x=black_crows.index, y=black_crows['Close'], mode='markers',
                                     marker=dict(color='red', size=13, symbol='circle'), name='Three Black Crows'))
    fig.update_layout(title=f"{symbol} Chart ({timeframe})",
                      template='plotly_dark', xaxis_rangeslider_visible=False)
    return fig


@app.callback(
    [Output('portfolio-graph', 'figure'), Output('trades-table', 'data'), Output('trades-table', 'columns'),
     Output('trades-table',
            'style_data_conditional'), Output('backtest-summary', 'children'),
     Output('trades-data-store', 'data')],
    Input('backtest-button', 'n_clicks'),
    [State('capital-input', 'value'),
     State('strategy-selector-dropdown', 'value'),
     State({'type': 'strategy-param-input', 'param': ALL}, 'value'),
     State({'type': 'strategy-param-input', 'param': ALL}, 'id')]
)
def run_backtest_callback(n_clicks, capital, strategy_name, param_values, param_ids):
    if n_clicks == 0 or not all([trader, not trader.data.empty, strategy_name]):
        return go.Figure(), [], [], [], "", {}
    strategy_class = STRATEGY_REGISTRY.get(strategy_name)
    if not strategy_class:
        return go.Figure(), [], [], [], "Strategy not found", {}
    params = {p['param']: v for p, v in zip(param_ids, param_values)}
    strategy_instance = strategy_class()
    df_with_signals = strategy_instance.generate_signals(
        trader.data.copy(), params)
    backtester = Backtester(initial_capital=float(capital))
    trades_df, portfolio_df = backtester.run_backtest(df_with_signals)
    summary_text = "No trades executed."
    if portfolio_df is not None and not portfolio_df.empty and trades_df is not None and not trades_df.empty:
        total_return = (
            (portfolio_df['Portfolio_Value'].iloc[-1] / float(capital)) - 1) * 100
        summary_text = f"Total Return: {total_return:.2f}%"
    fig = go.Figure()
    if portfolio_df is not None and not portfolio_df.empty:
        fig.add_trace(
            go.Scatter(x=portfolio_df['Date'], y=portfolio_df['Portfolio_Value'], mode='lines', name='Portfolio Value',
                       line=dict(color='#00BFFF', width=2)))
        if trades_df is not None and not trades_df.empty:
            entry_dates = pd.to_datetime(trades_df['Entry_Date'])
            entry_values = [portfolio_df.loc[portfolio_df['Date'].sub(date).abs().idxmin(), 'Portfolio_Value'] for date
                            in entry_dates]
            fig.add_trace(go.Scatter(
                x=entry_dates, y=entry_values, mode='markers', name='Trade Entry',
                marker=dict(symbol='triangle-up', size=10,
                            color='#4CAF50', line=dict(width=2, color='white'))
            ))
            exit_dates = pd.to_datetime(trades_df['Exit_Date'])
            exit_values = [portfolio_df.loc[portfolio_df['Date'].sub(date).abs().idxmin(), 'Portfolio_Value'] for date
                           in exit_dates]
            fig.add_trace(go.Scatter(
                x=exit_dates, y=exit_values, mode='markers', name='Trade Exit',
                marker=dict(symbol='triangle-down', size=10,
                            color='#F44336', line=dict(width=2, color='white'))
            ))
    fig.update_layout(title='Portfolio Value Over Time with Trade Markers', template='plotly_dark', xaxis_title='Date',
                      yaxis_title='Portfolio Value ($)', hovermode='x unified', legend=dict(x=0.02, y=0.98))
    trades_data, trades_cols = [], []
    if trades_df is not None and not trades_df.empty:
        trades_df_display = trades_df.copy()
        for col in ['Entry_Date', 'Exit_Date']:
            trades_df_display[col] = pd.to_datetime(
                trades_df_display[col]).dt.strftime('%Y-%m-%d %H:%M')
        trades_data = trades_df_display.to_dict('records')
        trades_cols = [{"name": i, "id": i} for i in trades_df_display.columns]
    style_data_conditional = [
        {'if': {'column_id': 'PnL', 'filter_query': '{PnL} > 0'},
         'backgroundColor': '#1e5631', 'color': '#4caf50', 'fontWeight': 'bold'},
        {'if': {'column_id': 'PnL', 'filter_query': '{PnL} < 0'},
         'backgroundColor': '#5c1e1e', 'color': '#f44336', 'fontWeight': 'bold'},
        {'if': {'column_id': 'PnL', 'filter_query': '{PnL} = 0'},
         'backgroundColor': '#2c2c2c', 'color': '#f0f0f0', 'fontWeight': 'bold'}]
    return fig, trades_data, trades_cols, style_data_conditional, summary_text, trades_data


# --- Live Config and Optimizer Callbacks ---
@app.callback(
    Output('save-config-confirmation', 'children'),
    Input('save-config-button', 'n_clicks'),
    [State('symbol-input', 'value'),
     State('timeframe-input', 'value'),
     # SL/TP states removed
     State('strategy-selector-dropdown',
           'value'), State({'type': 'strategy-param-input', 'param': ALL}, 'value'),
     State({'type': 'strategy-param-input', 'param': ALL}, 'id'),
     State('live-start-date-picker', 'value'), State('units-usdt-input', 'value'), State('leverage-input', 'value')],
    prevent_initial_call=True
)
def save_trade_config(n_clicks, symbol, bar_length,
                      strategy_name, param_values, param_ids, start_date, units, leverage):
    if n_clicks > 0:
        strategy_params = {p['param']: v for p,
                           v in zip(param_ids, param_values)}
        config_data = {
            "strategy_name": strategy_name, "symbol": symbol, "bar_length": bar_length,
            "start_date": start_date, "units_usdt": float(units), "leverage": int(leverage),
            **strategy_params
        }
        try:
            with open('trade_config.json', 'w') as f:
                json.dump(config_data, f, indent=4)
            return f"Config saved for '{strategy_name}' at {datetime.now().strftime('%H:%M:%S')}"
        except Exception as e:
            return f"Error saving config: {e}"
    return ""


@app.callback(
    [Output('symbol-input', 'value'),
     Output('timeframe-input', 'value')],
    [Input('live-opt-pair-dropdown', 'value'),
     Input('live-opt-params-dropdown', 'value')],
    [State('all-trials-store', 'data')],
    prevent_initial_call=True
)
def sync_live_config_to_manual(selected_pair, selected_index, opt_data):
    if not selected_pair or selected_index is None or not opt_data:
        return no_update, no_update
    try:
        df = pd.DataFrame(opt_data)
        selected_row = df.loc[selected_index]
        return (
            selected_row['Trading_Pair'], selected_row['Timeframe']
        )
    except (KeyError, IndexError):
        return no_update, no_update


@app.callback(
    Output('applied-params-store', 'data', allow_duplicate=True),
    Output('strategy-selector-dropdown', 'value'),
    Input('apply-to-manual-btn', 'n_clicks'),
    [State('live-opt-params-dropdown', 'value'),
     State('all-trials-store', 'data')],
    prevent_initial_call=True
)
def apply_params_to_manual(n_clicks, selected_index, data):
    if n_clicks == 0 or selected_index is None or not data:
        return no_update, no_update
    df = pd.DataFrame(data)
    try:
        selected_row = df.loc[selected_index]
        
        # Debug: Log available columns for troubleshooting
        available_cols = list(selected_row.index)
        add_optimization_log(f"Apply to Manual - Available columns: {available_cols}")
        
        # FIXED: Detect strategy type based on available columns
        # Check for both possible column name formats (ATR_Period or Atr_Period)
        if ('ATR_Period' in selected_row and 'ATR_Multiplier' in selected_row) or \
           ('Atr_Period' in selected_row and 'Atr_Multiplier' in selected_row):
            # ATR SuperTrend strategy
            strategy_name = "ATR SuperTrend"
            # Handle both possible column name formats
            atr_period_col = 'ATR_Period' if 'ATR_Period' in selected_row else 'Atr_Period'
            atr_mult_col = 'ATR_Multiplier' if 'ATR_Multiplier' in selected_row else 'Atr_Multiplier'
            params_to_apply = {
                'atr_period': int(selected_row[atr_period_col]),
                'atr_multiplier': float(selected_row[atr_mult_col]),
            }
            add_optimization_log(f"✅ Applied ATR SuperTrend params: {params_to_apply}")
        else:
            # Candlestick Patterns strategy (fallback)
            strategy_name = "Candlestick Patterns"
            params_to_apply = {
                'buy_signal_window': int(selected_row['Buy_Signal_Window']),
                'buy_pattern_lookback': int(selected_row['Buy_Pattern_Lookback']),
                'sell_signal_window': int(selected_row['Sell_Signal_Window']),
                'sell_pattern_lookback': int(selected_row['Sell_Pattern_Lookback']),
            }
            add_optimization_log(f"✅ Applied Candlestick Patterns params: {params_to_apply}")
        
        return params_to_apply, strategy_name
    except KeyError as e:
        add_optimization_log(f"KeyError in apply_params_to_manual: {e}")
        return no_update, no_update


# Callback removed - no SL/TP parameters to update
    return sl_value, tp_value, sl_action, tp_action


@app.callback(Output('applied-params-store', 'data', allow_duplicate=True),
              Input('strategy-selector-dropdown', 'value'),
              prevent_initial_call=True)
def clear_applied_params_on_strategy_change(_):
    return {}


@app.callback(Output('live-opt-pair-dropdown', 'options'), Input('all-trials-store', 'data'))
def update_live_pair_dropdown(data):
    if not data:
        return []
    df = pd.DataFrame(data)
    return [{'label': pair, 'value': pair} for pair in df['Trading_Pair'].unique()]


@app.callback(
    Output('live-opt-params-dropdown', 'options'),
    Input('live-opt-pair-dropdown', 'value'),
    State('all-trials-store', 'data')
)
def update_live_params_dropdown(selected_pair, data):
    if not selected_pair or not data:
        return []
    
    try:
        df = pd.DataFrame(data)
        pair_df = df[df['Trading_Pair'] == selected_pair].sort_values(
            by="Score", ascending=False)
        options = []
        
        for i, row in pair_df.iterrows():
            # Build label dynamically based on available columns
            label_parts = []
            
            # Core metrics (should always be present)
            if 'Score' in row:
                label_parts.append(f"Score: {row['Score']:.1f}")
            if 'Total_Return' in row:
                label_parts.append(f"Ret: {row['Total_Return']:.1f}%")
            if 'Win_Rate' in row:
                label_parts.append(f"WR: {row['Win_Rate']:.1f}%")
            if 'Total_Trades' in row:
                label_parts.append(f"Trd: {row['Total_Trades']}")
            
            # Strategy-specific parameters
            if all(col in row for col in ['Buy_Signal_Window', 'Buy_Pattern_Lookback', 'Sell_Signal_Window', 'Sell_Pattern_Lookback']):
                # Candlestick strategy parameters
                label_parts.append(f"B{row['Buy_Signal_Window']}/{row['Buy_Pattern_Lookback']} S{row['Sell_Signal_Window']}/{row['Sell_Pattern_Lookback']}")
            elif all(col in row for col in ['Atr_Period', 'Atr_Multiplier']):
                # SuperTrend strategy parameters
                label_parts.append(f"ATR{row['Atr_Period']}/{row['Atr_Multiplier']:.1f}")
            elif all(col in row for col in ['ATR_Period', 'ATR_Multiplier']):
                # Alternative column naming for SuperTrend
                label_parts.append(f"ATR{row['ATR_Period']}/{row['ATR_Multiplier']:.1f}")
            
            label = " | ".join(label_parts) if label_parts else f"Row {i}"
            options.append({'label': label, 'value': i})
        
        return options
        
    except Exception as e:
        print(f"Error in update_live_params_dropdown: {e}")
        return []


# --- Optimizer Callbacks ---
@app.callback(
    [Output('all-pairs-store', 'data'),
     Output('manual-pair-dropdown', 'options'),
     Output('pair-loading-error-message', 'children')],
    Input('optimizer-section', 'id')
)
def load_all_pairs(section_id):
    if trader:
        all_pairs = trader.get_all_usdt_futures_pairs()
        if all_pairs:
            options = [{'label': p['symbol'], 'value': p['symbol']}
                       for p in all_pairs]
            return all_pairs, options, ""
    error_message = "Failed to fetch trading pairs. Check API keys in config.json and internet connection."
    return [], [], error_message


@app.callback(
    Output('manual-pair-dropdown', 'value'),
    [Input('fill-volume-btn', 'n_clicks'),
     Input('fill-volatility-btn', 'n_clicks')],
    [State('all-pairs-store', 'data'),
     State('top-n-fill-input', 'value')],
    prevent_initial_call=True
)
def fill_top_pairs(volume_clicks, volatility_clicks, all_pairs_data, top_n):
    ctx = callback_context
    if not ctx.triggered or not all_pairs_data:
        return no_update

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    df_pairs = pd.DataFrame(all_pairs_data)

    if df_pairs.empty:
        return []

    sort_key = 'volume' if button_id == 'fill-volume-btn' else 'volatility'
    sorted_pairs = df_pairs.sort_values(by=sort_key, ascending=False)[
        'symbol'].tolist()

    return sorted_pairs[:top_n]


@app.callback(
    Output('manual-pair-dropdown', 'value', allow_duplicate=True),
    Input('refine-pairs-btn', 'n_clicks'),
    [State('all-trials-store', 'data'), State('refine-sort-by-dropdown', 'value'),
     State('refine-top-n-input', 'value')],
    prevent_initial_call=True
)
def refine_pairs_for_next_run(n_clicks, all_trials_data, sort_by, top_n):
    if n_clicks == 0 or not all_trials_data or not sort_by or not top_n:
        return no_update
    df = pd.DataFrame(all_trials_data)
    if sort_by not in df.columns:
        return no_update
    df_sorted = df.sort_values(by=sort_by, ascending=False)
    top_pairs = df_sorted['Trading_Pair'].unique()[:top_n]
    add_optimization_log(
        f"Refined pairs list: Loaded top {len(top_pairs)} pairs sorted by {sort_by}.")
    return top_pairs.tolist()


@app.callback(
    [Output('buy-window-range', 'value'), Output('buy-lookback-range', 'value'),
     Output('sell-window-range', 'value'), Output('sell-lookback-range', 'value'),
     Output('atr-period-range', 'value'), Output('atr-multiplier-range', 'value')],
    [Input('preset-fast-btn', 'n_clicks'), Input('preset-normal-btn', 'n_clicks'),
     Input('preset-deep-btn', 'n_clicks')],
    prevent_initial_call=True
)
def set_parameter_presets(fast, normal, deep):
    ctx = callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'preset-fast-btn':
        return '5,10,5', '2,3,1', '5,10,5', '2,3,1', '10,20,2', '2.0,4.0,0.5'
    if button_id == 'preset-normal-btn':
        return '5,15,2', '2,5,1', '5,15,2', '2,5,1', '10,30,2', '3.0,6.0,0.5'
    if button_id == 'preset-deep-btn':
        return '3,20,1', '1,5,1', '3,20,1', '1,5,1', '5,20,1', '1.0,4.0,0.25'
    return [no_update] * 6


# Stability optimization checkbox callback
@app.callback(
    Output('stability-weight-input', 'disabled'),
    Input('stability-optimization-checkbox', 'value'),
    prevent_initial_call=True
)
def toggle_stability_weight_input(checkbox_value):
    """Enable/disable stability weight input based on checkbox."""
    return 'enabled' not in (checkbox_value or [])


# Trade balance filter checkbox callback
@app.callback(
    Output('max-trade-ratio-input', 'disabled'),
    Input('trade-balance-filter-checkbox', 'value'),
    prevent_initial_call=True
)
def toggle_trade_ratio_input(checkbox_value):
    """Enable/disable trade ratio input based on checkbox."""
    return 'enabled' not in (checkbox_value or [])


# Dynamic parameter visibility callback
@app.callback(
    [Output('candlestick-params-container', 'style'),
     Output('atr-params-container', 'style'),
     Output('param-selection-checklist', 'options'),
     Output('param-selection-checklist', 'value')],
    Input('optimizer-strategy-selector', 'value'),
    prevent_initial_call=True
)
def update_optimizer_parameter_visibility(selected_strategy):
    """Show/hide parameter containers and update checklist based on selected strategy."""
    if not selected_strategy:
        # Default to Candlestick if no strategy selected
        return (
            {'display': 'block'},  # Show candlestick params
            {'display': 'none'},   # Hide ATR params
            [
                {'label': 'Buy Signal Window', 'value': 'buy_signal_window'},
                {'label': 'Buy Pattern Lookback', 'value': 'buy_pattern_lookback'},
                {'label': 'Sell Signal Window', 'value': 'sell_signal_window'},
                {'label': 'Sell Pattern Lookback', 'value': 'sell_pattern_lookback'},
            ],
            ['buy_signal_window', 'buy_pattern_lookback', 'sell_signal_window', 'sell_pattern_lookback']
        )
    
    # Get strategy class and its parameters
    strategy_class = STRATEGY_REGISTRY.get(selected_strategy)
    if not strategy_class:
        # Fallback to default
        return (
            {'display': 'block'},
            {'display': 'none'},
            [
                {'label': 'Buy Signal Window', 'value': 'buy_signal_window'},
                {'label': 'Buy Pattern Lookback', 'value': 'buy_pattern_lookback'},
                {'label': 'Sell Signal Window', 'value': 'sell_signal_window'},
                {'label': 'Sell Pattern Lookback', 'value': 'sell_pattern_lookback'},
            ],
            ['buy_signal_window', 'buy_pattern_lookback', 'sell_signal_window', 'sell_pattern_lookback']
        )
    
    strategy_params = strategy_class.get_parameters()
    param_keys = list(strategy_params.keys())
    
    # Create checklist options based on strategy parameters
    checklist_options = []
    default_values = []
    
    # Map parameter names to display labels
    param_labels = {
        'buy_signal_window': 'Buy Signal Window',
        'buy_pattern_lookback': 'Buy Pattern Lookback', 
        'sell_signal_window': 'Sell Signal Window',
        'sell_pattern_lookback': 'Sell Pattern Lookback',
        'atr_period': 'ATR Period',
        'atr_multiplier': 'ATR Multiplier',
        'rsi_period': 'RSI Period',
        'oversold_threshold': 'Oversold Threshold',
        'overbought_threshold': 'Overbought Threshold',
        'fast_ma_period': 'Fast MA Period',
        'middle_ma_period': 'Middle MA Period',
        'slow_ma_period': 'Slow MA Period',
        'ma_type': 'MA Type'
    }
    
    for param_key in param_keys:
        label = param_labels.get(param_key, param_key.replace('_', ' ').title())
        checklist_options.append({'label': label, 'value': param_key})
        default_values.append(param_key)
    
    # Determine container visibility
    has_candlestick_params = any(p in param_keys for p in ['buy_signal_window', 'buy_pattern_lookback', 'sell_signal_window', 'sell_pattern_lookback'])
    has_atr_params = any(p in param_keys for p in ['atr_period', 'atr_multiplier'])
    
    candlestick_style = {'display': 'block'} if has_candlestick_params else {'display': 'none'}
    atr_style = {'display': 'block'} if has_atr_params else {'display': 'none'}
    
    return candlestick_style, atr_style, checklist_options, default_values


# Best results priority callback
@app.callback(
    [Output('opt-results-table', 'data', allow_duplicate=True), 
     Output('opt-results-table', 'columns', allow_duplicate=True)],
    Input('best-results-priority-dropdown', 'value'),
    State('all-trials-store', 'data'),
    prevent_initial_call=True
)
def update_best_results_priority(priority_column, all_trials_data):
    """Update best results table based on selected priority metric."""
    if not all_trials_data or not priority_column:
        return no_update, no_update
    
    try:
        df_all = pd.DataFrame(all_trials_data)
        if priority_column not in df_all.columns:
            return no_update, no_update
        
        # Handle columns that should be sorted in ascending order (lower is better)
        ascending = priority_column in ['Max_Drawdown', 'Trade_Balance_Ratio', 'Trade_Difference', 'Avg_Unprofitable_Trade']
        
        # Get best result per pair based on selected priority
        if ascending:
            df_best = df_all.loc[df_all.groupby('Trading_Pair')[priority_column].idxmin()].copy()
        else:
            df_best = df_all.loc[df_all.groupby('Trading_Pair')[priority_column].idxmax()].copy()
        
        # Sort the results
        df_best = df_best.sort_values(by=priority_column, ascending=ascending).round(2)
        
        columns = [{"name": i.replace("_", " ").title(), "id": i} for i in df_best.columns]
        data = df_best.to_dict('records')
        
        add_optimization_log(f"🎯 Updated best results priority to: {priority_column}")
        
        return data, columns
    except Exception as e:
        add_optimization_log(f"❌ Error updating best results priority: {e}")
        return no_update, no_update


@app.callback(
    [Output('opt-status-store', 'data', allow_duplicate=True)],
    Input('stop-opt-button', 'n_clicks'),
    prevent_initial_call=True
)
def stop_optimization(n_clicks):
    if n_clicks > 0:
        OPTIMIZATION_STOP_EVENT.set()
        OPTIMIZATION_PAUSE_EVENT.clear()
        add_optimization_log(
            "!! STOP request received. Finishing current trials...")
        return ['stopped']
    return no_update


@app.callback(
    Output('opt-log-textarea', 'value', allow_duplicate=True),
    Input('export-to-config-btn', 'n_clicks'),
    [State('all-trials-store', 'data'), State('export-count-input', 'value'),
     State('refine-sort-by-dropdown',
           'value'), State('export-units-input', 'value'),
     State('export-leverage-input', 'value'), State('export-timeframe-dropdown', 'value')],
    prevent_initial_call=True
)
def export_to_config(n_clicks, opt_data, export_count, sort_by, units_usdt, leverage, timeframe):
    if n_clicks > 0 and opt_data:
        try:
            df = pd.DataFrame(opt_data)
            if not df.empty:
                # Use the input values for configuration
                count = int(export_count) if export_count else 5
                units = float(units_usdt) if units_usdt else 50.0
                lev = int(leverage) if leverage else 10
                tf = timeframe if timeframe else '15m'
                sort_column = sort_by if sort_by else 'Score'

                # Check if stability optimization was used
                if 'Stability_Score' in df.columns:
                    add_optimization_log("🎯 Using STABILITY-ENHANCED export")
                    success = trader.export_stable_pairs_to_config(
                        df, top_n=count, stability_threshold=-20.0,
                        units_usdt=units, leverage=lev, timeframe=tf)
                else:
                    success = trader.export_best_results_to_config_enhanced(
                        df, top_n=count, sort_by=sort_column,
                        units_usdt=units, leverage=lev, timeframe=tf)

                if success:
                    add_optimization_log(
                        f"🎯 Top {count} pairs exported to trade_config.json!")
                    add_optimization_log(f"   Sorted by: {sort_column}")
                    add_optimization_log(
                        f"   Settings: {units} USDT, {lev}x leverage, {tf} timeframe")
                    add_optimization_log(
                        "You can now run: python multi_symbol_trader.py")
                else:
                    add_optimization_log("❌ Failed to export results")
            else:
                add_optimization_log("❌ No optimization results to export")
        except Exception as e:
            add_optimization_log(f"❌ Export error: {e}")

    return '\n'.join(OPTIMIZATION_LOGS)


@app.callback(
    Output('opt-status-store', 'data', allow_duplicate=True),
    Input('pause-opt-button', 'n_clicks'),
    prevent_initial_call=True
)
def pause_optimization(n_clicks):
    if n_clicks > 0:
        OPTIMIZATION_PAUSE_EVENT.set()
        add_optimization_log(
            "⏸️ PAUSE request received. Pausing after current trials complete...")
        return ['paused']
    return no_update


@app.callback(
    Output('opt-status-store', 'data', allow_duplicate=True),
    Input('continue-opt-button', 'n_clicks'),
    prevent_initial_call=True
)
def continue_optimization(n_clicks):
    if n_clicks > 0:
        OPTIMIZATION_PAUSE_EVENT.clear()
        add_optimization_log(
            "▶️ CONTINUE request received. Resuming optimization...")
        return ['running']
    return no_update


@app.callback(
    Output('opt-log-textarea', 'value', allow_duplicate=True),
    Input('clear-log-button', 'n_clicks'),
    prevent_initial_call=True
)
def clear_log_button_click(n_clicks):
    if n_clicks > 0:
        OPTIMIZATION_LOGS.clear()
        if LOG_FILENAME:
            with open(LOG_FILENAME, 'a', encoding='utf-8') as f:
                f.write(
                    f"\n--- LOG CLEARED AT {datetime.now().strftime('%H:%M:%S')} ---\n\n")
        return ""
    return no_update


@app.callback(
    [Output('start-opt-button', 'disabled'), Output('stop-opt-button', 'disabled'),
     Output('pause-opt-button',
            'disabled'), Output('continue-opt-button', 'disabled'),
     Output('download-partial-results-btn', 'disabled')],
    [Input('opt-status-store', 'data')]
)
def toggle_opt_buttons(status):
    # start, stop, pause, continue, download_partial
    if status == 'running':
        return True, False, False, True, False
    if status == 'paused':
        return True, False, True, False, False
    # idle, stopped, finished
    return False, True, True, True, not PARTIAL_RESULTS_LIST


@app.callback(
    [Output('opt-status-store', 'data'),
     Output('optimization-settings-store', 'data'),
     Output('optimization-trigger-interval', 'max_intervals'),
     Output('optimization-trigger-interval', 'n_intervals'),
     Output('opt-progress-output', 'children', allow_duplicate=True)],
    Input('start-opt-button', 'n_clicks'),
    [State('manual-pair-dropdown', 'value'),                    # pairs
     State('param-selection-checklist', 'value'),               # selected_params
     State('buy-window-range', 'value'),                        # bw_str
     State('buy-lookback-range', 'value'),                      # bl_str
     State('sell-window-range', 'value'),                       # sw_str
     State('sell-lookback-range', 'value'),                     # slr_str
     State('atr-period-range', 'value'),                        # atr_period_str
     State('atr-multiplier-range', 'value'),                    # atr_mult_str
     State('max-combinations-input', 'value'),                  # n_trials
     State('min-trades-input', 'value'),                        # min_trades
     State('min-candles-input', 'value'),                       # min_candles
     State('optimizer-strategy-selector', 'value'),             # strategy_name
     State('opt-timeframe-dropdown', 'value'),                  # timeframe
     State('is-date-start', 'value'),                     # is_start
     State('is-date-end', 'value'),                       # is_end
     State('oos1-date-start', 'value'),                   # oos1_start
     State('oos1-date-end', 'value'),                     # oos1_end
     State('oos2-date-start', 'value'),                   # oos2_start
     State('oos2-date-end', 'value'),                     # oos2_end
     State('weight-return-input', 'value'),                     # weight_return
     State('weight-winrate-input', 'value'),                    # weight_winrate
     State('weight-trades-input', 'value'),                     # weight_trades
     State('stability-optimization-checkbox', 'value'),         # stability_checkbox
     State('stability-weight-input', 'value'),                  # stability_weight
     State('optimization-mode-dropdown', 'value'),              # optimization_mode
     State('trade-balance-filter-checkbox', 'value'),           # trade_balance_checkbox
     State('max-trade-ratio-input', 'value')],                  # max_trade_ratio
    prevent_initial_call=True
)
def start_optimization_trigger(n_clicks, pairs, selected_params, bw_str, bl_str, sw_str, slr_str, 
                               atr_period_str, atr_mult_str, n_trials, min_trades, min_candles, 
                               strategy_name, timeframe, is_start, is_end, oos1_start, oos1_end, 
                               oos2_start, oos2_end, weight_return, weight_winrate, weight_trades, 
                               stability_checkbox, stability_weight, optimization_mode,
                               trade_balance_checkbox, max_trade_ratio):
    
    # DEBUG: Add explicit debug logging to verify date alignment
    print(f"DEBUG: Optimization Trigger Received")
    print(f"DEBUG: IS Dates: {is_start} to {is_end}")
    print(f"DEBUG: OOS1 Dates: {oos1_start} to {oos1_end}")
    print(f"DEBUG: OOS2 Dates: {oos2_start} to {oos2_end}")
    print(f"DEBUG: Strategy: {strategy_name}, Timeframe: {timeframe}")
    print(f"DEBUG: Selected Pairs: {pairs}")
    print(f"DEBUG: Selected Params: {selected_params}")
    
    if n_clicks == 0:
        return no_update, no_update, no_update, no_update, no_update

    if not all([pairs, selected_params]):
        msg = "Please select pairs and parameters to optimize."
        return 'idle', no_update, 0, 0, msg

    if any(v is None for v in [n_trials, min_trades, min_candles]):
        msg = "Error: 'Number of Trials', 'Min Trades', and 'Min Candles' must have a numeric value."
        add_optimization_log(msg)
        return 'idle', no_update, 0, 0, msg

    OPTIMIZATION_LOGS.clear()
    add_optimization_log("--- Quick Validation ---")
    add_optimization_log(f"Validating {len(pairs)} pairs...")

    # Quick validation - assume most major pairs have data
    valid_pairs = []
    min_candles_val = int(min_candles)

    if not trader:
        msg = "Trader is not initialized."
        add_optimization_log(f"!! ERROR: {msg}")
        return 'idle', no_update, 0, 0, msg

    # Proper validation - check data availability for all pairs
    overall_start_date = is_start
    overall_end_date = oos2_end  # Use the end of OOS2 as the overall end date

    for symbol in pairs:
        try:
            # Check if data exists for the full date range
            data = trader.get_historical_data_for_symbol(
                symbol, timeframe, overall_start_date, overall_end_date)

            if data is not None and not data.empty:
                # Check if we have enough candles
                has_enough_candles = len(data) >= min_candles_val

                if has_enough_candles:
                    valid_pairs.append(symbol)
                    add_optimization_log(
                        f"✅ {symbol} - {len(data)} candles available")
                else:
                    add_optimization_log(
                        f"❌ {symbol} - Only {len(data)} candles (need {min_candles_val})")
            else:
                add_optimization_log(
                    f"❌ {symbol} - No data found for date range")

        except Exception as e:
            add_optimization_log(f"❌ {symbol} - Error: {str(e)[:50]}...")

    if not valid_pairs:
        add_optimization_log(
            f"❌ VALIDATION FAILED: None of the selected pairs have valid data.")
        add_optimization_log(f"Tried pairs: {', '.join(pairs)}")
        msg = f"Validation Failed: None of the {len(pairs)} selected pairs have sufficient data."
        return 'idle', no_update, 0, 0, msg

    add_optimization_log(
        f"✅ Validation Complete: {len(valid_pairs)} of {len(pairs)} pairs are valid.")

    skipped_pairs = [pair for pair in pairs if pair not in valid_pairs]
    if skipped_pairs:
        add_optimization_log(f"⚠️ Skipped pairs: {', '.join(skipped_pairs)}")

    try:
        # Check if stability optimization is enabled
        use_stability = 'enabled' in (stability_checkbox or [])
        stability_weight_val = float(stability_weight) if use_stability and stability_weight is not None else 0.0
        
        # Check if trade balance filter is enabled
        use_trade_balance_filter = 'enabled' in (trade_balance_checkbox or [])
        max_trade_ratio_val = float(max_trade_ratio) if use_trade_balance_filter and max_trade_ratio is not None else 3.0
        
        settings = {
            'pairs': valid_pairs, 'selected_params': selected_params, 'bw_str': bw_str, 'bl_str': bl_str,
            'sw_str': sw_str, 'slr_str': slr_str, 'atr_period_str': atr_period_str, 'atr_mult_str': atr_mult_str, 'n_trials': int(n_trials),
            'min_trades': int(min_trades), 'min_candles': min_candles_val, 'strategy_name': strategy_name, 'timeframe': timeframe,
            'is_start': is_start, 'is_end': is_end, 
            'oos1_start': oos1_start, 'oos1_end': oos1_end, 'oos2_start': oos2_start, 'oos2_end': oos2_end,
            'weight_return': weight_return, 'weight_winrate': weight_winrate,
            'weight_trades': weight_trades, 'use_stability': use_stability, 'stability_weight': stability_weight_val,
            'optimization_mode': optimization_mode or 'efficient', 'use_trade_balance_filter': use_trade_balance_filter,
            'max_trade_ratio': max_trade_ratio_val
        }
    except (ValueError, TypeError) as e:
        msg = f"Error: Invalid numeric input. Please check values. Details: {e}"
        add_optimization_log(msg)
        return 'idle', no_update, 0, 0, msg

    global LOG_FILENAME
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    LOG_FILENAME = f"optimization_log_{timestamp}.txt"
    PARTIAL_RESULTS_LIST.clear()
    OPTIMIZATION_STOP_EVENT.clear()
    OPTIMIZATION_PAUSE_EVENT.clear()
    add_optimization_log(f"Optimization started. Logging to {LOG_FILENAME}")

    return 'running', settings, 1, 0, f"Optimization is starting for {len(valid_pairs)} valid pairs..."


# --- TASK RUNNER CALLBACK ---
@app.callback(
    [Output('opt-results-table', 'data'), Output('opt-results-table', 'columns'),
     Output('opt-all-trials-table', 'data'), Output('opt-all-trials-table', 'columns'),
     Output('opt-progress-output', 'children'), Output('opt-results-store', 'data'),
     Output('optimized-pairs-store', 'data'), Output('all-trials-store', 'data'),
     Output('opt-status-store', 'data', allow_duplicate=True)],
    Input('optimization-trigger-interval', 'n_intervals'),
    State('optimization-settings-store', 'data'),
    prevent_initial_call=True
)
def run_optimization_task(n_intervals, settings):
    if n_intervals == 0 or not settings:
        return [no_update] * 9

    add_optimization_log(
        "Task runner received settings, proceeding with optimization.")

    try:
        pairs = settings['pairs']
        param_ranges = {
            'buy_signal_window': settings['bw_str'], 'buy_pattern_lookback': settings['bl_str'],
            'sell_signal_window': settings['sw_str'], 'sell_pattern_lookback': settings['slr_str'],
            'atr_period': settings['atr_period_str'], 'atr_multiplier': settings['atr_mult_str'],
        }
        weights = {
            'total_return': settings['weight_return'] or 0,
            'win_rate': settings['weight_winrate'] or 0,
            'total_trades': settings['weight_trades'] or 0
        }
    except KeyError as e:
        add_optimization_log(
            f"!! CRITICAL ERROR: Failed to unpack settings. Missing key: {e}")
        return [], [], f"Error: Missing key {e} in settings.", None, None, None, 'stopped'

    # Choose optimization method based on stability setting
    optimization_mode = settings.get('optimization_mode', 'efficient')
    strategy_name = settings.get('strategy_name', list(STRATEGY_REGISTRY.keys())[0])
    
    if settings.get('use_stability', False):
        add_optimization_log(f"🎯 Using STABILITY-BASED optimization ({optimization_mode.upper()} mode) with {strategy_name} strategy")
        df_results = trader.optimize_trading_pairs_with_stability(
            trading_pairs=pairs, param_ranges=param_ranges, selected_params=settings['selected_params'],
            is_start_date=settings['is_start'], is_end_date=settings['is_end'],
            oos1_start_date=settings['oos1_start'], oos1_end_date=settings['oos1_end'],
            oos2_start_date=settings['oos2_start'], oos2_end_date=settings['oos2_end'],
            timeframe=settings['timeframe'], min_trades=settings['min_trades'], n_trials=settings['n_trials'],
            weights=weights, min_candles=settings['min_candles'], stop_event=OPTIMIZATION_STOP_EVENT,
            pause_event=OPTIMIZATION_PAUSE_EVENT, stability_weight=settings['stability_weight'],
            optimization_mode=optimization_mode, strategy_name=strategy_name
        )
    else:
        add_optimization_log(f"📈 Using STANDARD optimization ({optimization_mode.upper()} mode) with {strategy_name} strategy")
        df_results = trader.optimize_trading_pairs(
            trading_pairs=pairs, param_ranges=param_ranges, selected_params=settings['selected_params'],
            is_start_date=settings['is_start'], is_end_date=settings['is_end'],
            oos1_start_date=settings['oos1_start'], oos1_end_date=settings['oos1_end'],
            oos2_start_date=settings['oos2_start'], oos2_end_date=settings['oos2_end'],
            timeframe=settings['timeframe'], min_trades=settings['min_trades'], n_trials=settings['n_trials'],
            weights=weights, min_candles=settings['min_candles'], stop_event=OPTIMIZATION_STOP_EVENT,
            pause_event=OPTIMIZATION_PAUSE_EVENT, optimization_mode=optimization_mode, strategy_name=strategy_name
        )

    add_optimization_log("Main optimizer function has completed.")
    final_status = 'finished'
    if OPTIMIZATION_STOP_EVENT.is_set():
        add_optimization_log("✅ Optimization stopped by user.")
        final_status = 'stopped'

    # Add trade difference calculation for all results
    if not df_results.empty and 'Long_Trades' in df_results.columns and 'Short_Trades' in df_results.columns:
        df_results['Trade_Difference'] = abs(df_results['Long_Trades'] - df_results['Short_Trades'])
        add_optimization_log(f"📊 Added trade difference calculation to {len(df_results)} results")

    # Store original results for file export (before filtering)
    df_all_results_for_export = df_results.copy() if not df_results.empty else pd.DataFrame()
    
    # Apply trade balance filter if enabled (for UI display only)
    if not df_results.empty and settings.get('use_trade_balance_filter', False):
        max_ratio = settings.get('max_trade_ratio', 3.0)
        initial_count = len(df_results)
        
        # Filter results based on trade balance ratio for UI display
        if 'Trade_Balance_Ratio' in df_results.columns:
            df_results_filtered = df_results[df_results['Trade_Balance_Ratio'] <= max_ratio]
            filtered_count = len(df_results_filtered)
            add_optimization_log(f"🎯 Trade Balance Filter: {filtered_count}/{initial_count} results meet criteria (max ratio: {max_ratio})")
            
            if filtered_count < initial_count:
                add_optimization_log(f"   {initial_count - filtered_count} strategies have unbalanced long/short trades")
                add_optimization_log(f"   All results will be included in exported files for manual review")
            
            # Use filtered results for UI display if any meet criteria
            if filtered_count > 0:
                df_results = df_results_filtered
            else:
                add_optimization_log("⚠️ No results meet trade balance criteria, showing all results")
        else:
            add_optimization_log("⚠️ Trade Balance Filter enabled but ratio data not available")

    if df_results.empty:
        message = "⚠️ Optimization finished, but no valid results were found that met the criteria."
        if not OPTIMIZATION_STOP_EVENT.is_set():
            send_telegram_notification(message)
        return [], [], [], [], "No valid results found.", None, pairs, None, final_status

    # Sort by appropriate score (stability-performance score if available, otherwise regular score)
    sort_column = 'Stability_Performance_Score' if 'Stability_Performance_Score' in df_results.columns else 'Score'
    df_results = df_results.sort_values(by=sort_column, ascending=False).round(2)
    
    # Use configurable priority for best results (default to sort_column if not specified)
    best_results_priority = sort_column  # This will be updated by callback
    df_best_results = df_results.loc[df_results.groupby('Trading_Pair')[best_results_priority].idxmax()].copy()
    all_trials_data = df_results.to_dict('records')
    df_display = df_best_results.copy()
    columns = [{"name": i.replace("_", " ").title(), "id": i}
               for i in df_display.columns]
    data = df_display.to_dict('records')

    if not OPTIMIZATION_STOP_EVENT.is_set():
        try:
            # Always send complete results file for manual filtering
            results_for_export = df_all_results_for_export if not df_all_results_for_export.empty else df_results
            message = f"✅ Optimization complete! Found {len(results_for_export)} total valid results from {len(pairs)} pairs."
            
            # Add trade balance info to message if filter was applied
            if settings.get('use_trade_balance_filter', False) and not df_results.empty:
                balanced_count = len(df_results)
                total_count = len(results_for_export)
                if balanced_count < total_count:
                    message += f"\n🎯 {balanced_count} results meet trade balance criteria, {total_count} total results in file for manual review."
            
            output_best = io.BytesIO()
            # Use complete results for best per pair calculation
            df_best_all = results_for_export.loc[results_for_export.groupby('Trading_Pair')['Score'].idxmax()].copy() if not results_for_export.empty else pd.DataFrame()
            if not df_best_all.empty:
                df_best_all.to_excel(output_best, index=False, sheet_name='Best_Results')
            file1 = {'object': output_best, 'filename': f"Best_Results_{settings['is_start']}_to_{settings['oos2_end']}.xlsx",
                     'caption': "Top result for each pair (all results included)."}
            
            output_all = io.BytesIO()
            if not results_for_export.empty:
                results_for_export.to_excel(output_all, index=False, sheet_name='All_Trials')
            file2 = {'object': output_all, 'filename': f"All_Trials_{settings['is_start']}_to_{settings['oos2_end']}.xlsx",
                     'caption': "All valid trials."}
            send_telegram_notification(message, files=[file1, file2])
        except Exception as e:
            add_optimization_log(
                f"!! ERROR during Telegram notification creation: {e}")

    final_message = f"Optimization Complete! Found {len(df_results)} total valid results."
    if OPTIMIZATION_STOP_EVENT.is_set():
        final_message += " (Process was stopped early by user)"

    # Add export reminder
    if len(df_results) > 0:
        add_optimization_log(
            "🎯 Ready to export! Use 'Export to trade_config.json' button below to save best results for trading.")

    # Prepare all trials table data
    all_trials_columns = [{"name": i.replace("_", " ").title(), "id": i} for i in df_results.columns]
    all_trials_data_display = df_results.to_dict('records')
    
    return data, columns, all_trials_data_display, all_trials_columns, final_message, df_best_results.to_dict('records'), pairs, all_trials_data, final_status


@app.callback(Output('opt-log-textarea', 'value'), Input('log-update-interval', 'n_intervals'))
def update_logs(n): return "\n".join(OPTIMIZATION_LOGS)


@app.callback(
    Output('download-opt-xlsx', 'data'),
    Input('export-opt-results-btn', 'n_clicks'),
    [State('all-trials-store', 'data'), State('is-date-start', 'value'),
     State('oos2-date-end', 'value')],
    prevent_initial_call=True
)
def export_opt_results(n_clicks, all_trials_data, start_date, end_date):
    if n_clicks is None or not all_trials_data:
        return no_update
    df_all = pd.DataFrame(all_trials_data)
    df_best = df_all.loc[df_all.groupby('Trading_Pair')['Score'].idxmax()]
    filename = f"OptResults_{start_date}_to_{end_date}.xlsx"
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_best.to_excel(writer, index=False, sheet_name='Best_Per_Pair')
        df_all.to_excel(writer, index=False, sheet_name='All_Trials')
    output.seek(0)
    return dcc.send_bytes(output.getvalue(), filename)


@app.callback(
    Output('download-all-trials-xlsx', 'data'),
    Input('export-all-trials-btn', 'n_clicks'),
    [State('all-trials-store', 'data'), State('is-date-start', 'value'), State('oos2-date-end', 'value')],
    prevent_initial_call=True
)
def export_all_trials_results(n_clicks, all_trials_data, start_date, end_date):
    if n_clicks is None or not all_trials_data:
        return no_update
    df_all = pd.DataFrame(all_trials_data)
    filename = f"AllTrials_{start_date}_to_{end_date}.xlsx"
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_all.to_excel(writer, index=False, sheet_name='All_Trials')
    output.seek(0)
    return dcc.send_bytes(output.getvalue(), filename)


@app.callback(
    Output('download-partial-opt-xlsx', 'data'),
    Input('download-partial-results-btn', 'n_clicks'),
    [State('is-date-start', 'value'),
     State('oos2-date-end', 'value')],
    prevent_initial_call=True
)
def export_partial_opt_results(n_clicks, start_date, end_date):
    if n_clicks is None or not PARTIAL_RESULTS_LIST:
        add_optimization_log(
            "Partial download requested, but no results are available yet.")
        return no_update

    add_optimization_log(
        f"Exporting {len(PARTIAL_RESULTS_LIST)} partial results...")
    df_all = pd.concat(PARTIAL_RESULTS_LIST, ignore_index=True)
    if df_all.empty:
        return no_update

    df_best = df_all.loc[df_all.groupby('Trading_Pair')['Score'].idxmax()]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"Partial_OptResults_{start_date}_to_{end_date}_{timestamp}.xlsx"

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_best.to_excel(writer, index=False,
                         sheet_name='Best_Per_Pair_Partial')
        df_all.to_excel(writer, index=False, sheet_name='All_Trials_Partial')
    output.seek(0)
    return dcc.send_bytes(output.getvalue(), filename)


# --- Main Execution Block ---
if __name__ == "__main__":
    if not os.path.exists("assets"):
        os.makedirs("assets")

    style_css_content = """
/* --- General Body & Theme --- */
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    background-color: #111111;
    color: #FFFFFF;
    padding-top: 50px; /* Add padding to body to prevent content from hiding behind fixed nav bar */
}
/* --- NEW: Pinned Navigation Bar --- */
.nav-bar {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 25px;
    background-color: rgba(30, 30, 30, 0.9);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid #00BFFF;
    padding: 10px;
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    z-index: 1000;
}
.nav-bar .nav-link {
    color: #FFFFFF;
    text-decoration: none;
    font-weight: bold;
    font-size: 16px;
    transition: color 0.2s ease-in-out;
}
.nav-bar .nav-link:hover {
    color: #00BFFF;
}
.nav-bar h2 {
    color: #FFFFFF !important;
    position: absolute;
    left: 20px;
}
h1, h2, h3, h4 {
    color: #00BFFF;
    text-shadow: 1px 1px 2px #000000;
    text-align: center;
}
/* --- Main Layout Containers --- */
.main-container {
    display: flex;
    flex-direction: column;
    gap: 20px;
}
.control-panel {
    flex: 1;
    min-width: 320px;
    border: 1px solid #333;
    border-radius: 8px;
    padding: 15px;
    background-color: #1e1e1e;
}
.control-panel-group {
    border: 1px solid #444;
    border-radius: 5px;
    padding: 15px;
    margin-bottom: 20px;
}
/* --- Collapsible Sections --- */
.collapsible-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 10px;
    border-bottom: 2px solid #00BFFF;
    padding-bottom: 8px;
}
/* --- Buttons --- */
.custom-button, .small-button {
    background-color: #007bff; /* Standard Blue */
    color: white;
    border: none;
    padding: 10px 18px;
    cursor: pointer;
    border-radius: 5px;
    font-weight: bold;
    font-size: 14px;
    transition: background-color 0.2s ease, transform 0.1s ease;
}
.custom-button:hover {
    background-color: #0056b3;
    transform: translateY(-1px);
}
.custom-button:disabled {
    background-color: #555;
    color: #aaa;
    cursor: not-allowed;
    transform: none;
}
.custom-button.clicked {
    background-color: #28a745;
    transform: translateY(1px);
}
.small-button {
    padding: 4px 8px;
    font-size: 12px;
}
/* --- Input Fields & Dropdowns --- */
.custom-input, .custom-input .Select-control {
    width: 100%;
    box-sizing: border-box;
    background-color: #2c2c2c !important;
    border: 1px solid #555 !important;
    border-radius: 4px;
    color: #f0f0f0 !important;
}
.custom-input input {
    padding: 8px;
    color: #f0f0f0 !important;
    background-color: #2c2c2c !important;
    border: none;
}
.custom-input .Select-placeholder, .custom-input .Select-value-label {
    color: #ccc !important;
}
/* --- Data Tables --- */
.dash-table-container .dash-spreadsheet-container { border: 1px solid #444; }
.dash-table-container .dash-header {
    background-color: #00BFFF;
    color: #000000;
    font-weight: bold;
}
.dash-table-container .dash-cell {
    background-color: #222;
    color: #eee;
    border: 1px solid #444;
}
/* --- Tabs --- */
.tab-container .dash-tabs {
    background-color: #1e1e1e;
    border-bottom: 2px solid #00BFFF;
}
.tab-container .dash-tab {
    background-color: #333;
    color: #ccc;
    border-radius: 5px 5px 0 0;
    border: 1px solid #444;
}
.tab-container .dash-tab--selected {
    background-color: #00BFFF;
    color: #000;
    font-weight: bold;
    border-color: #00BFFF;
}
/* --- Flexbox Utilities --- */
.flex-container {
    display: flex;
    justify-content: space-around;
    align-items: flex-end;
    flex-wrap: wrap;
    gap: 20px;
}
.flex-item {
    display: flex;
    flex-direction: column;
    gap: 5px;
    min-width: 180px;
    flex: 1;
}
.responsive-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 10px;
}
.responsive-checklist .rc-checklist-item {
    display: inline-block;
    margin-right: 15px;
}
/* --- Native date inputs (type=date) --- */
input[type="date"].custom-input {
    background-color: #2c2c2c !important;
    color: #f0f0f0 !important;
    -webkit-text-fill-color: #f0f0f0 !important;
    border: 1px solid #555 !important;
    border-radius: 4px;
    padding: 6px 8px;
    color-scheme: dark;
    min-width: 140px;
}
input[type="date"].custom-input::-webkit-calendar-picker-indicator {
    filter: invert(1);
    cursor: pointer;
}
/* --- Legacy react-dates pickers (if any remain) --- */
/* High-specificity, attribute-based selectors so react-dates' own injected
   styles cannot win. #react-entry-point is Dash's app root. */
#react-entry-point [class*="DateRangePicker"],
#react-entry-point [class*="SingleDatePicker"],
#react-entry-point [class*="DateInput"],
.DateRangePicker, .SingleDatePicker,
.DateRangePickerInput, .SingleDatePickerInput,
.DateInput {
    background-color: #2c2c2c !important;
    background: #2c2c2c !important;
}
#react-entry-point [class*="DateInput"] input,
#react-entry-point [class*="DatePicker"] input,
.DateRangePicker input,
.SingleDatePicker input,
.DateInput_input,
input[id*="date"], input[id*="Date"] {
    background-color: #2c2c2c !important;
    background: #2c2c2c !important;
    color: #f0f0f0 !important;
    -webkit-text-fill-color: #f0f0f0 !important;
    opacity: 1 !important;
    border: 1px solid #555 !important;
    font-weight: bold;
    text-align: center;
}
.DateInput_input__focused {
    border-color: #00BFFF !important;
}
.DateRangePickerInput_arrow svg,
.DateRangePickerInput_arrow,
.SingleDatePickerInput_calendarIcon svg {
    fill: #f0f0f0 !important;
    color: #f0f0f0 !important;
}
.DateRangePicker input::placeholder,
.SingleDatePicker input::placeholder,
.DateInput_input::placeholder {
    color: #aaa !important;
}
/* Calendar popup */
.CalendarMonthGrid, .CalendarMonth, .DayPicker,
.DayPicker_weekHeader, .CalendarDay__default {
    background-color: #2c2c2c !important;
    color: #f0f0f0 !important;
    border-color: #444 !important;
}
.CalendarDay__default:hover {
    background-color: #00BFFF !important;
    color: #000 !important;
}
.CalendarDay__selected, .CalendarDay__selected_span {
    background-color: #00BFFF !important;
    color: #000 !important;
}
    """
    with open("assets/style.css", "w") as f:
        f.write(style_css_content)

    scripts_js_content = """
    window.dash_clientside = Object.assign({}, window.dash_clientside, {
        clientside: {
            button_feedback: function(...args) {
                const ctx = dash_clientside.callback_context;
                if (ctx.triggered && ctx.triggered.length > 0) {
                    const triggeredButton = document.querySelector('[data-dash-is-loading]');
                    if (triggeredButton) {
                        triggeredButton.classList.add('clicked');
                        setTimeout(() => {
                            triggeredButton.classList.remove('clicked');
                        }, 250);
                    }
                }
                return '';
            }
        }
    });
    """
    with open("assets/scripts.js", "w") as f:
        f.write(scripts_js_content)

    config = {}
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        TELEGRAM_BOT_TOKEN = config.get("telegram_bot_token")
        TELEGRAM_CHAT_ID = config.get("telegram_chat_id")
    except Exception as e:
        logging.error(f"Could not load config.json: {e}")
        exit()
    client = None
    try:
        client = Client(api_key=config.get("api_key"),
                        api_secret=config.get("secret_key"), tld="com")
        client.futures_ping()
        logger.info("Successfully connected to Binance API.")
    except Exception as e:
        logger.error(f"Failed to connect to Binance API: {e}")

    trader = FuturesTrader(client, config)
    # Host/port configurable via env so the app is reachable remotely.
    # Defaults to 0.0.0.0:8080 (access via the server IP or a tunnel to 8080).
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", "8080"))
    app.run(debug=False, host=host, port=port)
