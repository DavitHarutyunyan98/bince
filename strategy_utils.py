"""Strategy utilities for trading bot."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from numba import jit


# ==============================================================================
#  1. BASE STRATEGY CLASS (THE BLUEPRINT)
# ==============================================================================
class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    @staticmethod
    @abstractmethod
    def name():
        """Returns the display name of the strategy."""

    @staticmethod
    @abstractmethod
    def get_parameters():
        """
        Returns a dictionary defining the parameters for this strategy.
        Format: {'param_name': {'type': 'text'/'number', 'default': value}}
        """

    @abstractmethod
    def generate_signals(self, data, params):
        """The core method to generate buy/sell signals."""


# ==============================================================================
#  NUMBA OPTIMIZED SUPERTREND CORE FUNCTION
# ==============================================================================
@jit(nopython=True)
def calculate_supertrend_core(close, high, low, atr, multiplier, n):
    """
    Fast Numba implementation of the iterative SuperTrend logic.
    """
    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    supertrend = np.full(n, np.nan)
    trend = np.full(n, 0)  # 1 for Bullish, -1 for Bearish

    # Calculate Basic Bands
    hl2 = (high + low) / 2
    basic_upper = hl2 + (multiplier * atr)
    basic_lower = hl2 - (multiplier * atr)

    # Find first valid index (skip NaNs)
    first_valid_idx = -1
    for i in range(n):
        if not np.isnan(atr[i]):
            first_valid_idx = i
            break

    if first_valid_idx == -1:
        return trend

    # Initialize first valid point
    final_upper[first_valid_idx] = basic_upper[first_valid_idx]
    final_lower[first_valid_idx] = basic_lower[first_valid_idx]

    # Determine initial trend
    if close[first_valid_idx] > final_upper[first_valid_idx]:
        trend[first_valid_idx] = 1
        supertrend[first_valid_idx] = final_lower[first_valid_idx]
    else:
        trend[first_valid_idx] = -1
        supertrend[first_valid_idx] = final_upper[first_valid_idx]

    # Iteration
    for i in range(first_valid_idx + 1, n):
        prev = i - 1

        # Calculate Final Upper Band
        if (basic_upper[i] < final_upper[prev]) or (close[prev] > final_upper[prev]):
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[prev]

        # Calculate Final Lower Band
        if (basic_lower[i] > final_lower[prev]) or (close[prev] < final_lower[prev]):
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[prev]

        # Determine Trend
        prev_trend = trend[prev]
        if prev_trend == 1:  # Was Bullish
            if close[i] < final_lower[i]:  # Strict Break Down
                trend[i] = -1
                supertrend[i] = final_upper[i]
            else:
                trend[i] = 1
                supertrend[i] = final_lower[i]
        elif prev_trend == -1:  # Was Bearish
            if close[i] > final_upper[i]:  # Strict Break Up
                trend[i] = 1
                supertrend[i] = final_lower[i]
            else:
                trend[i] = -1
                supertrend[i] = final_upper[i]
        else:
            trend[i] = 1
            supertrend[i] = final_lower[i]

    return trend


# ==============================================================================
#  2. CONCRETE STRATEGY IMPLEMENTATIONS
# ==============================================================================
class CandlestickStrategy(BaseStrategy):
    """Strategy based on Three White Soldiers / Three Black Crows.

    Entries: pattern-frequency over a rolling window (fully configurable, no
    hardcoded defaults). Exit: a configurable price-range expressed in percent
    around the entry price — the position is closed (to flat) as soon as price
    crosses outside that range.
    """

    @staticmethod
    def name():
        return "Candlestick Patterns"

    @staticmethod
    def get_parameters():
        # No hardcoded/default values: every value must be supplied by the user.
        return {
            'buy_signal_window': {'type': 'number', 'step': 1},
            'buy_pattern_lookback': {'type': 'number', 'step': 1},
            'sell_signal_window': {'type': 'number', 'step': 1},
            'sell_pattern_lookback': {'type': 'number', 'step': 1},
            'exit_minus_percent': {
                'type': 'number', 'step': 0.1,
                'help': ('lower exit bound in %: exit when the close price falls this far '
                         'below the entry candle open price. example: 2 → open 100 exits if close ≤ 98'),
            },
            'exit_plus_percent': {
                'type': 'number', 'step': 0.1,
                'help': ('upper exit bound in %: exit when the close price rises this far '
                         'above the entry candle open price. example: 3 → open 100 exits if close ≥ 103'),
            },
        }

    @staticmethod
    def _require(params, key, is_float=False):
        """Return the parameter value strictly — no silent defaults."""
        val = params.get(key)
        if val is None or val == '':
            raise ValueError(f"CRITICAL: Parameter '{key}' is required (no default).")
        try:
            num = float(val)
        except (ValueError, TypeError):
            raise ValueError(f"CRITICAL: Parameter '{key}' ({val}) is not a valid number.")
        if np.isnan(num):
            raise ValueError(f"CRITICAL: Parameter '{key}' is NaN.")
        return num if is_float else int(num)

    def _detect_pattern(self, data, is_bullish):
        """Detect candlestick patterns."""
        if is_bullish:
            c1 = data['Close'] > data['Open']
            c2 = data['Close'].shift(1) > data['Open'].shift(1)
            c3 = data['Close'].shift(2) > data['Open'].shift(2)
            increase = (
                (data['Close'] > data['Close'].shift(1)) &
                (data['Close'].shift(1) > data['Close'].shift(2))
            )
            basic_pattern = c1 & c2 & c3 & increase
        else:  # Bearish
            c1 = data['Close'] < data['Open']
            c2 = data['Close'].shift(1) < data['Open'].shift(1)
            c3 = data['Close'].shift(2) < data['Open'].shift(2)
            decrease = (
                (data['Close'] < data['Close'].shift(1)) &
                (data['Close'].shift(1) < data['Close'].shift(2))
            )
            basic_pattern = c1 & c2 & c3 & decrease

        return basic_pattern.astype(int)

    def generate_signals(self, data, params):
        """Generate trading signals based on candlestick patterns.

        Position is built statefully so the percent-range exit can reset the
        position to flat (0) when price crosses outside the configured range.
        """
        if data is None or data.empty:
            return pd.DataFrame()

        df = data.copy()

        df['ThreeWhiteSoldiers'] = self._detect_pattern(df, is_bullish=True)
        df['ThreeBlackCrows'] = self._detect_pattern(df, is_bullish=False)

        # Strict parameter extraction (no hardcoded fallbacks).
        buy_window = self._require(params, 'buy_signal_window')
        buy_lookback = self._require(params, 'buy_pattern_lookback')
        sell_window = self._require(params, 'sell_signal_window')
        sell_lookback = self._require(params, 'sell_pattern_lookback')

        # Exit bounds are optional at runtime; when both are absent, only
        # opposite-pattern flips close a position. Each bound is independent.
        minus_raw = params.get('exit_minus_percent')
        plus_raw = params.get('exit_plus_percent')
        exit_minus = None
        exit_plus = None
        if minus_raw is not None and minus_raw != '':
            exit_minus = self._require(params, 'exit_minus_percent', is_float=True)
        if plus_raw is not None and plus_raw != '':
            exit_plus = self._require(params, 'exit_plus_percent', is_float=True)

        buy_raw = (
            df['ThreeWhiteSoldiers'].rolling(window=buy_window).sum() >= buy_lookback
        ).to_numpy()
        sell_raw = (
            df['ThreeBlackCrows'].rolling(window=sell_window).sum() >= sell_lookback
        ).to_numpy()
        open_ = df['Open'].to_numpy()
        close = df['Close'].to_numpy()

        n = len(df)
        positions = np.zeros(n, dtype=int)
        pos = 0
        entry_price = 0.0  # reference = OPEN price of the candle the position opened

        for i in range(n):
            if pos == 0:
                # Look for a fresh entry (reference = this candle's OPEN).
                if buy_raw[i]:
                    pos, entry_price = 1, open_[i]
                elif sell_raw[i]:
                    pos, entry_price = -1, open_[i]
            else:
                # 1) Opposite pattern flips the position.
                if pos == 1 and sell_raw[i]:
                    pos, entry_price = -1, open_[i]
                elif pos == -1 and buy_raw[i]:
                    pos, entry_price = 1, open_[i]
                # 2) Percent-range exit: compare the entry OPEN price with the
                #    current CLOSE; exit to flat if CLOSE crosses the band
                #    [open - minus%, open + plus%].
                elif entry_price > 0:
                    hit_lower = exit_minus is not None and close[i] <= entry_price * (1 - exit_minus / 100.0)
                    hit_upper = exit_plus is not None and close[i] >= entry_price * (1 + exit_plus / 100.0)
                    if hit_lower or hit_upper:
                        pos, entry_price = 0, 0.0
            positions[i] = pos

        df["position"] = positions
        return df


class RsiStrategy(BaseStrategy):
    """Strategy based on RSI overbought/oversold signals."""

    @staticmethod
    def name():
        return "RSI Crossover"

    @staticmethod
    def get_parameters():
        return {
            'rsi_period': {'type': 'number', 'default': 14, 'step': 1},
            'oversold_threshold': {'type': 'number', 'default': 30, 'step': 1},
            'overbought_threshold': {'type': 'number', 'default': 70, 'step': 1}
        }

    def generate_signals(self, data, params):
        df = data.copy()
        rsi_period = int(params.get('rsi_period') or 14)
        oversold = float(params.get('oversold_threshold') or 30)
        overbought = float(params.get('overbought_threshold') or 70)

        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        df['position'] = 0
        df.loc[df['RSI'] < oversold, 'position'] = 1
        df.loc[df['RSI'] > overbought, 'position'] = -1
        df['position'] = df['position'].replace(0, np.nan).ffill().fillna(0)
        return df


class MACrossoverStrategy(BaseStrategy):
    """Strategy based on a fast, middle, and slow moving average crossover."""

    @staticmethod
    def name():
        return "Moving Average Crossover"

    @staticmethod
    def get_parameters():
        return {
            'fast_ma_period': {'type': 'number', 'default': 10, 'step': 1},
            'middle_ma_period': {'type': 'number', 'default': 20, 'step': 1},
            'slow_ma_period': {'type': 'number', 'default': 50, 'step': 1},
            'ma_type': {'type': 'text', 'default': 'EMA'},
        }

    def generate_signals(self, data, params):
        if data is None or data.empty:
            return pd.DataFrame()
        df = data.copy()

        fast_period = int(params.get('fast_ma_period') or 10)
        middle_period = int(params.get('middle_ma_period') or 20)
        slow_period = int(params.get('slow_ma_period') or 50)
        ma_type = params.get('ma_type') or 'EMA'
        ma_type = ma_type.upper()

        def ma_func(period):
            if ma_type == 'EMA':
                return df['Close'].ewm(span=period, adjust=False).mean()
            else:
                return df['Close'].rolling(window=period).mean()

        df['fast_ma'] = ma_func(fast_period)
        df['middle_ma'] = ma_func(middle_period)
        df['slow_ma'] = ma_func(slow_period)

        df['signal'] = 0
        buy_condition = (
            (df['fast_ma'] > df['middle_ma']) &
            (df['middle_ma'] > df['slow_ma'])
        )
        sell_condition = (
            (df['fast_ma'] < df['middle_ma']) &
            (df['middle_ma'] < df['slow_ma'])
        )

        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1

        df['position'] = df['signal'].replace(0, np.nan).ffill().fillna(0)

        return df


class SuperTrendStrategy(BaseStrategy):
    """Strategy based on ATR SuperTrend indicator (Numba Optimized)."""

    @staticmethod
    def name():
        return "ATR SuperTrend"

    @staticmethod
    def get_parameters():
        return {
            'atr_period': {'type': 'number', 'default': 10, 'step': 1},
            'atr_multiplier': {'type': 'number', 'default': 3.0, 'step': 0.1},
        }

    def _validate_param(self, params, key, is_float=False):
        """Strictly validates parameters. Raises Error if missing or NaN."""
        val = params.get(key)
        if val is None:
            raise ValueError(f"CRITICAL: Parameter '{key}' is missing or None.")
        try:
            num_val = float(val)
            if np.isnan(num_val):
                raise ValueError(f"CRITICAL: Parameter '{key}' is NaN.")
        except (ValueError, TypeError):
            raise ValueError(f"CRITICAL: Parameter '{key}' ({val}) is not a valid number.")
        return float(num_val) if is_float else int(num_val)

    def generate_signals(self, data, params):
        if data is None or data.empty:
            return pd.DataFrame()

        df = data.copy()

        # Strict Parameter Extraction
        atr_period = self._validate_param(params, 'atr_period', is_float=False)
        atr_multiplier = self._validate_param(params, 'atr_multiplier', is_float=True)

        if len(df) < atr_period + 1:
            df['position'] = 0
            return df

        # Vectorized ATR Calculation (Wilder's Smoothing)
        high = df['High']
        low = df['Low']
        close = df['Close']
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.ewm(alpha=1/atr_period, adjust=False).mean()

        # Numba Execution
        trend_array = calculate_supertrend_core(
            close.values,
            high.values,
            low.values,
            atr.values,
            atr_multiplier,
            len(df)
        )

        df['trend'] = trend_array
        df['position'] = df['trend'].fillna(0).astype(int)

        return df


# ==============================================================================
#  3. STRATEGY REGISTRY (THE ENGINE'S GEARBOX)
# ==============================================================================
STRATEGY_REGISTRY = {
    CandlestickStrategy.name(): CandlestickStrategy,
    RsiStrategy.name(): RsiStrategy,
    MACrossoverStrategy.name(): MACrossoverStrategy,
    SuperTrendStrategy.name(): SuperTrendStrategy,
}


# ==============================================================================
#  4. BACKTESTER CLASS (fixed initial-capital sizing, no compounding)
# ==============================================================================
class Backtester:
    """Handles the logic for running a backtest on historical data with trading signals."""

    def __init__(self, initial_capital=10000, fee_percent=0.05, sizing_mode='fixed'):
        self.initial_capital = initial_capital
        self.fee_percent = fee_percent / 100
        # 'fixed'      -> every trade sized off the initial capital (no compounding)
        # 'compound'   -> every trade sized off the running equity (compounding)
        self.sizing_mode = 'compound' if str(sizing_mode).lower() == 'compound' else 'fixed'
        self.trades = []
        self.portfolio_value = []

    def calculate_trading_fee(self, trade_value):
        """Calculates the trading fee for a given trade value."""
        return trade_value * self.fee_percent

    def run_backtest(self, df_with_signals):
        """Runs the backtest simulation with next-candle execution to eliminate lookahead bias."""
        if df_with_signals is None or df_with_signals.empty:
            return None, None

        self.trades = []
        self.portfolio_value = []
        capital = self.initial_capital   # running realized equity
        compounding = self.sizing_mode == 'compound'
        position = 0
        entry_price = 0
        entry_date = None
        entry_base = self.initial_capital  # size of the open trade

        # Convert to list for index-based access
        df_list = list(df_with_signals.itertuples())

        for i, row in enumerate(df_list):
            date = row.Index  # Get the date from the index
            current_price = row.Close  # Access using dot notation
            current_signal = getattr(row, 'position', 0)  # Safe access to position column

            # Portfolio value = realized equity + unrealized PnL on the trade's base
            if position == 1:
                portfolio_val = capital + (entry_base * (current_price / entry_price) - entry_base)
            elif position == -1:
                portfolio_val = capital + (entry_base * (1 + (entry_price - current_price) / entry_price) - entry_base)
            else:
                portfolio_val = capital
            self.portfolio_value.append({
                'Date': date,
                'Portfolio_Value': portfolio_val
            })

            exit_triggered = False
            exit_reason = None

            # Check for position exits - FIXED: Use next candle open for exit price
            if position != 0:
                if position == 1 and current_signal == -1:
                    exit_triggered, exit_reason = True, 'Signal Flip'
                elif position == -1 and current_signal == 1:
                    exit_triggered, exit_reason = True, 'Signal Flip'
                elif current_signal == 0:
                    # Strategy moved to flat (e.g. percent-range exit) -> close.
                    exit_triggered, exit_reason = True, 'Exit Signal'

                if exit_triggered:
                    # CRITICAL FIX: Use next candle open for exit price (same as entry logic)
                    if i + 1 < len(df_list):
                        next_row = df_list[i + 1]
                        exit_price = getattr(next_row, 'Open', next_row.Close)  # Next candle open
                        exit_date = next_row.Index  # Next candle timestamp
                    else:
                        # Fallback for last candle
                        exit_price = current_price
                        exit_date = date
                    
                    if position == 1:
                        position_value = entry_base * (exit_price / entry_price)
                    else:
                        position_value = entry_base * (
                            1 + (entry_price - exit_price) / entry_price
                        )

                    fee = self.calculate_trading_fee(position_value)
                    net_value = position_value - fee
                    pnl = net_value - entry_base
                    pnl_percent = (pnl / entry_base) * 100 if entry_base > 0 else 0

                    self.trades.append({
                        'Entry_Date': entry_date,
                        'Exit_Date': exit_date,  # Use next candle timestamp
                        'Entry_Price': entry_price,
                        'Exit_Price': exit_price,  # Use next candle open
                        'Position': 'Long' if position == 1 else 'Short',
                        'PnL': pnl,
                        'PnL %': pnl_percent,
                        'Exit_Reason': exit_reason
                    })
                    capital = capital + pnl  # accumulate realized PnL (linear)
                    position = 0

            # Check for new position entries - FIXED: Next-candle execution
            if position == 0:
                if current_signal == 1 or current_signal == -1:
                    # Check if we have a next candle for execution
                    if i + 1 < len(df_list):
                        next_row = df_list[i + 1]
                        # Execute trade on NEXT candle to eliminate lookahead bias
                        position = current_signal
                        entry_price = getattr(next_row, 'Open', next_row.Close)  # Use next candle's OPEN price
                        entry_date = next_row.Index  # Use next candle's timestamp
                        # Compounding sizes off current equity; fixed off initial.
                        entry_base = capital if compounding else self.initial_capital
                    # If no next candle available, skip the trade (edge case handling)

        # Close any open position at the end of the data
        if position != 0:
            last_row = df_with_signals.iloc[-1]
            exit_price = last_row['Close']
            exit_date = last_row.name
            exit_reason = 'End of Data'
            if position == 1:
                position_value = entry_base * (exit_price / entry_price)
            else:
                position_value = entry_base * (
                    1 + (entry_price - exit_price) / entry_price
                )
            fee = self.calculate_trading_fee(position_value)
            net_value = position_value - fee
            pnl = net_value - entry_base
            pnl_percent = (pnl / entry_base) * 100 if entry_base > 0 else 0
            capital = capital + pnl
            pos_type = 'Long' if position == 1 else 'Short'
            self.trades.append({
                'Entry_Date': entry_date,
                'Exit_Date': exit_date,
                'Entry_Price': entry_price,
                'Exit_Price': exit_price,
                'Position': pos_type,
                'PnL': pnl,
                'PnL %': pnl_percent,
                'Exit_Reason': exit_reason
            })

        return pd.DataFrame(self.trades), pd.DataFrame(self.portfolio_value)
