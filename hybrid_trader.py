#!/usr/bin/env python3
"""
Hybrid Multi-Symbol Trading Bot for pairs that don't work well with WebSocket.
Uses REST API polling for data collection and signal generation.
"""

import os
import time
import json
import logging
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

from binance.client import Client
from strategy_utils import STRATEGY_REGISTRY

# Setup logging


def setup_logging():
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_filename = f"hybrid_trader_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = os.path.join(log_dir, log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filepath, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return log_filepath


logger = logging.getLogger(__name__)


class HybridSymbolTrader:
    """Hybrid trader that uses REST API polling instead of WebSocket."""

    def __init__(self, trade_config, client, telegram_sender):
        self.symbol = trade_config["symbol"]
        self.client = client
        self.send_telegram_notification = telegram_sender
        self.trade_config = trade_config

        # Bot State
        self.position = 0
        self.data = pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Volume", "Complete"])
        self.prepared_data = pd.DataFrame()
        self.current_price = 0.0
        self.last_candle_time = None

        # Trading Parameters
        self.bar_length = self.trade_config["bar_length"]
        self.units_usdt = float(self.trade_config.get("units_usdt", 10.0))
        self.leverage = int(self.trade_config.get("leverage", 1))
        self.poll_interval = self._get_poll_interval()

        # Position tracking
        self.entry_price = 0.0
        self.position_start_time = None
        self.session_trades = 0
        self.session_pnl = 0.0
        
        # Flag to track if initial position has been opened
        self.initial_position_opened = False
        
        # ENHANCED: Position health monitoring
        self.last_position_check = datetime.now(timezone.utc)
        self.position_fix_count = 0
        self.last_signal_time = None
        self.consecutive_missing_positions = 0

        # Strategy setup
        strategy_name = self.trade_config.get("strategy_name")
        strategy_class = STRATEGY_REGISTRY.get(strategy_name)
        if not strategy_class:
            raise ValueError(
                f"[{self.symbol}] Strategy '{strategy_name}' not found.")

        self.strategy_instance = strategy_class()
        self.strategy_params = {key: self.trade_config.get(key)
                                for key in self.strategy_instance.get_parameters()}

        # Symbol precision
        self._setup_symbol_precision()

    def _get_poll_interval(self):
        """Get polling interval based on timeframe."""
        intervals = {
            '1m': 30,   # Poll every 30 seconds for 1m
            '5m': 60,   # Poll every 1 minute for 5m
            '15m': 180,  # Poll every 3 minutes for 15m
            '1h': 600,  # Poll every 10 minutes for 1h
        }
        return intervals.get(self.bar_length, 60)

    def _setup_symbol_precision(self):
        """Setup symbol precision info."""
        try:
            info = self.client.futures_exchange_info()
            symbol_info = next(
                item for item in info["symbols"] if item["symbol"] == self.symbol)
            self.price_precision = int(symbol_info['pricePrecision'])
            self.quantity_precision = int(symbol_info['quantityPrecision'])
            price_filter = next(
                f for f in symbol_info["filters"] if f["filterType"] == "PRICE_FILTER")
            self.tick_size = float(price_filter["tickSize"])
        except Exception as e:
            logger.error(f"[{self.symbol}] Could not fetch symbol info: {e}")
            self.price_precision = 6
            self.quantity_precision = 3
            self.tick_size = 0.000001

    def initialize(self):
        """Initialize the trader."""
        logger.info(f"[{self.symbol}] Initializing hybrid trader...")

        try:
            self.client.futures_change_leverage(
                symbol=self.symbol, leverage=self.leverage)
            logger.info(f"[{self.symbol}] Leverage set to {self.leverage}x.")
        except Exception as e:
            logger.warning(f"[{self.symbol}] Could not set leverage: {e}")

        self._check_open_position()
        self._load_initial_data()

        logger.info(
            f"[{self.symbol}] Hybrid trader initialized. Poll interval: {self.poll_interval}s")

    def _check_open_position(self):
        """Check initial position state."""
        try:
            position_info = self.client.futures_position_information(
                symbol=self.symbol)
            if position_info and len(position_info) > 0:
                pos_amt = float(position_info[0]["positionAmt"])
                self.position = 1 if pos_amt > 0 else -1 if pos_amt < 0 else 0
                logger.info(
                    f"[{self.symbol}] Initial position: {self.position}")
                
                # If we have an existing position, mark as opened
                if self.position != 0:
                    self.initial_position_opened = True
            else:
                self.position = 0
                logger.info(f"[{self.symbol}] No initial position")
        except Exception as e:
            logger.error(f"[{self.symbol}] Error checking position: {e}")
            self.position = 0

    def _sync_position_with_exchange(self):
        """Sync internal position tracking with actual exchange position."""
        try:
            position_info = self.client.futures_position_information(symbol=self.symbol)
            if position_info and len(position_info) > 0:
                pos_amt = float(position_info[0]["positionAmt"])
                exchange_position = 1 if pos_amt > 0 else -1 if pos_amt < 0 else 0
                
                if exchange_position != self.position:
                    logger.warning(f"[{self.symbol}] Position sync: Internal={self.position}, Exchange={exchange_position}")
                    self.position = exchange_position
                    return True
            return False
        except Exception as e:
            logger.error(f"[{self.symbol}] Error syncing position: {e}")
            return False

    def _load_initial_data(self):
        """Load initial historical data."""
        try:
            logger.info(f"[{self.symbol}] Loading initial data...")

            # Get last 500 candles for strategy initialization (increased from 100)
            # This ensures sufficient historical context for pattern detection
            klines = self.client.futures_klines(
                symbol=self.symbol,
                interval=self.bar_length,
                limit=500
            )

            for kline in klines:
                start_time = pd.to_datetime(kline[0], unit="ms")
                self.data.loc[start_time] = [
                    float(kline[1]),  # Open
                    float(kline[2]),  # High
                    float(kline[3]),  # Low
                    float(kline[4]),  # Close
                    float(kline[5]),  # Volume
                    True  # Complete
                ]

            self.current_price = float(klines[-1][4])
            self.last_candle_time = pd.to_datetime(klines[-1][0], unit="ms")

            logger.info(
                f"[{self.symbol}] Loaded {len(klines)} historical candles")

            # Generate initial signals and execute immediately if needed (using completed historical candles)
            self._generate_signals()
            self._execute_trades()  # Execute immediately on startup with historical data

        except Exception as e:
            logger.error(f"[{self.symbol}] Error loading initial data: {e}")

    def start_polling(self):
        """Start the REST API polling loop."""
        logger.info(f"[{self.symbol}] Starting REST API polling...")

        while True:
            try:
                self._poll_data()
                time.sleep(self.poll_interval)

            except Exception as e:
                logger.error(f"[{self.symbol}] Polling error: {e}")
                time.sleep(30)  # Wait 30s on error

    def _poll_data(self):
        """Poll for new data via REST API."""
        try:
            # Get latest 10 candles to ensure data consistency
            klines = self.client.futures_klines(
                symbol=self.symbol,
                interval=self.bar_length,
                limit=10
            )

            if not klines:
                return

            # Update all recent candles to maintain data consistency
            new_candle_detected = False
            latest_candle_time = None
            
            for i, kline in enumerate(klines):
                candle_time = pd.to_datetime(kline[0], unit="ms")
                is_latest_candle = (i == len(klines) - 1)
                
                # FIXED: Only mark candles as complete if they're not the latest (current) candle
                # The latest candle from the API is still forming and should not be used for signals
                is_complete = not is_latest_candle
                
                # Check if this is a new COMPLETE candle (not the forming one)
                if not is_latest_candle and (self.last_candle_time is None or candle_time > self.last_candle_time):
                    new_candle_detected = True
                    self.last_candle_time = candle_time
                
                # Update/add candle data
                self.data.loc[candle_time] = [
                    float(kline[1]),  # Open
                    float(kline[2]),  # High
                    float(kline[3]),  # Low
                    float(kline[4]),  # Close
                    float(kline[5]),  # Volume
                    is_complete  # Mark as complete only if it's not the current forming candle
                ]
                
                if is_latest_candle:
                    latest_candle_time = candle_time

            # Update current price from latest candle
            latest_kline = klines[-1]
            self.current_price = float(latest_kline[4])

            # Maintain historical buffer - keep last 1000 candles for sufficient context
            if len(self.data) > 1000:
                self.data = self.data.tail(1000)

            if new_candle_detected and self.last_candle_time is not None:
                candle_time_str = self.last_candle_time.strftime('%H:%M:%S')

                logger.info(f"[L] [{candle_time_str}] [{self.symbol}] NEW CANDLE CLOSED | "
                            f"Price: {self.current_price:.{self.price_precision}f}")

                # FIXED: Generate signals and execute trades ONLY on closed candles
                # Use only complete candles for signal generation
                self._generate_signals()
                self._execute_trades()
                
                # ADDED: Check if positions are opened correctly after each candle close
                self._verify_and_fix_positions()

            else:
                # Just a price update - NO signal generation, NO trade execution
                current_time = datetime.now(timezone.utc).strftime('%H:%M:%S')
                if latest_candle_time:
                    latest_candle_str = latest_candle_time.strftime('%H:%M:%S')
                    logger.debug(
                        f"[{current_time}] [{self.symbol}] Price update: {self.current_price:.{self.price_precision}f} | "
                        f"Live candle: {latest_candle_str} (incomplete)")
                else:
                    logger.debug(
                        f"[{current_time}] [{self.symbol}] Price update: {self.current_price:.{self.price_precision}f}")
                
                # FIXED: Do NOT generate signals during price updates - signals should only be based on closed candles
                # Generating signals with incomplete candle data causes premature signal flips
                
                # ENHANCED: Periodic position health check (every 5 minutes during price updates)
                self._periodic_position_health_check()

        except Exception as e:
            logger.error(f"[{self.symbol}] Error polling data: {e}")

    def _generate_signals(self):
        """Generate trading signals using only complete candles."""
        try:
            # Require more data for reliable signal generation
            min_data_required = max(50, self.strategy_params.get('buy_signal_window', 20) * 2)
            
            if len(self.data) < min_data_required:
                logger.debug(f"[{self.symbol}] Insufficient data for signals: {len(self.data)}/{min_data_required}")
                return

            # CRITICAL FIX: Use only complete candles for signal generation
            # Filter out incomplete candles to prevent signals based on live/forming candle data
            complete_data = self.data[self.data['Complete'] == True].copy()
            
            if len(complete_data) < min_data_required:
                logger.debug(f"[{self.symbol}] Insufficient complete candles for signals: {len(complete_data)}/{min_data_required}")
                return

            # Generate signals with complete candles only
            self.prepared_data = self.strategy_instance.generate_signals(
                complete_data, self.strategy_params)

            if not self.prepared_data.empty:
                latest_signal = self.prepared_data['position'].iloc[-1]
                complete_candles_used = len(complete_data)
                logger.info(f"[{self.symbol}] Signal generated: {latest_signal} (current position: {self.position}) using {complete_candles_used} complete candles")
            else:
                logger.warning(f"[{self.symbol}] No signals generated from strategy")

        except Exception as e:
            logger.error(f"[{self.symbol}] Signal generation error: {e}")
            import traceback
            logger.error(f"[{self.symbol}] Traceback: {traceback.format_exc()}")

    def _execute_trades(self):
        """Execute trades based on signals."""
        if self.prepared_data.empty:
            return

        # Sync position with exchange before comparison
        self._sync_position_with_exchange()

        desired_position = int(self.prepared_data["position"].iloc[-1])
        current_position = self.position

        logger.debug(f"[{self.symbol}] Signal check: Current={current_position}, Desired={desired_position}")

        # For initial position opening, don't wait for candle close
        if not self.initial_position_opened and desired_position != 0:
            logger.info(f"[{self.symbol}] INITIAL POSITION: Opening {desired_position} immediately")
            success = self._open_position_with_retry(desired_position)
            if success:
                self.initial_position_opened = True
            return

        if desired_position == current_position:
            return

        logger.info(
            f"[{self.symbol}] SIGNAL FLIP: {current_position} -> {desired_position}")

        # Close existing position
        if current_position != 0:
            self._close_position(
                f"Signal flip: {current_position} -> {desired_position}")

        # Open new position immediately with retry mechanism
        if desired_position != 0:
            self._open_position_with_retry(desired_position)
        
        # Update position tracking immediately
        self.position = desired_position
        logger.info(f"[{self.symbol}] Position updated to: {self.position}")

    def _open_position(self, signal):
        """Open a new position."""
        try:
            quantity = self._get_trade_quantity()
            if not quantity or quantity <= 0:
                return

            side = "BUY" if signal == 1 else "SELL"
            pos_type = "LONG" if signal == 1 else "SHORT"

            logger.info(f"[{self.symbol}] Opening {pos_type} position...")

            order = self.client.futures_create_order(
                symbol=self.symbol,
                side=side,
                type="MARKET",
                quantity=quantity
            )

            # Update position tracking immediately after order
            self.position = signal
            self.entry_price = self.current_price
            self.position_start_time = datetime.now(timezone.utc)
            self.session_trades += 1

            message = (
                f"[HYBRID] <b>{pos_type} Position Opened</b>\n\n"
                f"<b>Symbol:</b> <code>{self.symbol}</code>\n"
                f"<b>Entry Price:</b> <code>{self.entry_price:.{self.price_precision}f}</code>\n"
                f"<b>Position Size:</b> <code>{quantity:.{self.quantity_precision}f}</code>\n"
                f"<b>Method:</b> <code>REST API Polling</code>"
            )

            logger.info(
                f"[{self.symbol}] {pos_type} position opened at {self.entry_price:.{self.price_precision}f}")
            self.send_telegram_notification(message)

        except Exception as e:
            logger.error(f"[{self.symbol}] Error opening position: {e}")

    def _close_position(self, reason="Manual Close"):
        """Close current position."""
        if self.position == 0:
            return

        try:
            position_size = self._get_position_size()
            if position_size <= 0:
                return

            side = "SELL" if self.position == 1 else "BUY"
            pos_type = "LONG" if self.position == 1 else "SHORT"

            logger.info(f"[{self.symbol}] Closing {pos_type} position...")

            order = self.client.futures_create_order(
                symbol=self.symbol,
                side=side,
                type="MARKET",
                quantity=position_size
            )

            # Calculate PnL
            if self.position == 1:
                pnl = (self.current_price - self.entry_price) * position_size
            else:
                pnl = (self.entry_price - self.current_price) * position_size

            self.session_pnl += pnl
            # Update position tracking immediately after closing
            self.position = 0

            message = (
                f"[HYBRID] <b>{pos_type} Position Closed</b>\n\n"
                f"<b>Symbol:</b> <code>{self.symbol}</code>\n"
                f"<b>Exit Price:</b> <code>{self.current_price:.{self.price_precision}f}</code>\n"
                f"<b>PnL:</b> <code>{pnl:+.2f} USDT</code>\n"
                f"<b>Reason:</b> <code>{reason}</code>"
            )

            logger.info(
                f"[{self.symbol}] {pos_type} position closed. PnL: {pnl:+.2f} USDT")
            self.send_telegram_notification(message)

        except Exception as e:
            logger.error(f"[{self.symbol}] Error closing position: {e}")

    def _get_trade_quantity(self):
        """Calculate trade quantity."""
        try:
            if self.current_price <= 0:
                return None
            quantity = self.units_usdt / self.current_price
            return float(f"{quantity:.{self.quantity_precision}f}")
        except Exception as e:
            logger.error(f"[{self.symbol}] Error calculating quantity: {e}")
            return None

    def _get_position_size(self):
        """Get current position size."""
        try:
            position_info = self.client.futures_position_information(
                symbol=self.symbol)
            if position_info and len(position_info) > 0:
                return abs(float(position_info[0]["positionAmt"]))
            return 0
        except Exception:
            return 0

    def _verify_and_fix_positions(self):
        """ENHANCED: Verify if positions are opened correctly after candle close, open missing positions if needed."""
        try:
            if self.prepared_data.empty:
                return
                
            # Get desired position from strategy signal
            desired_position = int(self.prepared_data["position"].iloc[-1])
            
            # Sync with exchange to get actual position
            position_synced = self._sync_position_with_exchange()
            actual_position = self.position
            
            # Update last position check time
            self.last_position_check = datetime.now(timezone.utc)
            
            # Check if positions match
            if desired_position != actual_position:
                self.consecutive_missing_positions += 1
                logger.warning(f"[{self.symbol}] 🚨 POSITION MISMATCH #{self.consecutive_missing_positions}! Desired: {desired_position}, Actual: {actual_position}")
                
                # If we should have a position but don't, open it
                if desired_position != 0 and actual_position == 0:
                    logger.info(f"[{self.symbol}] 🔧 OPENING MISSING POSITION: {desired_position}")
                    success = self._open_position_with_retry(desired_position)
                    
                    if success:
                        self.position_fix_count += 1
                        self.consecutive_missing_positions = 0  # Reset counter on successful fix
                        
                        # Send notification about position fix
                        pos_type = "LONG" if desired_position == 1 else "SHORT"
                        message = (
                            f"[HYBRID] <b>⚠️ Position Fixed #{self.position_fix_count}</b>\n\n"
                            f"<b>Symbol:</b> <code>{self.symbol}</code>\n"
                            f"<b>Issue:</b> <code>Missing {pos_type} position</code>\n"
                            f"<b>Action:</b> <code>Opened {pos_type} at {self.current_price:.{self.price_precision}f}</code>\n"
                            f"<b>Reason:</b> <code>Position verification after candle close</code>\n"
                            f"<b>Consecutive Misses:</b> <code>{self.consecutive_missing_positions}</code>"
                        )
                        self.send_telegram_notification(message)
                    else:
                        logger.error(f"[{self.symbol}] ❌ FAILED to open missing position after retry!")
                    
                # If we have wrong position, close and open correct one
                elif desired_position != 0 and actual_position != 0 and desired_position != actual_position:
                    logger.info(f"[{self.symbol}] 🔄 FIXING WRONG POSITION: {actual_position} -> {desired_position}")
                    self._close_position("Position verification - wrong direction")
                    success = self._open_position_with_retry(desired_position)
                    if success:
                        self.position_fix_count += 1
                        self.consecutive_missing_positions = 0
                    
                # If we shouldn't have a position but do, close it
                elif desired_position == 0 and actual_position != 0:
                    logger.info(f"[{self.symbol}] 🗑️ CLOSING UNWANTED POSITION: {actual_position}")
                    self._close_position("Position verification - should be neutral")
                    self.consecutive_missing_positions = 0
                    
            else:
                # Positions match - reset consecutive counter
                if self.consecutive_missing_positions > 0:
                    logger.info(f"[{self.symbol}] ✅ Position verification OK after {self.consecutive_missing_positions} mismatches")
                    self.consecutive_missing_positions = 0
                else:
                    logger.debug(f"[{self.symbol}] ✅ Position verification OK: {actual_position}")
                
        except Exception as e:
            logger.error(f"[{self.symbol}] Error in position verification: {e}")

    def _periodic_position_health_check(self):
        """ENHANCED: Periodic health check to catch stuck pairs (every 5 minutes)."""
        try:
            current_time = datetime.now(timezone.utc)
            time_since_last_check = (current_time - self.last_position_check).total_seconds()
            
            # Run health check every 5 minutes (300 seconds)
            if time_since_last_check >= 300:
                logger.info(f"[{self.symbol}] 🔍 Running periodic position health check...")
                
                if not self.prepared_data.empty:
                    desired_position = int(self.prepared_data["position"].iloc[-1])
                    self._sync_position_with_exchange()
                    actual_position = self.position
                    
                    # Check for stuck pair scenario
                    if desired_position != 0 and actual_position == 0:
                        logger.warning(f"[{self.symbol}] 🚨 STUCK PAIR DETECTED! Should have {desired_position} but has {actual_position}")
                        
                        # Try to fix the stuck position
                        success = self._open_position_with_retry(desired_position)
                        if success:
                            self.position_fix_count += 1
                            pos_type = "LONG" if desired_position == 1 else "SHORT"
                            message = (
                                f"[HYBRID] <b>🔧 Stuck Pair Fixed</b>\n\n"
                                f"<b>Symbol:</b> <code>{self.symbol}</code>\n"
                                f"<b>Issue:</b> <code>Pair was stuck without {pos_type} position</code>\n"
                                f"<b>Action:</b> <code>Opened {pos_type} at {self.current_price:.{self.price_precision}f}</code>\n"
                                f"<b>Detection:</b> <code>Periodic health check</code>\n"
                                f"<b>Total Fixes:</b> <code>{self.position_fix_count}</code>"
                            )
                            self.send_telegram_notification(message)
                        else:
                            # Send alert about persistent stuck pair
                            message = (
                                f"[HYBRID] <b>🚨 ALERT: Persistent Stuck Pair</b>\n\n"
                                f"<b>Symbol:</b> <code>{self.symbol}</code>\n"
                                f"<b>Issue:</b> <code>Cannot open required position</code>\n"
                                f"<b>Desired:</b> <code>{desired_position}</code>\n"
                                f"<b>Actual:</b> <code>{actual_position}</code>\n"
                                f"<b>Action:</b> <code>Manual intervention may be needed</code>"
                            )
                            self.send_telegram_notification(message)
                    
                    elif desired_position != actual_position and actual_position != 0:
                        logger.warning(f"[{self.symbol}] 🔄 Wrong position detected in health check: {actual_position} should be {desired_position}")
                        self._close_position("Health check - wrong position")
                        if desired_position != 0:
                            self._open_position_with_retry(desired_position)
                
                self.last_position_check = current_time
                
        except Exception as e:
            logger.error(f"[{self.symbol}] Error in periodic health check: {e}")

    def _open_position_with_retry(self, signal, max_retries=3):
        """ENHANCED: Open position with retry mechanism for better reliability."""
        for attempt in range(max_retries):
            try:
                quantity = self._get_trade_quantity()
                if not quantity or quantity <= 0:
                    logger.error(f"[{self.symbol}] Invalid quantity calculated: {quantity}")
                    return False

                side = "BUY" if signal == 1 else "SELL"
                pos_type = "LONG" if signal == 1 else "SHORT"

                logger.info(f"[{self.symbol}] 🔄 Opening {pos_type} position (attempt {attempt + 1}/{max_retries})...")

                order = self.client.futures_create_order(
                    symbol=self.symbol,
                    side=side,
                    type="MARKET",
                    quantity=quantity
                )

                # Verify the order was successful
                if order and order.get('orderId'):
                    # Update position tracking immediately after order
                    self.position = signal
                    self.entry_price = self.current_price
                    self.position_start_time = datetime.now(timezone.utc)
                    self.session_trades += 1

                    # Double-check position was actually opened
                    time.sleep(1)  # Brief wait for order to settle
                    self._sync_position_with_exchange()
                    
                    if self.position == signal:
                        message = (
                            f"[HYBRID] <b>{pos_type} Position Opened</b>\n\n"
                            f"<b>Symbol:</b> <code>{self.symbol}</code>\n"
                            f"<b>Entry Price:</b> <code>{self.entry_price:.{self.price_precision}f}</code>\n"
                            f"<b>Position Size:</b> <code>{quantity:.{self.quantity_precision}f}</code>\n"
                            f"<b>Method:</b> <code>REST API Polling</code>\n"
                            f"<b>Attempt:</b> <code>{attempt + 1}/{max_retries}</code>"
                        )

                        logger.info(f"[{self.symbol}] ✅ {pos_type} position opened successfully at {self.entry_price:.{self.price_precision}f}")
                        self.send_telegram_notification(message)
                        return True
                    else:
                        logger.warning(f"[{self.symbol}] ⚠️ Order placed but position not confirmed. Retrying...")
                        
            except Exception as e:
                logger.error(f"[{self.symbol}] ❌ Error opening position (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retry
                    
        logger.error(f"[{self.symbol}] ❌ Failed to open position after {max_retries} attempts")
        return False

    def get_unrealized_pnl(self):
        """Get current unrealized PnL."""
        if self.position == 0 or self.entry_price <= 0:
            return 0.0

        try:
            position_size = self._get_position_size()
            if position_size <= 0:
                return 0.0

            if self.position == 1:  # LONG
                return (self.current_price - self.entry_price) * position_size
            else:  # SHORT
                return (self.entry_price - self.current_price) * position_size
        except Exception:
            return 0.0


class HybridTraderManager:
    """Manager for hybrid traders."""

    def __init__(self, trade_configs_path, api_config_path):
        self.log_file_path = setup_logging()
        logger.info("HybridTraderManager starting up...")

        self.api_config = self._load_json(api_config_path)
        self.trade_configs = [c for c in self._load_json(
            trade_configs_path) if c.get("enabled", True)]

        if not self.trade_configs or not self.api_config:
            raise ValueError(
                "Configuration files could not be loaded or are empty.")

        self.client = Client(
            api_key=self.api_config.get("api_key"),
            api_secret=self.api_config.get("secret_key"),
            tld="com"
        )

        self.telegram_bot_token = self.api_config.get("telegram_bot_token")
        self.telegram_chat_id = self.api_config.get("telegram_chat_id")
        self.traders = {}

        # Session tracking
        self.session_start_time = datetime.now(timezone.utc)
        self.last_report_time = datetime.now(timezone.utc)  # Changed from hourly to 30min tracking

    def start(self):
        """Start the hybrid trading session."""
        logger.info("=" * 80)
        logger.info("HYBRID TRADING SESSION STARTED")
        logger.info(f"Log File: {os.path.basename(self.log_file_path)}")

        enabled_symbols = [config['symbol'] for config in self.trade_configs]

        startup_message = (
            f"[HYBRID] <b>Multi-Symbol Trader Started</b>\n\n"
            f"<b>Method:</b> <code>REST API Polling</code>\n"
            f"<b>Symbols:</b> <code>{', '.join(enabled_symbols)}</code>\n"
            f"<b>Total Pairs:</b> <code>{len(enabled_symbols)}</code>\n\n"
            f"[BOT] Bot is now monitoring via REST API polling..."
        )
        self.send_telegram_notification(startup_message)

        # Initialize traders
        for config in self.trade_configs:
            symbol = config["symbol"]
            self.traders[symbol] = HybridSymbolTrader(
                config, self.client, self.send_telegram_notification)
            self.traders[symbol].initialize()

        # Start polling threads
        threads = []
        for symbol, trader in self.traders.items():
            thread = threading.Thread(target=trader.start_polling, daemon=True)
            thread.start()
            threads.append(thread)
            logger.info(f"[{symbol}] Polling thread started")

        logger.info("--- All hybrid traders started ---")

        # Send initial status report
        self._send_report()

        try:
            # Keep main thread alive
            while True:
                time.sleep(60)
                active_count = sum(
                    1 for trader in self.traders.values() if trader.position != 0)
                logger.info(
                    f"[BOT] STATUS: {active_count}/{len(self.traders)} pairs active")

                # CHANGED: Check if it's time for 30-minute report (was hourly)
                self._check_report_time()

        except KeyboardInterrupt:
            logger.info("Received KeyboardInterrupt, shutting down.")
            self.send_telegram_notification(
                "[HYBRID] <b>Bot Stopped Manually</b>")

    def _check_report_time(self):
        """CHANGED: Check if it's time to send 30-minute report (was hourly)."""
        current_time = datetime.now(timezone.utc)
        
        # Check if 30 minutes have passed since last report
        time_diff = current_time - self.last_report_time
        if time_diff.total_seconds() >= 1800:  # 30 minutes = 1800 seconds
            self.last_report_time = current_time
            self._send_report()

    def _send_report(self):
        """CHANGED: Send 30-minute trading report (was hourly)."""
        try:
            current_time = datetime.now(timezone.utc)
            report_time = current_time.strftime('%Y-%m-%d %H:%M UTC')

            # Build individual pair status
            pair_reports = []
            total_unrealized_pnl = 0.0
            total_session_pnl = 0.0
            total_trades = 0
            active_positions = 0

            for symbol, trader in self.traders.items():
                # Get current signal
                signal = 0
                if not trader.prepared_data.empty:
                    signal = trader.prepared_data['position'].iloc[-1]

                # Get unrealized PnL
                unrealized_pnl = trader.get_unrealized_pnl()

                # Position status
                if trader.position == 1:
                    status_icon = "🟢"
                    position_type = "LONG"
                    active_positions += 1
                elif trader.position == -1:
                    status_icon = "🔴"
                    position_type = "SHORT"
                    active_positions += 1
                else:
                    status_icon = "⚪"
                    position_type = "NEUTRAL"

                # Get strategy parameters
                strategy_params = []
                for key, value in trader.strategy_params.items():
                    if value is not None:
                        strategy_params.append(f"{key}={value}")
                params_str = ", ".join(strategy_params)

                # ENHANCED: Add position fix info to report
                fix_info = f" | Fixes: {trader.position_fix_count}" if trader.position_fix_count > 0 else ""
                miss_info = f" | Misses: {trader.consecutive_missing_positions}" if trader.consecutive_missing_positions > 0 else ""
                
                pair_report = (
                    f"{status_icon} {symbol}: {position_type} | Signal: {signal:+.1f} | "
                    f"PnL: {unrealized_pnl:+.2f} | Trades: {trader.session_trades}{fix_info}{miss_info}\n"
                    f"└─ Strategy: {trader.trade_config.get('strategy_name', 'Unknown')} | "
                    f"Params: {params_str}"
                )

                pair_reports.append(pair_report)
                total_unrealized_pnl += unrealized_pnl
                total_session_pnl += trader.session_pnl
                total_trades += trader.session_trades

            # Build complete report matching your format
            report_lines = [
                "📊 30-Minute Trading Report",  # CHANGED: Updated report title
                f"Time: {report_time}",
                "",
                "Individual Pair Status:"
            ]

            # Add each pair status
            for pair_report in pair_reports:
                report_lines.append(pair_report)

            # Add session summary
            report_lines.extend([
                "",
                "📈 Session Summary:",
                f"Active Positions: {active_positions}/{len(self.traders)}",
                f"Total Unrealized PnL: {total_unrealized_pnl:+.2f} USDT",
                f"Total Session PnL: {total_session_pnl:+.2f} USDT",
                f"Total Trades: {total_trades}"
            ])

            report = "\n".join(report_lines)

            self.send_telegram_notification(report)
            logger.info(
                f"[BOT] 30-minute report sent - Active: {active_positions}/{len(self.traders)}")  # CHANGED: Updated log message

        except Exception as e:
            logger.error(f"Error generating 30-minute report: {e}")  # CHANGED: Updated error message

    def send_status_report(self):
        """Send immediate status report."""
        self._send_report()  # CHANGED: Updated method call

    def send_telegram_notification(self, message):
        """Send Telegram notification."""
        if not self.telegram_bot_token or not self.telegram_chat_id:
            return

        try:
            import requests
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            payload = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'HTML'  # Changed to HTML for better formatting
            }
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code != 200:
                logger.warning(
                    f"Telegram API returned status {response.status_code}")
        except Exception as e:
            logger.error(f"Telegram notification failed: {e}")

    def _load_json(self, filepath):
        """Load JSON configuration."""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
            return None


if __name__ == "__main__":
    import sys

    API_CONFIG_FILE = "config.json"
    
    # Allow custom trade config file as command line argument
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        TRADE_CONFIGS_FILE = sys.argv[1]
        print(f"📊 Using custom trade config: {TRADE_CONFIGS_FILE}")
    else:
        TRADE_CONFIGS_FILE = "trade_config.json"

    if not os.path.exists(API_CONFIG_FILE) or not os.path.exists(TRADE_CONFIGS_FILE):
        print(
            f"❌ CRITICAL ERROR: Ensure '{API_CONFIG_FILE}' and '{TRADE_CONFIGS_FILE}' exist.")
        exit(1)

    # Check for command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        try:
            print("📊 Sending status report...")
            manager = HybridTraderManager(
                trade_configs_path=TRADE_CONFIGS_FILE,
                api_config_path=API_CONFIG_FILE
            )

            # Initialize traders without starting polling
            for config in manager.trade_configs:
                symbol = config["symbol"]
                manager.traders[symbol] = HybridSymbolTrader(
                    config, manager.client, manager.send_telegram_notification)
                manager.traders[symbol].initialize()

            manager.send_status_report()
            print("✅ Status report sent!")

        except Exception as e:
            print(f"❌ Error sending report: {e}")
        exit(0)

    try:
        print("🚀 Starting Hybrid Multi-Symbol Trading Bot...")
        print("📊 Method: REST API Polling for low-volume pairs")
        print("💡 Tip: Use 'python hybrid_trader.py --report' to send status report")

        manager = HybridTraderManager(
            trade_configs_path=TRADE_CONFIGS_FILE,
            api_config_path=API_CONFIG_FILE
        )
        manager.start()

    except KeyboardInterrupt:
        print("\n🛑 Hybrid trading stopped by user")
    except Exception as e:
        print(f"\n💥 Critical error: {e}")
        logger.error(f"Critical error: {e}", exc_info=True)
