import time
from datetime import datetime, timedelta

import psycopg2
import pandas as pd
from pybit.unified_trading import HTTP


class BackTest_Demo:
    """
    Demo trading runner (one-way mode) with:
      - Safe open: never adds to an existing position
      - Safe close: reduceOnly market
      - Immediate DB insert on OPEN, DB update on CLOSE
      - Robust exchange↔local reconciliation (prevents ghost state)
    """

    def __init__(self, signal_df, ohlvc_df, fee, tp_threshold, sl_threshold, symbol, time_horizon,exchange,
                 db_host="localhost", db_name="exchange", db_user="postgres", db_pass="shah7788", db_port=5432):

        # -------- Strategy config --------
        self.interval_minutes = self.timeframe_to_minutes(time_horizon)
        self.interval_hours= time_horizon
        self.tp_threshold = float(tp_threshold)
        self.sl_threshold = float(sl_threshold)
        self.fee_in_percentage = float(fee)
        self.base_symbol = symbol.upper()
        self.symbol = f"{self.base_symbol}USDT"
        self.signal_df = signal_df
        self.ohlvc_df = ohlvc_df
        self.exchange= exchange

        # -------- Runtime state --------
        self.position = None            # "Buy" for long, "Sell" for short, or None
        self.entry_price = None
        self.entry_time = None
        self.tp_price = None
        self.sl_price = None
        self.trade_quantity = None
        self.balance_before_trade = None
        self.cooldown_end = None
        self.entry_order_id = None      # UNIQUE KEY to link OPEN row with CLOSE update
        self.last_signal = None

        # -------- API client --------
        self.client = HTTP(
            demo=True,
            api_key="1r8nLf7jpFXz9SHwNy",
            api_secret="MNmOleKVIGwSwnMaDh57nNmMF0T3knHWyM8d"
        )


        # -------- Wallet --------
        self.initial_balance = self.get_balance_safely()

        # -------- DB setup --------
        self.conn = psycopg2.connect(host=db_host, 
                                     database=db_name, 
                                     user=db_user, 
                                     password=db_pass, 
                                     port=db_port)
        self.cursor = self.conn.cursor()
        self.table_name = f"bybit_demo_{self.base_symbol.lower()}"
        self.ensure_table()

    # ===================== Utilities =====================

    @staticmethod
    def timeframe_to_minutes(time_str: str) -> int:
        num = int(time_str[:-1])
        unit = time_str[-1].lower()
        if unit == "m":
            return num
        if unit == "h":
            return num * 60
        if unit == "d":
            return num * 1440
        raise ValueError(f"Unsupported timeframe: {time_str}")

    def ensure_table(self):
        q = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(32),
            position_type VARCHAR(10),
            status VARCHAR(16),               -- OPEN / CLOSED
            entry_order_id VARCHAR(64) UNIQUE,
            close_order_id VARCHAR(64),
            entry_time TIMESTAMP,
            close_time TIMESTAMP,
            entry_price FLOAT,
            exit_price FLOAT,
            quantity FLOAT,
            tp_price FLOAT,
            sl_price FLOAT,
            tp_hit BOOLEAN,
            sl_hit BOOLEAN,
            reason TEXT,
            balance_before FLOAT,
            balance_after FLOAT,
            pnl FLOAT,
            fee FLOAT,
            interval_hours VARCHAR(8),        -- NEW
            exchange VARCHAR(32)              -- NEW
        );
        """
        self.cursor.execute(q)
        self.conn.commit()


    def get_balance_safely(self, retries=3, delay=0.5) -> float:
        for i in range(retries):
            try:
                w = self.client.get_wallet_balance(accountType="UNIFIED", coin="USDT")
                return float(w["result"]["list"][0]["coin"][0]["walletBalance"])
            except Exception as e:
                if i == retries - 1:
                    raise
                time.sleep(delay)
        return 0.0

    def get_last_price(self) -> float:
        t = self.client.get_tickers(category="linear", symbol=self.symbol)
        return float(t["result"]["list"][0]["lastPrice"])

    def fetch_open_position(self):
        """
        Return a dict with current exchange position or None if flat.
        Assumes one-way mode (MergedSingle). For hedge mode, extend by side idx.
        """
        p = self.client.get_positions(category="linear", symbol=self.symbol)
        lst = p["result"]["list"]
        if not lst:
            return None
        row = lst[0]
        size = float(row.get("size", 0) or 0)
        if size == 0:
            return None
        side = row.get("side")  # "Buy" or "Sell"
        avg_price = float(row.get("avgPrice") or 0)  # average entry price
        return {"size": size, "side": side, "avg_price": avg_price}

    # ===================== DB helpers =====================

    def insert_open_row(self):
        q = f"""
        INSERT INTO {self.table_name}
        (symbol, position_type, status, entry_order_id, entry_time, entry_price, quantity,
        tp_price, sl_price, tp_hit, sl_hit, reason, balance_before, fee, interval_hours, exchange)
        VALUES
        (%s, %s, %s, %s, %s, %s, %s, %s, %s, FALSE, FALSE, %s, %s, %s, %s, %s);
        """
        open_fee = (self.entry_price * self.trade_quantity) * (self.fee_in_percentage / 100.0)
        self.cursor.execute(q, (
            self.symbol, self.position, "OPEN", self.entry_order_id, self.entry_time, self.entry_price,
            self.trade_quantity, self.tp_price, self.sl_price, "Opened", self.balance_before_trade, open_fee,
            self.interval_hours, self.exchange
        ))
        self.conn.commit()


    def update_close_row(self, exit_price, balance_after, reason, tp_hit=False, sl_hit=False, close_order_id=None):
        total_fee = ((self.entry_price * self.trade_quantity) + (exit_price * self.trade_quantity)) * (self.fee_in_percentage / 100.0)
        pnl = (exit_price - self.entry_price) * self.trade_quantity if self.position == "Buy" \
            else (self.entry_price - exit_price) * self.trade_quantity

        q = f"""
        UPDATE {self.table_name}
        SET status='CLOSED',
            close_order_id=%s,
            close_time=%s,
            exit_price=%s,
            balance_after=%s,
            pnl=%s,
            fee=%s,
            tp_hit=%s,
            sl_hit=%s,
            reason=%s
        WHERE entry_order_id=%s;
        """
        self.cursor.execute(q, (
            close_order_id,
            datetime.utcnow(),
            exit_price,
            balance_after,
            pnl,
            total_fee,
            tp_hit,
            sl_hit,
            reason,
            self.entry_order_id
        ))
        self.conn.commit()

    # ===================== Trading actions =====================

    def open_trade(self, signal, quantity):
        """
        Safe open:
          - Check exchange position first; only open if flat
          - Place MARKET order with TP/SL attached (trigger by LastPrice)
          - Insert DB row immediately
        """
        # Map signal to side (your original code had -1 -> Buy which is unusual; keeping same mapping)
        side = "Buy" if signal == -1 else "Sell"

        # 1) Prevent adding to an existing position
        live = self.fetch_open_position()
        if live is not None:
            print("⚠️ Exchange already has an open position; skipping open.")
            # Recover local state to match exchange if our local says None
            if self.position is None:
                self.recover_local_state_from_exchange(live)
            return

        # 2) Compute TP/SL based on live last price
        last_price = self.get_last_price()
        if side == "Buy":
            tp_price = last_price * (1 + self.tp_threshold / 100.0)
            sl_price = last_price * (1 - self.sl_threshold / 100.0)
        else:
            tp_price = last_price * (1 - self.tp_threshold / 100.0)
            sl_price = last_price * (1 + self.sl_threshold / 100.0)

        # 3) Place MARKET open order with TP/SL
        order = self.client.place_order(
            category="linear",
            symbol=self.symbol,
            side=side,
            orderType="Market",
            qty=quantity,
            timeInForce="GoodTillCancel",
            reduceOnly=False,
            takeProfit=str(round(tp_price, 6)),
            stopLoss=str(round(sl_price, 6)),
            tpTriggerBy="LastPrice",
            slTriggerBy="LastPrice",
            positionIdx=0  # one-way mode
        )
        print("OPEN ORDER RESPONSE:", order)
        entry_order_id = order["result"]["orderId"]

        # 4) Poll for avgFillPrice
        entry_price = None
        for _ in range(20):
            details = self.client.get_order_history(category="linear", symbol=self.symbol, orderId=entry_order_id)
            if details["retCode"] == 0 and details["result"]["list"]:
                row = details["result"]["list"][0]
                avg = row.get("avgFillPrice")
                if avg:
                    entry_price = float(avg)
                    break
            time.sleep(0.3)

        if entry_price is None:
            # Fallback to last price; better than crashing mid-flight
            entry_price = last_price

        # 5) Set local state
        self.position = side
        self.entry_price = entry_price
        self.tp_price = tp_price if side == "Buy" else tp_price
        self.sl_price = sl_price if side == "Buy" else sl_price
        self.entry_time = datetime.utcnow()
        self.trade_quantity = float(quantity)
        self.balance_before_trade = self.get_balance_safely()
        self.entry_order_id = entry_order_id

        # 6) DB insert immediately
        self.insert_open_row()

        print(f"✅ Opened {side} {self.symbol} qty={quantity} @ {entry_price} | TP={tp_price}, SL={sl_price}")

    def close_trade(self, reason, tp_hit=False, sl_hit=False):
        """
        Safe close:
          - Query exchange for current size
          - Place reduceOnly MARKET to flatten
          - Update DB row
        """
        live = self.fetch_open_position()
        if live is None:
            # Nothing open on exchange; if we think we have one, finalize from history
            if self.position is not None:
                print("ℹ️ No live position, reconciling from history…")
                self.reconcile_closed_from_history(default_reason=reason)
            return

        qty = live["size"]
        side = "Sell" if live["side"] == "Buy" else "Buy"

        close_order = self.client.place_order(
            category="linear",
            symbol=self.symbol,
            side=side,
            orderType="Market",
            qty=qty,
            reduceOnly=True,
            positionIdx=0
        )
        print("CLOSE ORDER RESPONSE:", close_order)
        close_order_id = close_order["result"]["orderId"]

        # Poll for close avg price
        exit_price = None
        for _ in range(20):
            h = self.client.get_order_history(category="linear", symbol=self.symbol, orderId=close_order_id)
            if h["retCode"] == 0 and h["result"]["list"]:
                r = h["result"]["list"][0]
                avg = r.get("avgFillPrice")
                if avg:
                    exit_price = float(avg)
                    break
            time.sleep(0.3)

        if exit_price is None:
            # Fallback to last price
            exit_price = self.get_last_price()

        balance_after = self.get_balance_safely()
        self.update_close_row(exit_price, balance_after, reason, tp_hit=tp_hit, sl_hit=sl_hit, close_order_id=close_order_id)

        print(f"✅ Closed {self.position} {self.symbol} @ {exit_price} ({reason})")

        # Reset local state
        self._reset_local_state()

        # Cooldown (optional): skip re-entry until next bar
        self.cooldown_end = datetime.utcnow() + timedelta(minutes=self.interval_minutes)

    # ===================== Reconciliation =====================

    def recover_local_state_from_exchange(self, live_pos):
        """If we restarted or lost state, rebuild local based on exchange info."""
        self.position = live_pos["side"]
        self.entry_price = live_pos["avg_price"]
        self.entry_time = datetime.utcnow()  # unknown; you could use executions to refine
        self.trade_quantity = live_pos["size"]
        # Recompute indicative TP/SL (unknown if they were changed on-exchange)
        if self.position == "Buy":
            self.tp_price = self.entry_price * (1 + self.tp_threshold / 100.0)
            self.sl_price = self.entry_price * (1 - self.sl_threshold / 100.0)
        else:
            self.tp_price = self.entry_price * (1 - self.tp_threshold / 100.0)
            self.sl_price = self.entry_price * (1 + self.sl_threshold / 100.0)
        self.balance_before_trade = self.get_balance_safely()
        self.entry_order_id = self.entry_order_id or f"RECOVERED-{int(time.time())}"

    def reconcile_closed_from_history(self, default_reason="Closed on exchange"):
        """
        If exchange is flat but we still have local state, fetch recent order history
        to get the last fill price and finalize DB.
        """
        # Try to find the most recent filled order for this symbol
        hist = self.client.get_order_history(category="linear", symbol=self.symbol, limit=5)
        exit_price = None
        close_order_id = None
        if hist["retCode"] == 0 and hist["result"]["list"]:
            # Take the most recent item with avgFillPrice
            for row in hist["result"]["list"]:
                avg = row.get("avgFillPrice")
                status = row.get("orderStatus")
                if avg and status in ("Filled", "PartiallyFilled"):  # conservative
                    exit_price = float(avg)
                    close_order_id = row.get("orderId")
                    break
        if exit_price is None:
            exit_price = self.get_last_price()

        balance_after = self.get_balance_safely()
        self.update_close_row(exit_price, balance_after, default_reason, tp_hit=False, sl_hit=False, close_order_id=close_order_id)
        print(f"ℹ️ Reconciled closed position from history @ {exit_price} ({default_reason})")
        self._reset_local_state()

    def _reset_local_state(self):
        self.position = None
        self.entry_price = None
        self.entry_time = None
        self.tp_price = None
        self.sl_price = None
        self.trade_quantity = None
        self.balance_before_trade = None
        self.entry_order_id = None

    # ===================== Monitoring & loop =====================

    def monitor_trade(self):
        """
        Keep local state in sync with exchange (detect TP/SL closure),
        and avoid duplicate open/close actions.
        """
        # 1) If we think we have a position, check the exchange truth
        live = self.fetch_open_position()

        if live is None:
            # Exchange flat
            if self.position is not None:
                # We thought it was open => it got closed on exchange; reconcile and reset
                self.reconcile_closed_from_history(default_reason="Closed on Bybit (TP/SL/Manual)")
            return
        else:
            # Exchange has a pos — if we lost local state, rebuild it
            if self.position is None:
                self.recover_local_state_from_exchange(live)

        # 2) Optional local TP/SL check (Bybit already has TP/SL attached)
        # Leave it as a no-op; Bybit will execute TP/SL. If you want a local fail-safe,
        # you can fetch last price here and call self.close_trade(...)

    def is_interval_end(self) -> bool:
        # Simple placeholder – always true, like your original
        return True

    def run_strategy(self, quantity):
        """
        Main loop:
          - Never opens if an exchange position exists
          - Opens only when flat and signal is in {1,-1}
          - Closes when signal flips
        """
        while True:
            try:
                now = datetime.utcnow()

                # Cooldown after a close
                if self.cooldown_end and now < self.cooldown_end:
                    self.monitor_trade()
                    time.sleep(2)
                    continue

                self.monitor_trade()

                # Current signal
                cur_signal = int(self.signal_df.iloc[-1]["aggregated_signal"])

                # If no local/exchange position, consider opening
                live = self.fetch_open_position()
                flat = (live is None)
                if flat and self.position is None:
                    if cur_signal in (1, -1):
                        self.open_trade(cur_signal, quantity)
                        self.last_signal = cur_signal
                else:
                    # We are in a position: exit if the signal flips
                    if self.last_signal is None:
                        self.last_signal = cur_signal
                    else:
                        if cur_signal != self.last_signal:
                            self.close_trade("Signal changed")
                            # Optionally re-open in the new direction immediately:
                            if cur_signal in (1, -1):
                                self.open_trade(cur_signal, quantity)
                            self.last_signal = cur_signal

                time.sleep(2)

            except Exception as e:
                print("⚠️ Loop error:", repr(e))
                time.sleep(2)

