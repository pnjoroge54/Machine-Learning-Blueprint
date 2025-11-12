"""
MQL5 Integration Bridge for AFML Cache System
Enables seamless communication between Python ML/caching and MQL5 trading.
"""

import json
import socket
import struct
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class SignalPacket:
    """Trading signal packet for MQL5."""

    timestamp: str
    symbol: str
    signal_type: str  # "BUY", "SELL", "CLOSE", "MODIFY"
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    strategy_name: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class MarketDataPacket:
    """Market data from MQL5."""

    timestamp: str
    symbol: str
    bid: float
    ask: float
    volume: float
    spread: float
    bars: Optional[List[Dict]] = None  # OHLCV bars


class MQL5Bridge:
    """
    Bridge between Python AFML cache system and MQL5.

    Features:
    - Socket-based communication (low latency)
    - Signal caching and replay
    - Market data collection
    - Performance tracking
    - Live/backtest mode switching
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9090,
        mode: str = "live",  # "live" or "backtest"
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize MQL5 bridge.

        Args:
            host: Server host
            port: Server port
            mode: Operating mode ("live" or "backtest")
            cache_dir: Cache directory (uses AFML cache if None)
        """
        from . import CACHE_DIRS

        self.host = host
        self.port = port
        self.mode = mode
        self.cache_dir = cache_dir or CACHE_DIRS["base"] / "mql5_bridge"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Communication
        self.server_socket: Optional[socket.socket] = None
        self.client_socket: Optional[socket.socket] = None
        self.is_running = False
        self.lock = threading.Lock()

        # Signal management
        self.signal_history: List[SignalPacket] = []
        self.pending_signals: List[SignalPacket] = []

        # Market data buffer
        self.market_data_buffer: Dict[str, List[MarketDataPacket]] = {}

        # Performance tracking
        self.signals_sent = 0
        self.signals_executed = 0
        self.connection_start_time: Optional[float] = None

        # Load cached signals if in backtest mode
        if self.mode == "backtest":
            self._load_cached_signals()

    def start_server(self):
        """Start the bridge server to accept MQL5 connections."""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            self.is_running = True

            logger.info(
                f"MQL5 Bridge server started on {self.host}:{self.port} (mode: {self.mode})"
            )

            # Start listener thread
            listener_thread = threading.Thread(target=self._accept_connections, daemon=True)
            listener_thread.start()

        except Exception as e:
            logger.error(f"Failed to start MQL5 bridge server: {e}")
            raise

    def _accept_connections(self):
        """Accept incoming MQL5 connections."""
        while self.is_running:
            try:
                client, addr = self.server_socket.accept()
                logger.info(f"MQL5 client connected from {addr}")

                with self.lock:
                    self.client_socket = client
                    self.connection_start_time = time.time()

                # Handle client in separate thread
                handler_thread = threading.Thread(
                    target=self._handle_client, args=(client,), daemon=True
                )
                handler_thread.start()

            except Exception as e:
                if self.is_running:
                    logger.error(f"Error accepting connection: {e}")

    def _handle_client(self, client: socket.socket):
        """Handle messages from MQL5 client."""
        buffer = b""

        while self.is_running:
            try:
                # Receive data
                data = client.recv(4096)
                if not data:
                    logger.warning("MQL5 client disconnected")
                    break

                buffer += data

                # Process complete messages (length-prefixed)
                while len(buffer) >= 4:
                    msg_length = struct.unpack("I", buffer[:4])[0]

                    if len(buffer) < 4 + msg_length:
                        break  # Incomplete message

                    msg_data = buffer[4 : 4 + msg_length]
                    buffer = buffer[4 + msg_length :]

                    # Process message
                    self._process_mql5_message(msg_data)

            except Exception as e:
                logger.error(f"Error handling MQL5 client: {e}")
                break

        client.close()
        with self.lock:
            self.client_socket = None

    def _process_mql5_message(self, data: bytes):
        """Process incoming message from MQL5."""
        try:
            message = json.loads(data.decode("utf-8"))
            msg_type = message.get("type")

            if msg_type == "market_data":
                self._handle_market_data(message)
            elif msg_type == "execution_report":
                self._handle_execution_report(message)
            elif msg_type == "heartbeat":
                self._send_heartbeat_response()
            elif msg_type == "request_signals":
                self._send_pending_signals()
            else:
                logger.warning(f"Unknown message type: {msg_type}")

        except Exception as e:
            logger.error(f"Error processing MQL5 message: {e}")

    def _handle_market_data(self, message: Dict):
        """Handle incoming market data from MQL5."""
        try:
            packet = MarketDataPacket(
                timestamp=message["timestamp"],
                symbol=message["symbol"],
                bid=message["bid"],
                ask=message["ask"],
                volume=message.get("volume", 0),
                spread=message.get("spread", 0),
                bars=message.get("bars"),
            )

            symbol = packet.symbol
            if symbol not in self.market_data_buffer:
                self.market_data_buffer[symbol] = []

            self.market_data_buffer[symbol].append(packet)

            # Keep only last 10000 packets per symbol
            if len(self.market_data_buffer[symbol]) > 10000:
                self.market_data_buffer[symbol] = self.market_data_buffer[symbol][-10000:]

            logger.debug(f"Received market data for {symbol}: {packet.bid}/{packet.ask}")

        except Exception as e:
            logger.error(f"Error handling market data: {e}")

    def _handle_execution_report(self, message: Dict):
        """Handle execution report from MQL5."""
        try:
            signal_id = message.get("signal_id")
            status = message.get("status")
            execution_price = message.get("execution_price")

            logger.info(
                f"Execution report - ID: {signal_id}, Status: {status}, Price: {execution_price}"
            )

            if status == "executed":
                self.signals_executed += 1

            # Track execution in cache
            self._save_execution_report(message)

        except Exception as e:
            logger.error(f"Error handling execution report: {e}")

    def send_signal(self, signal: SignalPacket) -> bool:
        """
        Send trading signal to MQL5.

        Args:
            signal: Signal packet to send

        Returns:
            True if sent successfully, False otherwise
        """
        try:
            # Add to history
            self.signal_history.append(signal)
            self.signals_sent += 1

            # Save to cache
            self._cache_signal(signal)

            # Send to MQL5 if connected
            if self.client_socket:
                message = {"type": "signal", "data": asdict(signal)}

                self._send_message(message)
                logger.info(f"Sent {signal.signal_type} signal for {signal.symbol}")
                return True
            else:
                logger.warning("No MQL5 client connected - signal queued")
                self.pending_signals.append(signal)
                return False

        except Exception as e:
            logger.error(f"Error sending signal: {e}")
            return False

    def _send_message(self, message: Dict):
        """Send message to MQL5 with length prefix."""
        try:
            data = json.dumps(message).encode("utf-8")
            length = struct.pack("I", len(data))

            with self.lock:
                if self.client_socket:
                    self.client_socket.sendall(length + data)

        except Exception as e:
            logger.error(f"Error sending message: {e}")
            with self.lock:
                self.client_socket = None

    def _send_pending_signals(self):
        """Send all pending signals to MQL5."""
        with self.lock:
            for signal in self.pending_signals:
                message = {"type": "signal", "data": asdict(signal)}
                self._send_message(message)

            logger.info(f"Sent {len(self.pending_signals)} pending signals")
            self.pending_signals.clear()

    def _send_heartbeat_response(self):
        """Send heartbeat response to MQL5."""
        message = {"type": "heartbeat_response", "timestamp": datetime.now().isoformat()}
        self._send_message(message)

    def get_market_data(
        self, symbol: str, as_dataframe: bool = True
    ) -> Optional[pd.DataFrame | List[MarketDataPacket]]:
        """
        Get buffered market data for symbol.

        Args:
            symbol: Symbol to retrieve
            as_dataframe: Return as DataFrame if True, else list

        Returns:
            Market data or None if not available
        """
        if symbol not in self.market_data_buffer:
            return None

        data = self.market_data_buffer[symbol]

        if not as_dataframe:
            return data

        # Convert to DataFrame
        records = []
        for packet in data:
            records.append(
                {
                    "timestamp": pd.to_datetime(packet.timestamp),
                    "bid": packet.bid,
                    "ask": packet.ask,
                    "volume": packet.volume,
                    "spread": packet.spread,
                }
            )

        df = pd.DataFrame(records)
        df.set_index("timestamp", inplace=True)
        return df

    def _cache_signal(self, signal: SignalPacket):
        """Cache signal to disk."""
        try:
            signals_file = self.cache_dir / f"signals_{self.mode}.jsonl"

            with open(signals_file, "a") as f:
                f.write(json.dumps(asdict(signal)) + "\n")

        except Exception as e:
            logger.error(f"Error caching signal: {e}")

    def _load_cached_signals(self):
        """Load cached signals for backtest mode."""
        try:
            signals_file = self.cache_dir / f"signals_{self.mode}.jsonl"

            if signals_file.exists():
                with open(signals_file, "r") as f:
                    for line in f:
                        signal_dict = json.loads(line)
                        signal = SignalPacket(**signal_dict)
                        self.signal_history.append(signal)

                logger.info(f"Loaded {len(self.signal_history)} cached signals")

        except Exception as e:
            logger.error(f"Error loading cached signals: {e}")

    def _save_execution_report(self, report: Dict):
        """Save execution report to cache."""
        try:
            reports_file = self.cache_dir / f"executions_{self.mode}.jsonl"

            with open(reports_file, "a") as f:
                f.write(json.dumps(report) + "\n")

        except Exception as e:
            logger.error(f"Error saving execution report: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get bridge performance statistics."""
        stats = {
            "mode": self.mode,
            "signals_sent": self.signals_sent,
            "signals_executed": self.signals_executed,
            "execution_rate": (
                self.signals_executed / self.signals_sent if self.signals_sent > 0 else 0
            ),
            "pending_signals": len(self.pending_signals),
            "connected": self.client_socket is not None,
            "uptime_seconds": (
                time.time() - self.connection_start_time if self.connection_start_time else 0
            ),
            "symbols_tracked": list(self.market_data_buffer.keys()),
        }
        return stats

    def stop(self):
        """Stop the bridge server."""
        self.is_running = False

        if self.client_socket:
            self.client_socket.close()

        if self.server_socket:
            self.server_socket.close()

        logger.info("MQL5 Bridge stopped")


class MQL5CachedStrategy:
    """
    Wrapper for cached ML strategies that integrates with MQL5.

    Combines AFML caching with MQL5 signal generation.
    """

    def __init__(
        self, strategy_func, bridge: MQL5Bridge, use_cache: bool = True, track_data: bool = True
    ):
        """
        Initialize cached strategy for MQL5.

        Args:
            strategy_func: Function that generates signals (should be cached)
            bridge: MQL5Bridge instance
            use_cache: Use AFML caching
            track_data: Track data access
        """
        from .backtest_cache import get_backtest_cache
        from .data_access_tracker import get_data_tracker

        self.strategy_func = strategy_func
        self.bridge = bridge
        self.use_cache = use_cache
        self.track_data = track_data

        self.backtest_cache = get_backtest_cache()
        self.data_tracker = get_data_tracker() if track_data else None

        self.last_signal_time: Optional[datetime] = None

    def generate_signals(
        self, market_data: pd.DataFrame, parameters: Dict[str, Any]
    ) -> List[SignalPacket]:
        """
        Generate trading signals using cached strategy.

        Args:
            market_data: Market data DataFrame
            parameters: Strategy parameters

        Returns:
            List of signal packets
        """
        try:
            # Track data access
            if self.track_data and isinstance(market_data.index, pd.DatetimeIndex):
                self.data_tracker.log_access(
                    dataset_name=f"mql5_{self.bridge.mode}",
                    start_date=market_data.index[0],
                    end_date=market_data.index[-1],
                    purpose="live_trading" if self.bridge.mode == "live" else "backtest",
                    data_shape=market_data.shape,
                )

            # Generate signals (cached by AFML)
            signals = self.strategy_func(market_data, parameters)

            # Convert to signal packets
            signal_packets = []
            for signal in signals:
                packet = self._convert_to_signal_packet(signal)
                if packet:
                    signal_packets.append(packet)

            # Send to MQL5
            for packet in signal_packets:
                self.bridge.send_signal(packet)

            self.last_signal_time = datetime.now()

            return signal_packets

        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return []

    def _convert_to_signal_packet(self, signal: Dict) -> Optional[SignalPacket]:
        """Convert strategy signal to MQL5 signal packet."""
        try:
            return SignalPacket(
                timestamp=signal.get("timestamp", datetime.now().isoformat()),
                symbol=signal["symbol"],
                signal_type=signal["type"],
                entry_price=signal["price"],
                stop_loss=signal.get("stop_loss"),
                take_profit=signal.get("take_profit"),
                position_size=signal.get("size"),
                metadata=signal.get("metadata"),
                strategy_name=self.strategy_func.__name__,
                confidence=signal.get("confidence"),
            )
        except Exception as e:
            logger.error(f"Error converting signal: {e}")
            return None


# Integration with AFML cache monitoring
def setup_mql5_monitoring(bridge: MQL5Bridge):
    """Setup integrated monitoring for MQL5 bridge and AFML cache."""
    from .cache_monitoring import get_cache_monitor

    monitor = get_cache_monitor()

    def print_integrated_report():
        """Print combined MQL5 and cache report."""
        print("\n" + "=" * 80)
        print("INTEGRATED MQL5 + AFML CACHE REPORT")
        print("=" * 80)

        # MQL5 stats
        mql5_stats = bridge.get_performance_stats()
        print(f"\nMQL5 Bridge Status:")
        print(f"  Mode: {mql5_stats['mode']}")
        print(f"  Connected: {mql5_stats['connected']}")
        print(f"  Signals Sent: {mql5_stats['signals_sent']}")
        print(f"  Signals Executed: {mql5_stats['signals_executed']}")
        print(f"  Execution Rate: {mql5_stats['execution_rate']:.1%}")
        print(f"  Uptime: {mql5_stats['uptime_seconds']:.0f}s")

        # Cache stats
        print(f"\nCache Performance:")
        monitor.print_health_report(detailed=False)

        print("=" * 80 + "\n")

    return print_integrated_report


__all__ = [
    "MQL5Bridge",
    "MQL5CachedStrategy",
    "SignalPacket",
    "MarketDataPacket",
    "setup_mql5_monitoring",
]
