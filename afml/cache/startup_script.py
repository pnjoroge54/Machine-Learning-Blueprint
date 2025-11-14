"""
Complete startup script for AFML Cache + MQL5 Integration
Handles proper initialization order and connection verification.
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO",
)
logger.add("mql5_bridge_{time:YYYY-MM-DD}.log", rotation="1 day", retention="7 days", level="DEBUG")

from afml.cache import initialize_cache_system, robust_cacheable
from afml.cache.mql5_bridge import MQL5Bridge, MQL5CachedStrategy, SignalPacket


def wait_for_connection(bridge: MQL5Bridge, timeout: int = 60) -> bool:
    """
    Wait for MQL5 to connect with timeout.

    Args:
        bridge: MQL5Bridge instance
        timeout: Maximum seconds to wait

    Returns:
        True if connected, False if timeout
    """
    logger.info(f"Waiting for MQL5 connection (timeout: {timeout}s)...")

    start_time = time.time()
    last_message = time.time()

    while time.time() - start_time < timeout:
        if bridge.client_socket is not None:
            logger.success("✅ MQL5 client connected!")
            return True

        # Print waiting message every 5 seconds
        if time.time() - last_message >= 5:
            elapsed = int(time.time() - start_time)
            logger.info(f"Still waiting... ({elapsed}s elapsed)")
            last_message = time.time()

        time.sleep(0.5)

    logger.error(f"❌ Connection timeout after {timeout}s")
    return False


def verify_server_listening(host: str, port: int) -> bool:
    """
    Verify that the Python server is actually listening on the port.

    Args:
        host: Server host
        port: Server port

    Returns:
        True if server is listening
    """
    import socket

    try:
        # Try to connect to ourselves to verify
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_socket.settimeout(2)

        # This should succeed if server is listening
        result = test_socket.connect_ex((host, port))
        test_socket.close()

        if result == 0:
            logger.success(f"✅ Server is listening on {host}:{port}")
            return True
        else:
            logger.error(f"❌ Server not listening on {host}:{port} (error: {result})")
            return False

    except Exception as e:
        logger.error(f"❌ Failed to verify server: {e}")
        return False


def check_port_available(port: int) -> bool:
    """
    Check if port is available before starting server.

    Args:
        port: Port to check

    Returns:
        True if available, False if in use
    """
    import socket

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", port))
        s.close()
        logger.success(f"✅ Port {port} is available")
        return True
    except OSError as e:
        logger.error(f"❌ Port {port} is in use: {e}")
        return False


@robust_cacheable
def generate_test_features(data: pd.DataFrame) -> pd.DataFrame:
    """Generate test features (cached)."""
    features = data.copy()
    features["sma_20"] = data["close"].rolling(20).mean()
    features["rsi"] = 50 + np.random.randn(len(data)) * 10  # Simplified
    return features.dropna()


def generate_test_signal(data: pd.DataFrame) -> SignalPacket:
    """Generate a test trading signal."""
    latest_price = data["close"].iloc[-1]

    return SignalPacket(
        timestamp=datetime.now().isoformat(),
        symbol="EURUSD",
        signal_type="BUY",
        entry_price=latest_price,
        stop_loss=latest_price - 0.0050,
        take_profit=latest_price + 0.0100,
        position_size=0.01,
        confidence=0.75,
        strategy_name="test_strategy",
        metadata={"test": True, "generated_at": datetime.now().isoformat()},
    )


def run_startup_checks() -> bool:
    """
    Run all startup checks before starting bridge.

    Returns:
        True if all checks pass
    """
    logger.info("Running startup checks...")

    # Check 1: Cache system
    try:
        initialize_cache_system()
        logger.success("✅ Cache system initialized")
    except Exception as e:
        logger.error(f"❌ Cache initialization failed: {e}")
        return False

    # Check 2: Port availability
    if not check_port_available(80):
        logger.error("❌ Port 80 is not available")
        logger.info("Try: lsof -i :80  (Linux/Mac) or netstat -ano | findstr :80 (Windows)")
        return False

    # Check 3: Test cache functionality
    try:
        test_data = pd.DataFrame({"close": np.random.randn(100) + 1.1000})
        features = generate_test_features(test_data)
        logger.success("✅ Cache functionality verified")
    except Exception as e:
        logger.error(f"❌ Cache test failed: {e}")
        return False

    logger.success("✅ All startup checks passed")
    return True


def print_startup_instructions():
    """Print clear instructions for user."""
    print("\n" + "=" * 70)
    print("MQL5 CONNECTION INSTRUCTIONS")
    print("=" * 70)
    print("\n📋 Follow these steps IN ORDER:\n")
    print("1. ✅ Python server is now running (you're seeing this message)")
    print("2. 🔌 Open MetaTrader 5")
    print("3. 📊 Open any chart (e.g., EURUSD, M5)")
    print("4. 🤖 Drag 'PythonBridgeEA' from Navigator → Expert Advisors")
    print("5. ⚙️  In the EA settings, verify:")
    print("     - PythonHost: 127.0.0.1")
    print("     - PythonPort: 80")
    print("     - EnableTrading: false (for testing)")
    print("6. ✅ Click OK")
    print("7. 🟢 Enable AutoTrading (Ctrl+E or click AutoTrading button)")
    print("8. 👀 Watch the 'Experts' tab for connection message")
    print("\n" + "=" * 70)
    print("⏳ Waiting for MQL5 to connect...")
    print("=" * 70 + "\n")


def main_live_mode():
    """
    Main function for live trading mode with proper connection handling.
    """
    logger.info("=" * 70)
    logger.info("AFML Cache + MQL5 Integration - LIVE MODE")
    logger.info("=" * 70)

    # Step 1: Run startup checks
    if not run_startup_checks():
        logger.error("Startup checks failed. Exiting.")
        return

    # Step 2: Create and start bridge
    logger.info("\nStarting MQL5 bridge...")
    bridge = MQL5Bridge(host="127.0.0.1", port=80, mode="live")

    try:
        bridge.start_server()
        time.sleep(1)  # Give server time to bind

        # Step 3: Verify server is actually listening
        if not verify_server_listening("127.0.0.1", 80):
            logger.error("Server failed to start properly")
            return

        # Step 4: Print instructions
        print_startup_instructions()

        # Step 5: Wait for connection
        if not wait_for_connection(bridge, timeout=120):  # 2 minute timeout
            logger.error("\n❌ MQL5 did not connect within timeout period")
            logger.info("\nTroubleshooting:")
            logger.info("1. Is MetaTrader 5 running?")
            logger.info("2. Is the EA attached to a chart?")
            logger.info("3. Is AutoTrading enabled? (Ctrl+E)")
            logger.info("4. Check the 'Experts' tab for error messages")
            logger.info("5. Check Windows Firewall settings")
            return

        # Step 6: Connection established - send test signal
        logger.info("\n" + "=" * 70)
        logger.info("CONNECTION ESTABLISHED - TESTING")
        logger.info("=" * 70)

        time.sleep(2)  # Let MQL5 settle

        # Generate and send test signal
        test_data = pd.DataFrame(
            {
                "close": 1.1000 + np.random.randn(100) * 0.0010,
                "high": 1.1000 + np.random.randn(100) * 0.0010 + 0.0005,
                "low": 1.1000 + np.random.randn(100) * 0.0010 - 0.0005,
            }
        )

        logger.info("Sending test signal...")
        test_signal = generate_test_signal(test_data)
        success = bridge.send_signal(test_signal)

        if success:
            logger.success("✅ Test signal sent successfully!")
        else:
            logger.warning("⚠️  Test signal queued (will send when MQL5 requests)")

        time.sleep(2)

        # Step 7: Print stats
        stats = bridge.get_performance_stats()
        logger.info("\n" + "=" * 70)
        logger.info("BRIDGE STATUS")
        logger.info("=" * 70)
        logger.info(f"Connected: {stats['connected']}")
        logger.info(f"Signals Sent: {stats['signals_sent']}")
        logger.info(f"Pending Signals: {stats['pending_signals']}")
        logger.info(f"Uptime: {stats['uptime_seconds']:.0f}s")
        logger.info("=" * 70)

        # Step 8: Keep running
        logger.info("\n✅ Bridge is running. Press Ctrl+C to stop.\n")

        # Main loop - keep bridge alive and send periodic updates
        update_counter = 0
        try:
            while True:
                time.sleep(10)
                update_counter += 1

                # Print status every 60 seconds
                if update_counter % 6 == 0:
                    stats = bridge.get_performance_stats()
                    logger.info(
                        f"Status: Connected={stats['connected']}, "
                        f"Signals={stats['signals_sent']}, "
                        f"Executed={stats['signals_executed']}"
                    )

                # Check if we lost connection
                if not bridge.client_socket:
                    logger.warning("⚠️  Lost connection to MQL5. Waiting for reconnect...")
                    if wait_for_connection(bridge, timeout=30):
                        logger.success("✅ Reconnected to MQL5")

        except KeyboardInterrupt:
            logger.info("\n\nShutting down gracefully...")

    except Exception as e:
        logger.error(f"Error in main loop: {e}")
        import traceback

        logger.error(traceback.format_exc())
    finally:
        bridge.stop()
        logger.info("Bridge stopped. Goodbye!")


def main_test_mode():
    """
    Test mode - just verify everything works without waiting for MQL5.
    """
    logger.info("=" * 70)
    logger.info("QUICK TEST MODE")
    logger.info("=" * 70)

    # Run checks
    if not run_startup_checks():
        return

    # Start bridge
    logger.info("\nStarting bridge...")
    bridge = MQL5Bridge(port=80, mode="live")
    bridge.start_server()
    time.sleep(1)

    # Verify listening
    if not verify_server_listening("127.0.0.1", 80):
        logger.error("Server not listening!")
        return

    # Try sending a test signal (will be queued)
    logger.info("\nTesting signal generation...")
    test_data = pd.DataFrame({"close": 1.1000 + np.random.randn(100) * 0.0010})

    test_signal = generate_test_signal(test_data)
    bridge.send_signal(test_signal)

    stats = bridge.get_performance_stats()
    logger.info(f"\nBridge Stats: {stats}")

    logger.success("\n✅ Test completed successfully!")
    logger.info("Run with --live flag to wait for MQL5 connection")

    bridge.stop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AFML MQL5 Bridge")
    parser.add_argument(
        "--mode",
        choices=["live", "test"],
        default="test",
        help="Run mode: 'live' waits for MQL5, 'test' just verifies setup",
    )

    args = parser.parse_args()

    if args.mode == "live":
        main_live_mode()
    else:
        main_test_mode()
