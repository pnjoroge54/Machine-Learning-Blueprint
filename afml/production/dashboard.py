import subprocess
import sys
import time
from typing import Optional

import psutil
from loguru import logger


def launch_optuna_dashboard(
    storage: str,
    host: str = "127.0.0.1",
    port: int = 8080,
    wait_for_server: bool = True,
    timeout: float = 10.0,
) -> Optional[psutil.Process]:
    """
    Launch the Optuna dashboard as a background process and return its psutil.Process.

    Args:
        storage: Database URL for Optuna storage (e.g., "sqlite:///example.db").
        host: Host address for the dashboard (default: 127.0.0.1).
        port: Port for the dashboard (default: 8080).
        wait_for_server: If True, wait until the dashboard is ready to accept connections.
        timeout: Maximum seconds to wait for the dashboard to become ready.

    Returns:
        psutil.Process object if the dashboard was successfully launched,
        otherwise None.

    Raises:
        FileNotFoundError: If the 'optuna' command is not found in PATH.
        subprocess.SubprocessError: If the subprocess fails to start.
        TimeoutError: If waiting for the server times out.
    """
    cmd = [
        "optuna",
        "dashboard",
        "--storage",
        storage,
        "--host",
        host,
        "--port",
        str(port),
    ]

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
    except FileNotFoundError:
        logger.error("'optuna' command not found. Is Optuna installed?")
        raise
    except Exception as e:
        logger.error(f"Failed to start Optuna dashboard: {e}")
        raise

    try:
        proc = psutil.Process(process.pid)
    except psutil.NoSuchProcess:
        logger.error("Process died immediately after launch.")
        return None

    if wait_for_server:
        start_time = time.time()
        while time.time() - start_time < timeout:
            if not proc.is_running():
                stderr = process.stderr.read()
                logger.error(f"Dashboard process died. stderr: {stderr}")
                raise RuntimeError(f"Dashboard process died. stderr: {stderr}")

            # Check if port is listening
            try:
                for conn in psutil.net_connections():
                    if conn.laddr.port == port and conn.status == "LISTEN":
                        logger.info(f"Dashboard ready at http://{host}:{port}")
                        return proc
            except (psutil.AccessDenied, psutil.Error):
                # Fallback: just sleep and retry
                pass

            time.sleep(0.2)

        raise TimeoutError(
            f"Dashboard did not become ready within {timeout} seconds."
        )

    return proc


if __name__ == "__main__":
    # Example usage
    proc = launch_optuna_dashboard("sqlite:///example.db", port=8080)
    if proc:
        logger.info(f"Optuna dashboard running (PID: {proc.pid})")
        logger.info("Open http://127.0.0.1:8080 in your browser.")
        input("Press Enter to stop the dashboard...")
        proc.terminate()
        proc.wait()
        logger.info("Dashboard stopped.")
