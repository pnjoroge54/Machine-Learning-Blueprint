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
    force_restart: bool = False,
) -> Optional[psutil.Process]:
    """
    Launch the Optuna dashboard as a background process and return its psutil.Process.

    Args:
        storage: Database URL for Optuna storage (e.g., "sqlite:///example.db").
        host: Host address for the dashboard (default: 127.0.0.1).
        port: Port for the dashboard (default: 8080).
        wait_for_server: If True, wait until the dashboard is ready to accept connections.
        timeout: Maximum seconds to wait for the dashboard to become ready.
        force_restart: If True, kill any existing process listening on (host, port)
                      before launching a new dashboard.

    Returns:
        psutil.Process object if the dashboard was successfully launched,
        otherwise None.

    Raises:
        FileNotFoundError: If the 'optuna' command is not found in PATH.
        subprocess.SubprocessError: If the subprocess fails to start.
        TimeoutError: If waiting for the server times out.
        RuntimeError: If force_restart is False and a dashboard is already running.
    """
    # If force_restart is enabled, kill any existing process on the port
    if force_restart:
        kill_process_on_port(port, host)
    else:
        # Check if something is already listening on the port
        if is_port_in_use(port, host):
            raise RuntimeError(
                f"Port {port} is already in use. Use force_restart=True to replace the existing dashboard."
            )

    # Build the command
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
                pass  # Fallback: just sleep and retry

            time.sleep(0.2)

        raise TimeoutError(
            f"Dashboard did not become ready within {timeout} seconds."
        )

    return proc


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """
    Check if a specific port is currently listening on the given host.
    Note: This only checks for IPv4 connections on the exact host address.
    """
    try:
        for conn in psutil.net_connections(kind="inet"):
            if conn.laddr.port == port and conn.laddr.ip == host:
                if conn.status == "LISTEN":
                    return True
    except psutil.AccessDenied:
        # If we lack permissions, assume port is not in use (but log warning)
        logger.warning("Insufficient permissions to inspect network connections.")
    except Exception as e:
        logger.warning(f"Error while checking port: {e}")
    return False


def kill_process_on_port(port: int, host: str = "127.0.0.1"):
    """
    Terminate any process listening on the given (host, port).
    It first checks if the process is an Optuna dashboard by inspecting its command line.
    """
    for conn in psutil.net_connections(kind="inet"):
        if conn.laddr.port == port and conn.laddr.ip == host and conn.status == "LISTEN":
            try:
                proc = psutil.Process(conn.pid)
                # Avoid killing ourselves
                if proc.pid == psutil.Process().pid:
                    logger.warning("The process listening on port is this script itself – skipping.")
                    continue

                # Check if it's likely an Optuna dashboard (optional safety)
                cmdline = " ".join(proc.cmdline())
                if "optuna" in cmdline and "dashboard" in cmdline:
                    logger.info(f"Killing existing Optuna dashboard (PID {conn.pid}) on port {port}...")
                    proc.terminate()
                    proc.wait(timeout=3)
                    logger.info("Existing dashboard terminated.")
                else:
                    logger.warning(
                        f"A non-Optuna process (PID {conn.pid}) is using port {port}. "
                        "Skipping termination to avoid affecting unrelated services."
                    )
            except psutil.NoSuchProcess:
                pass
            except psutil.TimeoutExpired:
                logger.warning(f"Process {conn.pid} did not terminate, force killing.")
                proc.kill()
            except Exception as e:
                logger.error(f"Error while terminating process on port {port}: {e}")


if __name__ == "__main__":
    # Example with force_restart=True
    proc = launch_optuna_dashboard(
        "sqlite:///example.db",
        port=8080,
        force_restart=True,  # will kill any existing dashboard on port 8080
    )
    if proc:
        logger.info(f"Optuna dashboard running (PID: {proc.pid})")
        logger.info("Open http://127.0.0.1:8080 in your browser.")
        input("Press Enter to stop the dashboard...")
        proc.terminate()
        proc.wait()
        logger.info("Dashboard stopped.")
