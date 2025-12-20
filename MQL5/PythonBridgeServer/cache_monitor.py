"""
Real-time Cache Monitoring Dashboard
Monitors MQL5 cache performance in real-time.

Usage:
    python cache_monitor.py --cache-stats MLCache/cache_stats.json
"""

import argparse
import json
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import matplotlib.animation as animation
import matplotlib.pyplot as plt


class CacheMonitor:
    """Real-time cache performance monitor."""

    def __init__(self, stats_file: Path, history_size: int = 100):
        self.stats_file = stats_file
        self.history_size = history_size

        # Data buffers
        self.timestamps = deque(maxlen=history_size)
        self.hit_rates = deque(maxlen=history_size)
        self.total_hits = deque(maxlen=history_size)
        self.total_misses = deque(maxlen=history_size)

        # Previous state for delta calculation
        self.prev_hits = 0
        self.prev_misses = 0

    def read_stats(self) -> Optional[Dict]:
        """Read current cache statistics."""
        try:
            if not self.stats_file.exists():
                return None

            with open(self.stats_file, "r") as f:
                stats = json.load(f)

            return stats

        except Exception as e:
            print(f"Error reading stats: {e}")
            return None

    def update_data(self, stats: Dict):
        """Update monitoring data."""
        current_time = datetime.now()

        # Extract metrics
        hits = stats.get("total_hits", 0)
        misses = stats.get("total_misses", 0)
        total = hits + misses

        hit_rate = (hits / total * 100) if total > 0 else 0

        # Update buffers
        self.timestamps.append(current_time)
        self.hit_rates.append(hit_rate)
        self.total_hits.append(hits)
        self.total_misses.append(misses)

        # Update previous state
        self.prev_hits = hits
        self.prev_misses = misses

    def start_monitoring(self, update_interval: int = 1):
        """Start real-time monitoring with live plot."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("MQL5 Cache Performance Monitor", fontsize=16)

        def animate(frame):
            # Read latest stats
            stats = self.read_stats()

            if stats:
                self.update_data(stats)

            # Clear all axes
            for ax in axes.flat:
                ax.clear()

            if len(self.timestamps) == 0:
                return

            # Plot 1: Hit Rate Over Time
            axes[0, 0].plot(self.timestamps, self.hit_rates, "b-", linewidth=2)
            axes[0, 0].set_title("Cache Hit Rate")
            axes[0, 0].set_ylabel("Hit Rate (%)")
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_ylim([0, 100])

            # Plot 2: Hits vs Misses
            axes[0, 1].plot(
                self.timestamps, self.total_hits, "g-", label="Hits", linewidth=2
            )
            axes[0, 1].plot(
                self.timestamps, self.total_misses, "r-", label="Misses", linewidth=2
            )
            axes[0, 1].set_title("Cumulative Hits vs Misses")
            axes[0, 1].set_ylabel("Count")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # Plot 3: Current Statistics (text)
            axes[1, 0].axis("off")
            if stats:
                stats_text = f"""
                Current Statistics:
                
                Total Hits: {stats.get("total_hits", 0):,}
                Total Misses: {stats.get("total_misses", 0):,}
                Hit Rate: {self.hit_rates[-1] if self.hit_rates else 0:.2f}%
                
                Cache Size: {stats.get("cache_size", 0)} / {stats.get("max_size", 0)}
                
                Last Update: {datetime.now().strftime("%H:%M:%S")}
                """
                axes[1, 0].text(
                    0.1,
                    0.5,
                    stats_text,
                    fontsize=12,
                    verticalalignment="center",
                    family="monospace",
                )

            # Plot 4: Hit Rate Distribution
            if len(self.hit_rates) > 10:
                axes[1, 1].hist(self.hit_rates, bins=20, alpha=0.7, edgecolor="black")
                axes[1, 1].set_title("Hit Rate Distribution")
                axes[1, 1].set_xlabel("Hit Rate (%)")
                axes[1, 1].set_ylabel("Frequency")
                axes[1, 1].grid(True, alpha=0.3, axis="y")

            plt.tight_layout()

        # Start animation
        ani = animation.FuncAnimation(
            fig, animate, interval=update_interval * 1000, cache_frame_data=False
        )

        plt.show()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Real-time MQL5 cache monitoring")

    parser.add_argument(
        "--cache-stats",
        type=str,
        required=True,
        help="Path to cache statistics file (JSON)",
    )

    parser.add_argument(
        "--interval", type=int, default=1, help="Update interval in seconds"
    )

    parser.add_argument(
        "--history",
        type=int,
        default=100,
        help="Number of data points to keep in history",
    )

    args = parser.parse_args()

    # Create monitor
    monitor = CacheMonitor(Path(args.cache_stats), history_size=args.history)

    # Start monitoring
    print(f"Starting cache monitor (updating every {args.interval}s)...")
    print(f"Reading from: {args.cache_stats}")
    print("Press Ctrl+C to stop")

    try:
        monitor.start_monitoring(update_interval=args.interval)
    except KeyboardInterrupt:
        print("\nMonitoring stopped")


if __name__ == "__main__":
    main()
