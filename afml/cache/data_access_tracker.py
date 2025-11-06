"""
Data Access Tracker for preventing test set contamination.
Logs every dataset access with temporal boundaries to detect data snooping.
"""

import inspect
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger


class DataAccessTracker:
    """
    Track every data access to detect test set contamination.

    Critical for detecting data snooping bias in ML development where
    developers unknowingly test on the same data repeatedly during
    iterative optimization.
    """

    def __init__(self, log_file: Optional[Path] = None):
        """
        Initialize data access tracker.

        Args:
            log_file: Path to access log CSV (None = use default in cache dir)
        """
        # Import at runtime to avoid circular imports
        from . import CACHE_DIRS

        self.log_file = log_file or CACHE_DIRS["base"] / "data_access_log.csv"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        self.access_log: List[Dict] = []
        self._load_existing_log()

    def _load_existing_log(self):
        """Load existing access log if it exists."""
        if self.log_file.exists():
            try:
                df = pd.read_csv(self.log_file)
                self.access_log = df.to_dict("records")
                logger.debug(f"Loaded {len(self.access_log)} existing access records")
            except Exception as e:
                logger.warning(f"Failed to load existing access log: {e}")
                self.access_log = []

    def log_access(
        self,
        dataset_name: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        purpose: str,
        data_shape: Optional[Tuple[int, int]] = None,
    ):
        """
        Log a dataset access with metadata.

        Args:
            dataset_name: Unique identifier for the dataset
            start_date: Start of temporal range
            end_date: End of temporal range
            purpose: One of 'train', 'test', 'validate', 'optimize', 'analyze'
            data_shape: Optional shape tuple (rows, cols)
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "dataset": dataset_name,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "purpose": purpose,
            "data_shape": str(data_shape) if data_shape else None,
            "caller": self._get_caller_info(),
        }

        self.access_log.append(entry)
        logger.debug(f"Logged access: {dataset_name} [{start_date} to {end_date}] for {purpose}")

    def save_log(self):
        """Persist access log to disk."""
        try:
            df = pd.DataFrame(self.access_log)
            df.to_csv(self.log_file, index=False)
            logger.info(f"Saved {len(self.access_log)} access records to {self.log_file}")
        except Exception as e:
            logger.error(f"Failed to save access log: {e}")

    def analyze_contamination(
        self, dataset_name: str, exclude_purposes: Optional[List[str]] = None
    ) -> Tuple[int, str, List[Dict]]:
        """
        Analyze how many times a dataset was accessed.

        Args:
            dataset_name: Dataset to analyze
            exclude_purposes: Purposes to exclude (e.g., ['validate'] for final check)

        Returns:
            Tuple of (access_count, warning_level, access_details)

        Warning Levels:
            - CLEAN: 0 accesses
            - ACCEPTABLE: 1-2 accesses (normal development)
            - WARNING: 3-10 accesses (risky, be careful)
            - CONTAMINATED: >10 accesses (severely compromised)
        """
        exclude_purposes = exclude_purposes or []

        matching_accesses = [
            entry
            for entry in self.access_log
            if entry["dataset"] == dataset_name and entry["purpose"] not in exclude_purposes
        ]

        access_count = len(matching_accesses)

        # Determine warning level
        if access_count == 0:
            warning_level = "CLEAN"
        elif access_count <= 2:
            warning_level = "ACCEPTABLE"
        elif access_count <= 10:
            warning_level = "WARNING"
        else:
            warning_level = "CONTAMINATED"

        return access_count, warning_level, matching_accesses

    def get_contamination_report(self) -> pd.DataFrame:
        """
        Generate comprehensive contamination report for all datasets.

        Returns:
            DataFrame with contamination analysis for each dataset
        """
        if not self.access_log:
            return pd.DataFrame()

        # Group by dataset
        df = pd.DataFrame(self.access_log)

        report_data = []
        for dataset_name in df["dataset"].unique():
            access_count, warning_level, accesses = self.analyze_contamination(dataset_name)

            purposes = df[df["dataset"] == dataset_name]["purpose"].value_counts().to_dict()

            report_data.append(
                {
                    "dataset": dataset_name,
                    "total_accesses": access_count,
                    "warning_level": warning_level,
                    "train_accesses": purposes.get("train", 0),
                    "test_accesses": purposes.get("test", 0),
                    "validate_accesses": purposes.get("validate", 0),
                    "optimize_accesses": purposes.get("optimize", 0),
                    "first_access": df[df["dataset"] == dataset_name]["timestamp"].min(),
                    "last_access": df[df["dataset"] == dataset_name]["timestamp"].max(),
                }
            )

        return pd.DataFrame(report_data)

    def print_contamination_report(self):
        """Print formatted contamination report to console."""
        report = self.get_contamination_report()

        if report.empty:
            print("No data accesses logged yet.")
            return

        print("\n" + "=" * 80)
        print("DATA CONTAMINATION REPORT")
        print("=" * 80)

        for _, row in report.iterrows():
            print(f"\nDataset: {row['dataset']}")
            print(f"  Warning Level: {row['warning_level']}")
            print(f"  Total Accesses: {row['total_accesses']}")
            print(f"  Breakdown:")
            print(f"    - Train: {row['train_accesses']}")
            print(f"    - Test: {row['test_accesses']} {'⚠️' if row['test_accesses'] > 2 else ''}")
            print(
                f"    - Validate: {row['validate_accesses']} {'⚠️' if row['validate_accesses'] > 2 else ''}"
            )
            print(f"    - Optimize: {row['optimize_accesses']}")
            print(f"  First Access: {row['first_access']}")
            print(f"  Last Access: {row['last_access']}")

            if row["warning_level"] in ["WARNING", "CONTAMINATED"]:
                print(f"\n  ⚠️  WARNING: This dataset may be contaminated!")
                print(f"  ⚠️  Test/validation results on this data are unreliable.")

        print("\n" + "=" * 80)

        # Summary recommendations
        contaminated = report[report["warning_level"].isin(["WARNING", "CONTAMINATED"])]
        if len(contaminated) > 0:
            print("\nRECOMMENDATIONS:")
            print("  1. Use truly held-out validation set for honest assessment")
            print("  2. Document all contaminated datasets in your results")
            print("  3. Consider collecting fresh validation data")
            print("  4. Apply multiple testing corrections (e.g., Bonferroni)")
        else:
            print("\n✓ No contamination detected - data hygiene looks good!")

        print("=" * 80 + "\n")

    def export_detailed_log(self, output_file: Path):
        """
        Export detailed access log with analysis.

        Args:
            output_file: Path for output JSON file
        """
        report_data = {
            "summary": self.get_contamination_report().to_dict("records"),
            "detailed_log": self.access_log,
            "generated_at": datetime.now().isoformat(),
            "total_accesses": len(self.access_log),
        }

        with open(output_file, "w") as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"Exported detailed access log to {output_file}")

    def _get_caller_info(self) -> str:
        """Get caller function/file for audit trail."""
        try:
            # Walk up the stack to find the calling function (skip tracker internals)
            frame = inspect.currentframe()
            for _ in range(3):  # Skip this method and decorator layers
                if frame is not None:
                    frame = frame.f_back

            if frame is not None:
                filename = Path(frame.f_code.co_filename).name
                lineno = frame.f_lineno
                funcname = frame.f_code.co_name
                return f"{filename}:{funcname}:{lineno}"
        except Exception:
            pass

        return "unknown"

    def clear_log(self):
        """Clear all access records (use with caution!)."""
        self.access_log = []
        if self.log_file.exists():
            self.log_file.unlink()
        logger.warning("Cleared all data access logs")


# =============================================================================
# Global instance and convenience functions
# =============================================================================

_global_tracker: Optional[DataAccessTracker] = None


def get_data_tracker() -> DataAccessTracker:
    """Get global data access tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = DataAccessTracker()
    return _global_tracker


def log_data_access(
    dataset_name: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    purpose: str,
    data_shape: Optional[Tuple[int, int]] = None,
):
    """Convenience function to log data access."""
    tracker = get_data_tracker()
    tracker.log_access(dataset_name, start_date, end_date, purpose, data_shape)


def print_contamination_report():
    """Convenience function to print contamination report."""
    tracker = get_data_tracker()
    tracker.print_contamination_report()


def save_access_log():
    """Convenience function to save access log."""
    tracker = get_data_tracker()
    tracker.save_log()


__all__ = [
    "DataAccessTracker",
    "get_data_tracker",
    "log_data_access",
    "print_contamination_report",
    "save_access_log",
]
