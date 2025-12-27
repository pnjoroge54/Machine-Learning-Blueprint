import hashlib
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import List, Union

from .model_export import export_model_to_onnx


class ConfigPathGenerator:
    """
    Generates filenames and directory structures based on configuration parameters.
    Creates human-readable, navigable paths for model storage and analysis.
    """

    def __init__(self, base_dir: str = "Models"):
        """
        Initialize the path generator.

        Parameters
        ----------
        base_dir : str, optional
            Base directory for all models (default: Path.home()/"Models").
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def sanitize_filename(self, text: str) -> str:
        """
        Sanitize text to be safe for filenames.

        Parameters
        ----------
        text : str
            Text to sanitize.

        Returns
        -------
        str
            Sanitized filename-safe string.
        """
        # Replace problematic characters
        replacements = {
            "/": "_",
            "\\": "_",
            ":": "-",
            "*": "",
            "?": "",
            '"': "",
            "<": "",
            ">": "",
            "|": "",
            " ": "_",
            ".": "_",
        }

        result = str(text)
        for old, new in replacements.items():
            result = result.replace(old, new)

        # Limit length
        if len(result) > 100:
            result = result[:100]

        return result

    def create_config_hash(self, config: dict) -> str:
        """
        Create a short hash from configuration for unique identification.

        Parameters
        ----------
        config : dict
            Configuration dictionary.

        Returns
        -------
        str
            8-character hash string.
        """
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def format_date_range(self, start_date: str, end_date: str) -> str:
        """
        Format date range for directory names.

        Parameters
        ----------
        start_date : str
            Start date in 'YYYY-MM-DD' format.
        end_date : str
            End date in 'YYYY-MM-DD' format.

        Returns
        -------
        str
            Formatted date range string.
        """
        # Convert to YYYYMMDD format
        start_clean = start_date.replace("-", "")
        end_clean = end_date.replace("-", "")
        return f"{start_clean}_{end_clean}"

    def create_directory_structure(self, config: dict) -> Path:
        """
        Create directory structure based on configuration.

        Parameters
        ----------
        config : dict
            Configuration dictionary. Expected keys:
            - strategy : str (strategy name)
            - symbol : str (trading symbol)
            - bar_type : str (bar type)
            - bar_size : str or int (bar size)
            - training_start : str (start date)
            - training_end : str (end date)
            - [optional] account_name : str
            - [optional] price : str
            - [optional] target_lookback : int
            - [optional] profit_target : float
            - [optional] stop_loss : float

        Returns
        -------
        Path
            Path object for the created directory.
        """
        # Extract key parameters
        strategy = self.sanitize_filename(config.get("strategy", "UnknownStrategy"))
        symbol = self.sanitize_filename(config.get("symbol", "UnknownSymbol")).upper()
        bar_type = self.sanitize_filename(config.get("bar_type", "UnknownBarType"))
        bar_size = self.sanitize_filename(str(config.get("bar_size", "UnknownSize")))
        account_name = self.sanitize_filename(config.get("account_name", "default"))

        # Create date range string
        date_range = self.format_date_range(
            config.get("training_start", "UnknownStart"),
            config.get("training_end", "UnknownEnd"),
        )

        # Create config hash for uniqueness
        config_hash = self.create_config_hash(config)

        # Build directory path
        dir_path = (
            self.base_dir
            / strategy
            / symbol
            / account_name
            / bar_type
            / bar_size
            / date_range
            / config_hash
        )

        # Create directory
        dir_path.mkdir(parents=True, exist_ok=True)

        return dir_path

    def generate_filename(
        self,
        config: dict,
        file_type: str,
        include_timestamp: bool = True,
        include_config_summary: bool = True,
    ) -> str:
        """
        Generate descriptive filename based on configuration.

        Parameters
        ----------
        config : dict
            Configuration dictionary.
        file_type : str
            Type of file (e.g., 'model', 'features', 'events', 'metrics', 'config').
        include_timestamp : bool, optional
            Include timestamp in filename (default: True).
        include_config_summary : bool, optional
            Include config summary in filename (default: True).

        Returns
        -------
        str
            Generated filename.
        """
        # Extract key parameters
        strategy = self.sanitize_filename(config.get("strategy", "UnknownStrategy"))
        symbol = self.sanitize_filename(config.get("symbol", "UnknownSymbol")).upper()
        bar_type = self.sanitize_filename(config.get("bar_type", "UnknownBarType"))
        bar_size = self.sanitize_filename(str(config.get("bar_size", "UnknownSize")))

        # Create config summary if requested
        if include_config_summary:
            # Include key parameters in filename
            summary_parts = [
                f"sym-{symbol}",
                f"bar-{bar_type}-{bar_size}",
            ]

            # Add optional parameters if they exist
            optional_params = ["price", "target_lookback", "profit_target", "stop_loss"]
            for param in optional_params:
                if param in config:
                    value = self.sanitize_filename(str(config[param]))
                    summary_parts.append(f"{param}-{value}")

            summary = "_".join(summary_parts)
        else:
            summary = f"{strategy}_{symbol}"

        # Add timestamp if requested
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{file_type}_{summary}_{timestamp}"
        else:
            filename = f"{file_type}_{summary}"

        # Add appropriate extension
        extensions = {
            "model": ".pkl",
            "feature_config": ".pkl",
            "feature_names": ".pkl",
            "features": ".parquet",
            "events": ".parquet",
            "metrics": ".json",
            "config": ".json",
            "feature_importance": ".csv",
            "weights": ".parquet",
            "plot": ".png",
            "report": ".html",
            "log": ".log",
            "strategy": ".pkl",
        }

        extension = extensions.get(file_type, ".dat")
        return filename + extension

    def create_model_filename(self, config: dict, model_type: str = "rf") -> str:
        """
        Create filename for model files.

        Parameters
        ----------
        config : dict
            Configuration dictionary.
        model_type : str, optional
            Type of model (default: "rf" for RandomForest).

        Returns
        -------
        str
            Model filename.
        """
        # Create comprehensive model filename
        symbol = self.sanitize_filename(config.get("symbol", "UnknownSymbol")).upper()
        strategy = self.sanitize_filename(config.get("strategy", "UnknownStrategy"))
        bar_type = self.sanitize_filename(config.get("bar_type", "UnknownBarType"))
        bar_size = self.sanitize_filename(str(config.get("bar_size", "UnknownSize")))

        # Date range
        date_range = self.format_date_range(
            config.get("training_start", "UnknownStart"),
            config.get("training_end", "UnknownEnd"),
        )

        # Optional parameters
        param_parts = []
        optional_params = ["profit_target", "stop_loss", "target_lookback"]
        for param in optional_params:
            if param in config:
                value = self.sanitize_filename(str(config[param]))
                param_parts.append(f"{param[0:2]}-{value}")

        params_str = "_".join(param_parts) if param_parts else "default"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"{model_type}_{strategy}_{symbol}_{bar_type}_{bar_size}_{date_range}_{params_str}_{timestamp}.joblib"

        return filename

    def create_summary_filename(
        self, config: dict, analysis_type: str = "summary"
    ) -> str:
        """
        Create filename for summary/analysis files.

        Parameters
        ----------
        config : dict
            Configuration dictionary.
        analysis_type : str, optional
            Type of analysis (default: "summary").

        Returns
        -------
        str
            Summary filename.
        """
        symbol = self.sanitize_filename(config.get("symbol", "UnknownSymbol")).upper()
        bar_type = self.sanitize_filename(config.get("bar_type", "UnknownBarType"))
        bar_size = self.sanitize_filename(str(config.get("bar_size", "UnknownSize")))

        date_range = self.format_date_range(
            config.get("training_start", "UnknownStart"),
            config.get("training_end", "UnknownEnd"),
        )

        timestamp = datetime.now().strftime("%Y%m%d")

        return f"{analysis_type}_{symbol}_{bar_type}_{bar_size}_{date_range}_{timestamp}.html"

    def get_standard_file_paths(self, config: dict) -> dict:
        """
        Get standard file paths for all model artifacts.

        Parameters
        ----------
        config : dict
            Configuration dictionary.

        Returns
        -------
        dict
            Dictionary with standard file paths.
        """
        # Create directory structure
        base_dir = self.create_directory_structure(config)

        # Generate filenames
        model_filename = self.create_model_filename(config)
        model_filename_onxx = model_filename.replace(".joblib", ".onnx")
        config_filename = self.generate_filename(
            config, "config", include_timestamp=False
        )
        metrics_filename = self.generate_filename(config, "metrics")
        features_filename = self.generate_filename(config, "features")
        events_filename = self.generate_filename(config, "events")
        feature_importance_filename = self.generate_filename(
            config, "feature_importance"
        )
        weights_filename = self.generate_filename(config, "weights")
        strategy_filename = self.generate_filename(config, "strategy")
        feature_config_filename = self.generate_filename(config, "feature_config")
        feature_names_filename = self.generate_filename(config, "feature_names")

        return {
            "base_dir": base_dir,
            "model": base_dir / model_filename,
            "model_onxx": base_dir / model_filename_onxx,
            "config": base_dir / config_filename,
            "metrics": base_dir / metrics_filename,
            "events": base_dir / events_filename,
            "feature_names": base_dir / feature_names_filename,
            "feature_config": base_dir / feature_config_filename,
            "feature_importance": base_dir / feature_importance_filename,
            "features": base_dir / features_filename,
            "weights": base_dir / weights_filename,
            "strategy": base_dir / strategy_filename,
            "logs": base_dir / "logs",
            "plots": base_dir / "plots",
            "reports": base_dir / "reports",
        }

    def create_navigation_index(self, config: dict, file_paths: dict = None) -> str:
        """
        Create HTML navigation index for easy browsing of model artifacts.

        Parameters
        ----------
        config : dict
            Configuration dictionary.
        file_paths : dict, optional
            Dictionary of file paths.

        Returns
        -------
        str
            HTML index content.
        """
        if file_paths is None:
            file_paths = self.get_standard_file_paths(config)

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Artifacts - {config.get("symbol", "Unknown")}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
                .config {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .files {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
                .file-item {{ margin: 10px 0; padding: 10px; border-left: 4px solid #007bff; }}
                h1 {{ color: #333; }}
                h2 {{ color: #555; }}
                pre {{ background-color: #f8f9fa; padding: 10px; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Model Artifacts</h1>
                <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>

            <div class="config">
                <h2>Configuration</h2>
                <pre>{json.dumps(config, indent=2, default=str)}</pre>
            </div>

            <div class="files">
                <h2>Files</h2>
        """

        # Add file links
        for file_type, file_path in file_paths.items():
            if isinstance(file_path, Path):
                if file_path.is_dir():
                    html += f'<div class="file-item"><strong>{file_type}:</strong> {file_path.name}/ (directory)</div>'
                else:
                    html += f'<div class="file-item"><strong>{file_type}:</strong> <a href="{file_path.name}">{file_path.name}</a></div>'

        html += """
            </div>
        </body>
        </html>
        """

        # Save HTML index
        index_path = file_paths["base_dir"] / "index.html"
        index_path.write_text(html)

        return html


class ModelFileManager:
    """
    Manages file operations for model development with organized structure.
    """

    def __init__(self, base_dir: str = "Models"):
        """
        Initialize file manager.

        Parameters
        ----------
        base_dir : str, optional
            Base directory for all models (default: "Models").
        """
        self.path_generator = ConfigPathGenerator(base_dir)
        self.current_paths = None

    def setup_model_directory(self, config: dict) -> dict:
        """
        Set up directory structure for a model.

        Parameters
        ----------
        config : dict
            Configuration dictionary.

        Returns
        -------
        dict
            Dictionary of file paths.
        """
        self.current_paths = self.path_generator.get_standard_file_paths(config)

        # Create subdirectories
        for subdir in ["logs", "plots", "reports"]:
            self.current_paths[subdir].mkdir(exist_ok=True)

        # Save configuration
        self.save_config(config)

        # Create navigation index
        self.path_generator.create_navigation_index(config, self.current_paths)

        return self.current_paths

    def save_config(self, config: dict):
        """Save configuration to file."""
        if self.current_paths:
            config_path = self.current_paths["config"]
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2, default=str)

    def save_model(self, model, metadata: dict = None):
        """Save model with metadata."""
        if self.current_paths:
            import joblib

            save_data = {
                "model": model,
                "metadata": metadata or {},
                "save_timestamp": datetime.now().isoformat(),
                "config_path": str(self.current_paths["config"]),
            }

            joblib.dump(save_data, self.current_paths["model"])

    def save_model_as_onxx(self, model, feature_names: List[str], metadata: dict = None):
        """Export model to ONNX format."""
        if self.current_paths:
            export_model_to_onnx(model, feature_names, self.current_paths["model_onxx"], metadata)

    def save_object(self, object, name: str):
        """Save objects to file."""
        if self.current_paths and name in self.current_paths:
            with open(self.current_paths[name], "wb") as f:
                pickle.dump(object, f)

    def save_metrics(self, metrics: dict):
        """Save metrics to file."""
        if self.current_paths:
            with open(self.current_paths["metrics"], "w") as f:
                json.dump(metrics, f, indent=2, default=str)

    def save_dataframe(self, df, name: str):
        """Save DataFrame to appropriate format."""
        if self.current_paths and name in self.current_paths:
            df.to_parquet(self.current_paths[name], engine="pyarrow", compression="zstd")

    def get_model_info(self, model_path: Union[str, Path]) -> dict:
        """
        Get information about a saved model.

        Parameters
        ----------
        model_path : str | Path
            Path to model file.

        Returns
        -------
        dict
            Model information.
        """
        # Extract info from filename and directory structure
        model_path = Path(model_path)
        parts = model_path.parts

        info = {
            "file_path": str(model_path),
            "file_name": model_path.name,
            "strategy": parts[-8] if len(parts) >= 8 else "Unknown",
            "symbol": parts[-7] if len(parts) >= 7 else "Unknown",
            "account": parts[-6] if len(parts) >= 6 else "Unknown",
            "bar_type": parts[-5] if len(parts) >= 5 else "Unknown",
            "bar_size": parts[-4] if len(parts) >= 4 else "Unknown",
            "date_range": parts[-3] if len(parts) >= 3 else "Unknown",
            "config_hash": parts[-2] if len(parts) >= 2 else "Unknown",
        }

        return info

    def find_models(self, search_criteria: dict = None, base_dir: str = None) -> list:
        """
        Find models matching search criteria.

        Parameters
        ----------
        search_criteria : dict, optional
            Dictionary of search criteria.
        base_dir : str, optional
            Base directory to search (default: configured base_dir).

        Returns
        -------
        list
            List of matching model files with their info.
        """
        if base_dir is None:
            base_dir = self.path_generator.base_dir

        search_dir = Path(base_dir)
        model_files = list(search_dir.rglob("*.joblib"))

        results = []
        for model_file in model_files:
            info = self.get_model_info(model_file)

            # Apply search criteria if provided
            if search_criteria:
                match = True
                for key, value in search_criteria.items():
                    if key in info and info[key] != value:
                        match = False
                        break
                if not match:
                    continue

            results.append(info)

        return results

    def load_artifacts(self, search_criteria: dict = None, base_dir: str = None) -> dict:
        import json
        import os

        import joblib
        import pandas as pd

        data = self.find_models(search_criteria, base_dir)
        result = {}

        for d in data:
            model_data = {}
            fname = Path(d["file_path"])
            folder = fname.parent

            model_dict = joblib.load(fname)
            for k, v in model_dict.items():
                model_data[k] = v

            for fp in os.listdir(folder):
                f = str(folder / fp)
                key = (
                    fp.split("_")[0]
                    if not fp.startswith(("feature_"))
                    else "_".join(fp.split("_")[:2])
                )
                if key not in model_data:
                    if f.endswith("csv"):
                        model_data[key] = pd.read_csv(f)
                    elif f.endswith("parquet"):
                        model_data[key] = pd.read_parquet(f)
                    elif f.endswith(("json", "pkl")):
                        with open(f, "rb") as g:
                            if f.endswith("json"):
                                model_data[key] = json.load(g)
                            elif f.endswith("pkl"):
                                model_data[key] = pickle.load(g)

            barriers = f'{d["date_range"]}' + "_".join(fname.name.split(d["date_range"])[1].split("_")[:-2])
            result.setdefault(barriers, {})
            result[barriers].setdefault(d["bar_size"], {})
            result[barriers][d["bar_size"]][d["bar_type"]] = model_data

        return result
