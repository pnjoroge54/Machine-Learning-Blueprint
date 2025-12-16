import hashlib
import json
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Dict

import structlog

# Create context variables for tracking
_current_pipeline_id = ContextVar("pipeline_id", default=None)
_current_step = ContextVar("current_step", default=None)


class ModelDevelopmentLogger:
    """Structured logging for model development pipeline"""

    def __init__(self, name: str = "model_dev"):
        self.logger = structlog.get_logger(name)
        self.metrics_buffer = []

    def log_step_start(self, step_name: str, step_data: Dict = None):
        """Log the start of a pipeline step"""
        log_data = {"step": step_name, "status": "started"}
        if step_data:
            log_data.update(step_data)

        self.logger.info("pipeline_step", **log_data)
        _current_step.set(step_name)

    def log_step_complete(
        self, step_name: str, metrics: Dict = None, duration: float = None
    ):
        """Log the completion of a pipeline step"""
        log_data = {"step": step_name, "status": "completed"}
        if metrics:
            log_data["metrics"] = metrics
        if duration:
            log_data["duration_seconds"] = duration

        self.logger.info("pipeline_step", **log_data)

    def log_hyperparameter_search(
        self, search_results: Dict, best_params: Dict, search_space: Dict
    ):
        """Log hyperparameter search results"""
        self.logger.info(
            "hyperparameter_search",
            n_iterations=search_results.get("n_iter", 0),
            best_score=search_results.get("best_score", 0),
            best_params=best_params,
            param_distribution_size=len(search_space),
            cv_folds=search_results.get("cv", 5),
        )

    def log_model_metrics(self, metrics: Dict, model_info: Dict):
        """Log model performance metrics"""
        self.logger.info(
            "model_performance",
            cv_score=metrics.get("cv_score", 0),
            feature_count=metrics.get("feature_count", 0),
            training_samples=metrics.get("training_samples", 0),
            model_type=model_info.get("model_type", "unknown"),
            strategy=model_info.get("strategy", "unknown"),
        )

    def save_logs_to_file(self, file_path: Path):
        """Save logs to JSONL file for analysis"""
        # This requires structlog configuration to output JSON
        pass


class ModelDevelopmentCache:
    """
    Cache for model development pipeline with dictionary-based keys.
    """

    def __init__(self):
        self._pipelines = {}  # config key -> ModelDevelopmentPipeline
        self._results = {}  # config key -> results

    @staticmethod
    def create_config_key(base_config, param_grid):
        """
        Create a hashable key from configuration.

        Parameters
        ----------
        base_config : dict
            Base configuration dictionary
        param_grid : dict
            Parameter grid with lists of values

        Returns
        -------
        tuple
            Hashable key
        """

        def normalize_value(v):
            """Normalize values for hashing."""
            if isinstance(v, (list, tuple)):
                return tuple(normalize_value(x) for x in v)
            elif isinstance(v, dict):
                return tuple(sorted((k, normalize_value(v2)) for k, v2 in v.items()))
            elif hasattr(v, "__dict__"):
                # For objects, use class name and string representation
                return (type(v).__name__, str(v))
            else:
                return v

        # Normalize both dictionaries
        normalized_base = normalize_value(base_config)
        normalized_grid = normalize_value(param_grid)

        # Create tuple key
        return (normalized_base, normalized_grid)

    def store_pipeline(self, base_config, param_grid, pipeline):
        """Store a pipeline in cache."""
        key = self.create_config_key(base_config, param_grid)
        self._pipelines[key] = pipeline

    def get_pipeline(self, base_config, param_grid):
        """Retrieve pipeline from cache."""
        key = self.create_config_key(base_config, param_grid)
        return self._pipelines.get(key)

    def store_results(self, base_config, param_grid, results):
        """Store results in cache."""
        key = self.create_config_key(base_config, param_grid)
        self._results[key] = results

    def get_results(self, base_config, param_grid):
        """Retrieve results from cache."""
        key = self.create_config_key(base_config, param_grid)
        return self._results.get(key)

    def find_similar_configs(self, base_config, param_grid, threshold=0.8):
        """
        Find configurations similar to the given one.
        Useful for parameter analysis.
        """
        from difflib import SequenceMatcher

        current_key_str = str(self.create_config_key(base_config, param_grid))
        similar = []

        for key in self._pipelines.keys():
            key_str = str(key)
            similarity = SequenceMatcher(None, current_key_str, key_str).ratio()
            if similarity >= threshold:
                similar.append((key, similarity, self._pipelines[key]))

        return sorted(similar, key=lambda x: x[1], reverse=True)


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
            "features": ".parquet",
            "events": ".parquet",
            "metrics": ".json",
            "config": ".json",
            "feature_importance": ".csv",
            "weights": ".parquet",
            "plot": ".png",
            "report": ".html",
            "log": ".log",
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

        filename = f"{model_type}_{strategy}_{symbol}_{bar_type}_{bar_size}_{date_range}_{params_str}_{timestamp}.pkl"

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

        return {
            "base_dir": base_dir,
            "model": base_dir / model_filename,
            "config": base_dir / config_filename,
            "metrics": base_dir / metrics_filename,
            "features": base_dir / features_filename,
            "events": base_dir / events_filename,
            "feature_importance": base_dir / feature_importance_filename,
            "weights": base_dir / weights_filename,
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

    def save_metrics(self, metrics: dict):
        """Save metrics to file."""
        if self.current_paths:
            import json

            with open(self.current_paths["metrics"], "w") as f:
                json.dump(metrics, f, indent=2, default=str)

    def save_dataframe(self, df, name: str):
        """Save DataFrame to appropriate format."""
        if self.current_paths and name in self.current_paths:
            df.to_parquet(self.current_paths[name])

    def get_model_info(self, model_path: Path) -> dict:
        """
        Get information about a saved model.

        Parameters
        ----------
        model_path : Path
            Path to model file.

        Returns
        -------
        dict
            Model information.
        """
        # Extract info from filename and directory structure
        parts = model_path.parts

        info = {
            "file_path": str(model_path),
            "file_name": model_path.name,
            "strategy": parts[-7] if len(parts) >= 7 else "Unknown",
            "symbol": parts[-6] if len(parts) >= 6 else "Unknown",
            "account": parts[-5] if len(parts) >= 5 else "Unknown",
            "bar_type": parts[-4] if len(parts) >= 4 else "Unknown",
            "bar_size": parts[-3] if len(parts) >= 3 else "Unknown",
            "date_range": parts[-2] if len(parts) >= 2 else "Unknown",
            "config_hash": parts[-1] if len(parts) >= 1 else "Unknown",
        }

        # Parse filename for more details
        filename_parts = model_path.stem.split("_")
        if len(filename_parts) >= 6:
            info.update(
                {
                    "model_type": filename_parts[0],
                    "strategy_from_file": filename_parts[1],
                    "symbol_from_file": filename_parts[2],
                    "bar_type_from_file": filename_parts[3],
                    "bar_size_from_file": filename_parts[4],
                    "date_range_from_file": filename_parts[5],
                }
            )

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
        model_files = list(search_dir.rglob("*.pkl"))

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
