"""
ML Bridge Server for MQL5 Integration
Receives features from MQL5, runs ML inference, returns predictions.

Usage:
    python ml_bridge_server.py --model random_forest --port 80

Features:
    - Socket-based communication (low latency)
    - Model hot-reloading
    - Request/response logging
    - Multiple model support
    - Graceful error handling
"""

import argparse
import json
import pickle
import socket
import struct
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

# =============================================================================
# Configuration
# =============================================================================


class ServerConfig:
    """Server configuration with sensible defaults."""

    def __init__(self):
        self.host = "127.0.0.1"
        self.port = 80
        self.buffer_size = 4096
        self.model_path = Path("models")
        self.log_path = Path("logs")
        self.model_type = "random_forest"  # or "xgboost", "tensorflow", "pytorch"
        self.enable_caching = True
        self.cache_size = 1000

        # Create directories
        self.model_path.mkdir(exist_ok=True)
        self.log_path.mkdir(exist_ok=True)


# =============================================================================
# Model Wrapper Classes
# =============================================================================


class BaseModelWrapper:
    """Base class for model wrappers."""

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model = None
        self.feature_names = None
        self.model_version = "unknown"
        self.load_time = None

    def load_model(self):
        """Load model from disk."""
        raise NotImplementedError

    def predict(self, features: np.ndarray) -> Tuple[float, str, float]:
        """
        Make prediction.

        Returns:
            (prediction_score, predicted_class, confidence)
        """
        raise NotImplementedError


class RandomForestWrapper(BaseModelWrapper):
    """Wrapper for sklearn RandomForest models."""

    def load_model(self):
        """Load sklearn model."""
        start_time = time.time()

        try:
            with open(self.model_path / "model.pkl", "rb") as f:
                self.model = pickle.load(f)

            # Load metadata if available
            metadata_path = self.model_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    self.feature_names = metadata.get("feature_names", [])
                    self.model_version = metadata.get("version", "1.0")

            self.load_time = time.time() - start_time

            logger.info(
                f"Loaded RandomForest model: {self.model_version} "
                f"({self.load_time:.3f}s)"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def predict(self, features: np.ndarray) -> Tuple[float, str, float]:
        """Make prediction with RandomForest."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Reshape if needed
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Get prediction probabilities
        proba = self.model.predict_proba(features)[0]

        # Get class prediction
        predicted_class_idx = np.argmax(proba)
        confidence = proba[predicted_class_idx]

        # Convert to BUY/SELL signal
        # Assuming binary classification: 0=SELL, 1=BUY
        if predicted_class_idx == 1:
            prediction_score = confidence
            predicted_class = "BUY"
        else:
            prediction_score = -confidence
            predicted_class = "SELL"

        return prediction_score, predicted_class, confidence


class XGBoostWrapper(BaseModelWrapper):
    """Wrapper for XGBoost models."""

    def load_model(self):
        """Load XGBoost model."""
        start_time = time.time()

        try:
            import xgboost as xgb

            self.model = xgb.Booster()
            self.model.load_model(str(self.model_path / "model.xgb"))

            # Load metadata
            metadata_path = self.model_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    self.feature_names = metadata.get("feature_names", [])
                    self.model_version = metadata.get("version", "1.0")

            self.load_time = time.time() - start_time

            logger.info(
                f"Loaded XGBoost model: {self.model_version} ({self.load_time:.3f}s)"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to load XGBoost model: {e}")
            return False

    def predict(self, features: np.ndarray) -> Tuple[float, str, float]:
        """Make prediction with XGBoost."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        import xgboost as xgb

        # Reshape if needed
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Create DMatrix
        dmatrix = xgb.DMatrix(features, feature_names=self.feature_names)

        # Get prediction
        prediction = self.model.predict(dmatrix)[0]

        # Convert to signal
        if prediction > 0.5:
            prediction_score = prediction
            predicted_class = "BUY"
            confidence = prediction
        else:
            prediction_score = -(1 - prediction)
            predicted_class = "SELL"
            confidence = 1 - prediction

        return prediction_score, predicted_class, confidence


class TensorFlowWrapper(BaseModelWrapper):
    """Wrapper for TensorFlow/Keras models."""

    def load_model(self):
        """Load TensorFlow model."""
        start_time = time.time()

        try:
            import tensorflow as tf

            self.model = tf.keras.models.load_model(str(self.model_path / "model.h5"))

            # Load metadata
            metadata_path = self.model_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    self.feature_names = metadata.get("feature_names", [])
                    self.model_version = metadata.get("version", "1.0")

            self.load_time = time.time() - start_time

            logger.info(
                f"Loaded TensorFlow model: {self.model_version} ({self.load_time:.3f}s)"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to load TensorFlow model: {e}")
            return False

    def predict(self, features: np.ndarray) -> Tuple[float, str, float]:
        """Make prediction with TensorFlow."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Reshape if needed
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Get prediction
        prediction = self.model.predict(features, verbose=0)[0]

        # Handle different output formats
        if len(prediction) == 1:
            # Single output (binary classification)
            prob = float(prediction[0])
            if prob > 0.5:
                prediction_score = prob
                predicted_class = "BUY"
                confidence = prob
            else:
                prediction_score = -(1 - prob)
                predicted_class = "SELL"
                confidence = 1 - prob
        else:
            # Multi-class output
            predicted_class_idx = np.argmax(prediction)
            confidence = float(prediction[predicted_class_idx])

            if predicted_class_idx == 1:
                prediction_score = confidence
                predicted_class = "BUY"
            else:
                prediction_score = -confidence
                predicted_class = "SELL"

        return prediction_score, predicted_class, confidence


# =============================================================================
# Prediction Cache
# =============================================================================


class PredictionCache:
    """Simple LRU cache for predictions."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, Tuple[float, str, float]] = {}
        self.access_order: List[str] = []
        self.hits = 0
        self.misses = 0

    def _generate_key(self, features: np.ndarray) -> str:
        """Generate cache key from features."""
        # Use hash of first/middle/last features for speed
        if len(features) > 0:
            key_parts = [
                str(features[0]),
                str(features[len(features) // 2]) if len(features) > 1 else "",
                str(features[-1]) if len(features) > 1 else "",
                str(len(features)),
            ]
            return "_".join(key_parts)
        return ""

    def get(self, features: np.ndarray) -> Optional[Tuple[float, str, float]]:
        """Get cached prediction."""
        key = self._generate_key(features)

        if key in self.cache:
            self.hits += 1

            # Update access order (move to end)
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)

            return self.cache[key]

        self.misses += 1
        return None

    def set(self, features: np.ndarray, prediction: Tuple[float, str, float]):
        """Cache prediction."""
        key = self._generate_key(features)

        # Evict if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            # Remove least recently used
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]

        self.cache[key] = prediction

        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0

        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self.cache),
            "max_size": self.max_size,
        }


# =============================================================================
# ML Bridge Server
# =============================================================================


class MLBridgeServer:
    """Main server class for ML inference."""

    def __init__(self, config: ServerConfig):
        self.config = config
        self.model_wrapper: Optional[BaseModelWrapper] = None
        self.cache: Optional[PredictionCache] = None
        self.server_socket: Optional[socket.socket] = None
        self.running = False

        # Statistics
        self.requests_received = 0
        self.predictions_made = 0
        self.errors = 0
        self.start_time = time.time()

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging."""
        log_file = self.config.log_path / f"server_{datetime.now():%Y%m%d}.log"

        logger.remove()
        logger.add(
            sys.stderr,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
            level="INFO",
        )
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            level="DEBUG",
            rotation="1 day",
            retention="7 days",
        )

    def initialize(self) -> bool:
        """Initialize server components."""
        logger.info("=" * 70)
        logger.info("ML BRIDGE SERVER INITIALIZATION")
        logger.info("=" * 70)

        # Load model
        if not self._load_model():
            return False

        # Initialize cache
        if self.config.enable_caching:
            self.cache = PredictionCache(max_size=self.config.cache_size)
            logger.info(f"Prediction cache initialized (size={self.config.cache_size})")

        # Create server socket
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.config.host, self.config.port))
            self.server_socket.listen(1)

            logger.info(f"Server listening on {self.config.host}:{self.config.port}")

        except Exception as e:
            logger.error(f"Failed to create server socket: {e}")
            return False

        logger.info("=" * 70)
        logger.info("SERVER READY - Waiting for MQL5 connections...")
        logger.info("=" * 70)

        return True

    def _load_model(self) -> bool:
        """Load ML model based on configuration."""
        logger.info(f"Loading {self.config.model_type} model...")

        try:
            if self.config.model_type == "random_forest":
                self.model_wrapper = RandomForestWrapper(self.config.model_path)
            elif self.config.model_type == "xgboost":
                self.model_wrapper = XGBoostWrapper(self.config.model_path)
            elif self.config.model_type == "tensorflow":
                self.model_wrapper = TensorFlowWrapper(self.config.model_path)
            else:
                logger.error(f"Unknown model type: {self.config.model_type}")
                return False

            return self.model_wrapper.load_model()

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def start(self):
        """Start the server."""
        self.running = True

        try:
            while self.running:
                # Accept connection
                logger.info("Waiting for MQL5 connection...")
                client_socket, address = self.server_socket.accept()
                logger.info(f"✓ Connected: {address}")

                # Handle client
                self._handle_client(client_socket)

        except KeyboardInterrupt:
            logger.info("\nShutdown signal received...")
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            self.stop()

    def _handle_client(self, client_socket: socket.socket):
        """Handle messages from connected MQL5 client."""
        buffer = b""

        try:
            while self.running:
                # Receive data
                data = client_socket.recv(self.config.buffer_size)

                if not data:
                    logger.warning("Client disconnected")
                    break

                buffer += data

                # Process complete messages (length-prefixed)
                while len(buffer) >= 4:
                    # Read message length
                    msg_length = struct.unpack("I", buffer[:4])[0]

                    if len(buffer) < 4 + msg_length:
                        break  # Incomplete message

                    # Extract message
                    msg_data = buffer[4 : 4 + msg_length]
                    buffer = buffer[4 + msg_length :]

                    # Process message
                    response = self._process_message(msg_data)

                    # Send response
                    if response:
                        self._send_response(client_socket, response)

        except Exception as e:
            logger.error(f"Error handling client: {e}")
        finally:
            client_socket.close()

    def _process_message(self, data: bytes) -> Optional[Dict]:
        """Process incoming message from MQL5."""
        try:
            message = json.loads(data.decode("utf-8"))
            self.requests_received += 1

            msg_type = message.get("type")

            if msg_type == "predict":
                return self._handle_prediction_request(message)

            elif msg_type == "heartbeat":
                return {"type": "heartbeat_response", "status": "ok"}

            elif msg_type == "stats":
                return self._get_server_stats()

            else:
                logger.warning(f"Unknown message type: {msg_type}")
                return {"type": "error", "message": "Unknown message type"}

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.errors += 1
            return {"type": "error", "message": str(e)}

    def _handle_prediction_request(self, message: Dict) -> Dict:
        """Handle prediction request from MQL5."""
        try:
            # Extract features
            features = np.array(message.get("features", []), dtype=np.float64)

            if len(features) == 0:
                return {"type": "error", "message": "No features provided"}

            # Check cache
            cached_result = None
            if self.cache:
                cached_result = self.cache.get(features)

            if cached_result:
                prediction_score, predicted_class, confidence = cached_result
                logger.debug(
                    f"Cache HIT: {predicted_class} (score={prediction_score:.4f})"
                )
            else:
                # Make prediction
                start_time = time.time()

                prediction_score, predicted_class, confidence = (
                    self.model_wrapper.predict(features)
                )

                inference_time = (time.time() - start_time) * 1_000_000  # microseconds

                # Cache result
                if self.cache:
                    self.cache.set(
                        features, (prediction_score, predicted_class, confidence)
                    )

                self.predictions_made += 1

                logger.info(
                    f"Prediction: {predicted_class} "
                    f"(score={prediction_score:.4f}, conf={confidence:.4f}, "
                    f"latency={inference_time:.0f}µs)"
                )

            # Build response
            response = {
                "type": "prediction",
                "prediction_score": float(prediction_score),
                "predicted_class": predicted_class,
                "confidence": float(confidence),
                "model_version": self.model_wrapper.model_version,
                "timestamp": datetime.now().isoformat(),
            }

            return response

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            self.errors += 1
            return {"type": "error", "message": str(e)}

    def _get_server_stats(self) -> Dict:
        """Get server statistics."""
        uptime = time.time() - self.start_time

        stats = {
            "type": "stats",
            "uptime_seconds": uptime,
            "requests_received": self.requests_received,
            "predictions_made": self.predictions_made,
            "errors": self.errors,
            "model_version": self.model_wrapper.model_version
            if self.model_wrapper
            else "unknown",
        }

        if self.cache:
            stats["cache"] = self.cache.get_stats()

        return stats

    def _send_response(self, client_socket: socket.socket, response: Dict):
        """Send response to MQL5 client."""
        try:
            # Serialize response
            data = json.dumps(response).encode("utf-8")

            # Send with length prefix
            length = struct.pack("I", len(data))
            client_socket.sendall(length + data)

        except Exception as e:
            logger.error(f"Error sending response: {e}")

    def stop(self):
        """Stop the server."""
        self.running = False

        if self.server_socket:
            self.server_socket.close()

        # Print final statistics
        logger.info("=" * 70)
        logger.info("SERVER SHUTDOWN")
        logger.info("=" * 70)

        stats = self._get_server_stats()
        logger.info(f"Total requests: {stats['requests_received']}")
        logger.info(f"Total predictions: {stats['predictions_made']}")
        logger.info(f"Total errors: {stats['errors']}")
        logger.info(f"Uptime: {stats['uptime_seconds']:.0f}s")

        if self.cache:
            cache_stats = self.cache.get_stats()
            logger.info(f"Cache hit rate: {cache_stats['hit_rate']:.2f}%")

        logger.info("=" * 70)


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ML Bridge Server for MQL5 Integration"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="random_forest",
        choices=["random_forest", "xgboost", "tensorflow"],
        help="Model type to load",
    )

    parser.add_argument("--port", type=int, default=80, help="Server port")

    parser.add_argument(
        "--model-path", type=str, default="models", help="Path to model files"
    )

    parser.add_argument(
        "--no-cache", action="store_true", help="Disable prediction caching"
    )

    parser.add_argument(
        "--cache-size", type=int, default=1000, help="Cache size (number of entries)"
    )

    args = parser.parse_args()

    # Create configuration
    config = ServerConfig()
    config.model_type = args.model
    config.port = args.port
    config.model_path = Path(args.model_path)
    config.enable_caching = not args.no_cache
    config.cache_size = args.cache_size

    # Create and start server
    server = MLBridgeServer(config)

    if not server.initialize():
        logger.error("Failed to initialize server")
        sys.exit(1)

    server.start()


if __name__ == "__main__":
    main()
