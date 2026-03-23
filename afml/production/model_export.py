import json
import sys
from datetime import datetime as dt
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import onnx
import onnxruntime
import sklearn
from loguru import logger
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def export_model_to_onnx(
    model, feature_names: List[str], output_path: Path, metadata: Dict[str, Any] = None
) -> bool:
    """
    Export trained model to ONNX format with comprehensive validation.

    Args:
        model: Trained sklearn model
        feature_names: List of feature names in exact order
        output_path: Where to save .onnx file (Path object)
        metadata: Additional metadata to embed

    Returns:
        bool: True if export and validation succeeded
    """
    print("\n" + "=" * 70)
    print("ONNX EXPORT PIPELINE")
    print("=" * 70)

    # Step 1: Prepare metadata
    print("\n[Step 1/5] Preparing metadata...")

    if metadata is None:
        metadata = {}

    metadata.update(
        {
            "feature_names": feature_names,
            "feature_count": len(feature_names),
            "model_type": type(model).__name__,
            "version": "1.0",
            "created_date": dt.now().isoformat(),
            "created_by": "AFML Production Pipeline",
        }
    )

    print(f"✓ Model type: {metadata['model_type']}")
    print(f"✓ Features: {len(feature_names)}")
    print(f"✓ Version: {metadata['version']}")

    # Step 2: Convert to ONNX
    print("\n[Step 2/5] Converting to ONNX format...")

    try:
        initial_type = [("float_input", FloatTensorType([None, len(feature_names)]))]
        onnx_model = convert_sklearn(
            model,
            initial_types=initial_type,
            target_opset=12,
            options={"zipmap": False},
        )
        onnx_model.doc_string = json.dumps(metadata, indent=2)
        print("✓ Conversion successful")
        print("✓ ONNX opset: 12 (MQL5 compatible)")
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        return False

    # Step 3: Save ONNX model
    print("\n[Step 3/5] Saving ONNX model...")

    try:
        onnx.save_model(onnx_model, str(output_path))
        file_size = output_path.stat().st_size / (1024**2)  # MB
        print(f"✓ Saved to: {output_path}")
        print(f"✓ File size: {file_size:.2f} MB")
    except Exception as e:
        print(f"✗ Save failed: {e}")
        return False

    # Step 4: Validate ONNX model
    print("\n[Step 4/5] Validating ONNX model...")
    try:
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model structure valid")
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        return False

    # Step 5: Compare predictions
    print("\n[Step 5/5] Comparing Python vs ONNX predictions...")
    validation_passed = validate_onnx_predictions(model, output_path, feature_names)

    if validation_passed:
        print("\n" + "=" * 70)
        print("✅ EXPORT SUCCESSFUL - Model ready for MQL5 deployment")
        print("=" * 70)
        return True
    else:
        print("\n" + "=" * 70)
        print("❌ EXPORT FAILED - Predictions don't match")
        print("=" * 70)
        return False


def validate_onnx_predictions(
    python_model, onnx_path: Path, feature_names: List[str], n_test_samples: int = 1000
) -> bool:
    """
    Validate that ONNX model produces identical predictions to Python.
    """
    print("\nGenerating test data...")
    np.random.seed(42)
    X_test = np.random.randn(n_test_samples, len(feature_names)).astype(np.float32)

    print("Computing Python predictions...")
    if hasattr(python_model, "predict_proba"):
        python_preds = python_model.predict_proba(X_test)[:, 1]
    else:
        python_preds = python_model.predict(X_test)

    print("Computing ONNX predictions...")
    session = onnxruntime.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    onnx_outputs = session.run(None, {input_name: X_test})

    # ... (rest of the function unchanged)
    # (Same code for handling outputs and comparison)
    return ...  # keep existing logic


def extract_onnx_metadata(onnx_path: Path) -> Dict[str, Any]:
    """
    Extract embedded metadata from ONNX model.
    """
    model = onnx.load(str(onnx_path))
    try:
        metadata = json.loads(model.doc_string)
        return metadata
    except Exception as e:
        logger.error(e)
        return {}


def complete_export_workflow(
    model,
    feature_names: List[str],
    output_dir: Path = Path("production_models"),
    model_name: str = "trading_model",
) -> Path:
    """
    Complete export workflow with versioning and documentation.
    Returns Path to exported ONNX file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_v{timestamp}.onnx"
    output_path = output_dir / filename

    metadata = {
        "model_name": model_name,
        "timestamp": timestamp,
        "sklearn_version": sklearn.__version__,
        "python_version": sys.version.split()[0],
        "training_date": dt.now().isoformat(),
    }

    success = export_model_to_onnx(model, feature_names, output_path, metadata)

    if success:
        doc_path = output_path.with_suffix(".txt").with_name(f"{filename}_info.txt")
        with doc_path.open("w") as f:
            f.write("=" * 70 + "\n")
            f.write("ONNX MODEL DOCUMENTATION\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Model File: {filename}\n")
            f.write(f"Created: {dt.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Features ({len(feature_names)}):\n")
            for i, feat in enumerate(feature_names, 1):
                f.write(f"  {i:2d}. {feat}\n")
            f.write("\n")
            f.write("Metadata:\n")
            for key, value in metadata.items():
                f.write(f"  {key}: {value}\n")
        print(f"\n✓ Documentation saved to: {doc_path}")
        return output_path
    else:
        return None
