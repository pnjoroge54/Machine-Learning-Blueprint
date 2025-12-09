import json
import os
import sys
from datetime import datetime as dt
from typing import Any, Dict, List

import numpy as np
import onnx
import onnxruntime
import sklearn
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def export_model_to_onnx(
    model, feature_names: List[str], output_path: str, metadata: Dict[str, Any] = None
) -> bool:
    """
    Export trained model to ONNX format with comprehensive validation.

    This ensures the ONNX model produces identical predictions to
    the Python model, preventing subtle bugs in production.

    Args:
        model: Trained sklearn model
        feature_names: List of feature names in exact order
        output_path: Where to save .onnx file
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
        # Define input type (float32 for MQL5 compatibility)
        initial_type = [("float_input", FloatTensorType([None, len(feature_names)]))]

        # Convert
        onnx_model = convert_sklearn(
            model, initial_types=initial_type, target_opset=12  # MQL5 supports opset 12
        )

        # Embed metadata in doc_string
        onnx_model.doc_string = json.dumps(metadata, indent=2)

        print(f"✓ Conversion successful")
        print(f"✓ ONNX opset: 12 (MQL5 compatible)")

    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        return False

    # Step 3: Save ONNX model
    print("\n[Step 3/5] Saving ONNX model...")

    try:
        onnx.save_model(onnx_model, output_path)
        file_size = os.path.getsize(output_path) / (1024**2)  # MB
        print(f"✓ Saved to: {output_path}")
        print(f"✓ File size: {file_size:.2f} MB")
    except Exception as e:
        print(f"✗ Save failed: {e}")
        return False

    # Step 4: Validate ONNX model
    print("\n[Step 4/5] Validating ONNX model...")

    try:
        # Check model is valid
        onnx.checker.check_model(onnx_model)
        print(f"✓ ONNX model structure valid")
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        return False

    # Step 5: Compare predictions (critical!)
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
    python_model, onnx_path: str, feature_names: List[str], n_test_samples: int = 1000
) -> bool:
    """
    Validate that ONNX model produces identical predictions to Python.

    This is CRITICAL - we must ensure production model behavior
    matches our backtested results exactly.
    """
    print("\nGenerating test data...")

    # Generate random test data that matches training distribution
    np.random.seed(42)
    X_test = np.random.randn(n_test_samples, len(feature_names)).astype(np.float32)

    # Python predictions
    print("Computing Python predictions...")
    if hasattr(python_model, "predict_proba"):
        python_preds = python_model.predict_proba(X_test)[:, 1]
    else:
        python_preds = python_model.predict(X_test)

    # ONNX predictions
    print("Computing ONNX predictions...")
    session = onnxruntime.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    onnx_preds = session.run(None, {input_name: X_test})[0]

    # If model outputs probabilities, extract positive class
    if onnx_preds.ndim > 1 and onnx_preds.shape[1] == 2:
        onnx_preds = onnx_preds[:, 1]

    # Compare predictions
    max_diff = np.max(np.abs(python_preds - onnx_preds))
    mean_diff = np.mean(np.abs(python_preds - onnx_preds))

    print(f"\nPrediction Comparison ({n_test_samples} samples):")
    print(f"  • Max difference:  {max_diff:.2e}")
    print(f"  • Mean difference: {mean_diff:.2e}")
    print(f"  • Std difference:  {np.std(np.abs(python_preds - onnx_preds)):.2e}")

    # Define tolerance (should be very small for production)
    tolerance = 1e-5

    if max_diff < tolerance:
        print(f"\n✅ VALIDATION PASSED - Predictions match within tolerance ({tolerance:.2e})")

        # Show some example predictions
        print(f"\nSample Predictions (first 5):")
        print(f"{'Index':<8} {'Python':<12} {'ONNX':<12} {'Diff':<12}")
        print("-" * 50)
        for i in range(5):
            diff = abs(python_preds[i] - onnx_preds[i])
            print(f"{i:<8} {python_preds[i]:<12.6f} {onnx_preds[i]:<12.6f} {diff:<12.2e}")

        return True
    else:
        print(
            f"\n❌ VALIDATION FAILED - Max difference {max_diff:.2e} exceeds tolerance {tolerance:.2e}"
        )

        # Find and report worst mismatches
        worst_indices = np.argsort(np.abs(python_preds - onnx_preds))[-5:]
        print(f"\nWorst 5 Mismatches:")
        print(f"{'Index':<8} {'Python':<12} {'ONNX':<12} {'Diff':<12}")
        print("-" * 50)
        for idx in worst_indices:
            diff = abs(python_preds[idx] - onnx_preds[idx])
            print(f"{idx:<8} {python_preds[idx]:<12.6f} {onnx_preds[idx]:<12.6f} {diff:<12.2e}")

        return False


def extract_onnx_metadata(onnx_path: str) -> Dict[str, Any]:
    """
    Extract embedded metadata from ONNX model.
    Useful for version checking in MQL5.
    """
    model = onnx.load(onnx_path)

    try:
        metadata = json.loads(model.doc_string)
        return metadata
    except:
        return {}


# Complete export workflow
def complete_export_workflow(
    model,
    feature_names: List[str],
    output_dir: str = "production_models",
    model_name: str = "trading_model",
) -> str:
    """
    Complete export workflow with versioning and documentation.

    Returns:
        str: Path to exported ONNX file
    """
    import os

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename with timestamp
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_v{timestamp}.onnx"
    output_path = os.path.join(output_dir, filename)

    # Prepare comprehensive metadata
    metadata = {
        "model_name": model_name,
        "timestamp": timestamp,
        "sklearn_version": sklearn.__version__,
        "python_version": sys.version.split()[0],
        "training_date": dt.now().isoformat(),
    }

    # Export with validation
    success = export_model_to_onnx(model, feature_names, output_path, metadata)

    if success:
        # Create accompanying documentation
        doc_path = output_path.replace(".onnx", "_info.txt")
        with open(doc_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("ONNX MODEL DOCUMENTATION\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Model File: {filename}\n")
            f.write(f"Created: {dt.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Features ({len(feature_names)}):\n")
            for i, feat in enumerate(feature_names, 1):
                f.write(f"  {i:2d}. {feat}\n")
            f.write("\n")
            f.write(f"Metadata:\n")
            for key, value in metadata.items():
                f.write(f"  {key}: {value}\n")

        print(f"\n✓ Documentation saved to: {doc_path}")

        return output_path
    else:
        return None
