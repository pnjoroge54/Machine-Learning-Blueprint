import re
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from ..backtest_statistics.performance_analysis import (
    evaluate_meta_labeling_performance,
)
from .training import ModelData, get_optimal_threshold


def labeling_reports(model_data, name="Primary Model"):
    """
    Generate meta-labeling report for primary strategy and meta-data.

    In meta-labeling framework:
    - Primary model generates signals (when to trade)
    - Meta-model filters signals (which trades to take)
    - Evaluation focuses on primary model's signal periods

    Args:
        model_data: Object containing test data and predictions with attributes:
            - y_test: True labels (whether primary signals were profitable)
            - pred: Meta-model predictions (filtered signals)
            - prob: Meta-model predicted probabilities
            - primary_signals: Primary strategy signals (Bollinger, MA, etc.)
            - w_test: Sample weights (optional)

    Returns: None
    """
    y_test, pred = (
        model_data.y_test,
        model_data.pred,
    )

    print(f"\nMODEL PERFORMANCE:")
    print("-" * 53)
    print(classification_report(y_test, pred))
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, pred)
    print(cm)


def meta_labeling_reports(model_data, name="Meta-Model", plot=False):
    """
    Generate meta-labeling report for primary strategy and meta-data.

    In meta-labeling framework:
    - Primary model generates signals (when to trade)
    - Meta-model filters signals (which trades to take)
    - Evaluation focuses on primary model's signal periods

    Args:
        model_data: Object containing test data and predictions with attributes:
            - y_test: True labels (whether primary signals were profitable)
            - pred: Meta-model predictions (filtered signals)
            - prob: Meta-model predicted probabilities
            - primary_signals: Primary strategy signals (Bollinger, MA, etc.)
            - w_test: Sample weights (optional)
        plot: If True, plots the ROC curve.

    Returns: None
    """
    y_test, pred = (
        model_data.y_test,
        model_data.pred,
    )
    # 1. Evaluate Primary Model Performance
    print("\nPRIMARY MODEL PERFORMANCE:")
    print("-" * 53)

    pred0 = np.ones_like(y_test)  # primary model predicts all signals as true
    print(classification_report(y_test, pred0))
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, pred0)
    print(cm)

    print(f"\nMETA-MODEL PERFORMANCE:")
    print("-" * 53)
    print(classification_report(y_test, pred))
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, pred)
    print(cm)

    fig = compare_roc_pr_curves([model_data], model_names=[name], show_baseline=True)

    # Plot ROC curve
    if plot:
        fig.show()


def compare_confusion_matrices(
    model_data_1: ModelData, model_data_2: ModelData, normalize=None, titles: List[str] = None
):
    cm0 = confusion_matrix(
        model_data_1.y_test,
        model_data_1.pred,
        normalize=normalize,
    )
    cm1 = confusion_matrix(
        model_data_2.y_test,
        model_data_2.pred,
        normalize=normalize,
    )
    if not titles:
        titles = [""] * 2

    fig, (ax0, ax1) = plt.subplots(1, 2, sharey=True, figsize=(7.5, 5), dpi=100)
    text_kw = dict(size=14)
    ConfusionMatrixDisplay(confusion_matrix=cm0).plot(
        cmap="Blues", colorbar=False, text_kw=text_kw, ax=ax0
    )
    ConfusionMatrixDisplay(confusion_matrix=cm1).plot(
        cmap="Blues", colorbar=False, text_kw=text_kw, ax=ax1
    )
    ax0.set_title(f"{titles[0]}")
    ax1.set_title(f"{titles[1]}")
    plt.style.use("dark_background")
    plt.tight_layout()
    return fig


def compare_roc_curves(
    model_data: List[ModelData],
    titles: List[str] = None,
    fig_title: str = None,
    columns: int = 1,
    height: float = 5,
):
    n = len(model_data)
    if not titles:
        titles = [""] * n

    nrows = int(np.ceil(n / columns))
    sharex = True if nrows <= 2 else False
    fig, ax = plt.subplots(
        nrows, ncols=columns, sharex=sharex, sharey=True, figsize=(7.5, height), dpi=100
    )
    ax = np.atleast_1d(ax).flatten()

    # Plot ROC curve
    for data, ax, title in zip((model_data), ax, titles):
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(data.y_test, data.prob)
        auc = roc_auc_score(data.y_test, data.prob)
        ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.2f})", color="blue")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{title}")
        ax.legend()

    plt.style.use("dark_background")
    plt.tight_layout()

    if fig_title:
        fig.suptitle(fig_title, fontsize=13)
        plt.subplots_adjust(top=0.88)

    return fig


def compare_pr_curves(
    model_data: List[tuple],  # ModelData or similar with y_test and prob
    titles: List[str] = None,
    fig_title: str = None,
    columns: int = 1,
    height: float = 5,
):
    """
    Compare Precision–Recall curves for multiple models/labeling methods.

    Parameters
    ----------
    model_data : list of ModelData-like
        Each element must have attributes `y_test` (true labels) and `prob` (predicted scores/probabilities).
    titles : list of str, optional
        Titles for each subplot.
    fig_title : str, optional
        Overall figure title.
    columns : int, default=1
        Number of subplot columns.
    height : float, default=5
        Height of the figure in inches.
    """
    n = len(model_data)
    if not titles:
        titles = [""] * n

    nrows = int(np.ceil(n / columns))
    sharex = True if nrows <= 2 else False
    fig, ax = plt.subplots(
        nrows, ncols=columns, sharex=sharex, sharey=True, figsize=(7.5, height), dpi=100
    )
    ax = np.atleast_1d(ax).flatten()

    # Plot PR curves
    for data, axis, title in zip(model_data, ax, titles):
        precision, recall, _ = precision_recall_curve(data.y_test, data.prob)
        ap = average_precision_score(data.y_test, data.prob)

        axis.plot(recall, precision, label=f"PR Curve (AP = {ap:.2f})", color="orange")
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        axis.set_title(title)
        axis.legend()
        axis.grid(True, linestyle="--", alpha=0.6)

        # Baseline: proportion of positives
        baseline = data.y_test.sum() / len(data.y_test)
        axis.hlines(baseline, 0, 1, colors="white", linestyles="--", label="Baseline")

    plt.style.use("dark_background")
    plt.tight_layout()

    if fig_title:
        fig.suptitle(fig_title, fontsize=13)
        plt.subplots_adjust(top=0.88)

    return fig


def compare_roc_pr_curves(
    model_data: List[ModelData],  # ModelData-like with y_test and prob
    model_names: List[str] = None,
    titles: List[str] = None,
    fig_title: str = None,
    columns: int = 1,
    width: float = 7.5,
    height: float = 5,
    show_baseline: bool = False,
):
    """
    Compare ROC and Precision–Recall curves for multiple models/labeling methods,
    marking the optimal threshold point on the PR curve and returning a summary table
    with both default and tuned threshold metrics.
    """
    n = len(model_data)
    if not model_names:
        model_names = [""] * n

    if not titles:
        titles = [""] * n

    nrows = int(np.ceil(n / columns))
    sharex = True if nrows <= 2 else False
    fig, ax = plt.subplots(
        nrows, ncols=columns, sharex=sharex, sharey=False, figsize=(width, height), dpi=100
    )

    ax = np.atleast_1d(ax).flatten()
    summary_rows = []

    for data, axis, title, name in zip(model_data, ax, titles, model_names):
        # --- ROC ---
        fpr, tpr, _ = roc_curve(data.y_test, data.prob)
        auc = roc_auc_score(data.y_test, data.prob)
        axis.plot(fpr, tpr, label=f"ROC (AUC = {auc:.2f})", color="skyblue", lw=2)
        axis.plot([0, 1], [0, 1], linestyle="--", color="gray")

        # --- Precision–Recall ---
        precision, recall, thresholds = precision_recall_curve(data.y_test, data.prob)
        ap = average_precision_score(data.y_test, data.prob)

        # Metric sweep
        opt_thresholds = get_optimal_threshold(data)
        best_threshold = opt_thresholds["threshold"]
        best_score = opt_thresholds["f1_score"]
        best_recall = opt_thresholds["recall"]
        best_precision = opt_thresholds["precision"]

        # Plot PR curve
        axis.plot(recall, precision, label=f"PR (AP = {ap:.2f})", color="orange", lw=2)
        axis.scatter(
            best_recall,
            best_precision,
            color="red",
            s=50,
            zorder=5,
            label=f"Best f1_score={best_score:.2f} @ thr={best_threshold:.2f}",
        )

        # Baseline: proportion of positives
        if show_baseline:
            baseline = data.y_test.sum() / len(data.y_test)
            axis.hlines(
                baseline, 0, 1, colors="gray", linestyles="-.", label=f"Baseline = {baseline:.2f}"
            )

        axis.set_xlabel("Recall / FPR")
        axis.set_ylabel("Precision / TPR")
        axis.set_title(title)
        axis.legend()
        axis.grid(True, linestyle="--", alpha=0.6)

    if fig_title:
        fig.suptitle(fig_title, fontsize=13)
        plt.subplots_adjust(top=0.86)

    plt.style.use("dark_background")
    plt.tight_layout()

    return fig


def plot_multi_pr_curves(y_true_dict, y_score_dict, title="Precision–Recall Comparison"):
    """
    Plot Precision–Recall curves for multiple labeling methods on one chart.

    Parameters
    ----------
    y_true_dict : dict
        Mapping of method name -> array-like of true binary labels.
    y_score_dict : dict
        Mapping of method name -> array-like of predicted probabilities/scores.
        Must have the same keys as y_true_dict.
    title : str
        Plot title.
    """
    plt.figure(figsize=(8, 6))

    for method in y_true_dict:
        y_true = y_true_dict[method]
        y_scores = y_score_dict[method]

        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)

        plt.plot(recall, precision, lw=2, label=f"{method} (AP = {ap:.3f})")

    # Baseline: proportion of positives
    all_y = list(y_true_dict.values())[0]
    baseline = sum(all_y) / len(all_y)
    plt.hlines(baseline, 0, 1, colors="gray", linestyles="--", label="Baseline")

    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower left")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def create_classification_report_image(
    y_true,
    y_pred,
    target_names=None,
    title="Classification Report",
    output_filename="classification_report.png",
    display=True,
    verbose=False,
):
    """
    Generates a classification report and saves it as a well-formatted PNG image.

    Args:
        y_true (list or array): True labels.
        y_pred (list or array): Predicted labels.
        target_names (list): List of target class names.
        title (str): The title of the report.
        output_filename (str): The name of the output PNG file.
        display (bool): If True, displays the image after saving.
    """
    # Set image title
    title = title.strip()

    # --- 1. Generate Classification Report as a string ---
    report_str = classification_report(y_true, y_pred, target_names=target_names)
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    if verbose:
        print("Classification Report (as text):")
        print(report_str)
        print("\nConfusion Matrix:")
        print(cm)
        print("\nAccuracy:")
        print(accuracy)

    # --- 2. Render the Report to an Image with a Tabular Format ---

    # Define image properties and colors
    bg_color = (245, 245, 245)  # Light gray background
    header_color = (70, 70, 70)  # Dark gray for text headers
    data_color = (40, 40, 40)  # Almost black for data
    padding = 40
    line_spacing = 5
    col_margin = 20  # Spacing between columns

    # Define font paths
    try:
        # A common monospaced font is better for tabular data
        font_path_mono = "cour.ttf"  # Courier New is often available
        font_path_regular = "arial.ttf"
        font_size_header = 16
        font_size_data = 14

        font_data = ImageFont.truetype(font_path_mono, font_size_data)
        font_header = ImageFont.truetype(font_path_regular, font_size_header)
    except IOError:
        print("Warning: Font files not found. Using default fonts.")
        font_data = ImageFont.load_default()
        font_header = ImageFont.load_default()

    # Parse the report string into a structured format
    lines = report_str.split("\n")
    lines = [line for line in lines if line.strip()]

    header_parts = lines[0].split()

    # CHANGE 1: Enhanced parsing to handle decimal support values and round them
    data_rows_raw = []
    for line in lines[1:]:
        parts = re.split(r"\s{2,}", line.strip())
        if parts:  # Only process non-empty splits
            # Round support values (last column) to 2 decimal places if it's a number
            if len(parts) >= 4:  # Has support column
                try:
                    support_val = float(parts[-1])
                    parts[-1] = f"{support_val:.0f}"
                except ValueError:
                    # Keep original if not a number (e.g., headers or text)
                    pass
            data_rows_raw.append(parts)

    # Split data rows into per-class metrics and summary metrics
    summary_index = -1
    for i, row in enumerate(data_rows_raw):
        if row[0] == "accuracy":
            summary_index = i
            break

    if summary_index == -1:
        class_data = data_rows_raw
        summary_data = []
    else:
        class_data = data_rows_raw[:summary_index]
        summary_data = data_rows_raw[summary_index:]

    # We need to handle cases where the first column is a label and the rest are data points
    labels = [row[0] for row in data_rows_raw]
    data_values = [row[1:] for row in data_rows_raw]

    # CHANGE 2: Improved column width calculation to handle decimal numbers properly
    col_widths = []
    for label in labels:
        col_widths.append(font_data.getlength(str(label)))

    # Account for header parts too
    for i, h in enumerate(header_parts):
        if i < len(col_widths) and font_data.getlength(str(h)) > col_widths[i]:
            col_widths[i] = font_data.getlength(str(h))

    # Find the max width for the label column
    label_col_width = max([font_data.getlength(str(label)) for label in labels])

    # CHANGE 3: More robust x position calculation that accounts for all data values
    x_positions = [padding + label_col_width + col_margin]

    # Calculate width needed for each data column
    column_widths = []
    for col_idx in range(len(header_parts)):
        max_width = font_data.getlength(str(header_parts[col_idx]))  # Start with header width

        # Check all data values in this column
        for row in data_values:
            if col_idx < len(row):
                width = font_data.getlength(str(row[col_idx]))
                max_width = max(max_width, width)

        column_widths.append(max_width)
        x_positions.append(x_positions[-1] + max_width + col_margin)

    # Calculate the image height with new sections
    total_height = int(
        padding * 2
        + (len(lines)) * (font_data.getbbox("A")[3] - font_data.getbbox("A")[1] + line_spacing)
        + 20
    )
    total_height += (
        font_data.getbbox("A")[3] - font_data.getbbox("A")[1]
    ) + 2 * line_spacing  # for confusion matrix title
    total_height += (font_data.getbbox("A")[3] - font_data.getbbox("A")[1] + line_spacing) * len(
        cm
    )  # for confusion matrix data
    total_height += (
        font_data.getbbox("A")[3] - font_data.getbbox("A")[1]
    ) + 2 * line_spacing  # for accuracy title
    total_height += (
        font_data.getbbox("A")[3] - font_data.getbbox("A")[1]
    ) + line_spacing  # for accuracy value

    # Add extra height for the empty row before accuracy
    total_height += (font_data.getbbox("A")[3] - font_data.getbbox("A")[1]) + line_spacing

    img_width = int(x_positions[-1] + padding)
    img_height = total_height
    img = Image.new("RGB", (img_width, img_height), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Draw the report title
    title_width = font_header.getlength(title)
    y_pos = padding / 2
    draw.text(((img_width - title_width) / 2, y_pos), title, font=font_header, fill=header_color)

    # Draw the table headers
    y_pos += (font_header.getbbox("A")[3] - font_header.getbbox("A")[1]) + 10

    # Draw headers with appropriate alignment
    for i, h in enumerate(header_parts):
        if h == "support":  # Right-align support header
            x_pos = x_positions[i + 1] - font_data.getlength(h)
        else:  # Left-align other headers
            x_pos = x_positions[i]
        draw.text((x_pos, y_pos), h, font=font_data, fill=header_color)

    # Draw a horizontal line under the header
    draw.line(
        [
            (padding, y_pos + (font_data.getbbox("A")[3] - font_data.getbbox("A")[1]) + 5),
            (
                img_width - padding,
                y_pos + (font_data.getbbox("A")[3] - font_data.getbbox("A")[1]) + 5,
            ),
        ],
        fill=header_color,
        width=2,
    )

    y_pos += (font_data.getbbox("A")[3] - font_data.getbbox("A")[1]) + line_spacing + 10

    # Draw the per-class data rows
    for i, row in enumerate(class_data):
        # Left-align class label
        draw.text((padding, y_pos), row[0], font=font_data, fill=data_color)

        # Draw data columns with appropriate alignment
        for j, col in enumerate(row[1:]):
            if j == len(row[1:]) - 1:  # Support column (last column) - right align
                x_pos = x_positions[j + 1] - font_data.getlength(str(col))
            else:  # Other columns - left align
                x_pos = x_positions[j]
            draw.text((x_pos, y_pos), col, font=font_data, fill=data_color)

        y_pos += (font_data.getbbox("A")[3] - font_data.getbbox("A")[1]) + line_spacing

    # Add empty row after per-class data (before summary metrics)
    y_pos += (font_data.getbbox("A")[3] - font_data.getbbox("A")[1]) + line_spacing

    # Draw the summary data rows (accuracy, macro avg, weighted avg)
    for i, row in enumerate(summary_data):
        # Left-align summary label
        draw.text((padding, y_pos), row[0], font=font_data, fill=data_color)

        # Special handling for 'accuracy' row to align with 'f1-score' and 'support'
        if row[0] == "accuracy":
            f1_score_col_index = header_parts.index("f1-score")
            support_col_index = header_parts.index("support")

            # Draw accuracy value in the f1-score column (left-aligned)
            x_pos_f1_score = x_positions[f1_score_col_index]
            draw.text((x_pos_f1_score, y_pos), row[1], font=font_data, fill=data_color)

            # Draw total support in the support column (right-aligned)
            x_pos_support = x_positions[support_col_index + 1] - font_data.getlength(row[2])
            draw.text((x_pos_support, y_pos), row[2], font=font_data, fill=data_color)

        else:
            # Draw other summary rows with appropriate alignment
            for j, col in enumerate(row[1:]):
                if j == len(row[1:]) - 1:  # Support column - right align
                    x_pos = x_positions[j + 1] - font_data.getlength(str(col))
                else:  # Other columns - left align
                    x_pos = x_positions[j]
                draw.text((x_pos, y_pos), col, font=font_data, fill=data_color)

        y_pos += (font_data.getbbox("A")[3] - font_data.getbbox("A")[1]) + line_spacing

    # Draw Confusion Matrix
    y_pos += 2 * line_spacing
    draw.text((padding, y_pos), "Confusion Matrix", font=font_data, fill=header_color)
    y_pos += (font_data.getbbox("A")[3] - font_data.getbbox("A")[1]) + line_spacing

    # Draw confusion matrix with proper spacing alignment
    # Find the maximum width needed for each column to align properly
    max_val_width = 0
    for row in cm:
        for val in row:
            val_width = font_data.getlength(str(val))
            max_val_width = max(max_val_width, val_width)

    # Add some padding between columns
    col_spacing = font_data.getlength("  ")  # Two spaces

    for row in cm:
        # Start with opening bracket
        cm_line = "["

        # Add each value with right-alignment within its allocated width
        for i, val in enumerate(row):
            val_str = str(val)
            if i > 0:  # Add spacing between values
                cm_line += " "

            # Right-align each number within the max width
            spaces_needed = max_val_width - font_data.getlength(val_str)
            spaces_count = int(spaces_needed / font_data.getlength(" "))
            cm_line += " " * spaces_count + val_str

        # Close with bracket
        cm_line += "]"

        # Draw the complete line
        draw.text((padding, y_pos), cm_line, font=font_data, fill=data_color)
        y_pos += (font_data.getbbox("A")[3] - font_data.getbbox("A")[1]) + line_spacing

    # Draw Accuracy
    y_pos += 2 * line_spacing
    draw.text((padding, y_pos), "Accuracy", font=font_data, fill=header_color)
    y_pos += (font_data.getbbox("A")[3] - font_data.getbbox("A")[1]) + line_spacing
    draw.text((padding, y_pos), f"{accuracy:.4f}", font=font_data, fill=data_color)

    # --- 3. Save the Image as a PNG File ---
    img.save(output_filename, "PNG")
    if display:
        img.show()

    logger.info(f"Successfully generated and saved '{output_filename}'")


def meta_labeling_classification_report_images(model_data, titles, output_filenames, dirpath):
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)

    for data, title, fname in zip(model_data, titles, output_filenames):
        create_classification_report_image(
            y_true=data.y_test,
            y_pred=np.ones_like(data.pred),
            title=f"{title} Primary Model",
            output_filename=dirpath / f"{fname}_primary_clf_report.png",
            display=False,
        )
        create_classification_report_image(
            y_true=data.y_test,
            y_pred=data.pred,
            title=f"{title} Meta-Model",
            output_filename=dirpath / f"{fname}_meta_clf_report.png",
            display=False,
        )
    logger.info("Classification reports saved.")


def meta_labeling_classification_report_tables(model_data, methods, dirpath):
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    report_frames = []

    for data, method in zip(model_data, methods):
        # Replace with actual model predictions per method
        y_true = data.y_test.values
        y_pred = data.pred.values

        rpt = classification_report(y_true, y_pred, output_dict=True)
        df = pd.DataFrame(rpt).iloc[:3, :2].T  # shape: (metrics, classes)
        # print(df)

        # Add labeling method as top-level index
        df.index = pd.MultiIndex.from_product([[method], df.index], names=["method", "class"])
        report_frames.append(df)

    # Concatenate all into one MultiIndex DataFrame
    combined_df = pd.concat(report_frames)

    # Step 1: Swap index levels so 'metric' is outermost
    df_swapped = combined_df.swaplevel(0, 1).sort_index()

    # Step 2: Stack columns to long format
    long_df = df_swapped.stack()

    # Step 3: Unstack labeling_method to compare across methods
    comparison_df = long_df.unstack(level=1)

    # Step 4: Optional — rename columns for clarity
    comparison_df.columns = [f"{method}" for method in comparison_df.columns]

    # Step 5: Reset index if needed
    # comparison_df = comparison_df.reset_index()

    # Step 6: Save as html
    styled = (
        comparison_df.style.format("{:.3f}")  # 3 decimal places
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [("text-align", "center"), ("background-color", "#f2f2f2")],
                },
                {"selector": "td", "props": [("text-align", "center")]},
            ]
        )
        .highlight_max(axis=1, color="lightgreen")  # highlight best per row
    )

    filename = dirpath / "classification_comparison.html"
    styled.to_html(filename)
    logger.info(f"Saved to {filename}")


def print_meta_labeling_comparison(results: dict, save_path: str = None):
    """
    Prints and optionally saves a comprehensive comparison of the primary strategy
    versus the meta-labeled strategy performance.

    Args:
        results: Output dictionary from `evaluate_meta_labeling_performance`
        save_path: If provided, saves the output to this file path
    """
    import io
    from contextlib import redirect_stdout

    # Capture output in a string buffer
    output_buffer = io.StringIO()

    with redirect_stdout(output_buffer):
        strategy_name = results["strategy_name"]
        primary_metrics = results["primary_metrics"]
        meta_metrics = results["meta_metrics"]

        print(f"\n{'='*100}")
        print(f"Meta-Labeling Performance Analysis: {strategy_name}")
        print(f"{'='*100}")

        # --- Signal Filtering Summary ---
        print(f"\nSignal Filtering Summary:")
        print(f"  Total Primary Signals: {results['total_primary_signals']:,}")
        print(f"  Filtered Signals: {results['filtered_signals']:,}")
        print(f"  Filter Rate: {meta_metrics['signal_filter_rate']:,.2%}")
        print(f"  Confidence Threshold: {meta_metrics['confidence_threshold']}")

        # --- Core Performance Metrics Table ---
        print(
            f"\n{'CORE PERFORMANCE METRICS':<30} {'Primary':<15} {'Meta-Labeled':<15} {'Improvement':<15}"
        )
        print("=" * 75)
        core_metrics = [
            ("Total Return", "total_return", "%"),
            ("Annualized Return", "annualized_return", "%"),
            ("Sharpe Ratio", "sharpe_ratio", "4f"),
            ("Sortino Ratio", "sortino_ratio", "4f"),
            ("Calmar Ratio", "calmar_ratio", "4f"),
            ("Information Ratio", "information_ratio", "4f"),
        ]
        for display_name, metric_key, fmt in core_metrics:
            if metric_key in primary_metrics and metric_key in meta_metrics:
                primary_val = primary_metrics.get(metric_key, 0)
                meta_val = meta_metrics.get(metric_key, 0)
                improvement = calculate_improvement(primary_val, meta_val, metric_key)
                primary_str = f"{primary_val:,.2%}" if fmt == "%" else f"{primary_val:,.4f}"
                meta_str = f"{meta_val:,.2%}" if fmt == "%" else f"{meta_val:,.4f}"
                improvement_str = f"{improvement:+.1f}%" if improvement != float("inf") else "N/A"
                print(f"{display_name:<30} {primary_str:<15} {meta_str:<15} {improvement_str:<15}")

        # --- Risk Metrics Table ---
        print(f"\n{'RISK METRICS':<30} {'Primary':<15} {'Meta-Labeled':<15} {'Improvement':<15}")
        print("=" * 75)
        risk_metrics = [
            ("Max Drawdown", "max_drawdown", "%"),
            ("Avg Drawdown", "avg_drawdown", "%"),
            ("Volatility (Ann.)", "volatility", "4f"),
            ("Downside Volatility", "downside_volatility", "4f"),
            ("Ulcer Index", "ulcer_index", "4f"),
            ("VaR (95%)", "var_95", "%"),
            ("CVaR (95%)", "cvar_95", "%"),
        ]
        for display_name, metric_key, fmt in risk_metrics:
            if metric_key in primary_metrics and metric_key in meta_metrics:
                primary_val = primary_metrics.get(metric_key, 0)
                meta_val = meta_metrics.get(metric_key, 0)
                improvement = calculate_improvement(primary_val, meta_val, metric_key)
                primary_str = f"{primary_val:,.2%}" if fmt == "%" else f"{primary_val:,.4f}"
                meta_str = f"{meta_val:,.2%}" if fmt == "%" else f"{meta_val:,.4f}"
                improvement_str = f"{improvement:+.1f}%" if improvement != float("inf") else "N/A"
                print(f"{display_name:<30} {primary_str:<15} {meta_str:<15} {improvement_str:<15}")

        # --- Trading Metrics Table ---
        print(f"\n{'TRADING METRICS':<30} {'Primary':<15} {'Meta-Labeled':<15} {'Improvement':<15}")
        print("=" * 75)
        trading_metrics = [
            ("Number of Bets", "bet_frequency", "0f"),
            ("Bets per Year", "bets_per_year", "0f"),
            # ("Number of Trades", "num_trades", "0f"),
            # ("Trades per Year", "trades_per_year", "0f"),
            ("Win Rate", "win_rate", "%"),
            ("Avg Win", "avg_win", "4%"),
            ("Avg Loss", "avg_loss", "4%"),
            ("Best Trade", "best_trade", "4%"),
            ("Worst Trade", "worst_trade", "4%"),
            ("Profit Factor", "profit_factor", "2f"),
            ("Expectancy", "expectancy", "4%"),
            ("Kelly Criterion", "kelly_criterion", "4f"),
            ("Max Consecutive Wins", "consecutive_wins", "0f"),
            ("Max Consecutive Losses", "consecutive_losses", "0f"),
        ]
        for display_name, metric_key, fmt in trading_metrics:
            if metric_key in primary_metrics and metric_key in meta_metrics:
                primary_val = primary_metrics[metric_key]
                meta_val = meta_metrics[metric_key]
                improvement = calculate_improvement(primary_val, meta_val, metric_key)

                if fmt == "%":
                    primary_str, meta_str = f"{primary_val:,.2%}", f"{meta_val:,.2%}"
                elif fmt == "4%":
                    primary_str, meta_str = f"{primary_val:,.4%}", f"{meta_val:,.4%}"
                elif fmt == "0f":
                    primary_str, meta_str = f"{primary_val:,.0f}", f"{meta_val:,.0f}"
                elif fmt == "1f":
                    primary_str, meta_str = f"{primary_val:,.1f}", f"{meta_val:,.1f}"
                elif fmt == "2f":
                    primary_str, meta_str = f"{primary_val:,.2f}", f"{meta_val:,.2f}"
                else:
                    primary_str, meta_str = f"{primary_val:,.4f}", f"{meta_val:,.4f}"

                improvement_str = f"{improvement:+.1f}%" if improvement != float("inf") else "N/A"
                print(f"{display_name:<30} {primary_str:<15} {meta_str:<15} {improvement_str:<15}")

        # --- Distribution Metrics Table ---
        print(
            f"\n{'DISTRIBUTION METRICS':<30} {'Primary':<15} {'Meta-Labeled':<15} {'Improvement':<15}"
        )
        print("=" * 75)
        dist_metrics = [("Skewness", "skewness", "4f"), ("Kurtosis", "kurtosis", "4f")]
        for display_name, metric_key, fmt in dist_metrics:
            if metric_key in primary_metrics and metric_key in meta_metrics:
                primary_val = primary_metrics[metric_key]
                meta_val = meta_metrics[metric_key]
                improvement = calculate_improvement(primary_val, meta_val, metric_key)
                primary_str = f"{primary_val:,.4f}"
                meta_str = f"{meta_val:,.4f}"
                improvement_str = f"{improvement:+.1f}%" if improvement != float("inf") else "N/A"
                print(f"{display_name:<30} {primary_str:<15} {meta_str:<15} {improvement_str:<15}")

        # --- Summary Assessment ---
        print(f"\n{'SUMMARY ASSESSMENT'}")
        print("=" * 50)
        key_improvements = []
        if "sharpe_ratio" in meta_metrics and "sharpe_ratio" in primary_metrics:
            sharpe_imp = calculate_improvement(
                primary_metrics["sharpe_ratio"],
                meta_metrics["sharpe_ratio"],
                "sharpe_ratio",
            )
            key_improvements.append(("Sharpe Ratio", sharpe_imp))
        if "total_return" in meta_metrics and "total_return" in primary_metrics:
            return_imp = calculate_improvement(
                primary_metrics["total_return"],
                meta_metrics["total_return"],
                "total_return",
            )
            key_improvements.append(("Total Return", return_imp))
        if "max_drawdown" in meta_metrics and "max_drawdown" in primary_metrics:
            dd_imp = calculate_improvement(
                primary_metrics["max_drawdown"],
                meta_metrics["max_drawdown"],
                "max_drawdown",
            )
            key_improvements.append(("Max Drawdown", dd_imp))

        avg_improvement = np.mean([imp for _, imp in key_improvements if imp != float("inf")])
        print(f"avg_improvement: {avg_improvement}")
        print([imp for _, imp in key_improvements if imp != float("inf")])

        if avg_improvement > 10:
            assessment = "✅ Meta-labeling shows SIGNIFICANT improvement"
        elif avg_improvement > 5:
            assessment = "✅ Meta-labeling shows GOOD improvement"
        elif avg_improvement > 0:
            assessment = "⚠️  Meta-labeling shows MODEST improvement"
        else:
            assessment = "❌ Meta-labeling DOES NOT improve performance"

        print(f"  {assessment}")
        for metric_name, improvement in key_improvements:
            if improvement != float("inf"):
                print(f"  {metric_name} Change: {improvement:+.1f}%")

        if "sharpe_ratio" in meta_metrics and meta_metrics["sharpe_ratio"] > primary_metrics.get(
            "sharpe_ratio", 0
        ):
            print(f"\n✅ Meta-labeling improves risk-adjusted returns")
        if "signal_filter_rate" in meta_metrics:
            print(
                f"\n📊 Signal filtering removed {meta_metrics['signal_filter_rate']:,.1%} of trades"
            )
            if meta_metrics["signal_filter_rate"] > 0.3:
                print(f"   High filtering rate suggests meta-model is selective")

    # Get the captured output
    output_text = output_buffer.getvalue()

    # Print to console
    print(output_text)

    # Save to file if path provided
    if save_path:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(output_text)
        print(f"\nOutput saved to: {save_path}")


def calculate_improvement(primary_val: float, meta_val: float, metric_key: str) -> float:
    """
    Calculates the percentage improvement of a metric from primary to meta.

    This function correctly handles the direction of improvement: for some
    metrics, higher is better (e.g., Sharpe Ratio), while for others,
    lower is better (e.g., Max Drawdown).

    Args:
        primary_val: The metric value for the primary strategy.
        meta_val: The metric value for the meta-labeled strategy.
        metric_key: The name of the metric, used to determine if lower is better.

    Returns:
        The percentage improvement.
    """
    if primary_val == 0:
        return 0 if meta_val == 0 else float("inf")

    # For metrics where a lower value is preferable (e.g., risk, losses).
    lower_is_better = [
        "max_drawdown",
        "avg_drawdown",
        "volatility",
        "downside_volatility",
        "avg_loss",
        "worst_trade",
        "consecutive_losses",
        "var_95",
        "cvar_95",
        "ulcer_index",
        "kurtosis",
    ]
    if metric_key in lower_is_better:
        if all(np.sign(x) for x in [primary_val, meta_val]):
            return (primary_val - meta_val) / primary_val * 100
        return (primary_val - meta_val) / abs(primary_val) * 100

    # For metrics where a higher value is better.
    return (meta_val - primary_val) / abs(primary_val) * 100


# --- Main Orchestrator Function ---


def run_meta_labeling_analysis(
    events: pd.DataFrame,
    meta_probabilities: pd.Series,
    close: pd.Series,
    confidence_threshold: float = 0.5,
    trading_days_per_year: int = 252,
    trading_hours_per_day: int = 24,
    strategy_name: str = "Strategy",
    save_path: str = None,
    bet_sizing: str = None,
    **kwargs,
):
    """
    A wrapper function to run a complete meta-labeling analysis.

    This function prepares the necessary returns data before calling the
    main `evaluate_meta_labeling_performance` function and then prints the results.

    Args:
        events: A DataFrame of trade events that contains:
            - index: Event start times
            - t1: Event end times, i.e., the time of first barrier touch
            - trgt: Target volatility
            - pt: Take-profit target
            - sl: Stop-loss target
            - side: Trade direction
        meta_probabilities: Probabilities from the meta-model for the test period.
        close: The full Series of historical price data.
        confidence_threshold: The probability threshold for filtering trades.
        strategy_name: The name for the strategy run.
        save_path: If provided, saves the output to this file path
        bet_sizing: Method used to size bets.
            Options are 'probability', 'budget', 'dynamic', 'reserve'.
        kwargs: Optional arguments passed to bet sizing functions.
    """
    results = evaluate_meta_labeling_performance(
        events,
        meta_probabilities,
        close,
        confidence_threshold,
        trading_days_per_year,
        trading_hours_per_day,
        strategy_name,
        bet_sizing,
        **kwargs,
    )
    print_meta_labeling_comparison(results, save_path)

    return results
