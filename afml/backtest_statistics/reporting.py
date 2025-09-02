import re
from collections import namedtuple
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def meta_labelling_reports(model_data, plot_roc=False):
    """
    Generate meta-labeling report for primary strategy and meta-model.

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
        plot_roc: If True, plots the ROC curve.

    Returns: None
    """
    y_test, pred, prob = (
        model_data.y_test,
        model_data.pred,
        model_data.prob,
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

    # Plot ROC curve
    if plot_roc:
        fpr, tpr, _ = roc_curve(y_test, prob)
        roc_auc = roc_auc_score(y_test, prob)

        plt.figure(figsize=(7.5, 5), dpi=100)
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", color="blue")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Chance")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.style.use("dark_background")
        plt.title("Receiver Operating Characteristic")
        plt.legend()


def compare_confusion_matrices(
    model_data_1: namedtuple, model_data_2: namedtuple, normalize=None, titles: List[str] = None
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
    model_data: List[namedtuple],
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
    ax = ax.flatten()

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
        plt.subplots_adjust(top=0.9)

    return fig


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


def meta_labelling_classification_reports(model_data, titles, output_filenames, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    for data, title, fname in zip(model_data, titles, output_filenames):
        create_classification_report_image(
            y_true=data.y_test,
            y_pred=np.ones_like(data.pred),
            title=f"{title} Primary Model",
            output_filename=path / f"{fname}_primary_clf_report.png",
            display=False,
        )
        create_classification_report_image(
            y_true=data.y_test,
            y_pred=data.pred,
            title=f"{title} Meta-Model",
            output_filename=path / f"{fname}_meta_clf_report.png",
            display=False,
        )
    logger.info("Classification reports saved.")
