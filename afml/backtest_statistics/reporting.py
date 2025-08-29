import re
from collections import namedtuple
from typing import List

import numpy as np
import pandas as pd
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


def meta_labelling_reports(y_test, w_test, pred, prob, plot_roc=False):
    """
    Generate meta-labeling report for both tick and time bar features.

    Args:
        model_template: Classifier to be used for meta-labeling.
        y_test: True labels for the test set.
        w_test: Sample weights for the test set.
        pred: Predictions from the meta-model.
        prob: Predicted probabilities from the meta-model.
        plot_roc: If True, plots the ROC curve.
    Returns: None
    """
    print("Primary-Model on Validation Set:")
    pred0 = np.ones_like(y_test)  # primary model predicts all signals as true
    w_test = w_test if np.unique(w_test).size > 1 else None
    print(classification_report(y_test, pred0, sample_weight=w_test))
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, pred0, sample_weight=w_test)
    print(cm)

    print("\nMeta-Model on Validation Set:")
    print(classification_report(y_test, pred, sample_weight=w_test))
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, pred, sample_weight=w_test)
    print(cm)

    # Plot ROC curve
    if plot_roc:
        fpr, tpr, _ = roc_curve(y_test, prob, sample_weight=w_test)
        roc_auc = roc_auc_score(y_test, prob, sample_weight=w_test)

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
        sample_weight=model_data_1.w_test,
        normalize=normalize,
    )
    cm1 = confusion_matrix(
        model_data_2.y_test,
        model_data_2.pred,
        sample_weight=model_data_2.w_test,
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
    model_data_1: namedtuple, model_data_2: namedtuple, titles: List[str] = None
):
    fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(7.5, 5), dpi=100)
    if not titles:
        titles = [""] * 2

    # Plot ROC curve
    for data, ax, title in zip((model_data_1, model_data_2), axes, titles):
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(data.y_test, data.prob, sample_weight=data.w_test)
        auc = roc_auc_score(data.y_test, data.prob, sample_weight=data.w_test)
        ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.2f})", color="blue")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{title}")
        ax.legend()

    plt.style.use("dark_background")
    plt.tight_layout()
    return fig


def create_classification_report_image(
    y_true,
    y_pred,
    target_names=None,
    title="Classification Report",
    output_filename="classification_report.png",
    display=True,
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
    if "classification report" not in title.lower():
        title = "Classification Report: " + title.strip().title().replace(":", "-")
    else:
        title = title.strip().title()

    # --- 1. Generate Classification Report as a string ---
    report_str = classification_report(y_true, y_pred, target_names=target_names)
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

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
    data_rows_raw = [re.split(r"\s{2,}", line.strip()) for line in lines[1:]]

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

    # Calculate column widths to ensure perfect alignment
    col_widths = [font_data.getlength(label) for label in labels]
    # Account for header parts too
    for i, h in enumerate(header_parts):
        if font_data.getlength(h) > col_widths[i]:
            col_widths[i] = font_data.getlength(h)

    # Find the max width for the label column
    label_col_width = max([font_data.getlength(label) for label in labels])

    # Calculate x positions for each column
    x_positions = [padding + label_col_width + col_margin]
    for i in range(len(header_parts)):
        # Safely get max width by checking if the row has a value at index i
        current_col_width = max(
            [font_data.getlength(header_parts[i])]
            + [font_data.getlength(row[i]) for row in data_values if i < len(row)]
        )
        x_positions.append(x_positions[-1] + current_col_width + col_margin)

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
    x_pos = padding + label_col_width + col_margin

    for h in header_parts:
        draw.text((x_pos, y_pos), h, font=font_data, fill=header_color)
        x_pos += font_data.getlength(h) + col_margin

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
        draw.text((padding, y_pos), row[0], font=font_data, fill=data_color)
        x_pos = padding + label_col_width + col_margin
        for j, col in enumerate(row[1:]):
            draw.text((x_pos, y_pos), col, font=font_data, fill=data_color)
            x_pos += font_data.getlength(header_parts[j]) + col_margin
        y_pos += (font_data.getbbox("A")[3] - font_data.getbbox("A")[1]) + line_spacing

    # Add a blank line for spacing
    y_pos += line_spacing

    # Draw the summary data rows
    for i, row in enumerate(summary_data):
        draw.text((padding, y_pos), row[0], font=font_data, fill=data_color)
        x_pos = padding + label_col_width + col_margin
        # Special handling for 'accuracy' row to align with 'f1-score' and 'support'
        if row[0] == "accuracy":
            f1_score_col_index = header_parts.index("f1-score")
            support_col_index = header_parts.index("support")

            # Draw accuracy value in the f1-score column
            x_pos_f1_score = padding + label_col_width + col_margin
            for j in range(f1_score_col_index):
                x_pos_f1_score += font_data.getlength(header_parts[j]) + col_margin
            draw.text((x_pos_f1_score, y_pos), row[1], font=font_data, fill=data_color)

            # Draw total support in the support column
            x_pos_support = padding + label_col_width + col_margin
            for j in range(support_col_index):
                x_pos_support += font_data.getlength(header_parts[j]) + col_margin
            draw.text((x_pos_support, y_pos), row[2], font=font_data, fill=data_color)

        else:
            for j, col in enumerate(row[1:]):
                draw.text((x_pos, y_pos), col, font=font_data, fill=data_color)
                x_pos += font_data.getlength(header_parts[j]) + col_margin
        y_pos += (font_data.getbbox("A")[3] - font_data.getbbox("A")[1]) + line_spacing

    # Draw Confusion Matrix
    y_pos += 2 * line_spacing
    draw.text((padding, y_pos), "Confusion Matrix", font=font_data, fill=header_color)
    y_pos += (font_data.getbbox("A")[3] - font_data.getbbox("A")[1]) + line_spacing

    for row in cm:
        draw.text((padding, y_pos), str(row), font=font_data, fill=data_color)
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

    print(f"\nSuccessfully generated and saved '{output_filename}'")
