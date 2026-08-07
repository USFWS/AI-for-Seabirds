
"""
Compare bounding boxes between two CSV files, grouped by image filename.

Expected CSV columns (adjust COLUMN NAMES below to match your files):
    filename, x, y, width, height

For every image that appears in both files, every box in file A is compared
against every box in file B, and overlap metrics are computed.
"""
import pandas as pd
import itertools
import config

gt_csv = config.CSV_ground_truth
pred_boxes = config.CSV_predictions

new_csv = config.NEW_CSV

# ---- CONFIG: adjust these to match your actual column names ----
FILENAME_COL = "unique_image_jpg"
X_COL = "xmin"
Y_COL = "ymin"
W_COL = "w"
H_COL = "h"

def xywh_to_xyxy(x, y, w, h):
    """Convert (x, y, width, height) -> (xmin, ymin, xmax, ymax).
    Assumes x, y is the top-left corner."""
    return x, y, x + w, y + h

def box_overlap(box_a, box_b):
    """
    Given two boxes in (xmin, ymin, xmax, ymax) format, compute:
      - intersection area
      - union area
      - IoU (Intersection over Union)
      - % of box A covered by the overlap
      - % of box B covered by the overlap
    Returns None-ish zeroed result if there is no overlap.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = area_a + area_b - inter_area

    iou = inter_area / union_area if union_area > 0 else 0.0
    pct_of_a = inter_area / area_a if area_a > 0 else 0.0
    pct_of_b = inter_area / area_b if area_b > 0 else 0.0

    return {
        "intersection_area": inter_area,
        "union_area": union_area,
        "iou": iou,
        "pct_of_box_a": pct_of_a,
        "pct_of_box_b": pct_of_b,
        "overlaps": inter_area > 0,
    }

def load_boxes(csv_path):
    """Load a CSV and return a dict: filename -> list of (xyxy box, original row index)."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]  # strip whitespace from headers

    required = [FILENAME_COL, X_COL, Y_COL, W_COL, H_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing expected columns {missing} in {csv_path}. "
            f"Found columns: {list(df.columns)}"
        )

    # Check for exact duplicate rows, which can cause confusing repeated comparisons
    dupe_mask = df.duplicated(subset=[FILENAME_COL, X_COL, Y_COL, W_COL, H_COL], keep=False)
    if dupe_mask.any():
        dupe_rows = df[dupe_mask]
        print(f"WARNING: {csv_path} contains {len(dupe_rows)} duplicate box rows:")
        print(dupe_rows.to_string())

    boxes_by_image = {}
    for idx, row in df.iterrows():
        fname = row[FILENAME_COL]
        box = xywh_to_xyxy(row[X_COL], row[Y_COL], row[W_COL], row[H_COL])
        boxes_by_image.setdefault(fname, []).append({"row_index": idx, "box": box})

    return boxes_by_image

def compare_gt_vs_pred(gt_csv_path, pred_csv_path, only_overlaps=False):
    """
    Compare every ground-truth box against every predicted box, grouped by image.
    Returns a pandas DataFrame with one row per (gt_box, pred_box) comparison.

    Every ground-truth box is guaranteed to appear at least once in the output,
    even if its image has no predictions at all, or none of the predictions on
    that image overlap it. In those cases pred_row/pred_box/iou etc. will be
    None/0, so GT boxes with no match are never silently dropped.

    Because GT and predictions are different roles, each (gt_row, pred_row) pair
    is a distinct, meaningful comparison -- there is no "reverse duplicate" here.
    """
    gt_boxes = load_boxes(gt_csv_path)
    pred_boxes = load_boxes(pred_csv_path)

    results = []

    for fname, gt_entries in gt_boxes.items():
        pred_entries = pred_boxes.get(fname, [])  # may be empty -- that's fine

        for gt_entry in gt_entries:
            if not pred_entries:
                # No predictions at all on this image -- record the GT box as unmatched
                results.append({
                    "filename": fname,
                    "gt_row": gt_entry["row_index"],
                    "pred_row": None,
                    "gt_box": gt_entry["box"],
                    "pred_box": None,
                    "intersection_area": 0,
                    "union_area": None,
                    "iou": 0.0,
                    "pct_of_box_a": 0.0,
                    "pct_of_box_b": None,
                    "overlaps": False,
                })
                continue

            matched_any = False
            for pred_entry in pred_entries:
                metrics = box_overlap(gt_entry["box"], pred_entry["box"])

                if only_overlaps and not metrics["overlaps"]:
                    continue

                matched_any = True
                results.append({
                    "filename": fname,
                    "gt_row": gt_entry["row_index"],
                    "pred_row": pred_entry["row_index"],
                    "gt_box": gt_entry["box"],
                    "pred_box": pred_entry["box"],
                    **metrics,
                })

            # If only_overlaps=True and none of the predictions overlapped,
            # still record the GT box once so it isn't silently dropped.
            if only_overlaps and not matched_any:
                results.append({
                    "filename": fname,
                    "gt_row": gt_entry["row_index"],
                    "pred_row": None,
                    "gt_box": gt_entry["box"],
                    "pred_box": None,
                    "intersection_area": 0,
                    "union_area": None,
                    "iou": 0.0,
                    "pct_of_box_a": 0.0,
                    "pct_of_box_b": None,
                    "overlaps": False,
                })

    return pd.DataFrame(results)


def best_match_per_gt(gt_csv_path, pred_csv_path):
    """
    For every ground-truth box, find the single predicted box with the highest IoU.

    Every ground-truth box appears exactly once in the output. If a GT box has
    no predictions on its image (or none overlap it), pred_row/pred_box will be
    None and iou will be 0.0 -- it will NOT be silently dropped.
    """
    all_comparisons = compare_gt_vs_pred(gt_csv_path, pred_csv_path, only_overlaps=False)

    if all_comparisons.empty:
        return all_comparisons

    # Sort so the highest IoU per gt_row comes first, then keep only that top row.
    # Unmatched GT boxes (iou == 0.0) will naturally sort last within their group,
    # but since each unmatched GT box has only one row, it's kept regardless.
    best = (
        all_comparisons
        .sort_values("iou", ascending=False)
        .drop_duplicates(subset=["filename", "gt_row"], keep="first")
        .sort_values(["filename", "gt_row"])
        .reset_index(drop=True)
    )

    return best


if __name__ == "__main__":
    # ---- Update these paths ----
    GT_CSV = gt_csv
    PRED_CSV = pred_boxes

    best_matches_df = best_match_per_gt(GT_CSV, PRED_CSV)

    if best_matches_df.empty:
        print("No matching filenames between the two files.")
    else:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 150)
        print(best_matches_df.to_string(index=False))

        best_matches_df.to_csv(new_csv, index=False)
        print(f"\nSaved {len(best_matches_df)} best-match records to best_match_per_gt.csv")