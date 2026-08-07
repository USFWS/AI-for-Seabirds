"""
Compare bounding boxes between two CSV files, grouped by image filename.

Expected CSV columns (adjust COLUMN NAMES below to match your files):
    filename, x, y, width, height

For every image that appears in both files, every box in file A is compared
against every box in file B, and overlap metrics are computed.
"""
import pandas as pd
import config

pred_boxes1 = config.CSV_A
pred_boxes2 = config.CSV_B

new_csv1 = config.NEW_CSV # this compares set 1 to 2
new_csv2 = config.NEW_CSV2 # this compares set 2 to 1

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

def compare_gt_vs_pred(csv_path_1, csv_path_2, only_overlaps=False):
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
    gt_boxes = load_boxes(csv_path_1)
    pred_boxes = load_boxes(csv_path_2)

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

def best_match_per_gt(csv_path_1, csv_path_2):
    """
    For every ground-truth box, find the single predicted box with the highest IoU.

    Every ground-truth box appears exactly once in the output. If a GT box has
    no predictions on its image (or none overlap it), pred_row/pred_box will be
    None and iou will be 0.0 -- it will NOT be silently dropped.
    """
    all_comparisons = compare_gt_vs_pred(csv_path_1, csv_path_2, only_overlaps=False)

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

def compare_two_prediction_sets(csv_path_1, csv_path_2, only_overlaps=False):
    """
    Compare two sets of predictions (e.g. from two different models), where
    neither set is "ground truth" -- both are treated symmetrically.

    Every box in set 1 is compared against every box in set 2, for each image.
    Every box in set 1 is guaranteed to appear at least once in the output,
    even if set 2 has no boxes on that image, or none overlap it.

    Unlike compare_gt_vs_pred, this does NOT also loop the other direction
    (set2 vs set1) -- that would produce the "reverse duplicate" rows you saw
    before. Use best_match_each_direction() below to get the full agreement
    picture in both directions without duplicating comparisons.
    """
    boxes_1 = load_boxes(csv_path_1)
    boxes_2 = load_boxes(csv_path_2)

    results = []

    for fname, entries_1 in boxes_1.items():
        entries_2 = boxes_2.get(fname, [])

        for entry_1 in entries_1:
            if not entries_2:
                results.append({
                    "filename": fname,
                    "set1_row": entry_1["row_index"],
                    "set2_row": None,
                    "set1_box": entry_1["box"],
                    "set2_box": None,
                    "intersection_area": 0,
                    "union_area": None,
                    "iou": 0.0,
                    "pct_of_box_a": 0.0,
                    "pct_of_box_b": None,
                    "overlaps": False,
                })
                continue

            matched_any = False
            for entry_2 in entries_2:
                metrics = box_overlap(entry_1["box"], entry_2["box"])

                if only_overlaps and not metrics["overlaps"]:
                    continue

                matched_any = True
                results.append({
                    "filename": fname,
                    "set1_row": entry_1["row_index"],
                    "set2_row": entry_2["row_index"],
                    "set1_box": entry_1["box"],
                    "set2_box": entry_2["box"],
                    **metrics,
                })

            if only_overlaps and not matched_any:
                results.append({
                    "filename": fname,
                    "set1_row": entry_1["row_index"],
                    "set2_row": None,
                    "set1_box": entry_1["box"],
                    "set2_box": None,
                    "intersection_area": 0,
                    "union_area": None,
                    "iou": 0.0,
                    "pct_of_box_a": 0.0,
                    "pct_of_box_b": None,
                    "overlaps": False,
                })

    return pd.DataFrame(results)

def best_match_each_direction(csv_path_1, csv_path_2):
    """
    Find the best (highest IoU) match for every box, in both directions:
      - best_from_1: for each box in set 1, its best match in set 2
      - best_from_2: for each box in set 2, its best match in set 1

    Returns (best_from_1_df, best_from_2_df).

    Every box from both sets appears exactly once in its respective result.
    A box counts as having "no match" in two situations, and both are
    normalized to look identical in the output (match columns set to None,
    iou = 0.0):
      1. There were no candidate boxes at all on that image.
      2. There were candidates, but none of them overlapped (best IoU = 0).
    """
    forward = compare_two_prediction_sets(csv_path_1, csv_path_2, only_overlaps=False)
    backward = compare_two_prediction_sets(csv_path_2, csv_path_1, only_overlaps=False)

    def reduce_to_best(df, row_col, match_row_col, match_box_col):
        if df.empty:
            return df

        best = (
            df.sort_values("iou", ascending=False)
            .drop_duplicates(subset=["filename", row_col], keep="first")
            .sort_values(["filename", row_col])
            .reset_index(drop=True)
        )

        # Normalize zero-IoU "best matches" to look the same as no-candidate-at-all
        # rows, so downstream filtering only ever needs to check one condition.
        zero_iou_mask = best["iou"] == 0.0
        best.loc[zero_iou_mask, match_row_col] = None
        best.loc[zero_iou_mask, match_box_col] = None
        best.loc[zero_iou_mask, "intersection_area"] = 0
        best.loc[zero_iou_mask, "union_area"] = None
        best.loc[zero_iou_mask, "pct_of_box_a"] = 0.0
        best.loc[zero_iou_mask, "pct_of_box_b"] = None
        best.loc[zero_iou_mask, "overlaps"] = False

        return best

    best_from_1 = reduce_to_best(forward, "set1_row", "set2_row", "set2_box")

    # backward was computed as "set2 vs set1", so its own set1_row/set1_box
    # actually correspond to ORIGINAL set 2, and its set2_row/set2_box
    # correspond to ORIGINAL set 1. Build best_from_2 as a fresh DataFrame
    # with explicit column assignment (no simultaneous rename) to avoid
    # any risk of columns overwriting each other.
    best_from_2_raw = reduce_to_best(backward, "set1_row", "set2_row", "set2_box")

    if best_from_2_raw.empty:
        best_from_2 = best_from_2_raw
    else:
        best_from_2 = pd.DataFrame({
            "filename": best_from_2_raw["filename"],
            "set2_row": best_from_2_raw["set1_row"],
            "set1_row": best_from_2_raw["set2_row"],
            "set2_box": best_from_2_raw["set1_box"],
            "set1_box": best_from_2_raw["set2_box"],
            "intersection_area": best_from_2_raw["intersection_area"],
            "union_area": best_from_2_raw["union_area"],
            "iou": best_from_2_raw["iou"],
            "pct_of_box_a": best_from_2_raw["pct_of_box_b"],
            "pct_of_box_b": best_from_2_raw["pct_of_box_a"],
            "overlaps": best_from_2_raw["overlaps"],
        })
        # Reorder columns to match best_from_1 for consistency
        best_from_2 = best_from_2[[
            "filename", "set1_row", "set2_row", "set1_box", "set2_box",
            "intersection_area", "union_area", "iou",
            "pct_of_box_a", "pct_of_box_b", "overlaps",
        ]]

    return best_from_1, best_from_2

def _self_test_unmatched_boxes():
    """
    Sanity check: verifies that boxes with NO real match -- whether because
    there were no candidates at all, or because the best candidate had zero
    overlap -- are both normalized to a missing set2_row/set1_row, and that
    they survive all the way through best_match_each_direction in BOTH
    directions. Run this after any change to the matching logic above.

    Uses the CURRENT configured column names (FILENAME_COL, X_COL, Y_COL,
    W_COL, H_COL) so it stays valid even after you rename columns in config.
    """
    import tempfile, os

    header = f"{FILENAME_COL},{X_COL},{Y_COL},{W_COL},{H_COL}"

    csv1_content = (
        f"{header}\n"
        "img1.jpg,0,0,10,10\n"
        "img1.jpg,100,100,10,10\n"
        "img2.jpg,5,5,10,10\n"   # img2 box has no counterpart in set 2 at all
    )

    csv2_content = (
        f"{header}\n"
        "img1.jpg,1,1,10,10\n"
        "img3.jpg,0,0,10,10\n"   # img3 exists only in set 2
    )

    with tempfile.TemporaryDirectory() as tmp:
        path1 = os.path.join(tmp, "set1.csv")
        path2 = os.path.join(tmp, "set2.csv")
        with open(path1, "w") as f:
            f.write(csv1_content)
        with open(path2, "w") as f:
            f.write(csv2_content)

        best_from_1, best_from_2 = best_match_each_direction(path1, path2)

        # set1 has 3 boxes total -> best_from_1 must have 3 rows
        assert len(best_from_1) == 3, f"Expected 3 rows in best_from_1, got {len(best_from_1)}"

        # set2 has 2 boxes total -> best_from_2 must have 2 rows
        assert len(best_from_2) == 2, f"Expected 2 rows in best_from_2, got {len(best_from_2)}"

        # Note: pandas often stores missing values as NaN rather than None once a
        # column mixes numbers with missing entries -- use pd.isna() instead of
        # `is None` so this doesn't break depending on dtype.

        # The img1.jpg box at (100,100) in set1 has no nearby match -> should show set2_row missing
        no_match_row = best_from_1[
            (best_from_1["filename"] == "img1.jpg") & (best_from_1["set1_row"] == 1)
        ]
        assert pd.isna(no_match_row.iloc[0]["set2_row"]), "Expected unmatched set1 box to have missing set2_row"

        # The img2.jpg box has no image match at all in set 2 -> should also show set2_row missing
        no_image_row = best_from_1[best_from_1["filename"] == "img2.jpg"]
        assert pd.isna(no_image_row.iloc[0]["set2_row"]), "Expected img2 box (no matching image) to have missing set2_row"

        # The img3.jpg box in set2 has no image match at all in set 1 -> should show set1_row missing
        no_image_row_2 = best_from_2[best_from_2["filename"] == "img3.jpg"]
        assert pd.isna(no_image_row_2.iloc[0]["set1_row"]), "Expected img3 box (no matching image) to have missing set1_row"

        print("All unmatched-box checks passed.")
        print("\nbest_from_1:")
        print(best_from_1.to_string(index=False))
        print("\nbest_from_2:")
        print(best_from_2.to_string(index=False))

if __name__ == "__main__":
    # Run this once to confirm unmatched boxes survive correctly after any edits
    _self_test_unmatched_boxes()
    print()

    # ---- Uses the paths from config.py ----
    best_from_1, best_from_2 = best_match_each_direction(pred_boxes1, pred_boxes2)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 150)

    print("=== Best match in set 2, for each box in set 1 ===")
    print(best_from_1.to_string(index=False))
    best_from_1.to_csv(new_csv1, index=False)

    print("\n=== Best match in set 1, for each box in set 2 ===")
    print(best_from_2.to_string(index=False))
    best_from_2.to_csv(new_csv2, index=False)

    # Boxes in set 1 with no reasonable match in set 2 (models disagree)
    unmatched_1 = best_from_1[best_from_1["set2_row"].isna()]
    print(f"\n{len(unmatched_1)} boxes in set 1 have no overlapping match in set 2.")

    unmatched_2 = best_from_2[best_from_2["set1_row"].isna()]
    print(f"{len(unmatched_2)} boxes in set 2 have no overlapping match in set 1.")