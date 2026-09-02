"""
Compare bounding boxes between two CSV files, grouped by image unique_image_jpg AND
class_id. Boxes of different classes are never compared against each other.

Expected CSV columns (adjust COLUMN NAMES below to match your files):
    filename, x, y, width, height, class_id

For every image that appears in both files, every ground-truth box is compared
against every SAME-CLASS predicted box, and overlap metrics are computed.
"""
import pandas as pd
import config

gt_csv = config.CSV_ground_truth
pred_boxes = config.CSV_predictions
new_csv = config.NEW_CSV # this compares set 1 to 2
iou_threshold = 0.50

# ---- csv config: ----
FILENAME_COL = "unique_image_jpg"
X_COL = "xmin"
Y_COL = "ymin"
W_COL = "w"
H_COL = "h"
CLASS_COL = "class_id"

def xywh_to_xyxy(x, y, w, h):
    """Convert (x, y, width, height) -> (xmin, ymin, xmax, ymax).
    """
    return x, y, x + w, y + h

def box_overlap(box_a, box_b):
    """
    Given two boxes in (xmin, ymin, xmax, ymax) format, compute:
      - IoU (Intersection over Union)
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

    return {
        "iou": iou,
        "iou_overlaps": iou > iou_threshold,
    }

def load_boxes(csv_path):
    """
    Load a CSV and return a dict keyed by (filename, class_id) -> list of
    {"row_index", "box"} entries.

    Keying by (filename, class_id) instead of just filename means boxes of
    different classes are never even considered as candidates for each other
    anywhere downstream -- a "no match" for one class can't accidentally get
    filled in by an overlapping box of a different class.
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]  # strip whitespace from headers

    required = [FILENAME_COL, X_COL, Y_COL, W_COL, H_COL, CLASS_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing expected columns {missing} in {csv_path}. "
            f"Found columns: {list(df.columns)}"
        )

    # Check for exact duplicate rows, which can cause confusing repeated comparisons
    dupe_mask = df.duplicated(subset=[FILENAME_COL, X_COL, Y_COL, W_COL, H_COL, CLASS_COL], keep=False)
    if dupe_mask.any():
        dupe_rows = df[dupe_mask]
        print(f"WARNING: {csv_path} contains {len(dupe_rows)} duplicate box rows:")
        print(dupe_rows.to_string())

    boxes_by_image_class = {}
    for idx, row in df.iterrows():
        fname = row[FILENAME_COL]
        class_id = row[CLASS_COL]
        box = xywh_to_xyxy(row[X_COL], row[Y_COL], row[W_COL], row[H_COL])
        key = (fname, class_id)
        boxes_by_image_class.setdefault(key, []).append({"row_index": idx, "box": box})

    return boxes_by_image_class

def _entries_for(boxes_by_image_class, fname, class_id):
    """Look up candidates for a specific (filename, class_id) pair. Empty list if none."""
    return boxes_by_image_class.get((fname, class_id), [])

def compare_gt_vs_pred(gt_csv_path, pred_csv_path, only_overlaps=False):
    """
    Compare every ground-truth box against every predicted box of the SAME
    CLASS, grouped by image. Returns a pandas DataFrame with one row per
    (gt_box, pred_box) comparison.

    A GT box is only ever compared against predictions sharing its class_id.
    Predictions of a different class are never considered candidates, so they
    can't accidentally satisfy a match for the wrong class.

    Every ground-truth box is guaranteed to appear at least once in the output,
    even if its image has no same-class predictions at all, or none of the
    same-class predictions on that image overlap it. In those cases pred_row/
    pred_box/iou etc. will be None/0, so GT boxes with no match are never
    silently dropped.
    """
    gt_boxes = load_boxes(gt_csv_path)
    pred_boxes = load_boxes(pred_csv_path)

    results = []

    for (fname, class_id), gt_entries in gt_boxes.items():
        pred_entries = _entries_for(pred_boxes, fname, class_id)  # same class only

        for gt_entry in gt_entries:
            if not pred_entries:
                # No same-class predictions at all on this image -- record as unmatched
                results.append({
                    "filename": fname,
                    "class_id": class_id,
                    "gt_row": gt_entry["row_index"],
                    "pred_row": None,
                    "gt_box": gt_entry["box"],
                    "pred_box": None,
                    "iou": 0.0,
                    "iou_overlaps": False,
                })
                continue

            matched_any = False
            for pred_entry in pred_entries:
                metrics = box_overlap(gt_entry["box"], pred_entry["box"])

                if only_overlaps and not metrics["iou_overlaps"]:
                    continue

                matched_any = True
                results.append({
                    "filename": fname,
                    "class_id": class_id,
                    "gt_row": gt_entry["row_index"],
                    "pred_row": pred_entry["row_index"],
                    "gt_box": gt_entry["box"],
                    "pred_box": pred_entry["box"],
                    **metrics,
                })

            # If only_overlaps=True and none of the same-class predictions overlapped,
            # still record the GT box once so it isn't silently dropped.
            if only_overlaps and not matched_any:
                results.append({
                    "filename": fname,
                    "class_id": class_id,
                    "gt_row": gt_entry["row_index"],
                    "pred_row": None,
                    "gt_box": gt_entry["box"],
                    "pred_box": None,
                    "iou": 0.0,
                    "iou_overlaps": False,
                })

    return pd.DataFrame(results)


def best_match_per_gt(gt_csv_path, pred_csv_path):
    """
    For every ground-truth box, find the single SAME-CLASS predicted box with
    the highest IoU.

    Every ground-truth box appears exactly once in the output. If a GT box has
    no same-class predictions on its image (or none overlap it), pred_row/
    pred_box will be None and iou will be 0.0 -- it will NOT be silently
    dropped, and it will NEVER be filled in by a different-class prediction.
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

def full_match_report(gt_csv_path, pred_csv_path):
    """
    Build one combined table covering both:
      - every ground-truth box (matched to its best same-class prediction,
        or unmatched -- a missed detection)
      - every prediction that was NOT claimed as the best match by any
        ground-truth box of the same class on the same image -- a false
        positive

    A "object_type" column distinguishes the two kinds of rows:
      "gt"          -- a ground-truth box (see gt_row/gt_box; pred_row/pred_box
                        are None if unmatched)
      "false_positive" -- a prediction nobody claimed (see pred_row/pred_box;
                        gt_row/gt_box are always None for these rows)

    Note: a prediction only counts as "claimed" if it was the highest-IoU
    match FOR SOME GT box (i.e. it appears as pred_row in the best_match_per_gt
    result with iou > 0). A prediction that overlapped a GT box but lost out
    to a better-matching prediction will show up here as a false positive too
    -- that's intentional, since only one prediction can be "the" match per
    GT box, and any others on that box are still spurious detections.
    """
    best = best_match_per_gt(gt_csv_path, pred_csv_path)

    gt_rows = best.copy()
    gt_rows.insert(0, "object_type", "gt")

    # Load raw predictions so we can find which ones were never claimed
    pred_boxes = load_boxes(pred_csv_path)

    # Build the set of (filename, class_id, pred_row) that WERE claimed
    claimed = set()
    if not best.empty:
        matched = best.dropna(subset=["pred_row"])
        for _, row in matched.iterrows():
            claimed.add((row["filename"], row["class_id"], row["pred_row"]))

    fp_rows = []
    for (fname, class_id), entries in pred_boxes.items():
        for entry in entries:
            key = (fname, class_id, entry["row_index"])
            if key not in claimed:
                fp_rows.append({
                    "object_type": "false_positive",
                    "filename": fname,
                    "class_id": class_id,
                    "gt_row": None,
                    "pred_row": entry["row_index"],
                    "gt_box": None,
                    "pred_box": entry["box"],
                    "iou": None,
                    "iou_overlaps": None,
                })

    fp_df = pd.DataFrame(fp_rows)

    combined = pd.concat([gt_rows, fp_df], ignore_index=True) if not fp_df.empty else gt_rows
    combined = combined.sort_values(["filename", "class_id", "object_type"]).reset_index(drop=True)

    return combined

def _self_test_class_filtering():
    """
    Sanity check: verifies that a ground-truth box never gets matched to a
    prediction of a different class, even when that wrong-class prediction
    is a perfect geometric overlap. Also checks that unmatched GT boxes
    (no candidates, or no same-class overlap) correctly show a missing
    pred_row, AND that a genuinely unclaimed prediction shows up as a
    false_positive row in full_match_report.

    Uses the CURRENT configured column names so it stays valid even after
    you rename columns in config.
    """
    import tempfile, os

    header = f"{FILENAME_COL},{X_COL},{Y_COL},{W_COL},{H_COL},{CLASS_COL}"

    gt_content = (
        f"{header}\n"
        "img1.jpg,0,0,10,10,0\n"    # class 0, has a real same-class match below
        "img2.jpg,0,0,10,10,1\n"    # class 1, perfectly overlaps a class-0 prediction -- must NOT match
        "img3.jpg,5,5,10,10,0\n"    # class 0, no predictions on this image at all
    )

    pred_content = (
        f"{header}\n"
        "img1.jpg,1,1,10,10,0\n"    # class 0, correctly matches the GT box above
        "img2.jpg,0,0,10,10,0\n"    # class 0, same geometry as GT above but WRONG class
        "img5.jpg,0,0,10,10,0\n"    # class 0, no GT box at all on this image -- pure false positive
    )

    with tempfile.TemporaryDirectory() as tmp:
        gt_path = os.path.join(tmp, "gt.csv")
        pred_path = os.path.join(tmp, "pred.csv")
        with open(gt_path, "w") as f:
            f.write(gt_content)
        with open(pred_path, "w") as f:
            f.write(pred_content)

        best = best_match_per_gt(gt_path, pred_path)

        assert len(best) == 3, f"Expected 3 GT boxes in output, got {len(best)}"

        # img1.jpg class-0 GT box should find its real match
        row1 = best[best["filename"] == "img1.jpg"]
        assert not pd.isna(row1.iloc[0]["pred_row"]), "Expected img1 GT box to find its same-class match"
        assert row1.iloc[0]["iou"] > 0, "Expected a positive IoU for the real match"

        # img2.jpg class-1 GT box must NOT match the class-0 prediction, despite perfect overlap
        row2 = best[best["filename"] == "img2.jpg"]
        assert pd.isna(row2.iloc[0]["pred_row"]), (
            "Class filtering failed: a class-1 GT box matched a class-0 prediction "
            "just because they overlap geometrically."
        )

        # img3.jpg class-0 GT box has no predictions on its image at all
        row3 = best[best["filename"] == "img3.jpg"]
        assert pd.isna(row3.iloc[0]["pred_row"]), "Expected img3 GT box (no predictions at all) to have missing pred_row"

        # Now check the combined report for false positives
        combined = full_match_report(gt_path, pred_path)

        # Should include: 3 GT rows + false positives for img2's prediction (unclaimed,
        # since it was wrong-class) and img5's prediction (no GT box at all) = 5 rows
        assert len(combined) == 5, f"Expected 5 combined rows, got {len(combined)}"

        fp_rows = combined[combined["object_type"] == "false_positive"]
        assert len(fp_rows) == 2, f"Expected 2 false-positive rows, got {len(fp_rows)}"

        fp_filenames = set(fp_rows["filename"])
        assert fp_filenames == {"img2.jpg", "img5.jpg"}, (
            f"Expected false positives on img2.jpg and img5.jpg, got {fp_filenames}"
        )

        print("All class-filtering, unmatched-GT, and false-positive checks passed.")
        print("\nbest_match_per_gt:")
        print(best.to_string(index=False))
        print("\nfull_match_report:")
        print(combined.to_string(index=False))


if __name__ == "__main__":
    # Run this once to confirm class filtering and false-positive detection
    # behave correctly after any edits
    _self_test_class_filtering()
    print()

    # ---- Update these paths ----
    GT_CSV = gt_csv
    PRED_CSV = pred_boxes

    combined_df = full_match_report(GT_CSV, PRED_CSV)

    if combined_df.empty:
        print("No matching filenames between the two files.")
    else:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 150)
        print(combined_df.to_string(index=False))

        combined_df.to_csv(new_csv, index=False)
        print(f"\nSaved {len(combined_df)} records to {new_csv}")

        missed = combined_df[(combined_df["object_type"] == "gt") & (combined_df[
                                                                         "pred_row"].isna())]
        print(f"\n{len(missed)} ground-truth boxes have no matching prediction of the same class (missed detections).")

        false_positives = combined_df[combined_df["object_type"] == "false_positive"]
        print(f"{len(false_positives)} predictions were not claimed by any ground-truth box (false positives).")