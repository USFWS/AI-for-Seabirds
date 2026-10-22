
import ast
import pandas as pd

# ---- Config ----
INPUT_CSV = "E:/yolo_box_comparison.csv"
OUTPUT_CSV = "E:/yolo_box_merged.csv"
iou_threshold_merge = 0.20

import ast
import pandas as pd

def parse_box(box_str):
    """Parse a string like '(4391, 1280, 4422, 1338)' into a tuple of ints.
    Returns None if the value is missing (NaN/empty) — this happens when a
    box in set1 or set2 has no counterpart at all.
    """
    if pd.isna(box_str):
        return None
    return tuple(ast.literal_eval(box_str))

def merge_boxes(box1, box2):
    """
    Merge two (x1, y1, x2, y2) boxes into their union:
    min of the mins, max of the maxes.
    """
    x1a, y1a, x2a, y2a = box1
    x1b, y1b, x2b, y2b = box2

    x1 = min(x1a, x1b)
    y1 = min(y1a, y1b)
    x2 = max(x2a, x2b)
    y2 = max(y2a, y2b)
    return (x1, y1, x2, y2)

def main():
    df = pd.read_csv(INPUT_CSV)

    # Convert iou_overlaps from 0/1 to proper boolean True/False
    df["iou_overlaps"] = df["iou_overlaps"].astype(bool)

    # Parse box strings into tuples
    df["set1_box_parsed"] = df["set1_box"].apply(parse_box)
    df["set2_box_parsed"] = df["set2_box"].apply(parse_box)

    # Determine which rows meet the IoU threshold
    meets_threshold = df["iou"] >= iou_threshold_merge

    # Compute merged box only where threshold is met; otherwise leave blank
    def compute_merged(row):
        if row["iou"] >= iou_threshold_merge:
            return merge_boxes(row["set1_box_parsed"], row["set2_box_parsed"])
        return None

    df["merged_box"] = df.apply(compute_merged, axis=1)
    df["merged"] = meets_threshold

    # final_box logic:
    # - If iou >= threshold: use the merged (union) box.
    # - If below threshold but only one of set1_box/set2_box exists
    #   (the other prediction set found nothing at that location),
    #   use whichever box is present as-is.
    # - If below threshold and both boxes exist (they just don't overlap
    #   enough), default to set1_box.
    def compute_final(row):
        if row["iou"] >= iou_threshold_merge:
            return row["merged_box"]
        if row["set1_box_parsed"] is None and row["set2_box_parsed"] is not None:
            return row["set2_box_parsed"]
        if row["set2_box_parsed"] is None and row["set1_box_parsed"] is not None:
            return row["set1_box_parsed"]
        # Both present but don't meet threshold -> default to set1_box
        return row["set1_box_parsed"]

    df["final_box"] = df.apply(compute_final, axis=1)

    # Split final_box (x1, y1, x2, y2) into xmin, ymin, w, h columns
    df["xmin"] = df["final_box"].apply(lambda b: b[0])
    df["ymin"] = df["final_box"].apply(lambda b: b[1])
    df["w"] = df["final_box"].apply(lambda b: b[2] - b[0])
    df["h"] = df["final_box"].apply(lambda b: b[3] - b[1])

    # unique_BB: filename (without ".jpg") + xmin + ymin + w + h + ".jpg",
    # joined with underscores
    def build_unique_bb(row):
        base_name = row["filename"].rsplit(".jpg", 1)[0]
        return f"{base_name}_{row['xmin']}_{row['ymin']}_{row['w']}_{row['h']}.jpg"

    df["unique_BB"] = df.apply(build_unique_bb, axis=1)

    # Remove duplicate rows based on unique_BB, keeping the first occurrence
    before_count = len(df)
    df = df.drop_duplicates(subset="unique_BB", keep="first")
    after_count = len(df)
    print(f"Removed {before_count - after_count} duplicate rows based on unique_BB "
          f"({before_count} -> {after_count})")

    # Drop the intermediate parsed-tuple columns if you don't need them downstream
    df = df.drop(columns=["set1_box_parsed", "set2_box_parsed"])

    # Quick summary
    total = len(df)
    merged_count = df["merged"].sum()
    print(f"Total rows: {total}")
    print(f"Rows merged (iou >= {iou_threshold_merge}): {merged_count}")
    print(f"Rows not merged: {total - merged_count}")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved output to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()