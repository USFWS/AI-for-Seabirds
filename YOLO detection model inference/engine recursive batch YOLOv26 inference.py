import csv
import torch
import os
import time
from pathlib import Path
from sahi import AutoDetectionModel
import sahi
from sahi.predict import get_sliced_prediction
import pickle
import tensorrt
from ultralytics import YOLO
import config

# Inputs:
# source_img       = ROOT folder containing subfolders of images
# new_csv_dir      = folder where per-subfolder detection CSVs will be written
# model_path       = path to YOLOv8 weights file
# project_name     = root project dir; each subfolder gets its own project/exp under here

source_root = Path(config.SOURCE_IMG)
new_csv_dir = Path(config.NEW_CSV)   # now treated as a directory, one CSV per subfolder
model_path = config.MODEL_PATH
project_root = Path(config.PROJECT_NAME)

image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

new_csv_dir.mkdir(parents=True, exist_ok=True)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# allow cuDNN to find the fastest conv algorithms for your hardware
torch.backends.cudnn.benchmark = True

overall_start = time.time()

# ─── Load model once ──────────────────────────────────────────────────────
yolo_model = YOLO(model_path, task='detect')

detection_model = AutoDetectionModel.from_pretrained(
    model_type='ultralytics',
    model=yolo_model,  # pass object, not the path
    model_path=model_path,
    device="cuda:0",
)

detection_model.model.overrides['half'] = True
detection_model.model.overrides['imgsz'] = 1024

# GPU warmup ────────────────────────────────────────────────────────────────
# TensorRT engines are slow on the first few forward passes while CUDA
# allocates memory and warms up kernels. Warming up before timing ensures
# the clock starts only when the GPU is at full speed.
print("Warming up GPU...")
dummy = torch.zeros(1, 3, 1024, 1024, dtype=torch.float16, device=device)
for _ in range(3):
    detection_model.model(dummy)
torch.cuda.synchronize()
print("Warmup done.")
# detection_model.model.half()

# ─── Gather subfolders ───────────────────────────────────────────────────
subfolders = sorted(p for p in source_root.iterdir() if p.is_dir())
print(f"Found {len(subfolders)} subfolders under {source_root}")

for subfolder in subfolders:
    images = [p for p in subfolder.iterdir() if p.suffix.lower() in image_extensions]
    if not images:
        print(f"Skipping {subfolder} (no images found)")
        continue

    folder_start = time.time()
    print(f"\n=== Processing folder: {subfolder.name} ({len(images)} images) ===")

    # Give each subfolder its own project directory so pickles don't collide
    folder_project = project_root / subfolder.name

    result = sahi.predict.predict(
        detection_model=detection_model,
        model_type='ultralytics',
        model_confidence_threshold=0.15,
        slice_height=1024,
        slice_width=1024,
        no_standard_prediction=True,
        no_sliced_prediction=False,
        overlap_height_ratio=0.1,
        overlap_width_ratio=0.1,
        model_device=device,
        source=str(subfolder),
        export_crop=False,
        export_pickle=True,
        novisual=True,
        verbose=0,  # 1 or 2
        project=str(folder_project),
    )

    torch.cuda.synchronize()
    folder_duration = time.time() - folder_start
    print(f"Folder {subfolder.name} completed in {folder_duration / 60:.2f} minutes")

    # ─── Reformat pickles for this subfolder ─────────────────────────────
    pickle_dir = folder_project / "exp" / "pickles"
    print("Reading pickles from:", pickle_dir)

    rows = []

    if pickle_dir.exists():
        for root, dirs, files in os.walk(pickle_dir):
            for filename in files:
                path = os.path.join(root, filename)

                with open(path, 'rb') as file:
                    detections = pickle.load(file)

                basename = os.path.splitext(filename)[0] + ".jpg"

                for det in detections:
                    score = det.score.value
                    cat = det.category.id
                    x, y, w, h = [int(v) for v in det.bbox.to_coco_bbox()]
                    rows.append([basename, x, y, w, h, cat, score])
    else:
        print(f"Warning: no pickle directory found at {pickle_dir}")

    folder_csv = new_csv_dir / f"{subfolder.name}.csv"
    with open(folder_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['unique_image_jpg', 'xmin', 'ymin', 'w', 'h', 'class', 'score'])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} detections to {folder_csv}")

# ─── Overall duration tracking ─────────────────────────────────────────────
torch.cuda.synchronize()
overall_duration_sec = time.time() - overall_start
print("\nAll folders completed.")
print("Total time, seconds:", overall_duration_sec)
print("Total time, minutes:", overall_duration_sec / 60)
print("Total time, hours:", overall_duration_sec / 3600)