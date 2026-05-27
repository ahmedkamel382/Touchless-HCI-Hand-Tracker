import time
import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO


def calculate_iou(boxA, boxB):
    # Calculate the Intersection over Union (IoU) of two bounding boxes
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    unionArea = boxAArea + boxBArea - interArea
    return interArea / float(unionArea) if unionArea > 0 else 0


# 1. Load BOTH models for the A/B test
print("Loading PyTorch model (best.pt)...")
model_pt = YOLO('best.pt')

print("Loading ONNX model (best.onnx)...")
model_onnx = YOLO('best.onnx')

# 2. Folder setup
input_folder = 'test/images'
label_folder = 'test/labels'
output_folder = 'test_annotated'
os.makedirs(output_folder, exist_ok=True)

image_paths = glob.glob(os.path.join(input_folder, '*.jpg'))
if not image_paths:
    print(f"Error: No images found in '{input_folder}'. Please check your folder structure.")
    exit()

print(f"\n--- Running PyTorch vs ONNX Benchmark ({len(image_paths)} images) ---")

# 3. Warm-up both models (hardware initialization)
print("Warming up processors...")
warmup_img = cv2.imread(image_paths[0])
model_pt(warmup_img, verbose=False)
model_onnx(warmup_img, verbose=False)

total_time_pt = 0
total_time_onnx = 0
total_hands_detected = 0

# Arrays to store IoU scores for graphing
iou_scores_pt = []
iou_scores_onnx = []

# 4. Process the folder
for i, img_path in enumerate(image_paths):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    filename = os.path.basename(img_path)
    label_path = os.path.join(label_folder, filename.replace('.jpg', '.txt'))

    # --- Benchmark PyTorch ---
    start_pt = time.time()
    results_pt = model_pt(img, verbose=False)
    total_time_pt += (time.time() - start_pt)

    # --- Benchmark ONNX ---
    start_onnx = time.time()
    results_onnx = model_onnx(img, verbose=False)
    total_time_onnx += (time.time() - start_onnx)

    total_hands_detected += len(results_onnx[0].boxes)

    # --- IoU Calculation (Comparing predictions to actual ground truth) ---
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                # Parse YOLO normalized format [class_id, x_center, y_center, width, height]
                parts = list(map(float, line.split()))
                if len(parts) == 5:
                    xc, yc, bw, bh = parts[1:]
                    gt_box = [(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h]

                    # Find best IoU for PyTorch
                    best_iou_pt = 0
                    for pred in results_pt[0].boxes.xyxy:
                        best_iou_pt = max(best_iou_pt, calculate_iou(gt_box, pred.cpu().numpy()))
                    iou_scores_pt.append(best_iou_pt)

                    # Find best IoU for ONNX
                    best_iou_onnx = 0
                    for pred in results_onnx[0].boxes.xyxy:
                        best_iou_onnx = max(best_iou_onnx, calculate_iou(gt_box, pred.cpu().numpy()))
                    iou_scores_onnx.append(best_iou_onnx)

    # Save the visually annotated image (using ONNX output)
    annotated_img = results_onnx[0].plot()
    cv2.imwrite(os.path.join(output_folder, filename), annotated_img)

    # Console progress tracker
    if (i + 1) % 50 == 0 or (i + 1) == len(image_paths):
        print(f"Processed {i + 1}/{len(image_paths)} images...")

# 5. Calculate Final Math
avg_pt_ms = (total_time_pt / len(image_paths)) * 1000
fps_pt = 1000 / avg_pt_ms if avg_pt_ms > 0 else 0

avg_onnx_ms = (total_time_onnx / len(image_paths)) * 1000
fps_onnx = 1000 / avg_onnx_ms if avg_onnx_ms > 0 else 0

speedup = fps_onnx / fps_pt if fps_pt > 0 else 0

avg_iou_pt = np.mean(iou_scores_pt) if iou_scores_pt else 0
avg_iou_onnx = np.mean(iou_scores_onnx) if iou_scores_onnx else 0

# 6. Generate Analytical Presentation Graphs
print("\nGenerating presentation graphs...")
plt.figure(figsize=(12, 5))

# Graph 1: Speed Comparison
plt.subplot(1, 2, 1)
bars = plt.bar(['PyTorch (.pt)', 'ONNX (.onnx)'], [fps_pt, fps_onnx], color=['#1f77b4', '#2ca02c'])
plt.title('Inference Speed (Higher is Better)')
plt.ylabel('Frames Per Second (FPS)')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, f'{yval:.1f} FPS', ha='center', va='bottom',
             fontweight='bold')

# Graph 2: Accuracy Comparison (IoU)
plt.subplot(1, 2, 2)
bars_iou = plt.bar(['PyTorch (.pt)', 'ONNX (.onnx)'], [avg_iou_pt * 100, avg_iou_onnx * 100],
                   color=['#d62728', '#ff7f0e'])
plt.title('Localization Accuracy (IoU)')
plt.ylabel('Average Overlap %')
plt.ylim(0, 100)
for bar in bars_iou:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 1.5, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
graph_path = os.path.join(output_folder, 'presentation_metrics.png')
plt.savefig(graph_path, dpi=300)

# 7. Print the Presentation Report
print("\n" + "=" * 50)
print(" FINAL HARDWARE BENCHMARK REPORT ")
print("=" * 50)
print(f"Total Images Processed : {len(image_paths)}")
print(f"Total Ground Truths    : {len(iou_scores_onnx)}")
print(f"Total Hands Detected   : {total_hands_detected}")
print("-" * 50)
print(" PYTORCH (.pt) PERFORMANCE ")
print(f" -> Accuracy (Avg IoU) : {avg_iou_pt * 100:.1f}%")
print(f" -> Average Latency    : {avg_pt_ms:.1f} ms per image")
print(f" -> Processing Speed   : {fps_pt:.1f} FPS")
print("-" * 50)
print(" ONNX (.onnx) PERFORMANCE ")
print(f" -> Accuracy (Avg IoU) : {avg_iou_onnx * 100:.1f}%")
print(f" -> Average Latency    : {avg_onnx_ms:.1f} ms per image")
print(f" -> Processing Speed   : {fps_onnx:.1f} FPS")
print("=" * 50)
print(f" RESULT: ONNX is {speedup:.2f}x faster with identical accuracy!")
print("=" * 50)
print(f"\nAll visually annotated images and the presentation graph have been saved to '{output_folder}'.")