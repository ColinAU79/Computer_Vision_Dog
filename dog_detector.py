"""
Live Dog vs Not-Dog Detector
Uses a pre-trained MobileNetV3-Small (ImageNet) — no training needed!
Dog breeds occupy ImageNet class indices 151–268.

Setup:
    pip install torch torchvision opencv-python Pillow

Run:
    python3 dog_detector.py

Controls:
    Q — quit
    S — save a screenshot
"""

import cv2
import torch
import torchvision.transforms as T
from torchvision import models
from PIL import Image

# ── Model ────────────────────────────────────────────────────────────────────
print("Loading model…")
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
model.eval()
print("Model ready!")

# ImageNet dog class range (151 = Chihuahua … 268 = Mexican hairless)
DOG_MIN, DOG_MAX = 151, 268

# ── Preprocessing ─────────────────────────────────────────────────────────────
preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])

# ── Webcam loop ───────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check that it's connected.")

screenshot_count = 0

print("Detector running — press Q to quit, S to save a screenshot.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Lost webcam feed.")
        break

    # ── Inference ──────────────────────────────────────────────────────────
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img   = Image.fromarray(rgb)
    inp   = preprocess(img).unsqueeze(0)          # shape: [1, 3, 224, 224]

    with torch.no_grad():
        logits = model(inp)

    probs      = torch.softmax(logits[0], dim=0)
    top_prob, top_cls = probs.topk(1)
    top_prob   = top_prob.item()
    top_cls    = top_cls.item()

    is_dog     = DOG_MIN <= top_cls <= DOG_MAX

    # ── Confidence bar ─────────────────────────────────────────────────────
    # Show dog-class probability (sum of all 118 dog classes)
    dog_conf   = probs[DOG_MIN:DOG_MAX + 1].sum().item()

    # ── Overlay ────────────────────────────────────────────────────────────
    h, w       = frame.shape[:2]
    color      = (34, 197, 94) if is_dog else (239, 68, 68)   # green / red (BGR)
    label      = f"{'DOG' if is_dog else 'NOT DOG'}  {dog_conf*100:.1f}%"

    # Border
    cv2.rectangle(frame, (4, 4), (w - 4, h - 4), color, 4)

    # Label background
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 1.1, 2)
    cv2.rectangle(frame, (10, 10), (20 + tw, 20 + th + 10), color, -1)
    cv2.putText(frame, label, (15, 15 + th),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

    # Confidence bar (bottom of frame)
    bar_w = int(dog_conf * (w - 20))
    cv2.rectangle(frame, (10, h - 25), (w - 10, h - 10), (60, 60, 60), -1)
    cv2.rectangle(frame, (10, h - 25), (10 + bar_w, h - 10), color, -1)
    cv2.putText(frame, "dog confidence", (12, h - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    cv2.imshow("Dog Detector  [Q=quit  S=screenshot]", frame)

    # ── Key handling ───────────────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        fname = f"screenshot_{screenshot_count:03d}.jpg"
        cv2.imwrite(fname, frame)
        print(f"Saved {fname}")
        screenshot_count += 1

cap.release()
cv2.destroyAllWindows()
print("Bye!")
