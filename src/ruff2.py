import sys
from pathlib import Path
import time
import torch
import numpy as np
import serial
import cv2
import threading
from queue import Queue

# --- YOLOv5 root ---
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

# --- Arduino setup ---
arduino_port = "/dev/ttyACM0"
baud_rate = 9600
try:
    arduino = serial.Serial(arduino_port, baud_rate, timeout=0.01)
    time.sleep(2)
    print("Arduino connected")
except Exception as e:
    print("Arduino not connected:", e)
    arduino = None

# --- YOLOv5 setup ---
weights_path = str(ROOT / "My/best_linux2.pt")
device = select_device("cpu")  # use "cuda" if GPU available
model = DetectMultiBackend(weights_path, device=device)
img_size = 224  # smaller for faster CPU
conf_thres = 0.25
iou_thres = 0.45
names = model.names

# --- Camera capture queue ---
frame_queue = Queue(maxsize=1)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

def camera_thread():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        if frame_queue.full():
            try:
                frame_queue.get_nowait()  # drop old frame
            except:
                pass
        frame_queue.put(frame)

threading.Thread(target=camera_thread, daemon=True).start()

# --- HSV color detection ---
def detect_color(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_red = cv2.inRange(hsv, (0,120,70), (10,255,255)) | cv2.inRange(hsv, (170,120,70), (180,255,255))
    mask_green = cv2.inRange(hsv, (40,70,70), (90,255,255))
    kernel = np.ones((5,5),np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
    red_count = cv2.countNonZero(mask_red)
    green_count = cv2.countNonZero(mask_green)
    if green_count > 500 and green_count > red_count * 1.5:
        return 'G'
    elif red_count > 500 and red_count > green_count * 1.5:
        return 'R'
    else:
        return None

# --- Main loop ---
frame_count = 0
start_time = time.time()
last_sent = None  # stores last color sent to Arduino

with torch.no_grad():
    while True:
        if frame_queue.empty():
            time.sleep(0.005)
            continue
        frame = frame_queue.get()

        # Prepare YOLO image
        img = cv2.resize(frame, (img_size, img_size))
        img = img[:, :, ::-1].transpose(2,0,1)  # BGR->RGB, HWC->CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device).float()/255.0
        if img.ndim == 3:
            img = img.unsqueeze(0)

        # YOLO inference
        pred = model(img)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        detected_color = None

        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in det:
                    label = names[int(cls)]
                    if label.lower() == "red":
                        detected_color = 'R'
                    elif label.lower() == "green":
                        detected_color = 'G'

        # Fallback to HSV color detection if YOLO missed
        if not detected_color:
            detected_color = detect_color(frame)

        # --- Send to Arduino only if changed ---
        if detected_color != last_sent:
            if detected_color and arduino:
                try:
                    arduino.write(detected_color.encode())
                    print(f"Sent '{detected_color}' to Arduino")
                except:
                    pass
            elif not detected_color and last_sent and arduino:
                # send 'N' for no object
                try:
                    arduino.write(b'N')
                    print("Sent 'N' to Arduino (no object)")
                except:
                    pass

            last_sent = detected_color

        # FPS logging every 30 frames
        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            print(f"FPS: {fps:.1f}, Last sent: {last_sent}")

        # tiny sleep to reduce CPU
        time.sleep(0.005)