import cv2
import torch
from ultralytics import YOLO
import tkinter as tk
from PIL import Image, ImageTk
import time
import os
from datetime import datetime
import subprocess
import threading


# ✅ Choose device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} for inference")

# ✅ Load custom model only (best.pt)
model = YOLO("yolov8n.pt")
model = YOLO("best.pt")

# ✅ Class counter
counts = {"Criollo": 0, "Forastero": 0, "Trinitario": 0}
prediction_interval = 3  # Increased to 3 seconds
last_pred_time = 0
last_predicted_frame = None
predicting = False
prediction_lock = threading.Lock()
detection_active = False
camera_ready = False
tk_image = None

# ✅ Ensure results folder exists
os.makedirs("results", exist_ok=True)

def save_frame_with_timestamp(frame, prefix="capture"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join("results", f"{prefix}_{timestamp}.jpg")
    cv2.imwrite(filename, frame)

def capture_image():
    global last_predicted_frame
    if last_predicted_frame is not None:
        save_frame_with_timestamp(last_predicted_frame, prefix="manual")

def open_results_folder():
    subprocess.Popen(f'explorer "{os.path.abspath("results")}"')

def show_logo():
    global tk_image
    try:
        logo = Image.open("cacao.jpg").resize((320, 320), Image.Resampling.LANCZOS)
        tk_image = ImageTk.PhotoImage(logo)
        video_label.configure(image=tk_image)
    except Exception as e:
        print(f"Logo load failed: {e}")

def start_detection():
    global detection_active
    detection_active = True
    detected_type_var.set("Detected: Starting...")

def stop_detection():
    global detection_active, last_predicted_frame
    detection_active = False
    detected_type_var.set("Detected: Stopped")
    last_predicted_frame = None
    show_logo()

def predict_and_update(frame):
    global predicting, last_predicted_frame, last_pred_time
    with prediction_lock:
        if predicting:
            return
        predicting = True
    try:
        # ✅ Run prediction on correct device with reduced size
        results = model(frame, imgsz=224, verbose=False, device=device)
        counts_local = {k: 0 for k in counts}
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                label = model.names[class_id].capitalize()
                confidence = float(box.conf[0])
                if confidence >= 0.25 and label in counts_local:
                    counts_local[label] += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    cv2.putText(frame, f"{label}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        counts.update(counts_local)
        criollo_var.set(f"Criollo: {counts['Criollo']}")
        forastero_var.set(f"Forastero: {counts['Forastero']}")
        trinitario_var.set(f"Trinitario: {counts['Trinitario']}")
        detected = max(counts, key=counts.get) if sum(counts.values()) else 'No beans'
        detected_type_var.set(f"Detected: {detected}")
        last_predicted_frame = frame.copy()
        last_pred_time = time.time()
    except Exception as e:
        print(f"Prediction error: {e}")
    predicting = False

frame_skip = 0  # Initialize frame skip variable

def update_frame():
    global last_pred_time, camera_ready, tk_image, frame_skip
    ret, frame = cap.read()
    if ret:
        frame_skip += 1
        # Skip every other frame
        if frame_skip % 2 != 0:
            root.after(30, update_frame)
            return
        if not camera_ready:
            camera_ready = True
        if detection_active and time.time() - last_pred_time >= prediction_interval:
            threading.Thread(target=predict_and_update, args=(frame.copy(),), daemon=True).start()
        display = last_predicted_frame if last_predicted_frame is not None else frame
        disp = cv2.resize(display, (320, 320))
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        tk_image = ImageTk.PhotoImage(Image.fromarray(rgb))
        video_label.configure(image=tk_image)
    else:
        show_logo()
    root.after(10, update_frame)  # ~10 FPS (reduced)

# ✅ GUI
root = tk.Tk()
root.title("Cacao Detection")
root.geometry("680x400")
root.configure(bg="#2E2E2E")

top_frame = tk.Frame(root, bg="#2E2E2E")
top_frame.pack(pady=10)
bottom_frame = tk.Frame(root, bg="#2E2E2E")
bottom_frame.pack(pady=10)

video_label = tk.Label(top_frame, bd=2, relief="solid")
video_label.grid(row=0, column=0, padx=10)
dashboard = tk.Frame(top_frame, bg="#2E2E2E", padx=10)
dashboard.grid(row=0, column=1, padx=10, sticky="n")

detected_type_var = tk.StringVar(value="Detected: Waiting")
criollo_var = tk.StringVar(value="Criollo: 0")
forastero_var = tk.StringVar(value="Forastero: 0")
trinitario_var = tk.StringVar(value="Trinitario: 0")

tk.Label(dashboard, text="Detection Summary", font=("Arial", 14, "bold"), fg="white", bg="#2E2E2E").pack(pady=(0,10))
for var in [criollo_var, forastero_var, trinitario_var, detected_type_var]:
    tk.Label(dashboard, textvariable=var, font=("Arial", 12), fg="white", bg="#2E2E2E").pack(pady=2)

tk.Button(dashboard, text="Capture Image", font=("Arial", 10), command=capture_image,
          bg="#1E90FF", fg="white", relief="flat", padx=8, pady=4, width=14).pack(pady=6)
tk.Button(dashboard, text="Results", font=("Arial", 10), command=open_results_folder,
          bg="#6A5ACD", fg="white", relief="flat", padx=8, pady=4, width=14).pack(pady=6)
tk.Button(dashboard, text="Exit", font=("Arial", 10), command=root.quit,
          bg="#FF6347", fg="white", relief="flat", padx=8, pady=4, width=14).pack(pady=(20,6))

tk.Button(bottom_frame, text="Start Detecting", font=("Arial",10,"bold"), command=start_detection,
          bg="#32CD32", fg="white", relief="flat", padx=12, pady=6, width=14).grid(row=0,column=0,padx=5)
tk.Button(bottom_frame, text="Stop Detecting", font=("Arial",10), command=stop_detection,
          bg="#A52A2A", fg="white", relief="flat", padx=12, pady=6, width=14).grid(row=0,column=1,padx=5)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
show_logo()
root.update()
update_frame()
root.mainloop()
cap.release()
