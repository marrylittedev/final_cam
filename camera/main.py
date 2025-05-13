import cv2
import torch
import tkinter as tk
from PIL import Image, ImageTk
import time
import os
from datetime import datetime
import threading
import subprocess
from ultralytics import YOLO
import servo  # Your custom servo.py module

# ✅ Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("best.pt")

# ✅ Initialize state
counts = {"Criollo": 0, "Forastero": 0, "Trinitario": 0}
detection_active = False
predicting = False
last_predicted_frame = None
last_pred_time = 0
prediction_interval = 3  # seconds
prediction_lock = threading.Lock()

# ✅ Results directory
os.makedirs("results", exist_ok=True)

# ✅ GUI setup
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

# ✅ Add dashboard labels
tk.Label(dashboard, text="Detection Summary", font=("Arial", 14, "bold"), fg="white", bg="#2E2E2E").pack(pady=(0,10))
for var in [criollo_var, forastero_var, trinitario_var, detected_type_var]:
    tk.Label(dashboard, textvariable=var, font=("Arial", 12), fg="white", bg="#2E2E2E").pack(pady=2)

def show_logo():
    try:
        logo = Image.open("cacao.jpg").resize((320, 320), Image.Resampling.LANCZOS)
        logo_image = ImageTk.PhotoImage(logo)
        video_label.configure(image=logo_image)
        video_label.image = logo_image
    except:
        pass

def save_frame(frame, prefix="capture"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("results", f"{prefix}_{timestamp}.jpg")
    cv2.imwrite(path, frame)

def capture_image():
    if last_predicted_frame is not None:
        save_frame(last_predicted_frame, prefix="manual")

def open_results_folder():
    subprocess.Popen(f'explorer "{os.path.abspath("results")}"')

def start_detection():
    global detection_active
    detection_active = True
    detected_type_var.set("Detected: Starting...")

def stop_detection():
    global detection_active
    detection_active = False
    detected_type_var.set("Detected: Stopped")
    show_logo()

def move_servo(label):
    if label == "Criollo":
        servo.move_criollo()
    elif label == "Forastero":
        servo.move_forastero()
    elif label == "Trinitario":
        servo.move_trinitario()

def predict_and_update(frame):
    global predicting, last_predicted_frame, last_pred_time
    with prediction_lock:
        if predicting:
            return
        predicting = True
    try:
        results = model(frame, imgsz=224, verbose=False, device=device)
        counts_local = {k: 0 for k in counts}
        detected = None
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id].capitalize()
                confidence = float(box.conf[0])
                if confidence >= 0.25 and label in counts_local:
                    counts_local[label] += 1
        counts.update(counts_local)
        criollo_var.set(f"Criollo: {counts['Criollo']}")
        forastero_var.set(f"Forastero: {counts['Forastero']}")
        trinitario_var.set(f"Trinitario: {counts['Trinitario']}")
        detected = max(counts_local, key=counts_local.get) if sum(counts_local.values()) else "No beans"
        detected_type_var.set(f"Detected: {detected}")
        last_predicted_frame = frame.copy()
        last_pred_time = time.time()

        # ✅ Automatic servo control
        if detected in counts_local and counts_local[detected] > 0:
            move_servo(detected)
    except Exception as e:
        print(f"Prediction error: {e}")
    predicting = False

frame_skip = 0
def update_frame():
    global last_pred_time, frame_skip
    ret, frame = cap.read()
    if not ret:
        show_logo()
        root.after(10, update_frame)
        return

    frame_skip += 1
    if frame_skip % 2 != 0:
        root.after(30, update_frame)
        return

    if detection_active and time.time() - last_pred_time >= prediction_interval:
        threading.Thread(target=predict_and_update, args=(frame.copy(),), daemon=True).start()

    display = last_predicted_frame if last_predicted_frame is not None else frame
    resized = cv2.resize(display, (320, 320))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_tk = ImageTk.PhotoImage(Image.fromarray(rgb))
    video_label.configure(image=img_tk)
    video_label.image = img_tk
    root.after(10, update_frame)

# ✅ Manual servo keys
def on_key(event):
    key = event.char.lower()
    if key == 'c':
        move_servo("Criollo")
    elif key == 'f':
        move_servo("Forastero")
    elif key == 't':
        move_servo("Trinitario")

# ✅ Buttons
tk.Button(dashboard, text="Capture Image", font=("Arial", 10), command=capture_image,
          bg="#1E90FF", fg="white", relief="flat", width=14).pack(pady=6)
tk.Button(dashboard, text="Results", font=("Arial", 10), command=open_results_folder,
          bg="#6A5ACD", fg="white", relief="flat", width=14).pack(pady=6)
tk.Button(dashboard, text="Exit", font=("Arial", 10), command=root.quit,
          bg="#FF6347", fg="white", relief="flat", width=14).pack(pady=(20,6))
tk.Button(bottom_frame, text="Start Detecting", font=("Arial", 10, "bold"), command=start_detection,
          bg="#32CD32", fg="white", relief="flat", width=14).grid(row=0, column=0, padx=5)
tk.Button(bottom_frame, text="Stop Detecting", font=("Arial", 10), command=stop_detection,
          bg="#A52A2A", fg="white", relief="flat", width=14).grid(row=0, column=1, padx=5)

# ✅ Video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

root.bind("<Key>", on_key)
show_logo()
root.after(100, update_frame)
root.mainloop()
cap.release()
