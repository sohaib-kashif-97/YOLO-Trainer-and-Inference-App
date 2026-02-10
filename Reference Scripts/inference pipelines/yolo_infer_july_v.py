import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
from ultralytics import YOLO
import threading
import os
import csv
from PIL import Image, ImageTk
import time
import subprocess

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

class YOLOApp:
    def __init__(self):
        self.model = None
        self.cap = None
        self.running = False
        self.selected_classes = []

        self.root = ctk.CTk()
        self.root.title("YOLO Inference & Training GUI")
        self.root.geometry("1400x800")

        self.left_frame = ctk.CTkFrame(self.root, width=300)
        self.left_frame.pack(side="left", fill="y", padx=10, pady=10)

        self.right_frame = ctk.CTkFrame(self.root)
        self.right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # Video frame
        self.video_label = ctk.CTkLabel(self.right_frame, text="")
        self.video_label.pack(padx=10, pady=10)

        # Log box
        self.log_box = ctk.CTkTextbox(self.right_frame, height=150)
        self.log_box.pack(fill="x", padx=10, pady=10)

        self.setup_controls()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()

    def setup_controls(self):
        ctk.CTkLabel(self.left_frame, text="Inference Controls", font=("Arial", 16)).pack(pady=10)
        self.load_button = ctk.CTkButton(self.left_frame, text="Load Model", command=self.load_model)
        self.load_button.pack(pady=5)

        self.file_button = ctk.CTkButton(self.left_frame, text="Select Video/Image", command=self.select_file)
        self.file_button.pack(pady=5)

        self.webcam_button = ctk.CTkButton(self.left_frame, text="Use Webcam", command=self.use_webcam)
        self.webcam_button.pack(pady=5)

        self.stop_button = ctk.CTkButton(self.left_frame, text="Stop", command=self.stop_video)
        self.stop_button.pack(pady=5)

        self.class_filter = ctk.CTkEntry(self.left_frame, placeholder_text="Filter classes (comma-separated)")
        self.class_filter.pack(pady=5)

        self.export_button = ctk.CTkButton(self.left_frame, text="Export Detections", command=self.export_csv)
        self.export_button.pack(pady=5)

        ctk.CTkLabel(self.left_frame, text="Training Controls", font=("Arial", 16)).pack(pady=20)
        self.yaml_entry = ctk.CTkEntry(self.left_frame, placeholder_text="YAML dataset file")
        self.yaml_entry.pack(pady=5)

        self.model_entry = ctk.CTkEntry(self.left_frame, placeholder_text="Model (e.g., yolov8n.pt)")
        self.model_entry.pack(pady=5)

        self.epochs_entry = ctk.CTkEntry(self.left_frame, placeholder_text="Epochs")
        self.epochs_entry.pack(pady=5)

        self.batch_entry = ctk.CTkEntry(self.left_frame, placeholder_text="Batch size")
        self.batch_entry.pack(pady=5)

        self.imgsz_entry = ctk.CTkEntry(self.left_frame, placeholder_text="Image size")
        self.imgsz_entry.pack(pady=5)

        self.project_entry = ctk.CTkEntry(self.left_frame, placeholder_text="Project name")
        self.project_entry.pack(pady=5)

        self.name_entry = ctk.CTkEntry(self.left_frame, placeholder_text="Experiment name")
        self.name_entry.pack(pady=5)

        self.device_entry = ctk.CTkEntry(self.left_frame, placeholder_text="Device (0=GPU, 1=CPU)")
        self.device_entry.pack(pady=5)

        self.train_button = ctk.CTkButton(self.left_frame, text="Train Model", command=self.train_model)
        self.train_button.pack(pady=10)

    def log(self, text):
        self.log_box.insert("end", text + "\n")
        self.log_box.see("end")

    def load_model(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.model = YOLO(file_path)
            self.log(f"Model loaded: {file_path}")

    def select_file(self):
        path = filedialog.askopenfilename()
        if path:
            self.run_video(path)

    def use_webcam(self):
        self.run_video(0)

    def stop_video(self):
        self.running = False

    def run_video(self, source):
        if self.model is None:
            messagebox.showwarning("Warning", "Please load a YOLO model first.")
            return

        self.cap = cv2.VideoCapture(source)
        self.running = True
        self.selected_classes = [x.strip().lower() for x in self.class_filter.get().split(',') if x.strip()]

        def process():
            frame_id = 0
            self.detections = []
            while self.running and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                results = self.model(frame)[0]
                annotated = frame.copy()
                for box in results.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = self.model.names[cls].lower()

                    if self.selected_classes and label not in self.selected_classes:
                        continue

                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    cv2.rectangle(annotated, tuple(xyxy[:2]), tuple(xyxy[2:]), (0,255,0), 2)
                    cv2.putText(annotated, f"{label} {conf:.2f}", tuple(xyxy[:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                    self.detections.append([frame_id, label])

                frame_id += 1
                image = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                img = ImageTk.PhotoImage(Image.fromarray(image))
                self.video_label.configure(image=img)
                self.video_label.image = img

                time.sleep(0.01)

            self.cap.release()
            self.running = False

        threading.Thread(target=process).start()

    def export_csv(self):
        if not hasattr(self, 'detections') or not self.detections:
            self.log("No detections to export.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".csv")
        if file_path:
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Frame", "Class"])
                writer.writerows(self.detections)
            self.log(f"Detections exported to {file_path}")

    def train_model(self):
        yaml = self.yaml_entry.get()
        model = self.model_entry.get()
        epochs = self.epochs_entry.get()
        batch = self.batch_entry.get()
        imgsz = self.imgsz_entry.get()
        project = self.project_entry.get()
        name = self.name_entry.get()
        device = self.device_entry.get()

        if not all([yaml, model, epochs, batch, imgsz, project, name, device]):
            messagebox.showwarning("Missing Info", "Please fill all training fields.")
            return

        cmd = [
            "yolo", "task=detect", "mode=train",
            f"model={model}",
            f"data={yaml}",
            f"epochs={epochs}",
            f"batch={batch}",
            f"imgsz={imgsz}",
            f"project={project}",
            f"name={name}",
            f"device={device}"
        ]

        def train():
            self.log("Starting training...\n")
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
            for line in process.stdout:
                self.log(line.strip())
            self.log("Training complete.")

        threading.Thread(target=train).start()

    def on_close(self):
        self.running = False
        self.root.destroy()

if __name__ == "__main__":
    YOLOApp()
