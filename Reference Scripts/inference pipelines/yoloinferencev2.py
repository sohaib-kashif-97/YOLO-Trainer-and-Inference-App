import os
import cv2
import threading
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import pandas as pd
import time

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class YOLOApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("YOLOv8 Inference and Training App")
        self.geometry("1350x900")

        self.model = None
        self.cap = None
        self.running = False
        self.class_filter = []

        self.sidebar = ctk.CTkFrame(self, width=250)
        self.sidebar.pack(side="left", fill="y")

        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(side="right", fill="both", expand=True)

        self.infer_tab = self.tabview.add("Inference")
        self.train_tab = self.tabview.add("Training")

        self.setup_sidebar()
        self.setup_infer_tab()
        self.setup_train_tab()

    def setup_sidebar(self):
        ctk.CTkLabel(self.sidebar, text="YOLOv8 GUI", font=("Arial", 20)).pack(pady=20)

        self.load_model_button = ctk.CTkButton(self.sidebar, text="Load YOLO Model", command=self.load_model)
        self.load_model_button.pack(pady=10)

        self.drag_drop_label = ctk.CTkLabel(self.sidebar, text="Drop Image/Video Here")
        self.drag_drop_label.pack(pady=10)

        self.webcam_button = ctk.CTkButton(self.sidebar, text="Start Webcam", command=self.toggle_webcam)
        self.webcam_button.pack(pady=10)

        self.class_entry = ctk.CTkEntry(self.sidebar, placeholder_text="Filter classes (e.g. person, car)")
        self.class_entry.pack(pady=10)

        self.export_button = ctk.CTkButton(self.sidebar, text="Export Summary", command=self.export_summary)
        self.export_button.pack(pady=10)

    def setup_infer_tab(self):
        self.video_frame = ctk.CTkLabel(self.infer_tab)
        self.video_frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.log_box = ctk.CTkTextbox(self.infer_tab, height=200)
        self.log_box.pack(padx=10, pady=(0, 10), fill="x")

        self.fps_label = ctk.CTkLabel(self.infer_tab, text="FPS: 0")
        self.fps_label.pack(pady=(0, 5))

    def setup_train_tab(self):
        self.train_frame = ctk.CTkFrame(self.train_tab)
        self.train_frame.pack(pady=20, padx=20)

        def browse_file(entry):
            path = filedialog.askopenfilename()
            entry.delete(0, "end")
            entry.insert(0, path)

        def browse_folder(entry):
            path = filedialog.askdirectory()
            entry.delete(0, "end")
            entry.insert(0, path)

        self.yaml_entry = ctk.CTkEntry(self.train_frame, placeholder_text="Dataset YAML path")
        self.yaml_entry.grid(row=0, column=0, padx=10, pady=5)
        ctk.CTkButton(self.train_frame, text="Browse", command=lambda: browse_file(self.yaml_entry)).grid(row=0, column=1, padx=5)

        self.model_entry = ctk.CTkEntry(self.train_frame, placeholder_text="Model (e.g., yolov8n.pt)")
        self.model_entry.grid(row=1, column=0, padx=10, pady=5)
        ctk.CTkButton(self.train_frame, text="Browse", command=lambda: browse_file(self.model_entry)).grid(row=1, column=1, padx=5)

        self.epochs_entry = ctk.CTkEntry(self.train_frame, placeholder_text="Epochs")
        self.epochs_entry.grid(row=2, column=0, padx=10, pady=5)

        self.batch_entry = ctk.CTkEntry(self.train_frame, placeholder_text="Batch Size")
        self.batch_entry.grid(row=3, column=0, padx=10, pady=5)

        self.imgsz_entry = ctk.CTkEntry(self.train_frame, placeholder_text="Image Size")
        self.imgsz_entry.grid(row=4, column=0, padx=10, pady=5)

        self.project_entry = ctk.CTkEntry(self.train_frame, placeholder_text="Project folder")
        self.project_entry.grid(row=5, column=0, padx=10, pady=5)
        ctk.CTkButton(self.train_frame, text="Browse", command=lambda: browse_folder(self.project_entry)).grid(row=5, column=1, padx=5)

        self.name_entry = ctk.CTkEntry(self.train_frame, placeholder_text="Experiment name")
        self.name_entry.grid(row=6, column=0, padx=10, pady=5)

        self.device_entry = ctk.CTkEntry(self.train_frame, placeholder_text="Device (0=GPU, 1=CPU)")
        self.device_entry.grid(row=7, column=0, padx=10, pady=5)

        self.start_train_button = ctk.CTkButton(self.train_frame, text="Start Training", command=self.train_model)
        self.start_train_button.grid(row=8, column=0, columnspan=2, pady=20)

    def load_model(self):
        model_path = filedialog.askopenfilename()
        if model_path:
            self.model = YOLO(model_path)
            self.log_box.insert("end", f"Loaded model: {model_path}\n")

    def toggle_webcam(self):
        if not self.running:
            self.running = True
            self.cap = cv2.VideoCapture(0)
            threading.Thread(target=self.infer_loop).start()
            self.webcam_button.configure(text="Stop Webcam")
        else:
            self.running = False
            self.cap.release()
            self.webcam_button.configure(text="Start Webcam")

    def infer_loop(self):
        prev_time = time.time()
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            results = self.model(frame)[0]
            frame = self.draw_boxes(frame, results)

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            self.fps_label.configure(text=f"FPS: {fps:.2f}")

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_frame.configure(image=imgtk)
            self.video_frame.image = imgtk

    def draw_boxes(self, frame, results):
        names = self.model.names
        selected = self.class_entry.get().split(',') if self.class_entry.get() else []
        for box in results.boxes:
            cls = int(box.cls[0])
            label = names[cls]
            if selected and label not in selected:
                continue
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(frame, tuple(xyxy[:2]), tuple(xyxy[2:]), (0, 255, 0), 2)
            cv2.putText(frame, label, tuple(xyxy[:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame

    def export_summary(self):
        summary = self.log_box.get("1.0", "end").strip()
        path = filedialog.asksaveasfilename(defaultextension=".txt")
        if path:
            with open(path, 'w') as f:
                f.write(summary)
            messagebox.showinfo("Export", "Summary exported successfully.")

    def train_model(self):
        yaml = self.yaml_entry.get()
        model = self.model_entry.get()
        epochs = self.epochs_entry.get()
        batch = self.batch_entry.get()
        imgsz = self.imgsz_entry.get()
        project = self.project_entry.get()
        name = self.name_entry.get()
        device = self.device_entry.get()

        cmd = f"yolo task=detect mode=train model={model} data={yaml} epochs={epochs} batch={batch} imgsz={imgsz} project={project} name={name} device={device}"
        self.log_box.insert("end", f"\nRunning: {cmd}\n")
        os.system(cmd)


if __name__ == "__main__":
    app = YOLOApp()
    app.mainloop()