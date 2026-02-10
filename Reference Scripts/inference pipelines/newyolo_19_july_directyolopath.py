import customtkinter as ctk
from tkinter import filedialog, messagebox, ttk
import threading
import os
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import subprocess
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class YoloGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("YOLOv8 GUI Trainer & Inference")
        self.geometry("1200x800")

        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(expand=True, fill="both", padx=10, pady=10)

        self.train_tab = self.tabview.add("Train")
        self.infer_tab = self.tabview.add("Inference")

        self.create_train_tab()
        self.create_infer_tab()

    def create_train_tab(self):
        # File selection frame
        file_frame = ctk.CTkFrame(self.train_tab)
        file_frame.pack(pady=10, padx=10, fill="x")

        self.yaml_path = ctk.StringVar()
        self.model_path = ctk.StringVar(value="D:/yolo/yolov8n.pt")

        ctk.CTkLabel(file_frame, text="Dataset YAML").grid(row=0, column=0, padx=5)
        ctk.CTkEntry(file_frame, textvariable=self.yaml_path, width=400).grid(row=0, column=1, padx=5)
        ctk.CTkButton(file_frame, text="Browse", command=self.browse_yaml).grid(row=0, column=2, padx=5)

        ctk.CTkLabel(file_frame, text="Pretrained Model").grid(row=1, column=0, padx=5)
        ctk.CTkEntry(file_frame, textvariable=self.model_path, width=400).grid(row=1, column=1, padx=5)
        ctk.CTkButton(file_frame, text="Browse", command=self.browse_model).grid(row=1, column=2, padx=5)

        # Parameters
        param_frame = ctk.CTkFrame(self.train_tab)
        param_frame.pack(pady=10, padx=10, fill="x")

        self.epochs = ctk.IntVar(value=10)
        self.batch_size = ctk.IntVar(value=16)
        self.img_size = ctk.IntVar(value=640)
        self.project = ctk.StringVar(value="runs/train")
        self.name = ctk.StringVar(value="exp")
        self.device = ctk.StringVar(value="0")

        for i, (label, var) in enumerate([
            ("Epochs", self.epochs),
            ("Batch Size", self.batch_size),
            ("Image Size", self.img_size),
            ("Project Name", self.project),
            ("Experiment Name", self.name)
        ]):
            ctk.CTkLabel(param_frame, text=label).grid(row=i, column=0, sticky="e", padx=5, pady=2)
            ctk.CTkEntry(param_frame, textvariable=var).grid(row=i, column=1, padx=5, pady=2)

        ctk.CTkLabel(param_frame, text="Device").grid(row=5, column=0, sticky="e", padx=5)
        ctk.CTkOptionMenu(param_frame, values=["0", "1"], variable=self.device).grid(row=5, column=1, padx=5)

        # Buttons & Logs
        action_frame = ctk.CTkFrame(self.train_tab)
        action_frame.pack(pady=10, padx=10, fill="x")

        ctk.CTkButton(action_frame, text="Train", command=self.start_train_thread).pack(pady=5)
        self.train_progress = ctk.CTkProgressBar(action_frame)
        self.train_progress.pack(pady=5, fill="x")

        self.train_log = ctk.CTkTextbox(self.train_tab, height=200)
        self.train_log.pack(padx=10, pady=10, fill="both")

        # Chart
        self.show_chart = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(self.train_tab, text="Show Accuracy Chart", variable=self.show_chart).pack()

        self.fig, self.ax = plt.subplots()
        self.canvas_chart = FigureCanvasTkAgg(self.fig, master=self.train_tab)
        self.canvas_widget = self.canvas_chart.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)

    def create_infer_tab(self):
        self.video_path = ctk.StringVar()
        self.detect_model_path = ctk.StringVar(value="D:/yolo/yolov8n.pt")
        self.conf_threshold = ctk.DoubleVar(value=0.25)
        self.output_folder = ctk.StringVar(value="output")

        frame = ctk.CTkFrame(self.infer_tab)
        frame.pack(pady=10, padx=10, fill="x")

        ctk.CTkLabel(frame, text="Video File").grid(row=0, column=0, padx=5)
        ctk.CTkEntry(frame, textvariable=self.video_path, width=400).grid(row=0, column=1)
        ctk.CTkButton(frame, text="Browse", command=self.browse_video).grid(row=0, column=2)

        ctk.CTkLabel(frame, text="Model").grid(row=1, column=0, padx=5)
        ctk.CTkEntry(frame, textvariable=self.detect_model_path, width=400).grid(row=1, column=1)
        ctk.CTkButton(frame, text="Browse", command=self.browse_detect_model).grid(row=1, column=2)

        ctk.CTkLabel(frame, text="Confidence Threshold").grid(row=2, column=0)
        ctk.CTkSlider(frame, variable=self.conf_threshold, from_=0.0, to=1.0).grid(row=2, column=1)

        ctk.CTkLabel(frame, text="Output Folder").grid(row=3, column=0)
        ctk.CTkEntry(frame, textvariable=self.output_folder).grid(row=3, column=1)

        ctk.CTkButton(self.infer_tab, text="Detect", command=self.start_detect_thread).pack(pady=10)

    def browse_yaml(self):
        path = filedialog.askopenfilename(filetypes=[("YAML Files", "*.yaml")])
        if path:
            self.yaml_path.set(path)

    def browse_model(self):
        path = filedialog.askopenfilename(filetypes=[("PT Files", "*.pt")])
        if path:
            self.model_path.set(path)

    def browse_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])
        if path:
            self.video_path.set(path)

    def browse_detect_model(self):
        path = filedialog.askopenfilename(filetypes=[("PT Files", "*.pt")])
        if path:
            self.detect_model_path.set(path)

    def start_train_thread(self):
        threading.Thread(target=self.train_model, daemon=True).start()

    def train_model(self):
        self.train_progress.set(0)
        self.train_log.delete("0.0", "end")
        try:
            model = YOLO(self.model_path.get())
            results = model.train(
                data=self.yaml_path.get(),
                epochs=self.epochs.get(),
                imgsz=self.img_size.get(),
                batch=self.batch_size.get(),
                project=self.project.get(),
                name=self.name.get(),
                device=self.device.get(),
            )
            self.train_progress.set(1)
            self.train_log.insert("end", str(results))

            if self.show_chart.get():
                self.ax.clear()
                self.ax.plot(results.metrics["epoch"], results.metrics["accuracy"], label="Accuracy")
                self.ax.legend()
                self.canvas_chart.draw()

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def start_detect_thread(self):
        threading.Thread(target=self.run_detection, daemon=True).start()

    def run_detection(self):
        try:
            model = YOLO(self.detect_model_path.get())
            results = model.predict(
                source=self.video_path.get(),
                conf=self.conf_threshold.get(),
                save=True,
                project=self.output_folder.get()
            )
            messagebox.showinfo("Done", "Detection complete!")
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    app = YoloGUI()
    app.mainloop()