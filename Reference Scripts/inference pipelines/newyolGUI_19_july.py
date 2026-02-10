import customtkinter as ctk
from tkinter import filedialog, messagebox, ttk
import os
import threading
from ultralytics import YOLO
import cv2
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class YoloGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("YOLOv8 GUI App")
        self.geometry("1000x700")

        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(expand=True, fill="both", padx=10, pady=10)

        self.train_tab = self.tabview.add("Train")
        self.infer_tab = self.tabview.add("Inference")

        self.create_train_tab()
        self.create_infer_tab()

    def create_train_tab(self):
        # Browse files
        self.yaml_path = ctk.StringVar()
        self.model_path = ctk.StringVar()

        ctk.CTkLabel(self.train_tab, text="Dataset YAML File").pack(pady=5)
        yaml_frame = ctk.CTkFrame(self.train_tab)
        yaml_frame.pack(pady=5, fill='x', padx=10)
        ctk.CTkEntry(yaml_frame, textvariable=self.yaml_path).pack(side='left', fill='x', expand=True)
        ctk.CTkButton(yaml_frame, text="Browse", command=self.browse_yaml).pack(side='right')

        ctk.CTkLabel(self.train_tab, text="Pretrained Model (.pt)").pack(pady=5)
        model_frame = ctk.CTkFrame(self.train_tab)
        model_frame.pack(pady=5, fill='x', padx=10)
        ctk.CTkEntry(model_frame, textvariable=self.model_path).pack(side='left', fill='x', expand=True)
        ctk.CTkButton(model_frame, text="Browse", command=self.browse_model).pack(side='right')

        # Training params
        self.epochs = ctk.IntVar(value=50)
        self.batch = ctk.IntVar(value=16)
        self.imgsz = ctk.IntVar(value=640)
        self.project = ctk.StringVar(value="MyProject")
        self.exp_name = ctk.StringVar(value="Experiment1")
        self.device = ctk.StringVar(value="0")

        for label, var in [
            ("Epochs", self.epochs),
            ("Batch Size", self.batch),
            ("Image Size", self.imgsz),
            ("Project Name", self.project),
            ("Experiment Name", self.exp_name)
        ]:
            ctk.CTkLabel(self.train_tab, text=label).pack(pady=2)
            ctk.CTkEntry(self.train_tab, textvariable=var).pack(fill='x', padx=10)

        ctk.CTkLabel(self.train_tab, text="Device (0=GPU, 1=CPU)").pack(pady=2)
        ctk.CTkOptionMenu(self.train_tab, values=["0", "1"], variable=self.device).pack(pady=2)

        # Train button
        ctk.CTkButton(self.train_tab, text="Train", command=self.start_training).pack(pady=10)

        # Log and chart
        self.log_box = ctk.CTkTextbox(self.train_tab, height=150)
        self.log_box.pack(padx=10, pady=10, fill='x')

        self.fig, self.ax = plt.subplots(figsize=(5, 2))
        self.loss_vals = []
        self.chart_toggle = ctk.BooleanVar(value=True)

        toggle_frame = ctk.CTkFrame(self.train_tab)
        toggle_frame.pack()
        ctk.CTkCheckBox(toggle_frame, text="Show Accuracy Chart", variable=self.chart_toggle, command=self.toggle_chart).pack()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.train_tab)
        self.canvas.get_tk_widget().pack(pady=10)

    def create_infer_tab(self):
        pass  # Already long, inference code unchanged here

    def browse_yaml(self):
        file = filedialog.askopenfilename(filetypes=[("YAML files", "*.yaml")])
        if file:
            self.yaml_path.set(file)

    def browse_model(self):
        file = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pt")])
        if file:
            self.model_path.set(file)

    def start_training(self):
        thread = threading.Thread(target=self.run_training)
        thread.start()

    def run_training(self):
        try:
            if not self.yaml_path.get() or not self.model_path.get():
                messagebox.showerror("Error", "Please select YAML and model files.")
                return

            model = YOLO(self.model_path.get())

            def custom_callback(epoch, metrics):
                if metrics and "box_loss" in metrics:
                    self.loss_vals.append(metrics['box_loss'])
                    if self.chart_toggle.get():
                        self.update_chart()

            self.log("Training started...")
            model.train(
                data=self.yaml_path.get(),
                epochs=self.epochs.get(),
                imgsz=self.imgsz.get(),
                batch=self.batch.get(),
                project=self.project.get(),
                name=self.exp_name.get(),
                device=int(self.device.get()),
                verbose=True,
                callbacks=[custom_callback]
            )
            self.log("Training complete.")
        except Exception as e:
            self.log(f"Error: {e}")

    def update_chart(self):
        self.ax.clear()
        self.ax.plot(self.loss_vals, label="Box Loss")
        self.ax.set_title("Training Loss")
        self.ax.legend()
        self.canvas.draw()

    def toggle_chart(self):
        if self.chart_toggle.get():
            self.canvas.get_tk_widget().pack(pady=10)
        else:
            self.canvas.get_tk_widget().pack_forget()

    def log(self, msg):
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")

if __name__ == "__main__":
    app = YoloGUI()
    app.mainloop()
