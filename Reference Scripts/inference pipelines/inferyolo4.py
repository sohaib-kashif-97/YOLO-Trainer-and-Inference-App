import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from ultralytics import YOLO
import cv2
from PIL import Image, ImageTk, ImageDraw, ImageFont
import threading
import os
import datetime

class YOLOApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 Inference UI")
        self.root.geometry("1200x800")
        self.model = None
        self.stop_flag = False
        self.cap = None
        self.video_thread = None

        self.model_path = tk.StringVar()
        self.source_path = tk.StringVar()
        self.device = tk.StringVar(value="0")
        self.confidence = tk.DoubleVar(value=0.70)
        self.iou = tk.DoubleVar(value=0.45)
        self.output_project = tk.StringVar(value="runs/detect")
        self.output_name = tk.StringVar(value="tankoutput")
        self.save_output = tk.BooleanVar(value=True)

        self.class_counter = {}

        self.setup_ui()

    def setup_ui(self):
        config_frame = tk.LabelFrame(self.root, text="Inference Configuration", padx=10, pady=10)
        config_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(config_frame, text="Model (.pt):").grid(row=0, column=0, sticky="e")
        tk.Entry(config_frame, textvariable=self.model_path, width=50).grid(row=0, column=1)
        tk.Button(config_frame, text="Browse", command=self.browse_model).grid(row=0, column=2)

        tk.Label(config_frame, text="Source Video:").grid(row=1, column=0, sticky="e")
        tk.Entry(config_frame, textvariable=self.source_path, width=50).grid(row=1, column=1)
        tk.Button(config_frame, text="Browse", command=self.browse_source).grid(row=1, column=2)

        tk.Label(config_frame, text="Device:").grid(row=2, column=0, sticky="e")
        tk.Entry(config_frame, textvariable=self.device, width=10).grid(row=2, column=1, sticky="w")

        tk.Label(config_frame, text="Confidence:").grid(row=3, column=0, sticky="e")
        tk.Scale(config_frame, variable=self.confidence, from_=0, to=1, resolution=0.01, orient="horizontal", length=300).grid(row=3, column=1, sticky="w")

        tk.Label(config_frame, text="IoU:").grid(row=4, column=0, sticky="e")
        tk.Scale(config_frame, variable=self.iou, from_=0, to=1, resolution=0.01, orient="horizontal", length=300).grid(row=4, column=1, sticky="w")

        tk.Label(config_frame, text="Output Project:").grid(row=5, column=0, sticky="e")
        tk.Entry(config_frame, textvariable=self.output_project, width=30).grid(row=5, column=1, sticky="w")

        tk.Label(config_frame, text="Output Name:").grid(row=6, column=0, sticky="e")
        tk.Entry(config_frame, textvariable=self.output_name, width=30).grid(row=6, column=1, sticky="w")

        tk.Checkbutton(config_frame, text="Save Output Video", variable=self.save_output).grid(row=7, column=1, sticky="w")

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        self.start_btn = tk.Button(btn_frame, text="Run Inference", bg="green", fg="white", command=self.start_inference)
        self.start_btn.pack(side="left", padx=10)
        self.stop_btn = tk.Button(btn_frame, text="Stop Inference", bg="red", fg="white", command=self.stop_inference, state="disabled")
        self.stop_btn.pack(side="left", padx=10)

        self.video_frame = tk.Label(self.root)
        self.video_frame.pack(pady=10)

        self.counter_text = scrolledtext.ScrolledText(self.root, height=5, width=30)
        self.counter_text.pack(side="right", padx=10, pady=5)

        self.console = scrolledtext.ScrolledText(self.root, height=10)
        self.console.pack(fill="both", padx=10, pady=5)

    def browse_model(self):
        path = filedialog.askopenfilename(filetypes=[("YOLO Model", "*.pt")])
        if path:
            self.model_path.set(path)

    def browse_source(self):
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi")])
        if path:
            self.source_path.set(path)

    def log(self, msg):
        self.console.insert(tk.END, msg + "\n")
        self.console.see(tk.END)

    def update_class_counter(self, names):
        self.class_counter.clear()
        for name in names:
            self.class_counter[name] = self.class_counter.get(name, 0) + 1
        self.counter_text.delete(1.0, tk.END)
        for cls, count in self.class_counter.items():
            self.counter_text.insert(tk.END, f"{cls}: {count}\n")

    def start_inference(self):
        if not os.path.exists(self.model_path.get()) or not os.path.exists(self.source_path.get()):
            messagebox.showerror("Error", "Model or video file not found.")
            return

        self.stop_flag = False
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.console.delete(1.0, tk.END)
        self.counter_text.delete(1.0, tk.END)

        self.model = YOLO(self.model_path.get())
        self.cap = cv2.VideoCapture(self.source_path.get())

        if self.save_output.get():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_name = f"{self.output_project.get()}/{self.output_name.get()}_{datetime.datetime.now().strftime('%H%M%S')}.mp4"
            self.video_writer = cv2.VideoWriter(out_name, fourcc, 30.0,
                                                 (int(self.cap.get(3)), int(self.cap.get(4))))
        else:
            self.video_writer = None

        self.video_thread = threading.Thread(target=self.run_video_inference)
        self.video_thread.start()

    def stop_inference(self):
        self.stop_flag = True
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.log("Inference stopped by user.")

    def run_video_inference(self):
        while self.cap.isOpened() and not self.stop_flag:
            ret, frame = self.cap.read()
            if not ret:
                break

            results = self.model.predict(
                source=frame,
                conf=self.confidence.get(),
                iou=self.iou.get(),
                device=self.device.get(),
                save=False,
                stream=False
            )

            annotated_frame = results[0].plot()
            names = [results[0].names[int(cls)] for cls in results[0].boxes.cls.cpu().numpy()]
            self.update_class_counter(names)

            # Draw top-right object count
            count_overlay = annotated_frame.copy()
            cv2.rectangle(count_overlay, (0, 0), (250, 30), (0, 0, 0), -1)
            label_text = f"Objects: {len(names)}"
            cv2.putText(count_overlay, label_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            annotated_frame = count_overlay

            if self.video_writer:
                self.video_writer.write(annotated_frame)

            img = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            self.video_frame.configure(image=img_tk)
            self.video_frame.image = img_tk

        self.cap.release()
        if self.video_writer:
            self.video_writer.release()

        self.stop_btn.config(state="disabled")
        self.start_btn.config(state="normal")
        self.log("Inference completed.")

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOApp(root)
    root.mainloop()