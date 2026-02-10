# YOLO Inference App with customtkinter, enhanced UI, drag-drop, export, and live detection overlay
import customtkinter as ctk
import tkinter.filedialog as fd
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2
import threading
import os
import datetime
import csv

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class YOLOApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("YOLOv8 Inference - Pro UI")
        self.geometry("1350x900")

        # State
        self.model = None
        self.cap = None
        self.stop_flag = False
        self.video_writer = None
        self.class_counter = {}
        self.detections_log = []

        # Variables
        self.model_path = ctk.StringVar()
        self.source_path = ctk.StringVar()
        self.device = ctk.StringVar(value="0")
        self.conf = ctk.DoubleVar(value=0.5)
        self.iou = ctk.DoubleVar(value=0.45)
        self.output_project = ctk.StringVar(value="runs/detect")
        self.output_name = ctk.StringVar(value="output")
        self.save_output = ctk.BooleanVar(value=True)

        self.create_ui()
        #self.bind("<Drop>", self.drop_handler)

    def create_ui(self):
        tabview = ctk.CTkTabview(self)
        tabview.pack(expand=True, fill="both", padx=10, pady=10)

        config_tab = tabview.add("Configuration")
        video_tab = tabview.add("Video Feed")
        log_tab = tabview.add("Logs")

        # --- Config Tab ---
        ctk.CTkLabel(config_tab, text="Model Path:").grid(row=0, column=0, sticky="e")
        ctk.CTkEntry(config_tab, textvariable=self.model_path, width=400).grid(row=0, column=1, padx=5)
        ctk.CTkButton(config_tab, text="Browse", command=self.browse_model).grid(row=0, column=2)

        ctk.CTkLabel(config_tab, text="Video Source:").grid(row=1, column=0, sticky="e")
        ctk.CTkEntry(config_tab, textvariable=self.source_path, width=400).grid(row=1, column=1, padx=5)
        ctk.CTkButton(config_tab, text="Browse", command=self.browse_video).grid(row=1, column=2)

        ctk.CTkLabel(config_tab, text="Device:").grid(row=2, column=0, sticky="e")
        ctk.CTkEntry(config_tab, textvariable=self.device, width=60).grid(row=2, column=1, sticky="w")

        ctk.CTkLabel(config_tab, text="Confidence Threshold:").grid(row=3, column=0, sticky="e")
        ctk.CTkSlider(config_tab, from_=0.0, to=1.0, variable=self.conf).grid(row=3, column=1, sticky="we")

        ctk.CTkLabel(config_tab, text="IoU Threshold:").grid(row=4, column=0, sticky="e")
        ctk.CTkSlider(config_tab, from_=0.0, to=1.0, variable=self.iou).grid(row=4, column=1, sticky="we")

        ctk.CTkLabel(config_tab, text="Output Folder:").grid(row=5, column=0, sticky="e")
        ctk.CTkEntry(config_tab, textvariable=self.output_project).grid(row=5, column=1, sticky="we")

        ctk.CTkLabel(config_tab, text="Output Name:").grid(row=6, column=0, sticky="e")
        ctk.CTkEntry(config_tab, textvariable=self.output_name).grid(row=6, column=1, sticky="we")

        ctk.CTkCheckBox(config_tab, text="Save Output Video", variable=self.save_output).grid(row=7, column=1, sticky="w")

        ctk.CTkButton(config_tab, text="▶ Start Inference", command=self.start_inference).grid(row=8, column=1, pady=10)
        ctk.CTkButton(config_tab, text="⛔ Stop", fg_color="red", command=self.stop_inference).grid(row=8, column=2, pady=10)

        ctk.CTkButton(config_tab, text="⭳ Export Detection Summary", command=self.export_csv).grid(row=9, column=1, pady=10)

        # --- Video Tab ---
        self.video_frame = ctk.CTkLabel(video_tab)
        self.video_frame.pack(pady=10)

        self.counter_box = ctk.CTkTextbox(video_tab, width=300, height=100)
        self.counter_box.pack(pady=5)

        # --- Log Tab ---
        self.log_text = ctk.CTkTextbox(log_tab, wrap="word")
        self.log_text.pack(expand=True, fill="both")

    def drop_handler(self, event):
        dropped_file = event.data.strip('{}')
        if dropped_file.endswith(".pt"):
            self.model_path.set(dropped_file)
        elif dropped_file.endswith(('.mp4', '.avi')):
            self.source_path.set(dropped_file)

    def browse_model(self):
        path = fd.askopenfilename(filetypes=[("YOLO Weights", "*.pt")])
        if path:
            self.model_path.set(path)

    def browse_video(self):
        path = fd.askopenfilename(filetypes=[("Videos", "*.mp4 *.avi")])
        if path:
            self.source_path.set(path)

    def log(self, msg):
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")

    def update_class_counter(self, names):
        self.class_counter.clear()
        for name in names:
            self.class_counter[name] = self.class_counter.get(name, 0) + 1
        self.counter_box.delete("1.0", "end")
        for cls, count in self.class_counter.items():
            self.counter_box.insert("end", f"{cls}: {count}\n")

    def export_csv(self):
        filename = f"{self.output_name.get()}_detections.csv"
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Frame", "Class"])
            writer.writerows(self.detections_log)
        self.log(f"Exported detections to {filename}")

    def start_inference(self):
        if not os.path.exists(self.model_path.get()) or not os.path.exists(self.source_path.get()):
            self.log("Model or video path invalid.")
            return

        self.stop_flag = False
        self.model = YOLO(self.model_path.get())
        self.cap = cv2.VideoCapture(self.source_path.get())
        self.detections_log.clear()

        if self.save_output.get():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_name = f"{self.output_project.get()}/{self.output_name.get()}_{datetime.datetime.now().strftime('%H%M%S')}.mp4"
            self.video_writer = cv2.VideoWriter(out_name, fourcc, 30.0, (int(self.cap.get(3)), int(self.cap.get(4))))
        else:
            self.video_writer = None

        threading.Thread(target=self.run_inference).start()

    def stop_inference(self):
        self.stop_flag = True
        self.log("Stopped by user.")

    def run_inference(self):
        frame_id = 0
        while self.cap.isOpened() and not self.stop_flag:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame_id += 1

            results = self.model.predict(
                source=frame,
                conf=self.conf.get(),
                iou=self.iou.get(),
                device=self.device.get(),
                save=False,
                stream=False
            )

            annotated_frame = results[0].plot()
            names = [results[0].names[int(cls)] for cls in results[0].boxes.cls.cpu().numpy()]
            self.update_class_counter(names)
            self.detections_log.extend([[frame_id, name] for name in names])

            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (0, 0), (300, 30), (0, 0, 0), -1)
            cv2.putText(overlay, f"Objects: {len(names)}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            annotated_frame = overlay

            if self.video_writer:
                self.video_writer.write(annotated_frame)

            img = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = img.resize((960, 540))
            img_tk = ImageTk.PhotoImage(image=img)
            self.video_frame.configure(image=img_tk)
            self.video_frame.image = img_tk

        self.cap.release()
        if self.video_writer:
            self.video_writer.release()
        self.log("Inference complete.")

if __name__ == '__main__':
    app = YOLOApp()
    app.mainloop()
