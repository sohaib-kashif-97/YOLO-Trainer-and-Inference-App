import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import queue
import subprocess
import sys
import os
import re
from PIL import Image, ImageTk, UnidentifiedImageError
import time
import signal
import logging
import datetime # For elapsed time formatting
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import cv2 # Added for video frame reading/display if needed
import glob # Added for finding result files
from ultralytics import YOLO
import cv2
try:
    import psutil
except ImportError:
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Dependency Missing", "The 'psutil' library is required for CPU/RAM monitoring.\nPlease install it using:\n\npip install psutil")
        root.destroy()
    except Exception:
        print("ERROR: The 'psutil' library is required but not found.")
        print("Please install it using: pip install psutil")
    sys.exit(1)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s [%(threadName)s] - %(message)s')

# --- Configuration ---
DEFAULT_PROJECT_NAME = "runs/train"
DEFAULT_INFER_PROJECT_NAME = "runs/detect" # Separate default for inference
YOLO_MODELS_V8 = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
OUTPUT_UPDATE_INTERVAL_MS = 100
LIVE_DATA_UPDATE_INTERVAL_MS = 1000
# VALIDATION_IMAGE_CHECK_DELAY_MS = 200 # Not needed
# VALIDATION_PREVIEW_SIZE = (320, 240) # Not needed
# VALIDATION_HIGHLIGHT_COLOR = "#4CAF50" # Not needed
# VALIDATION_HIGHLIGHT_DURATION_MS = 1500 # Not needed
GPU_MONITOR_INTERVAL_S = 2
ICON_SIZE = (20, 20)
SEPARATOR_HEIGHT = 2
PAD_X = 10
PAD_Y = 5
PAD_Y_SECTION = (10, 5)
INFER_OUTPUT_DISPLAY_SIZE = (640, 480) # Target size for inference display label

# Image/Video Extensions
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff']
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']


# Status Indicator Colors
STATUS_COLORS = {
    "Idle": "gray",
    "Training": "#2196F3", # Blue
    "Stopping": "#FF9800", # Orange
    "Finished": "#4CAF50", # Green
    "Error": "#F44336", # Red
    "Starting": "#00BCD4", # Cyan
    "Inferring": "#9C27B0", # Purple
}

# System Status Bar Colors
SYSTEM_BAR_GREEN = "#4CAF50"
SYSTEM_BAR_BACKGROUND = "gray30"
SYSTEM_BAR_BORDER_WIDTH = 1
SYSTEM_BAR_HEIGHT = 12 # Define height for consistency

# --- Helper Functions ---
def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def is_int(string):
    try:
        int(string)
        return True
    except ValueError:
        return False

def clean_ansi_codes(text): ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])'); return ansi_escape.sub('', text)
def format_time(seconds):
    if seconds is None or seconds < 0: return "--:--:--"
    return str(datetime.timedelta(seconds=int(seconds)))

def load_icon(filename, size=ICON_SIZE):
    # Allow searching in a potential 'icons' subdirectory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(script_dir, filename),
        os.path.join(script_dir, 'icons', filename),
        os.path.abspath(filename)
    ]
    filepath = None
    for p in possible_paths:
        if os.path.exists(p):
            filepath = p
            break

    if not filepath:
        logging.warning(f"Icon file not found: {filename} (searched {possible_paths})")
        return None
    try:
        image = Image.open(filepath).resize(size, Image.Resampling.LANCZOS)
        if image.mode != 'RGBA': image = image.convert('RGBA')
        return ctk.CTkImage(light_image=image, dark_image=image, size=size)
    except Exception as e:
        logging.error(f"Error loading icon {filename} from {filepath}: {e}")
        return None

def get_source_type(filepath):
    """Determines if a file path points to an image or video."""
    if not filepath or not isinstance(filepath, str):
        return None
    _, ext = os.path.splitext(filepath.lower())
    if ext in IMAGE_EXTENSIONS:
        return 'image'
    if ext in VIDEO_EXTENSIONS:
        return 'video'
    return None # Unknown or unsupported


# --- Main Application Class ---
class YoloTrainerApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("YOLO Live Trainer & Infer Pro") # Updated Title
        self.geometry("1350x1000")
        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")

        # --- Load Icons ---
        self.icon_play = load_icon("play_icon.png")
        self.icon_stop = load_icon("stop_icon.png")
        self.icon_folder = load_icon("folder_icon.png")
        self.icon_browse = load_icon("browse_icon.png")
        self.icon_infer = load_icon("infer_icon.png") # Add inference icon if available

        # --- State Variables ---
        # Training States
        self.training_thread = None; self.training_process = None
        self.log_queue = queue.Queue(); self.stop_event = threading.Event()
        self.current_experiment_path = None; self.is_training = False
        self.custom_model_path = None; self.plot_update_job = None
        self.live_data_update_job = None;
        self.current_epoch = 0; self.total_epochs = 0
        self.start_time = None; self.elapsed_time_str = "00:00:00"
        self.etr_str = "--:--:--"; self.current_speed_str = "N/A"
        self.latest_map50 = None; self.latest_map50_95 = None
        self.metrics_data = { k: [] for k in ['epoch', 'train_box_loss', 'train_cls_loss', 'train_dfl_loss', 'val_map50', 'val_map50_95'] }
        self.last_val_image_epoch = -1
        self.current_train_status = "Idle" # Renamed from current_status

        # Inference States
        self.inference_thread = None; self.inference_process = None
        self.infer_log_queue = queue.Queue(); self.infer_stop_event = threading.Event()
        self.infer_results_path = None; self.is_inferring = False
        self.custom_infer_model_path = None
        self.current_infer_status = "Idle"
        self.infer_source_path = None
        self.infer_source_type = None # 'image' or 'video'
        self.infer_output_update_job = None
        self.last_displayed_frame_num = -1
        self.current_inference_frame_count = 0 # For video progress
        self.total_inference_frames = 0 # For video progress

        # Shared/System States
        self.gpu_monitor_thread = None
        self.gpu_stop_event = threading.Event()
        self.system_stats = {}
        self.system_status_widgets = {}
        self.appearance_mode_tracker = ctk.StringVar(value=ctk.get_appearance_mode())

        # --- UI Creation ---
        # Main layout: Configure grid for tabview
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Create Tabview
        self.tab_view = ctk.CTkTabview(self, anchor="nw")
        self.tab_view.grid(row=0, column=0, padx=20, pady=(0, 20), sticky="nsew")

        self.tab_view.add("Train")
        self.tab_view.add("Infer")

        # Set weights for tab content frames
        self.tab_view.tab("Train").grid_columnconfigure(1, weight=1) # Status panel takes more space
        self.tab_view.tab("Train").grid_rowconfigure(0, weight=1)
        self.tab_view.tab("Infer").grid_columnconfigure(1, weight=3) # Output panel takes more space
        self.tab_view.tab("Infer").grid_rowconfigure(0, weight=1)


        # --- Populate Tabs ---
        self._create_train_tab_content(self.tab_view.tab("Train"))
        self._create_infer_tab_content(self.tab_view.tab("Infer"))

        # --- Initial State ---
        self._update_train_status_indicator("Idle")
        self._update_infer_status_indicator("Idle")
        self.queue_check_job = self.after(OUTPUT_UPDATE_INTERVAL_MS, self._process_queues) # Combined queue processor
        self._start_system_monitor()

        self.infer_run_id = 0  # Unique ID for each inference run

    #+++++++++++++++++++++++++ display class++++++++++++++++++++++++++++

    #+++++++++++++++++++++++++display class end++++++++++++++++++++++++

    def _create_train_tab_content(self, tab_frame):
        """Creates the configuration and status panels within the Train tab."""
        self._create_config_panel(tab_frame) # Pass the tab frame as parent
        self._create_status_panel(tab_frame) # Pass the tab frame as parent

    def _create_infer_tab_content(self, tab_frame):
        """Creates the inference configuration and output panels within the Infer tab."""
        self._create_infer_config_panel(tab_frame) # Pass the tab frame as parent
        self._create_infer_output_panel(tab_frame) # Pass the tab frame as parent

    # --- (_create_section_header remains the same) ---
    def _create_section_header(self, parent, text):
        header = ctk.CTkLabel(parent, text=text, font=ctk.CTkFont(size=16, weight="bold"), anchor="w")
        return header

    # --- Training Tab UI Creation (Modified to accept parent) ---
    def _create_config_panel(self, parent_frame): # Added parent_frame argument
        config_outer_frame = ctk.CTkFrame(parent_frame, width=380, corner_radius=10)
        config_outer_frame.grid(row=0, column=0, padx=(0, 10), pady=0, sticky="nsew") # Use parent padding
        config_outer_frame.grid_rowconfigure(1, weight=1) # Allow content frame to expand
        config_outer_frame.grid_propagate(False)

        config_header = self._create_section_header(config_outer_frame, "Training Configuration")
        config_header.pack(fill="x", padx=PAD_X, pady=PAD_Y_SECTION)

        config_frame = ctk.CTkFrame(config_outer_frame, fg_color="transparent")
        config_frame.pack(fill="both", expand=True, padx=0, pady=0)
        config_frame.grid_columnconfigure(1, weight=1)
        config_frame.grid_rowconfigure(12, weight=1) # Spacer row before system monitor

        # --- Configuration Fields (use self. variables) ---
        row_idx = 0
        ctk.CTkLabel(config_frame, text="Dataset YAML:", anchor="w").grid(row=row_idx, column=0, padx=(PAD_X, PAD_Y), pady=PAD_Y, sticky="w")
        self.dataset_yaml_entry = ctk.CTkEntry(config_frame, placeholder_text="path/to/dataset.yaml")
        self.dataset_yaml_entry.grid(row=row_idx, column=1, padx=PAD_Y, pady=PAD_Y, sticky="ew")
        ctk.CTkButton(config_frame, text="Browse", image=self.icon_browse, compound="left", width=90, command=self._browse_dataset_yaml).grid(row=row_idx, column=2, padx=(PAD_Y, PAD_X), pady=PAD_Y)
        row_idx += 1
        ctk.CTkLabel(config_frame, text="Model/Weights:", anchor="w").grid(row=row_idx, column=0, padx=(PAD_X, PAD_Y), pady=PAD_Y, sticky="w")
        self.model_combobox = ctk.CTkComboBox(config_frame, values=YOLO_MODELS_V8 + ["Custom..."])
        self.model_combobox.grid(row=row_idx, column=1, padx=PAD_Y, pady=PAD_Y, sticky="ew")
        self.model_combobox.set(YOLO_MODELS_V8[1])
        ctk.CTkButton(config_frame, text="Browse PT", image=self.icon_browse, compound="left", width=90, command=self._browse_model_pt).grid(row=row_idx, column=2, padx=(PAD_Y, PAD_X), pady=PAD_Y)
        row_idx += 1
        self.custom_model_label = ctk.CTkLabel(config_frame, text="", text_color="gray", wraplength=280, anchor="w", justify="left")
        self.custom_model_label.grid(row=row_idx, column=1, columnspan=2, padx=(PAD_Y, PAD_X), pady=(0, PAD_Y), sticky="ew")
        row_idx += 1
        ctk.CTkLabel(config_frame, text="Epochs:", anchor="w").grid(row=row_idx, column=0, padx=(PAD_X, PAD_Y), pady=PAD_Y, sticky="w")
        self.epochs_entry = ctk.CTkEntry(config_frame, placeholder_text="e.g., 100")
        self.epochs_entry.grid(row=row_idx, column=1, columnspan=2, padx=(PAD_Y, PAD_X), pady=PAD_Y, sticky="ew")
        row_idx += 1
        ctk.CTkLabel(config_frame, text="Batch Size:", anchor="w").grid(row=row_idx, column=0, padx=(PAD_X, PAD_Y), pady=PAD_Y, sticky="w")
        self.batch_entry = ctk.CTkEntry(config_frame, placeholder_text="e.g., 16 or -1 (Auto)")
        self.batch_entry.grid(row=row_idx, column=1, columnspan=2, padx=(PAD_Y, PAD_X), pady=PAD_Y, sticky="ew")
        self.batch_entry.insert(0, "-1")
        row_idx += 1
        ctk.CTkLabel(config_frame, text="Image Size (px):", anchor="w").grid(row=row_idx, column=0, padx=(PAD_X, PAD_Y), pady=PAD_Y, sticky="w")
        self.img_size_entry = ctk.CTkEntry(config_frame, placeholder_text="e.g., 640")
        self.img_size_entry.grid(row=row_idx, column=1, columnspan=2, padx=(PAD_Y, PAD_X), pady=PAD_Y, sticky="ew")
        self.img_size_entry.insert(0, "640")
        row_idx += 1
        ctk.CTkLabel(config_frame, text="Project Name:", anchor="w").grid(row=row_idx, column=0, padx=(PAD_X, PAD_Y), pady=PAD_Y, sticky="w")
        self.project_name_entry = ctk.CTkEntry(config_frame)
        self.project_name_entry.grid(row=row_idx, column=1, columnspan=2, padx=(PAD_Y, PAD_X), pady=PAD_Y, sticky="ew")
        self.project_name_entry.insert(0, DEFAULT_PROJECT_NAME)
        row_idx += 1
        ctk.CTkLabel(config_frame, text="Experiment Name:", anchor="w").grid(row=row_idx, column=0, padx=(PAD_X, PAD_Y), pady=PAD_Y, sticky="w")
        self.exp_name_entry = ctk.CTkEntry(config_frame, placeholder_text="e.g., my_cool_run")
        self.exp_name_entry.grid(row=row_idx, column=1, columnspan=2, padx=(PAD_Y, PAD_X), pady=PAD_Y, sticky="ew")
        self.exp_name_entry.insert(0, "exp")
        row_idx += 1
        ctk.CTkLabel(config_frame, text="Device:", anchor="w").grid(row=row_idx, column=0, padx=(PAD_X, PAD_Y), pady=PAD_Y, sticky="w")
        self.device_combobox = ctk.CTkComboBox(config_frame, values=["cpu", "0", "0,1", "mps"]) # Share this? Maybe separate device for infer? Keep shared for now.
        self.device_combobox.grid(row=row_idx, column=1, columnspan=2, padx=(PAD_Y, PAD_X), pady=PAD_Y, sticky="ew")
        try: subprocess.check_output("nvidia-smi -L", shell=True, stderr=subprocess.DEVNULL); self.device_combobox.set("0")
        except (subprocess.CalledProcessError, FileNotFoundError):
            try: import torch; self.device_combobox.set("mps" if torch.backends.mps.is_available() else "cpu")
            except ImportError: self.device_combobox.set("cpu")
        row_idx += 1
        self.resume_checkbox = ctk.CTkCheckBox(config_frame, text="Resume training")
        self.resume_checkbox.grid(row=row_idx, column=0, columnspan=3, padx=PAD_X, pady=PAD_Y*2, sticky="w")
        row_idx += 1
        config_frame.grid_rowconfigure(row_idx, weight=1) # Spacer Row
        row_idx +=1

        # --- Separator ---
        ctk.CTkFrame(config_frame, height=SEPARATOR_HEIGHT, fg_color="gray50").grid(row=row_idx, column=0, columnspan=3, sticky="ew", padx=PAD_X, pady=(PAD_Y * 2, PAD_Y))
        row_idx += 1

        # --- System Monitor Section (shared display logic, created once) ---
        ctk.CTkLabel(config_frame, text="System Status", font=ctk.CTkFont(size=14, weight="bold"), anchor="w").grid(row=row_idx, column=0, columnspan=3, padx=PAD_X, pady=(PAD_Y, 0), sticky="w")
        row_idx += 1
        self.system_status_frame = ctk.CTkFrame(config_frame, fg_color="transparent") # Store ref to frame
        self.system_status_frame.grid(row=row_idx, column=0, columnspan=3, padx=PAD_X, pady=(0, PAD_Y), sticky="new")
        self.system_status_frame.grid_columnconfigure(0, weight=0)  # Label
        self.system_status_frame.grid_columnconfigure(1, weight=1)  # Bar
        self.system_status_frame.grid_columnconfigure(2, weight=0)  # Text%
        # Initial placeholder, _update_system_display will populate it
        ctk.CTkLabel(self.system_status_frame, text="Loading system info...", text_color="gray").grid(row=0, column=0, columnspan=3, sticky="ew")
        row_idx += 1

        # --- Theme Switcher (shared display logic, created once) ---
        self.theme_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        self.theme_frame.grid(row=row_idx, column=0, columnspan=3, padx=PAD_X, pady=(PAD_Y*2, PAD_Y), sticky="w")
        ctk.CTkLabel(self.theme_frame, text="Theme:").pack(side="left", padx=(0, PAD_Y))
        self.theme_switch = ctk.CTkSwitch(self.theme_frame, text="", width=0, command=self._toggle_theme)
        self.theme_switch.pack(side="left")
        self._update_theme_switch_text()

    def _create_status_panel(self, parent_frame): # Added parent_frame argument
        status_frame = ctk.CTkFrame(parent_frame, corner_radius=10)
        status_frame.grid(row=0, column=1, padx=(0, 0), pady=0, sticky="nsew") # Use parent padding
        status_frame.grid_columnconfigure(0, weight=1)
        status_frame.grid_rowconfigure(3, weight=4) # Plots frame (Adjusted row index)
        status_frame.grid_rowconfigure(5, weight=2) # Console frame (Adjusted row index)

        current_row = 0

        # --- 1. Control Bar ---
        control_bar_frame = ctk.CTkFrame(status_frame, fg_color="transparent")
        control_bar_frame.grid(row=current_row, column=0, padx=PAD_X, pady=(PAD_Y*2, PAD_Y), sticky="ew")
        control_bar_frame.grid_columnconfigure(3, weight=1) # Spacer col
        self.start_button = ctk.CTkButton(control_bar_frame, text="Start Training", image=self.icon_play, compound="left", width=140, command=self._start_training)
        self.start_button.grid(row=0, column=0, padx=(0, PAD_Y), pady=PAD_Y)
        self.stop_button = ctk.CTkButton(control_bar_frame, text="Stop Training", image=self.icon_stop, compound="left", width=140, command=self._stop_training, state="disabled", fg_color="#D32F2F", hover_color="#C62828")
        self.stop_button.grid(row=0, column=1, padx=PAD_Y, pady=PAD_Y)
        self.results_button = ctk.CTkButton(control_bar_frame, text="Open Results", image=self.icon_folder, compound="left", width=140, command=self._open_results_folder, state="disabled")
        self.results_button.grid(row=0, column=4, padx=(PAD_Y, 0), pady=PAD_Y) # Aligned right
        current_row += 1

        # --- 2. Live Dashboard Section ---
        self._create_section_header(status_frame, "Live Training Dashboard").grid(row=current_row, column=0, padx=PAD_X, pady=PAD_Y_SECTION, sticky="ew")
        current_row += 1
        dashboard_frame = ctk.CTkFrame(status_frame, corner_radius=5)
        dashboard_frame.grid(row=current_row, column=0, padx=PAD_X, pady=(0, PAD_Y_SECTION[1]), sticky="ew")
        dashboard_frame.grid_columnconfigure((1, 3, 5, 7), weight=1)
        dashboard_frame.grid_columnconfigure((0, 2, 4, 6), weight=0)

        # Status Indicator (use specific variable)
        self.train_status_indicator_label = ctk.CTkLabel(dashboard_frame, text="Idle", font=ctk.CTkFont(weight="bold"), width=80, anchor='w')
        self.train_status_indicator_label.grid(row=0, column=0, padx=(PAD_X,PAD_Y), pady=PAD_Y, sticky="w")

        self.epoch_label = ctk.CTkLabel(dashboard_frame, text="Epoch: 0 / 0", anchor='w')
        self.epoch_label.grid(row=0, column=2, padx=PAD_Y, pady=PAD_Y, sticky="w")
        self.progress_bar = ctk.CTkProgressBar(dashboard_frame, width=120, height=12)
        self.progress_bar.grid(row=0, column=3, padx=PAD_Y, pady=PAD_Y, sticky="ew")
        self.progress_bar.set(0)
        self.time_label = ctk.CTkLabel(dashboard_frame, text="Time: 00:00:00", anchor='w')
        self.time_label.grid(row=0, column=4, padx=PAD_Y, pady=PAD_Y, sticky="w")
        self.etr_label = ctk.CTkLabel(dashboard_frame, text="ETR: --:--:--", anchor='w')
        self.etr_label.grid(row=0, column=5, padx=PAD_Y, pady=PAD_Y, sticky="w")
        self.speed_label = ctk.CTkLabel(dashboard_frame, text="Speed: N/A", anchor='e')
        self.speed_label.grid(row=0, column=7, padx=(PAD_Y, PAD_X), pady=PAD_Y, sticky="e")
        ctk.CTkLabel(dashboard_frame, text="mAP@.50:", font=ctk.CTkFont(size=12), anchor='e').grid(row=1, column=2, padx=PAD_Y, pady=PAD_Y, sticky="e")
        self.latest_map50_label = ctk.CTkLabel(dashboard_frame, text="N/A", font=ctk.CTkFont(size=13, weight="bold"), anchor='w')
        self.latest_map50_label.grid(row=1, column=3, padx=PAD_Y, pady=PAD_Y, sticky="w")
        ctk.CTkLabel(dashboard_frame, text="mAP@.5-.95:", font=ctk.CTkFont(size=12), anchor='e').grid(row=1, column=4, padx=PAD_Y, pady=PAD_Y, sticky="e")
        self.latest_map50_95_label = ctk.CTkLabel(dashboard_frame, text="N/A", font=ctk.CTkFont(size=13, weight="bold"), anchor='w')
        self.latest_map50_95_label.grid(row=1, column=5, padx=PAD_Y, pady=PAD_Y, sticky="w")
        current_row += 1

        # --- 3. Training Plots Section ---
        self._create_section_header(status_frame, "Training Plots").grid(row=current_row, column=0, padx=PAD_X, pady=PAD_Y_SECTION, sticky="ew")
        current_row += 1
        plots_frame = ctk.CTkFrame(status_frame, corner_radius=5)
        plots_frame.grid(row=current_row, column=0, padx=PAD_X, pady=(0, PAD_Y_SECTION[1]), sticky="nsew")
        plots_frame.grid_columnconfigure(0, weight=1); plots_frame.grid_rowconfigure(0, weight=1)
        # Matplotlib Figure and Axes (specific for training)
        self.fig = Figure(figsize=(8, 3.5), dpi=100)
        self.fig.patch.set_facecolor(self._get_plot_bg_color())
        self.ax1, self.ax2 = self.fig.subplots(2, 1, sharex=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plots_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True, padx=PAD_Y, pady=PAD_Y)
        self._setup_plots() # Initialize training plots
        current_row += 1

        # --- 4. Console Output Section ---
        self._create_section_header(status_frame, "Training Console Output").grid(row=current_row, column=0, padx=PAD_X, pady=PAD_Y_SECTION, sticky="ew")
        current_row += 1
        console_frame = ctk.CTkFrame(status_frame, corner_radius=5)
        console_frame.grid(row=current_row, column=0, padx=PAD_X, pady=(0, PAD_X), sticky="nsew")
        console_frame.grid_columnconfigure(0, weight=1); console_frame.grid_rowconfigure(0, weight=1)
        self.console_log = ctk.CTkTextbox(console_frame, wrap="word", state="disabled", font=("Courier New", 10), border_width=0, fg_color="transparent")
        self.console_log.grid(row=0, column=0, padx=1, pady=1, sticky="nsew")
        current_row += 1

    # --- Inference Tab UI Creation ---
    def _create_infer_config_panel(self, parent_frame):
        """Creates the configuration panel for the Inference tab."""
        config_outer_frame = ctk.CTkFrame(parent_frame, width=380, corner_radius=10)
        config_outer_frame.grid(row=0, column=0, padx=(0, 10), pady=0, sticky="nsew")
        config_outer_frame.grid_rowconfigure(1, weight=1) # Allow content frame to expand
        config_outer_frame.grid_propagate(False)

        config_header = self._create_section_header(config_outer_frame, "Inference Configuration")
        config_header.pack(fill="x", padx=PAD_X, pady=PAD_Y_SECTION)

        config_frame = ctk.CTkFrame(config_outer_frame, fg_color="transparent")
        config_frame.pack(fill="both", expand=True, padx=0, pady=0)
        config_frame.grid_columnconfigure(1, weight=1)
        config_frame.grid_rowconfigure(10, weight=1) # Spacer row

        row_idx = 0

        # --- Model Selection ---
        ctk.CTkLabel(config_frame, text="Model (.pt):", anchor="w").grid(row=row_idx, column=0, padx=(PAD_X, PAD_Y), pady=PAD_Y, sticky="w")
        self.infer_model_combobox = ctk.CTkComboBox(config_frame, values=YOLO_MODELS_V8 + ["Custom..."])
        self.infer_model_combobox.grid(row=row_idx, column=1, padx=PAD_Y, pady=PAD_Y, sticky="ew")
        self.infer_model_combobox.set(YOLO_MODELS_V8[1]) # Default
        ctk.CTkButton(config_frame, text="Browse PT", image=self.icon_browse, compound="left", width=90, command=self._browse_infer_model_pt).grid(row=row_idx, column=2, padx=(PAD_Y, PAD_X), pady=PAD_Y)
        row_idx += 1
        self.custom_infer_model_label = ctk.CTkLabel(config_frame, text="", text_color="gray", wraplength=280, anchor="w", justify="left")
        self.custom_infer_model_label.grid(row=row_idx, column=1, columnspan=2, padx=(PAD_Y, PAD_X), pady=(0, PAD_Y), sticky="ew")
        row_idx += 1

        # --- Source Selection ---
        ctk.CTkLabel(config_frame, text="Source:", anchor="w").grid(row=row_idx, column=0, padx=(PAD_X, PAD_Y), pady=PAD_Y, sticky="w")
        self.infer_source_entry = ctk.CTkEntry(config_frame, placeholder_text="path/to/image.jpg or video.mp4")
        self.infer_source_entry.grid(row=row_idx, column=1, padx=PAD_Y, pady=PAD_Y, sticky="ew")
        ctk.CTkButton(config_frame, text="Browse", image=self.icon_browse, compound="left", width=90, command=self._browse_infer_source).grid(row=row_idx, column=2, padx=(PAD_Y, PAD_X), pady=PAD_Y)
        row_idx += 1

        # --- Device Selection ---
        ctk.CTkLabel(config_frame, text="Device:", anchor="w").grid(row=row_idx, column=0, padx=(PAD_X, PAD_Y), pady=PAD_Y, sticky="w")
        # **** CHANGE START ****
        # Create a SEPARATE combobox for inference
        self.infer_device_combobox = ctk.CTkComboBox(config_frame, values=["cpu", "0", "0,1", "mps"])
        self.infer_device_combobox.grid(row=row_idx, column=1, columnspan=2, padx=(PAD_Y, PAD_X), pady=PAD_Y, sticky="ew")

        # Set a default for the inference device (can be same logic as training one)
        try:
            subprocess.check_output("nvidia-smi -L", shell=True, stderr=subprocess.DEVNULL)
            self.infer_device_combobox.set("0")
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                # Check for MPS availability if PyTorch is installed
                import torch
                if torch.backends.mps.is_available():
                    self.infer_device_combobox.set("mps")
                else:
                    self.infer_device_combobox.set("cpu")
            except ImportError:
                 # Fallback to CPU if torch isn't available
                self.infer_device_combobox.set("cpu")
        # **** CHANGE END ****
        row_idx += 1


        # --- Confidence Threshold ---
        # Using a frame to group label, slider, value label
        conf_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        conf_frame.grid(row=row_idx, column=0, columnspan=3, padx=PAD_X, pady=PAD_Y, sticky="ew")
        conf_frame.grid_columnconfigure(1, weight=1) # Slider takes available space
        ctk.CTkLabel(conf_frame, text="Confidence:", anchor="w").grid(row=0, column=0, padx=(0, PAD_Y))
        self.infer_conf_slider = ctk.CTkSlider(conf_frame, from_=0.01, to=1.0, number_of_steps=99, command=lambda v: self.infer_conf_label.configure(text=f"{v:.2f}"))
        self.infer_conf_slider.grid(row=0, column=1, padx=(0, PAD_Y), sticky="ew")
        self.infer_conf_slider.set(0.25) # Default confidence
        self.infer_conf_label = ctk.CTkLabel(conf_frame, text="0.25", width=35, anchor="e") # Fixed width for value
        self.infer_conf_label.grid(row=0, column=2)
        row_idx += 1


        # --- IoU Threshold ---
        # Using a frame to group label, slider, value label
        iou_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        iou_frame.grid(row=row_idx, column=0, columnspan=3, padx=PAD_X, pady=PAD_Y, sticky="ew")
        iou_frame.grid_columnconfigure(1, weight=1) # Slider takes available space
        ctk.CTkLabel(iou_frame, text="IoU:", anchor="w").grid(row=0, column=0, padx=(0, PAD_Y))
        self.infer_iou_slider = ctk.CTkSlider(iou_frame, from_=0.01, to=1.0, number_of_steps=99, command=lambda v: self.infer_iou_label.configure(text=f"{v:.2f}"))
        self.infer_iou_slider.grid(row=0, column=1, padx=(0, PAD_Y), sticky="ew")
        self.infer_iou_slider.set(0.45) # Default IoU
        self.infer_iou_label = ctk.CTkLabel(iou_frame, text="0.45", width=35, anchor="e") # Fixed width for value
        self.infer_iou_label.grid(row=0, column=2)
        row_idx += 1

        # --- Project/Experiment Name (Optional for Inference) ---
        ctk.CTkLabel(config_frame, text="Output Project:", anchor="w").grid(row=row_idx, column=0, padx=(PAD_X, PAD_Y), pady=PAD_Y, sticky="w")
        self.infer_project_name_entry = ctk.CTkEntry(config_frame)
        self.infer_project_name_entry.grid(row=row_idx, column=1, columnspan=2, padx=(PAD_Y, PAD_X), pady=PAD_Y, sticky="ew")
        self.infer_project_name_entry.insert(0, DEFAULT_INFER_PROJECT_NAME)
        row_idx += 1
        ctk.CTkLabel(config_frame, text="Output Name:", anchor="w").grid(row=row_idx, column=0, padx=(PAD_X, PAD_Y), pady=PAD_Y, sticky="w")
        self.infer_exp_name_entry = ctk.CTkEntry(config_frame, placeholder_text="e.g., inference_run_1")
        self.infer_exp_name_entry.grid(row=row_idx, column=1, columnspan=2, padx=(PAD_Y, PAD_X), pady=PAD_Y, sticky="ew")
        self.infer_exp_name_entry.insert(0, "exp") # Simple default
        row_idx += 1

        # --- Control Buttons ---
        # Store ref to frame for potential use in _set_infer_config_panel_state
        self.infer_control_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        self.infer_control_frame.grid(row=row_idx, column=0, columnspan=3, padx=PAD_X, pady=(PAD_Y * 2, PAD_Y), sticky="ew")
        self.infer_run_button = ctk.CTkButton(self.infer_control_frame, text="Run Inference", image=self.icon_infer or self.icon_play, compound="left", command=self._start_inference)
        self.infer_run_button.pack(side="left", padx=(0, PAD_Y))
        self.infer_stop_button = ctk.CTkButton(self.infer_control_frame, text="Stop Inference", image=self.icon_stop, compound="left", command=self._stop_inference, state="disabled", fg_color="#D32F2F", hover_color="#C62828")
        self.infer_stop_button.pack(side="left", padx=PAD_Y)
        self.infer_results_button = ctk.CTkButton(self.infer_control_frame, text="Open Output", image=self.icon_folder, compound="left", command=self._open_infer_results_folder, state="disabled")
        self.infer_results_button.pack(side="right", padx=(PAD_Y, 0))
        row_idx += 1

        # --- Status Indicator ---
        self.infer_status_label = ctk.CTkLabel(config_frame, text="Status: Idle", text_color="gray", anchor="w")
        self.infer_status_label.grid(row=row_idx, column=0, columnspan=3, padx=PAD_X, pady=(PAD_Y, PAD_Y_SECTION[1]), sticky="ew")
        row_idx += 1

        # Spacer Row
        config_frame.grid_rowconfigure(row_idx, weight=1)

    def _create_infer_output_panel(self, parent_frame):
        """Creates the output display panel for the Inference tab."""
        output_frame = ctk.CTkFrame(parent_frame, corner_radius=10)
        output_frame.grid(row=0, column=1, padx=(0, 0), pady=0, sticky="nsew")
        output_frame.grid_columnconfigure(0, weight=1)
        output_frame.grid_rowconfigure(1, weight=1) # Output display area takes most space
        output_frame.grid_rowconfigure(3, weight=0) # Console log smaller

        current_row = 0

        # --- Header ---
        self._create_section_header(output_frame, "Inference Output").grid(row=current_row, column=0, padx=PAD_X, pady=PAD_Y_SECTION, sticky="ew")
        current_row += 1

        # --- Output Display Label ---
        self.infer_output_label = None
        self.infer_output_photoimage = None
        self._output_frame_ref = output_frame
        self._output_frame_row = current_row
        self._create_new_infer_output_label()
        current_row += 1

        # --- Video Progress Bar (Initially Hidden) ---
        self.infer_progress_bar = ctk.CTkProgressBar(output_frame, height=10)
        # Don't grid it initially, only show for videos
        # self.infer_progress_bar.grid(row=current_row, column=0, padx=PAD_X, pady=(0, PAD_Y), sticky="ew")
        self.infer_progress_bar.set(0)
        # current_row += 1 # Increment only when shown

        # --- Console Output ---
        self._create_section_header(output_frame, "Inference Console").grid(row=current_row, column=0, padx=PAD_X, pady=(PAD_Y_SECTION[0], PAD_Y), sticky="ew")
        current_row += 1 # Increment after header
        infer_console_frame = ctk.CTkFrame(output_frame, corner_radius=5, height=150) # Limit height
        infer_console_frame.grid(row=current_row, column=0, padx=PAD_X, pady=(0, PAD_X), sticky="nsew")
        infer_console_frame.grid_propagate(False) # Prevent resizing based on content
        infer_console_frame.grid_columnconfigure(0, weight=1)
        infer_console_frame.grid_rowconfigure(0, weight=1)
        self.infer_console_log = ctk.CTkTextbox(infer_console_frame, wrap="word", state="disabled", font=("Courier New", 9), border_width=0, fg_color="transparent")
        self.infer_console_log.grid(row=0, column=0, padx=1, pady=1, sticky="nsew")
        current_row += 1

    def _create_new_infer_output_label(self):
        # Destroy old label if it exists
        if self.infer_output_label is not None:
            try:
                self.infer_output_label.destroy()
            except Exception:
                pass
        # Create new label and grid it
        self.infer_output_label = ctk.CTkLabel(self._output_frame_ref, text="Output will appear here", text_color="gray", anchor="center")
        self.infer_output_label.grid(row=self._output_frame_row, column=0, padx=PAD_X, pady=PAD_Y, sticky="nsew")
        self.infer_output_photoimage = None
        print(f"[DEBUG] Created new infer_output_label: id={id(self.infer_output_label)}")

    # --- Plot Setup/Color Getters (remain mostly the same, used by training tab) ---
    def _setup_plots(self):
        plot_text_color = self._get_plot_text_color()
        grid_color = self._get_plot_grid_color()
        face_color = self._get_plot_bg_color()
        self.fig.patch.set_facecolor(face_color)
        for ax in [self.ax1, self.ax2]:
            ax.clear(); ax.set_facecolor(face_color)
            ax.tick_params(axis='x', colors=plot_text_color); ax.tick_params(axis='y', colors=plot_text_color)
            for spine in ax.spines.values(): spine.set_color(plot_text_color)
            ax.grid(True, color=grid_color, linestyle=':', linewidth=0.5)
        self.ax1.set_title('Training Loss Components', color=plot_text_color, fontsize=10); self.ax1.set_ylabel('Loss', color=plot_text_color, fontsize=9); self.ax1.set_ylim(bottom=0); self.ax1.tick_params(axis='both', which='major', labelsize=8)
        self.ax2.set_title('Validation mAP', color=plot_text_color, fontsize=10); self.ax2.set_xlabel('Epoch', color=plot_text_color, fontsize=9); self.ax2.set_ylabel('mAP', color=plot_text_color, fontsize=9); self.ax2.set_ylim(0, 1.05); self.ax2.tick_params(axis='both', which='major', labelsize=8)
        self.fig.tight_layout(pad=2.0); self.canvas.draw_idle()

    def _get_plot_bg_color(self):
        try: return ctk.ThemeManager.theme["CTkFrame"]["fg_color"][self.appearance_mode_tracker.get()]
        except: return "#FFFFFF" if ctk.get_appearance_mode() == "Light" else "#2B2B2B"
    def _get_plot_text_color(self):
        try: return ctk.ThemeManager.theme["CTkLabel"]["text_color"][self.appearance_mode_tracker.get()]
        except: return "#000000" if ctk.get_appearance_mode() == "Light" else "#DCE4EE"
    def _get_plot_grid_color(self): return "#CCCCCC" if ctk.get_appearance_mode() == "Light" else "#444444"

    # --- Theme Switcher Logic (Update plots + system display) ---
    def _update_theme_switch_text(self):
        mode = ctk.get_appearance_mode(); self.theme_switch.configure(text=mode)
    def _toggle_theme(self):
        mode = "Dark" if self.theme_switch.get() == 1 else "Light"; ctk.set_appearance_mode(mode); self.appearance_mode_tracker.set(mode)
        self._update_theme_switch_text();
        # Update Training Plots
        plot_bg = self._get_plot_bg_color()
        self.fig.patch.set_facecolor(plot_bg); self.ax1.set_facecolor(plot_bg); self.ax2.set_facecolor(plot_bg)
        self._setup_plots(); self._update_matplotlib_plots();
        # Update System Display Colors
        self._update_system_display(self.system_stats);
        # Update Status Indicator Colors
        self._update_train_status_indicator(self.current_train_status)
        self._update_infer_status_indicator(self.current_infer_status)
        # Update Inference Output Placeholder Color (if needed)
        # self.infer_output_label.configure(...)
    

    # --- UI Actions (Browsing, Opening Folder, Validation) ---
    def _browse_dataset_yaml(self):
        filepath = filedialog.askopenfilename(title="Select Dataset YAML File", filetypes=[("YAML files", "*.yaml")])
        if filepath: self.dataset_yaml_entry.delete(0, tk.END); self.dataset_yaml_entry.insert(0, filepath)

    def _browse_model_pt(self): # For Training Tab
        filepath = filedialog.askopenfilename(title="Select Model .pt File", filetypes=[("PyTorch Model", "*.pt")])
        if filepath: self.model_combobox.set("Custom..."); self.custom_model_path = filepath; self.custom_model_label.configure(text=f"Using: {os.path.basename(filepath)}")
        elif self.model_combobox.get() == "Custom...": self.custom_model_path = None; self.custom_model_label.configure(text="")

    def _browse_infer_model_pt(self): # For Inference Tab
        filepath = filedialog.askopenfilename(title="Select Model .pt File", filetypes=[("PyTorch Model", "*.pt")])
        if filepath: self.infer_model_combobox.set("Custom..."); self.custom_infer_model_path = filepath; self.custom_infer_model_label.configure(text=f"Using: {os.path.basename(filepath)}")
        elif self.infer_model_combobox.get() == "Custom...": self.custom_infer_model_path = None; self.custom_infer_model_label.configure(text="")

    def _browse_infer_source(self):
        filetypes = [("Media Files", f"{' '.join(IMAGE_EXTENSIONS)} {' '.join(VIDEO_EXTENSIONS)}"),
                     ("Image Files", ' '.join(IMAGE_EXTENSIONS)),
                     ("Video Files", ' '.join(VIDEO_EXTENSIONS)),
                     ("All files", "*.*")]
        filepath = filedialog.askopenfilename(title="Select Image or Video Source", filetypes=filetypes)
        if filepath:
            self.infer_source_entry.delete(0, tk.END)
            self.infer_source_entry.insert(0, filepath)
            self.infer_source_type = get_source_type(filepath)
            logging.info(f"Selected inference source: {filepath}, type: {self.infer_source_type}")
            # Clear previous output on new source selection
            self._clear_inference_output()

    def _open_results_folder(self): # Training results
        path_to_open = self.current_experiment_path
        logging.info(f"Attempting to open training results folder: {path_to_open}")
        self._open_folder_native(path_to_open)

    def _open_infer_results_folder(self): # Inference results
        path_to_open = self.infer_results_path
        logging.info(f"Attempting to open inference results folder: {path_to_open}")
        self._open_folder_native(path_to_open)

    def _open_folder_native(self, folder_path):
        """Platform-independent way to open a folder."""
        if folder_path and os.path.isdir(folder_path):
            try:
                if sys.platform == "win32": os.startfile(os.path.realpath(folder_path))
                elif sys.platform == "darwin": subprocess.Popen(["open", folder_path])
                else: subprocess.Popen(["xdg-open", folder_path])
            except Exception as e: logging.error(f"Could not open folder '{folder_path}': {e}"); messagebox.showerror("Error", f"Could not open folder:\n{e}")
        else: logging.warning(f"Folder not found or not set: {folder_path}"); messagebox.showwarning("Warning", f"Folder not found or process not run yet.\nPath: {folder_path}")

    def _validate_train_inputs(self): # Renamed
        if not self.dataset_yaml_entry.get() or not os.path.isfile(self.dataset_yaml_entry.get()) or not self.dataset_yaml_entry.get().lower().endswith(".yaml"): messagebox.showerror("Validation Error", "Please provide a valid Dataset YAML file."); return False
        if self.model_combobox.get() == "Custom..." and (not self.custom_model_path or not os.path.isfile(self.custom_model_path)): messagebox.showerror("Validation Error", "Custom training model selected, but no valid .pt file provided via 'Browse PT'."); return False
        elif not self.model_combobox.get(): messagebox.showerror("Validation Error", "Please select a base training model or provide a custom one."); return False
        if not self.epochs_entry.get() or not is_int(self.epochs_entry.get()) or int(self.epochs_entry.get()) <= 0: messagebox.showerror("Validation Error", "Epochs must be a positive integer."); return False
        batch_str = self.batch_entry.get();
        if not batch_str or not is_int(batch_str) or (int(batch_str) <= 0 and int(batch_str) != -1): messagebox.showerror("Validation Error", "Batch size must be a positive integer or -1."); return False
        if not self.img_size_entry.get() or not is_int(self.img_size_entry.get()) or int(self.img_size_entry.get()) <= 0: messagebox.showerror("Validation Error", "Image Size must be a positive integer (e.g., 640)."); return False
        if not self.project_name_entry.get().strip(): messagebox.showerror("Validation Error", "Project Name cannot be empty."); return False
        if not self.exp_name_entry.get().strip(): messagebox.showerror("Validation Error", "Experiment Name cannot be empty."); return False
        return True

    def _validate_infer_inputs(self):
        if self.infer_model_combobox.get() == "Custom..." and (not self.custom_infer_model_path or not os.path.isfile(self.custom_infer_model_path)): messagebox.showerror("Validation Error", "Custom inference model selected, but no valid .pt file provided via 'Browse PT'."); return False
        elif not self.infer_model_combobox.get(): messagebox.showerror("Validation Error", "Please select a base inference model or provide a custom one."); return False
        source = self.infer_source_entry.get()
        if not source or not os.path.exists(source): messagebox.showerror("Validation Error", "Please provide a valid Source file (image or video)."); return False
        self.infer_source_type = get_source_type(source) # Re-check type
        if self.infer_source_type not in ['image', 'video']: messagebox.showerror("Validation Error", f"Unsupported source file type: {os.path.splitext(source)[1]}. Use common image or video formats."); return False
        # Conf/IoU are sliders, always valid within range.
        if not self.infer_project_name_entry.get().strip(): messagebox.showerror("Validation Error", "Output Project Name cannot be empty."); return False
        if not self.infer_exp_name_entry.get().strip(): messagebox.showerror("Validation Error", "Output Experiment Name cannot be empty."); return False
        return True


    # --- Status Indicator Updates (Specific for each tab) ---
    def _update_train_status_indicator(self, status_text):
        self.current_train_status = status_text; color = STATUS_COLORS.get(status_text, "gray")
        try: self.train_status_indicator_label.configure(text=status_text, text_color=color)
        except tk.TclError: logging.warning("TclError updating train status indicator (window closing?)")
        except AttributeError: logging.warning("Train status indicator label not yet created?") # Avoid error during init

    def _update_infer_status_indicator(self, status_text):
        self.current_infer_status = status_text; color = STATUS_COLORS.get(status_text, "gray")
        try: self.infer_status_label.configure(text=f"Status: {status_text}", text_color=color)
        except tk.TclError: logging.warning("TclError updating infer status indicator (window closing?)")
        except AttributeError: logging.warning("Inference status label not yet created?") # Avoid error during init

    # --- Training Logic (Start, Run, Process Queue, Logging, Data Parsing, Path Check, Key Metrics - Mostly same) ---
    def _start_training(self):
        logging.info("Start training button clicked.")
        if not self._validate_train_inputs(): logging.warning("Training input validation failed."); return
        if self.is_training: logging.warning("Start clicked but training is already active."); messagebox.showwarning("Training Active", "A training process is already running."); return
        if self.is_inferring: messagebox.showwarning("Inference Active", "An inference process is running. Please stop it before starting training."); return

        self._clear_log(self.console_log) # Clear training console
        if self.plot_update_job: self.after_cancel(self.plot_update_job); self.plot_update_job = None
        if self.live_data_update_job: self.after_cancel(self.live_data_update_job); self.live_data_update_job = None
        self._update_train_status_indicator("Starting"); self.progress_bar.set(0)
        self.stop_event.clear(); self.results_button.configure(state="disabled")
        self.current_experiment_path = None; self.is_training = False; self.last_val_image_epoch = -1
        self.current_epoch = 0; self.total_epochs = int(self.epochs_entry.get()); self.start_time = time.time()
        self.elapsed_time_str = "00:00:00"; self.etr_str = "Calculating..."; self.current_speed_str = "Starting..."
        self.latest_map50 = None; self.latest_map50_95 = None; self.metrics_data = { key: [] for key in self.metrics_data }
        self._update_stats_display(); self._update_key_metrics_display(); self._setup_plots()

        # Get parameters from UI
        data_yaml = self.dataset_yaml_entry.get(); epochs = self.epochs_entry.get(); batch = self.batch_entry.get()
        imgsz = self.img_size_entry.get(); project = self.project_name_entry.get(); name = self.exp_name_entry.get()
        device = self.device_combobox.get(); resume = self.resume_checkbox.get() == 1
        model_selection = self.model_combobox.get()
        weights = self.custom_model_path if model_selection == "Custom..." else model_selection

        # Build command
        yolo_cmd_base = ["yolo", "detect", "train"] # Assuming YOLO CLI is in PATH
        yolo_args = yolo_cmd_base + [
            f"model={weights}", f"data={data_yaml}", f"epochs={epochs}",
            f"batch={batch}", f"imgsz={imgsz}", f"project={project}", f"name={name}",
            f"device={device}", "exist_ok=True", "save_period=1", # Save weights every epoch for resume/preview
            "plots=True" # Generate result plots
        ]
        if resume: yolo_args.append("resume=True")
        # Ensure no stream=True for training
        # yolo_args.append("stream=False") # Not a valid arg for train

        log_cmd_str = ' '.join(map(str, yolo_args))
        logging.info(f"Constructed YOLO Training command: {log_cmd_str}")
        self._log(f"Starting training with command:\n{log_cmd_str}\n" + "="*30 + "\n", self.console_log) # Log to training console

        # Update State & UI
        self.is_training = True; self.start_button.configure(state="disabled"); self.stop_button.configure(state="normal"); self._set_train_config_panel_state("disabled"); self._update_train_status_indicator("Training")

        # Start Thread
        self.training_thread = threading.Thread(target=self._run_training_process, args=(yolo_args,), daemon=True, name="YoloTrainThread")
        self.training_thread.start(); logging.info("Training thread started.")
        self._update_live_data() # Start the live data update loop

    def _run_training_process(self, command):
        self._run_generic_process(command, self.training_process, self.stop_event, self.log_queue, "Training")

    # --- Combined Queue Processing ---
    def _process_queues(self):
        """Processes both training and inference log queues."""
        # Process Training Queue
        try:
            while True:
                line = self.log_queue.get_nowait()
                if line is None: logging.info("Received None sentinel in training log queue."); self._training_finished(); break
                else: self._log(line, self.console_log); self._parse_train_log_line(line); self._check_and_set_train_experiment_path(line)
        except queue.Empty: pass
        except Exception as e: logging.exception(f"Error processing training queue: {e}")

        # Process Inference Queue
        try:
            while True:
                line = self.infer_log_queue.get_nowait()
                if line is None: logging.info("Received None sentinel in inference log queue."); self._inference_finished(); break
                else: self._log(line, self.infer_console_log); self._parse_infer_log_line(line)
        except queue.Empty: pass
        except Exception as e: logging.exception(f"Error processing inference queue: {e}")

        # Reschedule
        self.queue_check_job = self.after(OUTPUT_UPDATE_INTERVAL_MS, self._process_queues)

    # --- Generic Process Runner ---
    def _run_generic_process(self, command, process_attr_ref, stop_event_ref, queue_ref, process_name):
        """Runs a subprocess and streams output to a queue."""
        process_handle = None
        try:
            logging.info(f"Starting {process_name} subprocess with command: {' '.join(command)}")
            startupinfo = None; creationflags = 0; preexec_fn = None
            if sys.platform == "win32":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
                creationflags = subprocess.CREATE_NO_WINDOW
            else:
                 # Use process group for better signal handling on Linux/macOS
                 preexec_fn=os.setsid

            process_handle = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1, # Line buffered
                universal_newlines=True,
                startupinfo=startupinfo,
                creationflags=creationflags,
                preexec_fn=preexec_fn # Assign preexec_fn here
            )
            # Store the process handle in the instance attribute
            if process_name == "Training": self.training_process = process_handle
            elif process_name == "Inference": self.inference_process = process_handle

            pid_info = f"PID: {process_handle.pid}"
            if sys.platform != "win32": pid_info += f" PGID: {os.getpgid(process_handle.pid)}"
            logging.info(f"{process_name} Subprocess started - {pid_info}")

            if process_handle.stdout:
                for line in iter(process_handle.stdout.readline, ''):
                    if stop_event_ref.is_set():
                        logging.info(f"{process_name}: Stop event detected in process loop.")
                        self._log(f"\n[INFO] {process_name}: Stop requested. Terminating...\n", self.console_log if process_name == "Training" else self.infer_console_log)
                        self._terminate_process_nicely(process_handle, process_name)
                        break # Exit loop after attempting termination
                    queue_ref.put(line) # Put raw line in queue for main thread processing

                # Close stdout only after the loop finishes or breaks
                if process_handle and process_handle.stdout:
                    try: process_handle.stdout.close()
                    except Exception as close_e: logging.warning(f"Error closing {process_name} stdout: {close_e}")

            # Wait for the process to finish naturally if stop wasn't requested or termination failed
            return_code = process_handle.wait() if process_handle else -999 # Use a distinct code for process not started/found
            logging.info(f"{process_name} process finished with return code: {return_code}")

            if stop_event_ref.is_set(): queue_ref.put(f"\n[INFO] {process_name} stopped by user.")
            elif return_code == 0: queue_ref.put(f"\n[INFO] {process_name} finished successfully.")
            else: queue_ref.put(f"\n[ERROR] {process_name} process exited with code {return_code}.")

        except FileNotFoundError:
            err_msg = f"\n[ERROR] Command not found: '{command[0]}'. Is 'yolo' installed and in PATH?"
            logging.error(err_msg)
            queue_ref.put(err_msg)
        except Exception as e:
            err_msg = f"\n[ERROR] Exception launching/monitoring {process_name}: {e}"
            logging.exception(f"Exception in _run_{process_name.lower()}_process:")
            queue_ref.put(err_msg)
        finally:
            logging.info(f"Signaling end of {process_name} process (sending None to queue).")
            queue_ref.put(None) # Sentinel value MUST be sent
            # Clear the process handle on the instance
            if process_name == "Training": self.training_process = None
            elif process_name == "Inference": self.inference_process = None

    def _terminate_process_nicely(self, process_handle, process_name):
        """Attempts to terminate a process group gracefully, then forcefully."""
        if not process_handle or process_handle.poll() is not None:
            logging.info(f"{process_name}: Process already terminated.")
            return

        pid = process_handle.pid
        pgid = None
        if sys.platform != "win32":
            try: pgid = os.getpgid(pid)
            except ProcessLookupError: logging.warning(f"{process_name}: Process {pid} not found for PGID lookup."); return

        logging.info(f"{process_name}: Attempting graceful termination (PID: {pid}, PGID: {pgid})...")
        try:
            if sys.platform == "win32":
                # Send CTRL_BREAK_EVENT to the process group on Windows
                os.kill(pid, signal.CTRL_BREAK_EVENT)
            else:
                # Send SIGINT (like Ctrl+C) to the process group on Unix-like systems
                os.killpg(pgid, signal.SIGINT)

            # Wait a bit for graceful shutdown
            process_handle.wait(timeout=10)
            logging.info(f"{process_name}: Process terminated gracefully after signal.")
        except ProcessLookupError:
            logging.warning(f"{process_name}: Process {pid} already terminated when sending signal.")
        except subprocess.TimeoutExpired:
            logging.warning(f"{process_name}: Process did not terminate gracefully after SIGINT/Break. Forcing kill.")
            try:
                if sys.platform != "win32": os.killpg(pgid, signal.SIGKILL) # SIGKILL for the group
                else: process_handle.kill() # process.kill() on Windows
                process_handle.wait(timeout=3) # Short wait after kill
                logging.info(f"{process_name}: Process killed forcefully.")
            except ProcessLookupError: logging.warning(f"{process_name}: Process already terminated before force kill.")
            except subprocess.TimeoutExpired: logging.error(f"{process_name}: Process did not terminate even after SIGKILL/kill.")
            except Exception as kill_e: logging.error(f"Error during {process_name} force kill: {kill_e}")
        except Exception as e:
            logging.error(f"Error during {process_name} termination signal/wait: {e}")
            # Final attempt to kill if something went wrong
            try:
                 if process_handle.poll() is None:
                     logging.warning(f"{process_name}: Final kill attempt.")
                     if sys.platform != "win32": os.killpg(pgid, signal.SIGKILL)
                     else: process_handle.kill()
                     process_handle.wait(timeout=2)
            except: pass # Ignore errors during final cleanup


    def _log(self, message, textbox_widget): # Added widget argument
        """Logs message to the specified CTkTextbox."""
        try:
            textbox_widget.configure(state="normal")
            textbox_widget.insert(tk.END, message)
            textbox_widget.see(tk.END)
            textbox_widget.configure(state="disabled")
        except tk.TclError as e: logging.warning(f"Failed to write to textbox: {e}")
        except AttributeError: logging.warning("Textbox widget is not valid.")

    def _clear_log(self, textbox_widget): # Added widget argument
        try:
            textbox_widget.configure(state="normal")
            textbox_widget.delete('1.0', tk.END)
            textbox_widget.configure(state="disabled")
        except tk.TclError as e: logging.warning(f"Failed to clear textbox: {e}")
        except AttributeError: logging.warning("Textbox widget is not valid.")

    def _parse_train_log_line(self, line): # Renamed
        cleaned_line = clean_ansi_codes(line).strip()
        # Epoch/Loss parsing (same as before)
        epoch_loss_match = re.match(r'\s*(\d+)/(\d+)\s+[\d.]+[A-Z]?G?\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+', cleaned_line)
        if epoch_loss_match:
            try:
                epoch = int(epoch_loss_match.group(1)); total_epochs_log = int(epoch_loss_match.group(2)); box_loss, cls_loss, dfl_loss = map(float, epoch_loss_match.groups()[2:5])
                self.current_epoch = epoch; self.total_epochs = total_epochs_log
                if not self.metrics_data['epoch'] or epoch > self.metrics_data['epoch'][-1]:
                     self.metrics_data['epoch'].append(epoch); self.metrics_data['train_box_loss'].append(box_loss); self.metrics_data['train_cls_loss'].append(cls_loss); self.metrics_data['train_dfl_loss'].append(dfl_loss)
                     self.metrics_data['val_map50'].append(np.nan); self.metrics_data['val_map50_95'].append(np.nan)
                elif epoch in self.metrics_data['epoch']:
                     idx = self.metrics_data['epoch'].index(epoch); self.metrics_data['train_box_loss'][idx] = box_loss; self.metrics_data['train_cls_loss'][idx] = cls_loss; self.metrics_data['train_dfl_loss'][idx] = dfl_loss
            except (ValueError, IndexError) as e: logging.warning(f"Error parsing epoch/loss line: {e} | Line: '{cleaned_line}'")

        # Validation mAP parsing (same as before)
        val_map_match = re.search(r"\s*all\s+\d+\s+\d+\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+([\d.]+)", cleaned_line)
        if val_map_match:
            logging.debug(f"Validation mAP line match: '{cleaned_line}'")
            try:
                map50, map50_95 = map(float, val_map_match.groups()); self.latest_map50 = map50; self.latest_map50_95 = map50_95
                self._update_key_metrics_display() # Update training dashboard
                if self.metrics_data['epoch']:
                    last_recorded_epoch = self.metrics_data['epoch'][-1]
                    try:
                        epoch_index = self.metrics_data['epoch'].index(last_recorded_epoch)
                        # Ensure lists are long enough (pad with nan if needed)
                        while len(self.metrics_data['val_map50']) <= epoch_index: self.metrics_data['val_map50'].append(np.nan)
                        while len(self.metrics_data['val_map50_95']) <= epoch_index: self.metrics_data['val_map50_95'].append(np.nan)
                        self.metrics_data['val_map50'][epoch_index] = map50; self.metrics_data['val_map50_95'][epoch_index] = map50_95
                        self.last_val_image_epoch = last_recorded_epoch
                    except (ValueError, IndexError) as e: logging.error(f"Index error storing validation data for epoch {last_recorded_epoch}: {e}")
                else: logging.warning("Parsed validation line, but no epochs recorded yet.")
            except (ValueError, IndexError) as e: logging.warning(f"Error parsing/storing validation data: {e} | Line: '{cleaned_line}'")

        # Speed parsing (same as before)
        speed_match = re.search(r"Speed:\s*([\d.]+ms)\s*pre.*,\s*([\d.]+ms)\s*infer.*,\s*([\d.]+ms)\s*post.*", cleaned_line)
        if speed_match: pre, infer, post = speed_match.groups(); self.current_speed_str = f"{infer}/step"

    def _check_and_set_train_experiment_path(self, line): # Renamed
        if self.current_experiment_path is None:
            # Match "Results saved to runs/detect/exp<number>" or similar paths
            path_match = re.search(r"Results saved to\s+([\w/\\]+[\w\\]*)", line.strip())
            if path_match:
                 found_path_raw = path_match.group(1).strip()
                 # Clean potential ANSI codes just in case
                 exp_path_cleaned = clean_ansi_codes(found_path_raw)
                 # Make path absolute and normalized
                 abs_exp_path = os.path.abspath(os.path.normpath(exp_path_cleaned))

                 # Check if this path looks like a train path (heuristics)
                 if self.project_name_entry.get().replace('\\','/') in abs_exp_path.replace('\\','/'): # Check if project name is part of path
                     # Schedule a check, as the directory might not exist *immediately*
                     self.after(100, self._confirm_and_set_path, abs_exp_path, "train")
                 else:
                     logging.debug(f"Ignoring potential path '{abs_exp_path}' as it doesn't match training project name.")


    # --- Inference Logic ---
    def _draw_boxes_on_frame(self, frame, results):
        """Draw bounding boxes and labels on a frame using YOLO results."""
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = results.names[cls] if hasattr(results, 'names') else str(cls)
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(frame, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        return frame

    def _run_live_video_inference(self, model_path, source_path, conf, iou, device, run_id):
        import time
        try:
            model = YOLO(model_path)
            cap = cv2.VideoCapture(source_path)
            if not cap.isOpened():
                self._log(f"[ERROR] Could not open video: {source_path}\n", self.infer_console_log)
                self._update_infer_status_indicator("Error")
                self.is_inferring = False
                return
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.total_inference_frames = total_frames
            self.after(0, lambda: self.infer_progress_bar.grid(row=2, column=0, padx=10, pady=(0, 5), sticky="ew"))
            frame_num = 0
            persistent_size = None
            new_photoimage_needed = True
            while cap.isOpened() and not self.infer_stop_event.is_set() and run_id == self.infer_run_id:
                ret, frame = cap.read()
                if not ret:
                    break
                results = model(frame, conf=conf, iou=iou, device=device)[0]
                frame = self._draw_boxes_on_frame(frame, results)
                # Convert BGR to RGB for PIL
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                # Resize to fit display
                img_w, img_h = pil_img.size
                target_w, target_h = INFER_OUTPUT_DISPLAY_SIZE
                ratio = min(target_w / img_w, target_h / img_h)
                new_w = int(img_w * ratio)
                new_h = int(img_h * ratio)
                resized_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                # Always create a new PhotoImage on the first frame after restart or if size changes
                if new_photoimage_needed or persistent_size != (new_w, new_h):
                    def create_and_assign_photoimage():
                        print(f"[DEBUG] Creating new PhotoImage: label id={id(self.infer_output_label)}, exists={self.infer_output_label.winfo_exists()}, size=({new_w},{new_h})")
                        try:
                            self.infer_output_label.configure(image=None)
                        except tk.TclError:
                            print("[DEBUG] TclError when clearing label image in create_and_assign_photoimage")
                            pass
                        self.infer_output_photoimage = ImageTk.PhotoImage(resized_img)
                        try:
                            self.infer_output_label.configure(image=self.infer_output_photoimage, text="")
                        except tk.TclError:
                            print("[DEBUG] TclError when assigning new PhotoImage to label")
                            pass
                        self.infer_output_label.update_idletasks()
                        print(f"[DEBUG] Assigned new PhotoImage: label id={id(self.infer_output_label)}, exists={self.infer_output_label.winfo_exists()}")
                    self.after(0, create_and_assign_photoimage)
                    persistent_size = (new_w, new_h)
                    new_photoimage_needed = False
                else:
                    def update_photoimage():
                        if self.infer_output_photoimage is not None:
                            try:
                                self.infer_output_photoimage.paste(resized_img)
                                self.infer_output_label.configure(image=self.infer_output_photoimage, text="")
                            except tk.TclError:
                                print("[DEBUG] TclError when updating PhotoImage in update_photoimage")
                                pass
                            self.infer_output_label.update_idletasks()
                            print(f"[DEBUG] Updated PhotoImage: label id={id(self.infer_output_label)}, exists={self.infer_output_label.winfo_exists()}")
                    self.after(0, update_photoimage)
                frame_num += 1
                self.current_inference_frame_count = frame_num
                if total_frames > 0:
                    progress = min(1.0, frame_num / total_frames)
                    self.after(0, self.infer_progress_bar.set, progress)
                time.sleep(0.03)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            self.is_inferring = False
            self.after(0, self._inference_finished)
        except Exception as e:
            self._log(f"[ERROR] Exception during live video inference: {e}\n", self.infer_console_log)
            self.is_inferring = False
            self.after(0, self._inference_finished)

    def _update_infer_output_label(self, frame_rgb, run_id):
        # Debug print to diagnose UI update issues
        print(f"[DEBUG] is_inferring={self.is_inferring}, label_exists={self.infer_output_label.winfo_exists()}, run_id={run_id}, current_run_id={self.infer_run_id}")
        # Only update if inference is running, label exists, and run_id matches
        if not self.is_inferring or not self.infer_output_label.winfo_exists() or run_id != self.infer_run_id:
            return
        try:
            pil_img = Image.fromarray(frame_rgb)
            img_w, img_h = pil_img.size
            target_w, target_h = INFER_OUTPUT_DISPLAY_SIZE
            ratio = min(target_w / img_w, target_h / img_h)
            new_w = int(img_w * ratio)
            new_h = int(img_h * ratio)
            resized_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(resized_img)
            self.infer_output_photoimage = photo
            self.infer_output_label.configure(image=photo, text="")
            self.infer_output_label.update_idletasks()
            print(f"[DEBUG] Frame displayed on UI (run_id={run_id})")
        except tk.TclError:
            print(f"[DEBUG] TclError in _update_infer_output_label (run_id={run_id})")
            pass

    def _start_inference(self):
        logging.info("Start inference button clicked.")
        if not self._validate_infer_inputs(): logging.warning("Inference input validation failed."); return
        if self.is_inferring: logging.warning("Start clicked but inference is already active."); messagebox.showwarning("Inference Active", "An inference process is already running."); return
        if self.is_training: messagebox.showwarning("Training Active", "A training process is running. Please stop it before starting inference."); return

        self._clear_log(self.infer_console_log) # Clear inference console
        self._clear_inference_output() # Clear output display
        self._update_infer_status_indicator("Starting")
        self.infer_stop_event.clear()
        self.infer_results_button.configure(state="disabled")
        self.infer_results_path = None
        self.is_inferring = False
        self.last_displayed_frame_num = -1
        self.current_inference_frame_count = 0
        self.total_inference_frames = 0
        self.infer_progress_bar.set(0)
        self.infer_progress_bar.grid_forget() # Hide progress bar initially
        self.infer_run_id += 1  # Increment run ID for each new inference
        run_id = self.infer_run_id  # Capture the current run ID to pass to the thread

        # Wait for previous inference thread to finish if still alive
        if hasattr(self, 'inference_thread') and self.inference_thread is not None and self.inference_thread.is_alive():
            try:
                logging.info("Waiting for previous inference thread to finish before starting new one...")
                self.inference_thread.join(timeout=2.0)
                if self.inference_thread.is_alive():
                    logging.warning("Previous inference thread did not finish cleanly.")
            except Exception as e:
                logging.error(f"Error joining previous inference thread: {e}")
        self.inference_thread = None

        # Get parameters from Inference UI
        self.infer_source_path = self.infer_source_entry.get() # Already validated
        conf = self.infer_conf_slider.get()
        iou = self.infer_iou_slider.get()
        device = self.infer_device_combobox.get()
        project = self.infer_project_name_entry.get()
        name = self.infer_exp_name_entry.get()
        model_selection = self.infer_model_combobox.get()
        weights = self.custom_infer_model_path if model_selection == "Custom..." else model_selection

        if self.infer_source_type == 'video':
            # Use live video inference
            self.is_inferring = True
            self.infer_run_button.configure(state="disabled")
            self.infer_stop_button.configure(state="normal")
            self._set_infer_config_panel_state("disabled")
            self._update_infer_status_indicator("Inferring")
            self.inference_thread = threading.Thread(
                target=self._run_live_video_inference,
                args=(weights, self.infer_source_path, conf, iou, device, run_id),
                daemon=True, name="YoloLiveVideoInferThread"
            )
            self.inference_thread.start()
            logging.info("Live video inference thread started.")
        else:
            # ... existing code for image inference ...
            # Build command for yolo predict
            yolo_cmd_base = ["yolo", "predict"] # Assuming YOLO CLI is in PATH
            yolo_args = yolo_cmd_base + [
                f"model={weights}",
                f"source={self.infer_source_path}",
                f"conf={conf:.2f}",
                f"iou={iou:.2f}",
                f"device={device}",
                f"project={project}",
                f"name={name}",
                "save=True", # Save output images/video
                "exist_ok=True", # Allow overwriting previous run with same name
                "save_txt=False", # Optionally save labels
                "save_conf=False" # Optionally save confidence scores in labels
            ]
            log_cmd_str = ' '.join(map(str, yolo_args))
            logging.info(f"Constructed YOLO Inference command: {log_cmd_str}")
            self._log(f"Starting inference with command:\n{log_cmd_str}\n" + "="*30 + "\n", self.infer_console_log)
            self.is_inferring = True
            self.infer_run_button.configure(state="disabled")
            self.infer_stop_button.configure(state="normal")
            self._set_infer_config_panel_state("disabled")
            self._update_infer_status_indicator("Inferring")
            self.inference_thread = threading.Thread(target=self._run_inference_process, args=(yolo_args,), daemon=True, name="YoloInferThread")
            self.inference_thread.start()
            logging.info("Inference thread started.")

    def _run_inference_process(self, command):
        self._run_generic_process(command, self.inference_process, self.infer_stop_event, self.infer_log_queue, "Inference")


    def _parse_infer_log_line(self, line):
        """Parses inference output for save path and frame progress."""
        cleaned_line = clean_ansi_codes(line).strip()
        # logging.debug(f"Infer Line: {cleaned_line}") # Verbose

        # 1. Check for the results save path (usually printed near the start)
        if self.infer_results_path is None:
            path_match = re.search(r"Results saved to\s+([\w/\\]+[\w\\]*)", cleaned_line)
            if path_match:
                found_path_raw = path_match.group(1).strip()
                exp_path_cleaned = clean_ansi_codes(found_path_raw)
                abs_exp_path = os.path.abspath(os.path.normpath(exp_path_cleaned))
                 # Schedule a check, as the directory might not exist *immediately*
                self.after(100, self._confirm_and_set_path, abs_exp_path, "infer")

        # 2. Check for image/video processing progress
        # Example Image Line: image 1/1 /path/to/image.jpg: 640x480 1 person, 1 dog, 10.5ms
        # Example Video Line: video 1/1 (1/300) /path/to/video.mp4: 640x384 2 persons, 11.8ms
        progress_match = None
        if self.infer_source_type == 'image':
            progress_match = re.search(r"image \d+/\d+ .*?:", cleaned_line) # Simpler match for image done
        elif self.infer_source_type == 'video':
            progress_match = re.search(r"video \d+/\d+ \((\d+)/\d+\) .*?:", cleaned_line) # Capture frame number

        if progress_match:
            frame_num = 0 # Default for image
            if self.infer_source_type == 'video':
                try: frame_num = int(progress_match.group(1)); self.current_inference_frame_count = frame_num
                except (IndexError, ValueError): pass # Ignore if group not found or not int

            # Update progress bar for video
            if self.total_inference_frames > 0 and self.infer_source_type == 'video':
                progress = min(1.0, frame_num / self.total_inference_frames)
                self.after(0, self.infer_progress_bar.set, progress) # Schedule UI update

            # Attempt to display the output frame/image IF the results path is known
            if self.infer_results_path:
                # Don't update too frequently for videos, maybe every N frames or if it's the first time
                should_update_display = (self.infer_source_type == 'image') or \
                                        (frame_num > self.last_displayed_frame_num) # or \
                                        # (frame_num % 5 == 0) # Example: update every 5 frames
                                        # (self.last_displayed_frame_num == -1) # Always show first frame

                if should_update_display:
                    # Schedule the display update from the main thread
                    self.after(0, self._display_inference_output, frame_num)
                    self.last_displayed_frame_num = frame_num

    def _confirm_and_set_path(self, path_to_check, path_type, attempt=1):
        """Checks if a directory exists and sets the corresponding path variable."""
        if os.path.isdir(path_to_check):
            if path_type == "train":
                self.current_experiment_path = path_to_check
                log_msg = f"[INFO] Detected training results directory: {self.current_experiment_path}"
                logging.info(log_msg)
                self._log(log_msg + "\n", self.console_log)
                self.after(0, lambda: self.results_button.configure(state="normal")) # Enable train results button
                logging.debug("Enabled training results button.")
            elif path_type == "infer":
                self.infer_results_path = path_to_check
                log_msg = f"[INFO] Detected inference results directory: {self.infer_results_path}"
                logging.info(log_msg)
                self._log(log_msg + "\n", self.infer_console_log)
                self.after(0, lambda: self.infer_results_button.configure(state="normal")) # Enable infer results button
                logging.debug("Enabled inference results button.")
                # Maybe trigger initial display if it's an image result?
                if self.infer_source_type == 'image' and not self.is_inferring:
                     self.after(50, self._display_inference_output, 0) # Display after short delay

        elif attempt < 5: # Retry a few times
            logging.debug(f"{path_type} path check failed for '{path_to_check}', retrying (attempt {attempt+1})...")
            self.after(300, self._confirm_and_set_path, path_to_check, path_type, attempt + 1)
        else:
            logging.warning(f"Detected {path_type} path string '{path_to_check}', but directory check failed after multiple attempts.")


    def _display_inference_output(self, frame_num=0):
        """Loads and displays the inference result (image or specific video frame)."""
        if not self.infer_results_path:
            logging.debug("Cannot display output: Inference results path not set yet.")
            return

        output_file_path = None
        base_source_name = os.path.splitext(os.path.basename(self.infer_source_path))[0]

        if self.infer_source_type == 'image':
            base_name = os.path.splitext(os.path.basename(self.infer_source_path))[0]

            # Simple case: output image has the same name as input
            output_file_path = os.path.join(self.infer_results_path, base_name + ".jpg")
        elif self.infer_source_type == 'video':
            # Complex case: find the frame image saved by YOLO
            # Default YOLO 'save=True' for videos saves the processed *video* file,
            # AND if stream=True, it also often saves *individual frames* in a subdirectory.
            # Let's prioritize showing the individual frame if possible.
            frame_subdir = os.path.join(self.infer_results_path, base_source_name + "_frames") # Heuristic subdir name
            frame_filename = f"frame_{frame_num:04d}.jpg" # Heuristic filename pattern (needs confirmation based on actual YOLO output)

            # Look for specific frame file - THIS MIGHT BE WRONG depending on YOLO version/settings
            # Common output: <project>/<name>/frame_0001.jpg, frame_0002.jpg etc. directly in the exp folder
            potential_frame_path = os.path.join(self.infer_results_path, f"frame_{frame_num:04d}.jpg") # Check directly in results path
            potential_frame_path_alt = os.path.join(self.infer_results_path, f"{base_source_name}_{frame_num}.jpg") # Alternative pattern

            if os.path.exists(potential_frame_path):
                 output_file_path = potential_frame_path
            elif os.path.exists(potential_frame_path_alt):
                 output_file_path = potential_frame_path_alt
            else:
                 # Fallback: If inference is finished, try showing the saved video file (or its first frame)
                 if not self.is_inferring:
                     output_video_path = os.path.join(self.infer_results_path, os.path.basename(self.infer_source_path))
                     if os.path.exists(output_video_path):
                         logging.info(f"Video processing finished. Trying to display first frame of saved video: {output_video_path}")
                         output_file_path = self._extract_first_frame(output_video_path)
                     else:
                          logging.warning(f"Could not find individual frame image or final saved video: {output_video_path}")
                          # Maybe try finding *any* image in the results folder as a last resort?
                          images = glob.glob(os.path.join(self.infer_results_path, "*.jpg"))
                          if images:
                              output_file_path = images[0] # Display the first one found
                              logging.info(f"Displaying first found image in results: {output_file_path}")


        if output_file_path and os.path.exists(output_file_path):
            try:
                logging.debug(f"Attempting to display: {output_file_path}")
                image = Image.open(output_file_path)

                # Calculate aspect ratio and resize to fit INFER_OUTPUT_DISPLAY_SIZE
                img_w, img_h = image.size
                target_w, target_h = INFER_OUTPUT_DISPLAY_SIZE
                ratio = min(target_w / img_w, target_h / img_h)
                new_w = int(img_w * ratio)
                new_h = int(img_h * ratio)

                # Use LANCZOS for resizing
                resized_image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

                # Convert PIL image to PhotoImage for Tkinter
                self.infer_output_photoimage = ImageTk.PhotoImage(resized_image)

                # Update the CTkLabel
                self.infer_output_label.configure(image=self.infer_output_photoimage, text="") # Clear placeholder text
                # Force update sometimes needed
                self.infer_output_label.update_idletasks()

            except (FileNotFoundError, UnidentifiedImageError, Exception) as e:
                logging.error(f"Error loading or displaying image {output_file_path}: {e}")
                self._clear_inference_output(error=True) # Show error state
        else:
            # Don't clear if file just not found yet (might appear later) unless inference is done
            if not self.is_inferring:
                 logging.warning(f"Output file not found for display: {output_file_path}")
                 self._clear_inference_output() # Clear if finished and no file found


    def _extract_first_frame(self, video_path):
        """Extracts the first frame of a video and saves it as a temporary image."""
        temp_frame_path = None
        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logging.error(f"Cannot open video file: {video_path}")
                return None

            ret, frame = cap.read()
            if ret:
                # Save the frame to a temporary file (e.g., in the results dir)
                temp_frame_path = os.path.join(os.path.dirname(video_path), "_temp_first_frame.jpg")
                cv2.imwrite(temp_frame_path, frame)
                logging.info(f"Saved first frame to temporary file: {temp_frame_path}")
                return temp_frame_path
            else:
                logging.warning(f"Could not read first frame from video: {video_path}")
                return None
        except Exception as e:
            logging.error(f"Error extracting first frame from {video_path}: {e}")
            return None
        finally:
            if cap:
                cap.release()

    def _clear_inference_output(self, error=False):
        """Clears the inference output display area."""
        try:
            print(f"[DEBUG] Clearing output: label id={id(self.infer_output_label)}, exists={self.infer_output_label.winfo_exists()}")
            # Destroy and recreate the label for a fresh start
            self._create_new_infer_output_label()
            try:
                self.infer_output_label.configure(image=None)
            except tk.TclError:
                print("[DEBUG] TclError when clearing label image")
                pass
            placeholder_text = "Error processing output" if error else "Output will appear here"
            self.infer_output_label.configure(text=placeholder_text)
            self.infer_output_photoimage = None # Clear PhotoImage reference
            self.last_displayed_frame_num = -1 # Reset frame tracking
            self.infer_progress_bar.set(0) # Reset progress bar
            self.infer_output_label.update_idletasks()
            self.update_idletasks()
            print(f"[DEBUG] Output cleared: label id={id(self.infer_output_label)}, exists={self.infer_output_label.winfo_exists()}")
        except tk.TclError: pass # Ignore if widget destroyed
        except AttributeError: pass # Ignore if not created yet


    def _update_key_metrics_display(self): # Training specific
        map50_text = f"{self.latest_map50:.4f}" if self.latest_map50 is not None else "N/A"; map50_95_text = f"{self.latest_map50_95:.4f}" if self.latest_map50_95 is not None else "N/A"
        try: self.latest_map50_label.configure(text=map50_text); self.latest_map50_95_label.configure(text=map50_95_text)
        except tk.TclError as e: logging.warning(f"TclError updating key metric labels: {e}")
        except AttributeError: pass # Ignore if widgets not ready

    # --- Live Data Update Loop (_update_live_data, _update_stats_display - Training specific) ---
    def _update_live_data(self):
        if not self.is_training: logging.debug("Live data update loop stopping (not training)."); self.live_data_update_job = None; return
        if self.start_time and self.current_epoch > 0 and self.total_epochs > 0:
            elapsed_seconds = time.time() - self.start_time; time_per_epoch = elapsed_seconds / self.current_epoch; remaining_epochs = self.total_epochs - self.current_epoch; etr_seconds = remaining_epochs * time_per_epoch; self.etr_str = format_time(etr_seconds)
        elif self.is_training: self.etr_str = "Calculating..."
        else: self.etr_str = "--:--:--"
        if self.start_time: self.elapsed_time_str = format_time(time.time() - self.start_time)
        self._update_stats_display(); self._update_matplotlib_plots()
        self.live_data_update_job = self.after(LIVE_DATA_UPDATE_INTERVAL_MS, self._update_live_data)

    def _update_stats_display(self): # Training specific
        try:
            self.epoch_label.configure(text=f"Epoch: {self.current_epoch} / {self.total_epochs}")
            self.time_label.configure(text=f"Time: {self.elapsed_time_str}")
            self.etr_label.configure(text=f"ETR: {self.etr_str}")
            self.speed_label.configure(text=f"Speed: {self.current_speed_str}")
            if self.total_epochs > 0: progress = min(1.0, (self.current_epoch) / self.total_epochs) if self.current_epoch > 0 else 0; self.progress_bar.set(progress)
            else: self.progress_bar.set(0)
        except tk.TclError: logging.warning("TclError updating stats display")
        except AttributeError: pass # Ignore if widgets not ready


    # --- Matplotlib Plot Update (Training specific) ---
    def _update_matplotlib_plots(self):
        try:
            epochs = self.metrics_data['epoch']; max_len = len(epochs)
            if not epochs: return # Nothing to plot
            epochs_arr = np.array(epochs); plot_data = {}
            # Ensure all metric lists have the same length as epochs, padding with nan
            for key in self.metrics_data:
                data = self.metrics_data[key][:]
                while len(data) < max_len: data.append(np.nan)
                # Take only up to max_len to handle potential race conditions
                plot_data[key] = np.array(data[:max_len])

            self._setup_plots() # Re-applies all styles
            # Plot loss components
            self.ax1.plot(epochs_arr, plot_data['train_box_loss'], label='Box', marker='.', linestyle='-', markersize=4)
            self.ax1.plot(epochs_arr, plot_data['train_cls_loss'], label='Cls', marker='.', linestyle='-', markersize=4)
            self.ax1.plot(epochs_arr, plot_data['train_dfl_loss'], label='DFL', marker='.', linestyle='-', markersize=4)
            if any(not np.isnan(x) for x in plot_data['train_box_loss']): self.ax1.legend(fontsize='x-small')
            # Adjust Y limits dynamically but keep bottom at 0
            valid_losses = np.concatenate([plot_data['train_box_loss'], plot_data['train_cls_loss'], plot_data['train_dfl_loss']])
            valid_losses = valid_losses[~np.isnan(valid_losses)]
            if len(valid_losses) > 0:
                 current_ymin, current_ymax = self.ax1.get_ylim()
                 ymax_data = np.nanmax(valid_losses)
                 self.ax1.set_ylim(bottom=0, top=max(ymax_data * 1.1, current_ymax, 0.1)) # Add 10% margin, ensure at least 0.1 height
            else: self.ax1.set_ylim(bottom=0, top=0.1)


            # Plot validation mAP
            self.ax2.plot(epochs_arr, plot_data['val_map50'], label='mAP@.50', marker='o', linestyle='-', markersize=4)
            self.ax2.plot(epochs_arr, plot_data['val_map50_95'], label='mAP@.5-.95', marker='s', linestyle='--', markersize=4)
            if any(not np.isnan(x) for x in plot_data['val_map50']): self.ax2.legend(fontsize='x-small')
            # Set X limits based on epochs plotted
            if len(epochs_arr)>0: self.ax2.set_xlim(left=max(0, epochs_arr[0]-1), right=epochs_arr[-1]+1)
            # Y limits are fixed 0 to 1.05

            self.fig.tight_layout(pad=2.0) # Recalculate layout
            self.canvas.draw_idle() # Redraw efficiently
        except Exception as e: logging.exception(f"Error updating matplotlib plots: {e}")


    # --- Stop/Finish Logic (Separate for Train/Infer) ---
    def _stop_training(self):
        logging.info("Stop training button clicked.")
        if not self.is_training or self.training_thread is None or not self.training_thread.is_alive():
             logging.warning("Stop clicked but no active training."); messagebox.showinfo("Info", "No active training process to stop.")
             self._update_train_status_indicator("Idle"); self.start_button.configure(state="normal"); self.stop_button.configure(state="disabled", text="Stop Training"); self._set_train_config_panel_state("normal")
             return

        # Check if inference is also running (shouldn't happen based on start checks, but safety)
        if self.is_inferring:
             messagebox.showwarning("Inference Active", "Inference is also running. Stopping training only.")

        confirm = messagebox.askyesno("Stop Training", "Are you sure you want to stop the training process?\nThis attempts a graceful stop.")
        if confirm:
            logging.info("User confirmed training stop."); self._log("\n[USER] Initiating training stop...\n", self.console_log); self._update_train_status_indicator("Stopping"); self.stop_event.set();
            self.stop_button.configure(state="disabled", text="Stopping...")
            # Stop the live data updates for training
            if self.live_data_update_job: self.after_cancel(self.live_data_update_job); self.live_data_update_job = None
        else: logging.info("User cancelled training stop.")

    def _stop_inference(self):
        logging.info("Stop inference button clicked.")
        if not self.is_inferring or self.inference_thread is None or not self.inference_thread.is_alive():
             logging.warning("Stop clicked but no active inference."); messagebox.showinfo("Info", "No active inference process to stop.")
             self._update_infer_status_indicator("Idle"); self.infer_run_button.configure(state="normal"); self.infer_stop_button.configure(state="disabled", text="Stop Inference"); self._set_infer_config_panel_state("normal")
             return

        confirm = messagebox.askyesno("Stop Inference", "Are you sure you want to stop the inference process?")
        if confirm:
            logging.info("User confirmed inference stop."); self._log("\n[USER] Initiating inference stop...\n", self.infer_console_log); self._update_infer_status_indicator("Stopping"); self.infer_stop_event.set();
            self.infer_stop_button.configure(state="disabled", text="Stopping...")
            # No specific live update loop for inference currently, but could cancel output updates if added
            # if self.infer_output_update_job: self.after_cancel(self.infer_output_update_job); self.infer_output_update_job = None
        else: logging.info("User cancelled inference stop.")


    def _training_finished(self):
        logging.info("Training finished signal received."); self.is_training = False
        if self.live_data_update_job: self.after_cancel(self.live_data_update_job); self.live_data_update_job = None; logging.info("Live data update loop stopped.")

        # Determine final status based on stop event and console log
        final_status = "Finished";
        try: log_content = self.console_log.get("1.0", tk.END)
        except: log_content = "" # Handle potential error getting log

        if self.stop_event.is_set(): final_status = "Stopped"
        elif "[ERROR]" in log_content or "Traceback" in log_content or \
             ("process exited with code" in log_content and "exited with code 0" not in log_content):
             final_status = "Error"

        self._update_train_status_indicator(final_status)
        self.current_speed_str = final_status; self.etr_str = "--:--:--" # Update final dashboard stats
        self._update_stats_display(); self._update_key_metrics_display(); self._update_matplotlib_plots() # Final plot update

        # Reset UI elements
        self.start_button.configure(state="normal"); self.stop_button.configure(state="disabled", text="Stop Training"); self._set_train_config_panel_state("normal")

        # Enable results button if path is valid
        if self.current_experiment_path and os.path.isdir(self.current_experiment_path):
            self.results_button.configure(state="normal")
        else:
            self.results_button.configure(state="disabled")
            logging.warning("Training finished, but results path not confirmed.")

        # Final progress bar state
        if final_status == "Finished": self.progress_bar.set(1.0); logging.info("Setting progress to 1.0")
        elif final_status == "Stopped": logging.info("Training stopped, progress bar left as is.")
        else: logging.warning("Training finished with errors, progress bar left as is.")


    def _inference_finished(self):
        logging.info("Inference finished signal received."); self.is_inferring = False
        # Clear output image and reference to avoid Tkinter image errors
        self._clear_inference_output()
        self.inference_thread = None
        try:
             print(f"[DEBUG] Forcing label reset after inference finished: label id={id(self.infer_output_label)}, exists={self.infer_output_label.winfo_exists()}")
             self.infer_output_label.configure(image=None)
             self.infer_output_label.update_idletasks()
             self.update_idletasks()
             print(f"[DEBUG] Label reset after inference finished: label id={id(self.infer_output_label)}, exists={self.infer_output_label.winfo_exists()}")
        except tk.TclError:
             print("[DEBUG] TclError when forcing label reset after inference finished")
             pass
        self.infer_output_photoimage = None
        # Could cancel any periodic output update jobs here if they existed

        # Determine final status
        final_status = "Finished"
        try: log_content = self.infer_console_log.get("1.0", tk.END)
        except: log_content = ""

        if self.infer_stop_event.is_set(): final_status = "Stopped"
        elif "[ERROR]" in log_content or "Traceback" in log_content or \
             ("process exited with code" in log_content and "exited with code 0" not in log_content):
            final_status = "Error"

        self._update_infer_status_indicator(final_status)

        # Reset UI elements for inference tab
        self.infer_run_button.configure(state="normal")
        self.infer_stop_button.configure(state="disabled", text="Stop Inference")
        self._set_infer_config_panel_state("normal")

        # Enable results button if path is valid
        if self.infer_results_path and os.path.isdir(self.infer_results_path):
            self.infer_results_button.configure(state="normal")
            # Attempt one final display update in case the last frame wasn't shown
            self.after(100, self._display_inference_output, self.current_inference_frame_count)
        else:
            self.infer_results_button.configure(state="disabled")
            logging.warning("Inference finished, but results path not confirmed.")
            if final_status == "Error":
                 self._clear_inference_output(error=True)

        # Final progress bar state for video
        if self.infer_source_type == 'video':
            if final_status == "Finished":
                self.infer_progress_bar.set(1.0)
            elif final_status == "Error" and self.total_inference_frames > 0:
                # Set progress based on last processed frame
                progress = min(1.0, self.current_inference_frame_count / self.total_inference_frames)
                self.infer_progress_bar.set(progress)
            # For "Stopped", leave progress as is


    # --- Enable/Disable Config Panels (Separate for Train/Infer) ---
    def _set_train_config_panel_state(self, state):
        """Sets the state of widgets in the training config panel."""
        try:
            # Find the config panel within the "Train" tab
            train_tab = self.tab_view.tab("Train")
            config_outer_frame = train_tab.winfo_children()[0] # Assuming it's the first child
            config_frame = config_outer_frame.winfo_children()[1] # Assuming content frame is second child of outer frame
            logging.debug(f"Setting training config panel state to: {state}")

            # Widgets to ignore (System Status and Theme are shared, handled separately if needed)
            ignore_widgets = {self.system_status_frame, self.theme_frame}
            # Add dynamically created system widgets to ignore list
            for key, widget_group in self.system_status_widgets.items():
                if isinstance(widget_group, dict): ignore_widgets.update(widget_group.values())
                else: ignore_widgets.add(widget_group)

            for widget in config_frame.winfo_children():
                 if widget in ignore_widgets: continue # Skip system widgets
                 # Skip headers and separators
                 is_header = isinstance(widget, ctk.CTkLabel) and widget.cget("font") and "bold" in widget.cget("font").actual("weight")
                 is_separator = isinstance(widget, ctk.CTkFrame) and widget.cget("height") == SEPARATOR_HEIGHT
                 if is_header or is_separator: continue
                 # Skip the specific Resume checkbox if needed during resume? (No, disable all config)
                 # if widget == self.resume_checkbox and state == "disabled": continue

                 if isinstance(widget, (ctk.CTkEntry, ctk.CTkButton, ctk.CTkComboBox, ctk.CTkCheckBox, ctk.CTkSlider)):
                     if widget == self.theme_switch: continue # Always keep theme switch enabled
                     try: widget.configure(state=state)
                     except (tk.TclError, AttributeError): pass # Ignore errors if widget destroyed or not configurable

        except (IndexError, AttributeError, Exception) as e:
            logging.exception(f"Error setting training config panel state: {e}")

    def _set_infer_config_panel_state(self, state):
        """Sets the state of widgets in the inference config panel."""
        try:
            # Find the config panel within the "Infer" tab
            infer_tab = self.tab_view.tab("Infer")
            config_outer_frame = infer_tab.winfo_children()[0] # Assuming it's the first child
            config_frame = config_outer_frame.winfo_children()[1] # Assuming content frame is second
            logging.debug(f"Setting inference config panel state to: {state}")

            for widget in config_frame.winfo_children():
                 # Skip headers and separators
                 is_header = isinstance(widget, ctk.CTkLabel) and widget.cget("font") and "bold" in widget.cget("font").actual("weight")
                 is_separator = isinstance(widget, ctk.CTkFrame) and widget.cget("height") == SEPARATOR_HEIGHT
                 # Skip status label and control button frames (handled separately)
                 if widget in [self.infer_status_label] or isinstance(widget.master, ctk.CTkFrame) and widget.master.winfo_parent() == config_frame.winfo_id(): # Check if parent is the main config frame to find button/slider frames
                     parent_info = widget.master.grid_info() if isinstance(widget.master, ctk.CTkFrame) else {}
                     is_control_frame = 'infer_control_frame' in str(widget.master) or 'conf_frame' in str(widget.master) or 'iou_frame' in str(widget.master) # Hacky check

                     # More robust check: Find the specific frames we created
                     if widget.master == getattr(self, 'infer_control_frame', None) or \
                        widget.master == getattr(self, 'conf_frame', None) or \
                        widget.master == getattr(self, 'iou_frame', None):
                         # Check children of these specific frames
                         if isinstance(widget, (ctk.CTkSlider, ctk.CTkLabel)): # Allow labels within frames to stay enabled
                              if not isinstance(widget, ctk.CTkSlider): # Keep labels enabled
                                  continue
                         # Fallthrough to disable sliders etc. inside these frames

                 if is_header or is_separator: continue

                 # Enable/Disable actual input widgets
                 if isinstance(widget, (ctk.CTkEntry, ctk.CTkButton, ctk.CTkComboBox, ctk.CTkSlider)):
                      # Exclude the Stop button and Results button from this generic disabling
                      if widget in [self.infer_stop_button, self.infer_results_button]: continue
                      try: widget.configure(state=state)
                      except (tk.TclError, AttributeError): pass # Ignore errors

        except (IndexError, AttributeError, Exception) as e:
            logging.exception(f"Error setting inference config panel state: {e}")


    # --- System Monitor Methods (Fetching CPU/RAM/GPU - same logic, updates UI) ---
    def _start_system_monitor(self):
        if self.gpu_monitor_thread is None or not self.gpu_monitor_thread.is_alive():
            self.gpu_stop_event.clear()
            self.gpu_monitor_thread = threading.Thread(target=self._run_system_monitor, daemon=True, name="SystemMonitorThread")
            self.gpu_monitor_thread.start()
            logging.info("System Monitor thread started.")

    def _stop_system_monitor(self):
        if self.gpu_monitor_thread and self.gpu_monitor_thread.is_alive():
            logging.info("Stopping System monitor thread..."); self.gpu_stop_event.set()

    def _run_system_monitor(self):
        """Fetches system stats periodically and schedules UI update."""
        while not self.gpu_stop_event.is_set():
            current_stats = {'gpu': {}, 'cpu': None, 'ram': None, 'error': None}
            try:
                # CPU Usage
                current_stats['cpu'] = psutil.cpu_percent(interval=None) # Non-blocking

                # RAM Usage
                ram_info = psutil.virtual_memory()
                current_stats['ram'] = {
                    'used_gb': ram_info.used / (1024**3),
                    'total_gb': ram_info.total / (1024**3),
                    'percent': ram_info.percent
                }

                # GPU Usage (only if nvidia-smi works and device is set to GPU)
                device_setting = self.device_combobox.get() # Check the shared device setting
                gpu_indices_to_monitor = []
                if re.match(r"^\d+(,\d+)*$", device_setting): # Check if it's numeric GPU indices
                    try: gpu_indices_to_monitor = [int(idx) for idx in device_setting.split(',')]
                    except ValueError: gpu_indices_to_monitor = [] # Invalid format

                if gpu_indices_to_monitor: # Only run smi if GPUs are selected
                    try:
                        # Use a timeout for nvidia-smi to prevent hangs
                        smi_command = ['nvidia-smi', '--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits']
                        result = subprocess.run(smi_command, capture_output=True, text=True, check=True, timeout=5, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
                        lines = result.stdout.strip().split('\n')
                        for line in lines:
                            if not line: continue
                            try:
                                parts = [p.strip() for p in line.split(',')]
                                idx, name, temp_str, util_str, mem_used_str, mem_total_str = parts
                                idx = int(idx)
                                # Only include GPUs selected in the device combobox
                                if idx in gpu_indices_to_monitor:
                                    current_stats['gpu'][idx] = {
                                        'name': name,
                                        'temp': int(temp_str) if temp_str.isdigit() else None,
                                        'util': int(util_str) if util_str.isdigit() else None,
                                        'mem_used': int(mem_used_str) if mem_used_str.isdigit() else None,
                                        'mem_total': int(mem_total_str) if mem_total_str.isdigit() else None
                                    }
                            except (ValueError, IndexError) as parse_err:
                                logging.warning(f"Error parsing nvidia-smi line: '{line}'. Error: {parse_err}")
                                continue # Skip this line

                    except FileNotFoundError: current_stats['error'] = "nvidia-smi not found"; logging.warning("nvidia-smi not found, disabling GPU monitoring for this cycle."); # Don't stop the whole monitor
                    except subprocess.TimeoutExpired: current_stats['error'] = "nvidia-smi timeout"; logging.warning("nvidia-smi command timed out.")
                    except subprocess.CalledProcessError as e: current_stats['error'] = f"nvidia-smi error ({e.returncode})"; logging.error(f"nvidia-smi failed: {e}")
                    except Exception as e: current_stats['error'] = "GPU Parsing error"; logging.exception("Error parsing nvidia-smi.")
                else:
                     logging.debug(f"Device set to '{device_setting}', skipping nvidia-smi.")


            except psutil.Error as e: current_stats['error'] = f"psutil Error: {e}"; logging.error(f"Error fetching system stats: {e}")
            except Exception as e: current_stats['error'] = "System stat error"; logging.exception("Error fetching system stats.")

            self.system_stats = current_stats # Store latest data
            # Schedule the UI update to run on the main thread
            self.after(0, self._update_system_display, current_stats)

            # Wait for the interval, interruptible by the stop event
            self.gpu_stop_event.wait(GPU_MONITOR_INTERVAL_S)

        logging.info("System Monitor thread finished.")


    # --- _update_system_display (Updates the shared system status frame) ---
    def _update_system_display(self, data):
        """Updates the shared CPU, RAM, and GPU status display."""
        try:
            # --- Ensure the frame exists ---
            if not hasattr(self, 'system_status_frame') or not self.system_status_frame.winfo_exists():
                logging.warning("System status frame not available for update.")
                return

            # --- Clear previous widgets ---
            for widget in self.system_status_frame.winfo_children():
                widget.destroy()
            self.system_status_widgets.clear() # Clear the references

            current_row = 0
            label_width = 40 # Fixed width for "CPU:", "RAM:", "GPU X:" labels

            # --- Display Errors ---
            if data.get('error'):
                err_label = ctk.CTkLabel(self.system_status_frame, text=f"Sys Status: {data['error']}", text_color="orange", wraplength=300, anchor='w', font=ctk.CTkFont(size=10))
                err_label.grid(row=current_row, column=0, columnspan=3, sticky="ew", pady=(0, PAD_Y))
                self.system_status_widgets['error'] = err_label
                # Don't increment row, error replaces normal display for simplicity OR increment if showing below
                # current_row += 1 # Increment if error should be above stats

            # --- Display CPU ---
            cpu_percent = data.get('cpu')
            cpu_value = cpu_percent / 100.0 if cpu_percent is not None else 0
            cpu_text = f"{cpu_percent:.1f}%" if cpu_percent is not None else "N/A"

            cpu_title_label = ctk.CTkLabel(self.system_status_frame, text="CPU:", anchor="w", width=label_width, font=ctk.CTkFont(size=11))
            cpu_title_label.grid(row=current_row, column=0, sticky="w", padx=(0, PAD_Y), pady=1)

            cpu_bar = ctk.CTkProgressBar(self.system_status_frame,
                                         height=SYSTEM_BAR_HEIGHT,
                                         border_width=SYSTEM_BAR_BORDER_WIDTH,
                                         border_color=SYSTEM_BAR_GREEN,
                                         fg_color=SYSTEM_BAR_BACKGROUND,
                                         progress_color=SYSTEM_BAR_GREEN)
            cpu_bar.set(cpu_value)
            cpu_bar.grid(row=current_row, column=1, sticky="ew", padx=(0, PAD_Y), pady=1)

            cpu_text_label = ctk.CTkLabel(self.system_status_frame, text=cpu_text, anchor="w", width=40, font=ctk.CTkFont(size=11)) # Anchor W for consistency
            cpu_text_label.grid(row=current_row, column=2, sticky="w", padx=(0, 0), pady=1)

            self.system_status_widgets['cpu'] = {'title': cpu_title_label, 'bar': cpu_bar, 'text': cpu_text_label}
            current_row += 1

            # --- Display RAM ---
            ram_data = data.get('ram')
            if ram_data:
                ram_value = ram_data['percent'] / 100.0
                ram_text = f"{ram_data['used_gb']:.1f}/{ram_data['total_gb']:.1f} GiB ({ram_data['percent']:.1f}%)"
            else:
                ram_value = 0
                ram_text = "N/A"

            ram_title_label = ctk.CTkLabel(self.system_status_frame, text="RAM:", anchor="w", width=label_width, font=ctk.CTkFont(size=11))
            ram_title_label.grid(row=current_row, column=0, sticky="w", padx=(0, PAD_Y), pady=1)

            ram_bar = ctk.CTkProgressBar(self.system_status_frame,
                                         height=SYSTEM_BAR_HEIGHT,
                                         border_width=SYSTEM_BAR_BORDER_WIDTH,
                                         border_color=SYSTEM_BAR_GREEN,
                                         fg_color=SYSTEM_BAR_BACKGROUND,
                                         progress_color=SYSTEM_BAR_GREEN)
            ram_bar.set(ram_value)
            ram_bar.grid(row=current_row, column=1, sticky="ew", padx=(0, PAD_Y), pady=1)

            # Use the full text here, allow it to take necessary width
            ram_text_label = ctk.CTkLabel(self.system_status_frame, text=ram_text, anchor="w", font=ctk.CTkFont(size=10))
            ram_text_label.grid(row=current_row, column=2, sticky="w", padx=(0, 0), pady=1)

            self.system_status_widgets['ram'] = {'title': ram_title_label, 'bar': ram_bar, 'text': ram_text_label}
            current_row += 1

            # --- Display GPUs (if any and no major error) ---
            gpu_data = data.get('gpu', {})
            if not data.get('error') or "nvidia-smi" not in data.get('error',''): # Show GPUs even if psutil failed, maybe?
                 if not gpu_data:
                      # Only show 'No GPU' if device combo expected one
                      device_setting = self.device_combobox.get()
                      if re.match(r"^\d+(,\d+)*$", device_setting):
                          no_gpu_label = ctk.CTkLabel(self.system_status_frame, text="Selected GPU(s) not found by nvidia-smi.", text_color="gray", anchor='w', font=ctk.CTkFont(size=10))
                          no_gpu_label.grid(row=current_row, column=0, columnspan=3, sticky="ew")
                          self.system_status_widgets['no_gpu'] = no_gpu_label
                          current_row +=1
                 else:
                     # Add a small vertical space before GPUs if CPU/RAM are shown
                     if data.get('cpu') is not None or data.get('ram') is not None:
                         ctk.CTkFrame(self.system_status_frame, height=3, fg_color="transparent").grid(row=current_row, column=0, columnspan=3)
                         current_row += 1

                     for gpu_index in sorted(gpu_data.keys()):
                         stats = gpu_data[gpu_index]
                         temp = f"{stats['temp']}°C" if stats['temp'] is not None else "N/A"
                         util_val = stats['util'] if stats['util'] is not None else 0
                         util_pct_text = f"{util_val}%"
                         mem_used = stats['mem_used'] if stats['mem_used'] is not None else 0
                         mem_total = stats['mem_total'] if stats['mem_total'] is not None else 0
                         mem_text = f"{mem_used}/{mem_total}MiB" if mem_total > 0 else "N/A"
                         # name = stats.get('name', f'GPU {gpu_index}'); max_name_len = 18 # Keep name short
                         # display_name = name if len(name) <= max_name_len else name[:max_name_len-3] + "..." # Name removed for space

                         # Label: "GPU X:"
                         gpu_title_label = ctk.CTkLabel(self.system_status_frame, text=f"GPU {gpu_index}:", anchor="w", width=label_width, font=ctk.CTkFont(size=11))
                         gpu_title_label.grid(row=current_row, column=0, sticky="w", padx=(0, PAD_Y), pady=1)

                         # Progress Bar for Utilization
                         util_bar = ctk.CTkProgressBar(self.system_status_frame,
                                                       height=SYSTEM_BAR_HEIGHT,
                                                       border_width=SYSTEM_BAR_BORDER_WIDTH,
                                                       border_color=SYSTEM_BAR_GREEN,
                                                       fg_color=SYSTEM_BAR_BACKGROUND,
                                                       progress_color=SYSTEM_BAR_GREEN)
                         util_bar.set(util_val / 100.0)
                         util_bar.grid(row=current_row, column=1, sticky="ew", padx=(0, PAD_Y), pady=1)

                         # Label: "Util% | Temp | Mem" (Reordered for priority)
                         stats_text = f"{util_pct_text} | {temp} | {mem_text}"
                         stats_label = ctk.CTkLabel(self.system_status_frame, text=stats_text, anchor="w", font=ctk.CTkFont(size=10))
                         stats_label.grid(row=current_row, column=2, sticky="w", padx=(0, 0), pady=1)

                         self.system_status_widgets[f'gpu_{gpu_index}'] = {'title': gpu_title_label, 'bar': util_bar, 'stats': stats_label}
                         current_row += 1

        except tk.TclError as e: logging.warning(f"TclError updating system display: {e}")
        except Exception as e: logging.exception("Error updating system display")


    # --- Window Closing (Handle both Train and Infer) ---
    def on_closing(self):
        logging.info("Window close requested.")
        active_processes = []
        if self.is_training: active_processes.append("Training")
        if self.is_inferring: active_processes.append("Inference")

        confirm_exit = True
        if active_processes:
            process_list = " and ".join(active_processes)
            confirm_exit = messagebox.askyesno("Exit Confirmation",
                                               f"{process_list} is in progress. Are you sure you want to exit?\n"
                                               "Running processes will be terminated.")

        if confirm_exit:
            logging.info("Proceeding with shutdown.")
            # 1. Stop monitor threads first
            self._stop_system_monitor()
            if self.queue_check_job: self.after_cancel(self.queue_check_job); self.queue_check_job = None
            if self.live_data_update_job: self.after_cancel(self.live_data_update_job); self.live_data_update_job = None # Stop training updates
            # if self.infer_output_update_job: self.after_cancel(self.infer_output_update_job); self.infer_output_update_job = None # Stop any inference updates

            # 2. Signal running processes to stop
            if self.is_training:
                logging.info("Signalling training process to stop...")
                self.stop_event.set()
                # Attempt termination directly here as well
                self._terminate_process_nicely(self.training_process, "Training")
            if self.is_inferring:
                logging.info("Signalling inference process to stop...")
                self.infer_stop_event.set()
                self._terminate_process_nicely(self.inference_process, "Inference")

            # 3. Wait briefly for threads to potentially finish
            threads_to_join = [self.gpu_monitor_thread, self.training_thread, self.inference_thread]
            for thread in threads_to_join:
                if thread and thread.is_alive():
                    try:
                        logging.debug(f"Joining thread {thread.name}...")
                        thread.join(timeout=1.0) # Short timeout
                        if thread.is_alive():
                             logging.warning(f"Thread {thread.name} did not join cleanly.")
                    except Exception as e:
                        logging.error(f"Error joining thread {thread.name}: {e}")

            # 4. Destroy the window
            logging.info("Destroying main window.")
            self.destroy()

        else:
            logging.info("User cancelled exit.")
            # No need to restart monitor if it wasn't stopped

# --- Main Execution ---
if __name__ == "__main__":
    # Configure logging (consider adding FileHandler as well)
    log_format = '%(asctime)s - %(levelname)s [%(threadName)s] - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)
    # Example File Handler:
    # file_handler = logging.FileHandler("yolo_trainer_app.log")
    # file_handler.setFormatter(logging.Formatter(log_format))
    # logging.getLogger().addHandler(file_handler)

    logging.info("Starting YOLO Live Trainer & Infer Pro Application.")
    app = YoloTrainerApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing) # Register closing handler
    try:
        app.mainloop()
    except Exception as e:
        logging.exception("Unhandled exception in main loop:")
        try: messagebox.showerror("Fatal Error", f"An unexpected error occurred:\n{e}\n\nCheck logs for details.")
        except: pass # If tkinter itself is broken
    finally:
        logging.info("Application closing.")
