import os
import psutil
from tkinter import messagebox
import customtkinter as ctk
# from controller.training import TrainController
# from controller.inference import InferController
# from controller.exporter import ModelExporter
from utils import load_yaml_config, save_projects_in_json, set_status_in_json, add_time_stamp


# Load configuration from YAML + Set all Global Variables
yaml_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app.yaml')
config = load_yaml_config(yaml_file)
ICON_SIZE = config.get('icon_size', (20, 20))
SCREEN_WIDTH = config.get('screen_width', 1000)
SCREEN_HEIGHT = config.get('screen_height', 800)
BTN_COLOR = config.get('color_scheme', {}).get('sky-blue', "#2196F3")
TEXT_COLOR = config.get('color_scheme', {}).get('white', "#FFFFFF")


class ProjectWindow(ctk.CTkToplevel):
    def __init__(self, master, name, project_path, status):
        super().__init__(master)
        self.title(f"{name}")
        self.geometry(f"{SCREEN_WIDTH}x{SCREEN_HEIGHT}")
        
        self.project_name = name
        self.project_path = project_path
        self.project_status = status
        # self.trainer = TrainController
        # self.infer = InferController
        # self.exporter = ModelExporter
        
        if self.project_status == 'closed':
            self.create_project_menu_widgets()
            self.project_status = 'open'
        else:
            messagebox.showerror("Error","Project Already Opened")

    
    def create_project_menu_widgets(self):
        
        # ------- ADD: UI LAYOUT ---------
        
        # App Frame Configuration
        self.grid_rowconfigure(0, weight=0)     # Project Window - Header and Buttons Frame
        self.grid_rowconfigure(2, weight=1)     # Project Window - Tabs Frame with Content
        self.grid_columnconfigure(0, weight=1)  # All Frames in a Single Column
        
        # Project Menu Heading
        self.header = ctk.CTkLabel(self, text=f"Project Menu -- {self.project_name}", font=ctk.CTkFont(size=20, weight="bold"))
        self.header.pack(padx=5, pady=5)
        
        # 'Back to Main Menu' button in the top right corner
        btns_frame = ctk.CTkFrame(self)
        btns_frame.pack(side="top", fill="x")
        back_btn = ctk.CTkButton(btns_frame, text="Back to Main Menu", command=self.confirm_exit, fg_color = BTN_COLOR, text_color = TEXT_COLOR)
        back_btn.pack(side="right", padx=5, pady=5)
        
        project_frame = ctk.CTkFrame(self)
        # label = ctk.CTkLabel(self, text="Welcome to the Project Menu")
        # label.pack(pady=20)
        
        
        
        # Main container frame
        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.pack(pady=30, padx=40, fill="both", expand=True)
        
        tabview = ctk.CTkTabview(
            frame,
            width=720,
            height=520,
            corner_radius=12,
            fg_color="#181818",                        # very dark main background
            border_width=0,                            # or 1 if you want outer border
            border_color="#404040",
            # ── Segmented button styling (tabs header) ──
            segmented_button_fg_color="transparent",   # unselected tabs transparent → clean look
            segmented_button_selected_color=BTN_COLOR, # your blue for active tab
            segmented_button_selected_hover_color="#0D47A1",
            segmented_button_unselected_color="transparent",
            segmented_button_unselected_hover_color="#2A2A2A",
            text_color=("gray92", "gray92"),           # bright text for visibility
            text_color_disabled="gray60"
            # IMPORTANT: removed segmented_button_border_width & segmented_button_border_color
        )
        tabview.pack(fill="both", expand=True, padx=20, pady=10)

        # Add your tabs (same as before)
        tabview.add("Datasets")
        tabview.add("Train")
        tabview.add("Inference")
        tabview.add("Results")

        tabview.set("Datasets")  # start on Datasets
        
        # ────────────────────────────────────────────────
        # Placeholder content for each tab (replace later)
        # ────────────────────────────────────────────────
        
        # Datasets tab
        datasets_frame = tabview.tab("Datasets")
        ctk.CTkLabel(
            datasets_frame,
            text="Dataset Management\n(Images • Labels • Classes • Augmentation)",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(40, 20))
        
        ctk.CTkButton(
            datasets_frame,
            text="Open Dataset Folder",
            width=220, height=50, corner_radius=10,
            font=ctk.CTkFont(size=15, weight="medium")
        ).pack(pady=12)
        
        # Train tab
        train_frame = tabview.tab("Train")
        ctk.CTkLabel(
            train_frame,
            text="Training Controls\n(Start • Monitor • Hyperparameters)",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(40, 20))
        
        ctk.CTkButton(
            train_frame,
            text="Start Training",
            width=220, height=50, corner_radius=10,
            font=ctk.CTkFont(size=15, weight="medium")
        ).pack(pady=12)
        
        # Inference tab
        inference_frame = tabview.tab("Inference")
        ctk.CTkLabel(
            inference_frame,
            text="Run Predictions\n(Single Image • Batch • Webcam)",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(40, 20))
        
        ctk.CTkButton(
            inference_frame,
            text="Run Inference",
            width=220, height=50, corner_radius=10,
            font=ctk.CTkFont(size=15, weight="medium")
        ).pack(pady=12)
        
        # Results tab
        results_frame = tabview.tab("Results")
        ctk.CTkLabel(
            results_frame,
            text="Training & Inference Results\n(Metrics • Visualizations • Exports)",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(40, 20))
        
        ctk.CTkButton(
            results_frame,
            text="Open Project Folder",
            width=220, height=50, corner_radius=10,
            font=ctk.CTkFont(size=15, weight="medium")
        ).pack(pady=12)
    
      
    def confirm_exit(self):
        # Dialog box to confirm before exiting
        dialog_msg = "Are you sure you want to go back to the Main Menu? Any unsaved changes will be lost."
        if messagebox.askyesno("Confirm Exit", dialog_msg):
            self.project_status = 'closed'
            # Save the status in the Projects JSON File
            set_status_config(config, self.project_status)
            self.destroy()  
    
    
    """
    ALL FUNCTIONS WITH WINDOW BOXES
    """


    def show_system_info(self):
        self.info_box.delete("0.0", "end")
        cpu = psutil.cpu_percent(interval=1)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        info = f"CPU Usage: {cpu:.1f}%\n"
        info += f"Memory: {mem.percent}% used ({mem.used/(1024**3):.1f} GB / {mem.total/(1024**3):.1f} GB)\n"
        info += f"Disk: {disk.percent}% used"
        
        self.info_box.insert("0.0", info)
        self.status.configure(text="System info updated")