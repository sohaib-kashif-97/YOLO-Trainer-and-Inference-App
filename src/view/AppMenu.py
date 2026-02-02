import logging
import os
import psutil
import customtkinter as ctk
from tkinter import filedialog, messagebox  # for future image loading extensions
# from src.view.ProjectMenu import ProjectMenu  # Assuming this exists for project handling


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("CV Trainer App for YOLO Models")
        self.geometry("1000x800")
        self.minsize(500, 400)
        
        # Enlisting all CV Projects
        self.projects_dir = os.path.join(os.getcwd(), "projects")
        self.projects = []                      # List of project paths
        if os.path.exists(self.projects_dir):
            self.projects = [os.path.join(self.projects_dir, f) for f in os.listdir(self.projects_dir) if os.path.isdir(os.path.join(self.projects_dir, f))]
            
        #Create all Widgets for the Main Menu
        self.create_main_menu_widgets()

    def create_main_menu_widgets(self):
        
        # App Frame Configuration
        self.grid_rowconfigure(0, weight=0)     # Header  
        self.grid_rowconfigure(1, weight=1)     # Main Menu
        self.grid_columnconfigure(0, weight=1)  # Single Column         
        
        # App Header Frame
        self.header_frame = ctk.CTkFrame(self)
        self.header_frame.grid(row=0, column=0, padx=0, pady=10, sticky="nsew")    
        self.header = ctk.CTkLabel(self.header_frame, text="CV Trainer App", padx=400, pady=10,
                                  font=ctk.CTkFont(size=20, weight="bold"))
        self.header.pack(padx=50, pady=10) 
        
        # Main Menu Heading (With 'Edit Project' and 'Add Project' Buttons)
        self.main_menu_frame = ctk.CTkFrame(self)
        self.main_menu_frame.grid(row=1, column=4, padx=10, pady=10, sticky="nsew")
        self.header = ctk.CTkLabel(self.main_menu_frame, text="Main Menu", 
                                  font=ctk.CTkFont(size=20, weight="bold"))
        self.header.grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.edit_btn = ctk.CTkButton(self.main_menu_frame, text="Edit Project", width=100, 
                                     command=self.edit_project)  # Placeholder command
        self.edit_btn.grid(row=1, column=2, padx=10, pady=10, sticky="e")
        self.add_btn = ctk.CTkButton(self.main_menu_frame, text="Add Project", width=100, 
                                    command=self.add_project)  # Placeholder command
        self.add_btn.grid(row=1, column=3, padx=10, pady=10, sticky="e")

    def edit_project(self):
        pass

    def add_project(self):
        pass

    def on_closing(self):
        """Called when user clicks × / Alt+F4"""
        if messagebox.askokcancel("Quit", "Do you want to quit the trainer?"):
            logging.info("User confirmed exit")
            self.destroy()
        else:
            # Do nothing → window stays open
            return
    
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