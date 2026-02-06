import psutil
import customtkinter as ctk
import os

class ProjectMenu(ctk.CTkFrame):
    
    def __init__(self, parent, controller, project_path):
        super().__init__(parent)
        self.controller = controller
        self.project_path = project_path
        self.project_name = os.path.basename(project_path)
        
        # Project Menu Heading
        self.header = ctk.CTkLabel(self, text=f"Project: {self.project_name}", 
                                  font=ctk.CTkFont(size=20, weight="bold"))
        self.header.pack(padx=10, pady=5)
        
        # Create all Widgets for the Project Menu
        self.create_project_menu_widgets()
        
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