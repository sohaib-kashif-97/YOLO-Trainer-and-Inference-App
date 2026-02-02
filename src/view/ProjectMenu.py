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