import os
import yaml
import json
import psutil
import logging
import customtkinter as ctk
from tkinter import filedialog, messagebox  # for future image loading extensions
# from src.view.ProjectMenu import ProjectMenu  # Assuming this exists for project handling
from utils import load_yaml_config, add_time_stamp

# Load configuration from YAML + Set all Global Variables
yaml_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app.yaml')
config = load_yaml_config(yaml_file)
ICON_SIZE = config.get('icon_size', (20, 20))
SCREEN_WIDTH = config.get('screen_width', 1000)
SCREEN_HEIGHT = config.get('screen_height', 800)
PX, PY = config.get('padding', {}).get('frame', (10, 10))
SX, SY = config.get('padding', {}).get('section', (10, 5))
BX, BY = config.get('padding', {}).get('button', (10, 10))
TABLE_FG_COLOR = config.get('color_scheme', {}).get('transparent', '#808080')
TABLE_HOVER_COLOR = config.get('color_scheme', {}).get('light-gray', '#B1ABAB')
TABLE_BORDER_COLOR = config.get('color_scheme', {}).get('black', '#000000')
BTN_COLOR = config.get('color_scheme', {}).get('sky-blue', "#2196F3")
SCREEN_PADDING = config.get('padding', {}).get('screen', (20,20))
SCREEN_STRETCHED = SCREEN_WIDTH - SCREEN_PADDING[0]


"""APP MAIN CLASS"""
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("CV Trainer App for YOLO Models")
        self.geometry(f"{SCREEN_WIDTH}x{SCREEN_HEIGHT}")
        self.maxsize(SCREEN_WIDTH, SCREEN_HEIGHT)
        self.minsize(int(SCREEN_WIDTH / 2), int(SCREEN_HEIGHT / 2))
        
        # Creating and Ensuring folders exist w.r.t. Base Directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # go up from src/
        self.resources_dir = os.path.join(base_dir, "resources")
        self.projects_dir  = os.path.join(base_dir, "projects")
        os.makedirs(self.resources_dir, exist_ok=True)
        os.makedirs(self.projects_dir,  exist_ok=True)

        # Creating a JSON File (if not already existing)
        self.projects_json = os.path.join(self.resources_dir, "projects.json")

        # Creating variables for Projects (with entry template)
        self.proj_entry_template = {
            'idx': 0,
            'name': '',
            'path': ''
        }
        self.projects = []           # now list of dicts
        self.load_projects()         # will load list of dicts

        # Create all Widgets for the Main Menu
        self.create_main_menu_widgets()

    """
    WIDGET LAYOUT MENTIONED IN THE MAIN MENU
    """
    def create_main_menu_widgets(self):
        
        # App Frame Configuration
        self.grid_rowconfigure(0, weight=0)     # Header Frame
        self.grid_rowconfigure(1, weight=0)     # Main Menu Frame (Heading and Buttons)
        self.grid_rowconfigure(2, weight=1)     # Table Frame (With Scrollbar)
        self.grid_columnconfigure(0, weight=1)  # All Frames in a Single Column 

        # App Header Frame
        self.header_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.header_frame.grid(row=0, column=0, padx=PX, pady=(8, 4), sticky="ew")
        self.header = ctk.CTkLabel(self.header_frame, text="CV Trainer App", font=ctk.CTkFont(size=26, weight="bold"))
        self.header.pack(pady=(10, 6))          

        # Main Menu Frame (With Project Management Buttons)
        self.main_menu_frame = ctk.CTkFrame(self, fg_color="transparent", width=SCREEN_STRETCHED)
        self.main_menu_frame.grid(row=1, column=0, padx=PX, pady=(2, 6), sticky="ew")
        self.main_menu_frame.grid_columnconfigure(0, weight=1)
        self.main_menu_frame.grid_columnconfigure(1, weight=0)

        # Left Container: Heading
        self.main_menu_label = ctk.CTkLabel( self.main_menu_frame, text="Main Menu", font=ctk.CTkFont(size=18, weight="bold"))
        self.main_menu_label.grid(row=0, column=0, padx=10, pady=6, sticky="w")

        # Left Container: Heading
        self.btns_frame = ctk.CTkFrame(self.main_menu_frame, fg_color="transparent")
        self.btns_frame.grid(row=0, column=1, sticky="e")

        btn_kwargs = {
            "fg_color": BTN_COLOR,
            "text_color": "#ffffff",
            "hover_color": "#1976D2",
            "width": 105,
            "height": 32,
            "corner_radius": 8
        }

        # 'Edit Project' and 'Add Project' Buttons
        self.refresh_btn = ctk.CTkButton(self.btns_frame, text="Refresh all Projects", command=self.refresh_projects, **btn_kwargs)
        self.refresh_btn.pack(side="right", padx=(6, 0))
        self.add_btn = ctk.CTkButton(self.btns_frame, text="Add Project", command=self.add_project, **btn_kwargs)
        self.add_btn.pack(side="right", padx=6)

        # Table Frame for enlisting all the CV Projects
        self.table_frame = ctk.CTkScrollableFrame(self, fg_color=TABLE_FG_COLOR, scrollbar_button_hover_color=TABLE_HOVER_COLOR,border_color=TABLE_BORDER_COLOR, border_width=1, corner_radius=10)
        self.table_frame.grid(row=2, column=0, padx=PX, pady=(4, PY), sticky="nsew")  
        
        # If projects already exist, then display the projects as cards
        self.display_projects()
        
    
    """
    ALL BUTTON COMMAND FUNCTIONS BELOW
    """
    def load_projects(self):
        if os.path.exists(self.projects_json):
            try:
                with open(self.projects_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Filter only projects whose folder still exists
                valid_projects = []
                for entry in data:
                    if isinstance(entry, dict) and 'path' in entry:
                        if os.path.exists(entry['path']):
                            valid_projects.append(entry)
                
                self.projects = valid_projects
                
                # Re-number indices if you want (optional)
                for i, proj in enumerate(self.projects, 1):
                    proj['idx'] = i
                
                if len(valid_projects) != len(data):
                    self.save_projects_in_json()   # clean up removed projects
                
            except Exception as e:
                print(f"Error loading projects.json: {e}")
                self.projects = []
        else:
            self.projects = []
            
        # DEBUG: Do projects exist yet?    
        print(self.projects)


    def save_projects_in_json(self):
        try:
            # Sort by idx before saving (optional)
            sorted_projects = sorted(self.projects, key=lambda x: x.get('idx', 0))
            with open(self.projects_json, 'w', encoding='utf-8') as f:
                json.dump(sorted_projects, f, indent=4)
        except Exception as e:
            print(f"Error saving projects.json: {e}")


    def add_project(self):
        stamp = add_time_stamp()
        print(f"Add Project button clicked --- {stamp}")
        
        project_name = ctk.CTkInputDialog(
            title="New Project",
            text="Enter project name:"
        ).get_input()
        
        # Validate Input for Project Name
        if not project_name:
            self.show_error("Error", "Project name cannot be empty!")
            return
        else:
            project_name = project_name.strip()
            project_path = os.path.join(self.projects_dir, project_name)
        
        # Validate if the Project by name already exists
        if os.path.exists(project_path):
            self.show_error("Error", f"Project '{project_name}' already exists!")
            return
        
        # Create new Project
        try:
            print("try")
            # Create new project directory with sub-folders
            os.makedirs(project_path, exist_ok=False)
            os.makedirs(os.path.join(project_path, "images"), exist_ok=True)
            os.makedirs(os.path.join(project_path, "labels"), exist_ok=True)
            os.makedirs(os.path.join(project_path, "runs"), exist_ok=True)

            # Optional: create config.yaml stub
            config_path = os.path.join(project_path, "project.yaml")
            with open(config_path, "w", encoding="utf-8") as f:
                f.write("# Basic project config\n")
                f.write(f"name: {project_name}\n")
                f.write("created: auto\n")

            # Success Message for Project Creation
            self.show_info_msg_box("Success", f"Project '{project_name}' created successfully!")

            # Create new entry using template in the 'projects.json' file
            new_entry = self.proj_entry_template.copy()
            new_entry['idx']  = len(self.projects) + 1
            new_entry['name'] = project_name
            new_entry['path'] = project_path
            self.projects.append(new_entry)
            self.save_projects_in_json()

            # Display Refreshed Project List
            self.refresh_projects()
            self.display_projects()   
           
        except Exception as e:
            print(e)
            # logging.error(f"Error creating project '{project_name}': {e}")
            self.show_error("Error", f"Failed to create project '{project_name}'.\n{e}")

        
    def display_projects(self):
        
        # Clearing all old widgets
        for widget in self.table_frame.winfo_children():
            widget.destroy()
            
        if not self.projects:
             # Create a Placeholder when empty
            self.empty_label = ctk.CTkLabel(
                self.table_frame,
                text="No projects yet.\nClick «Add Project» to begin.",
                font=ctk.CTkFont(size=14),
                text_color="gray70"
            )
            self.empty_label.pack(expand=True, pady=80)
        else:
            for project_path in self.projects:
                self.create_project_card(project_path)

            
    def create_project_card(self, project_dict):
        
        # Process Values from Dictionary Entry
        name = project_dict.get('name', 'Unnamed')
        path = project_dict.get('path', '')
        idx  = project_dict.get('idx', '?')
        
        # Create card with three columns: Sr. Number, Project Folder Name, Project Path, and Button to open the Project.
        card_frame = ctk.CTkFrame(self.table_frame, corner_radius=10, fg_color="#2B2B2B", border_width=1, border_color="#404040")
        card_frame.pack(pady=6, padx=12, fill="x")
        ctk.CTkLabel(card_frame, text=f"#{idx}  {name}", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w")
        ctk.CTkLabel(card_frame, text=path,text_color="gray60",font=ctk.CTkFont(size=12)).pack(anchor="w", pady=(2, 0))
        
        # Action Buttons for every Project
        btn_frame = ctk.CTkFrame(card_frame, fg_color="transparent")
        btn_frame.place(relx=1.0, rely=0.5, anchor="e", x=-15)
        action_btn_kwargs = {
            "fg_color": BTN_COLOR,
            "text_color": "#ffffff",
            "hover_color": "#1976D2",
            "width": 90,
            "height": 32,
            "corner_radius": 8
        }
        ctk.CTkButton(btn_frame, text="Open", command=lambda p=path: self.open_project(p), **action_btn_kwargs).pack(side="right", padx=(6, 0))
        ctk.CTkButton(btn_frame, text="Edit", command=lambda p=path: self.edit_project(p), **action_btn_kwargs).pack(side="right", padx=6)


    def refresh_projects(self):
        self.load_projects()
        print(f"Projects refreshed — found {len(self.projects)} projects")

 
    def open_project(self, project_path):
        name = self.get_project_name_from_path(project_path)
        print(f"Opening project: {name} → {project_path}")
        messagebox.showinfo("Open Project", f"Opening project:\n{name}")


    def edit_project(self, project_path):
        name = self.get_project_name_from_path(project_path)
        print(f"Editing project: {name} → {project_path}")
        messagebox.showinfo("Edit Project", f"Editing project:\n{name}")

            
    def get_project_name_from_path(self, path):
        return os.path.basename(os.path.normpath(path))


    """
    ALL FUNCTIONS WITH WINDOW BOXES
    """
    def on_closing(self):
        """Called when user clicks × / Alt+F4"""
        if messagebox.askokcancel("Quit", "Do you want to quit the trainer?"):
            logging.info("User confirmed exit")
            self.destroy()
        else:
            # Do nothing → window stays open
            return
 
    
    def show_error(self, title, message):
        messagebox.showerror(title, message)


    def show_info_msg_box(self, title, message):
        messagebox.showinfo(title, message)
    
