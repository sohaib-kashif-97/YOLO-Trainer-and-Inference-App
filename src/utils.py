import os
import re
import pytz
import yaml
import json
import logging
import datetime
from pathlib import Path
from datetime import date, datetime
from PIL import Image
import customtkinter as ctk


# Defining all Resources and Constants
ICON_SIZE = (20, 20)
YOLO_MODELS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff']

# All helping Functions Enlisted
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

def clean_ansi_codes(text):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def format_time(seconds):
    if seconds is None or seconds < 0: return "--:--:--"
    return str(datetime.timedelta(seconds=int(seconds)))

def add_time_stamp():
    # Define the date
    date_today = date.today()
    date_str = date_today.isoformat()

    # Get the current time in the specified timezone
    timezone = pytz.timezone('Asia/Karachi')
    current_time_str = datetime.now(timezone).strftime("%H:%M:%S")

    # Concatenating Date and Time onto TimeStamp
    dt_stamp_msg = f" [{date_str} | {current_time_str}] "

    return dt_stamp_msg

def load_icon(filename, size=ICON_SIZE):
    """ Allow searching in a potential 'icons' subdirectory """
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

def load_yaml_config(config_path):
    """Loads a YAML configuration file and returns its contents as a dictionary."""
    if not os.path.exists(config_path):
        logging.warning(f"Config file not found: {config_path}")
        return {}
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            return config if config else {}
    except Exception as e:
        logging.error(f"Error loading config file {config_path}: {e}")
        return {}

def save_projects_in_json(projects, json_file):
    try:
        # Sort by idx before saving (optional)
        sorted_projects = sorted(projects, key=lambda x: x.get('idx', 0))
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(sorted_projects, f, indent=4)
    except Exception as e:
        print(f"Error saving projects.json: {e}")

def load_projects_in_json(json_file, projects):
        if os.path.exists(json_file):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Filter only projects whose folder still exists
                valid_projects = []
                for entry in data:
                    if isinstance(entry, dict) and 'path' in entry:
                        if os.path.exists(entry['path']):
                            valid_projects.append(entry)
                
                projects = valid_projects
                
                # Re-number indices if you want (optional)
                for i, proj in enumerate(projects, 1):
                    proj['idx'] = i
                
                if len(valid_projects) != len(data):
                    save_projects_in_json()   # CHECK: clean up removed projects
                
            except Exception as e:
                print(f"Error loading projects.json: {e}")
                projects = []
        else:
            projects = []

def set_status_in_json(json_file, idx, name, status):
    """Loads a JSON configuration file and updates its contents."""
    
    json_path = Path(json_file)
    
    try:
        with open(json_path, 'r') as file:
            projects = json.safe_load(file)
            
        for project in projects:
            if project['name'] == name and project['idx'] == idx:
                project['status'] = status
        
        
        return save_projects_in_json(projects, json_path)
            
    except Exception as e:
        logging.error(f"Error loading config file {json_file}: {e}")
        return

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