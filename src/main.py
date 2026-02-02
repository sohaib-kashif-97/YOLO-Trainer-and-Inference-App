# Importing necessary libraries
import os
import sys
import logging
import customtkinter as ctk
from tkinter import messagebox
from view.AppMenu import App


if __name__ == "__main__":
    
    # Configure logging (consider adding FileHandler as well)
    log_format = '%(asctime)s - %(levelname)s [%(threadName)s] - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)
    
    # Follows OS dark/light mode
    ctk.set_appearance_mode("System")  
    ctk.set_default_color_theme("blue")
    
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_closing) # Register closing handler
    try:
        app.mainloop()
    except Exception as e:
        logging.exception("Unhandled exception in main loop:")
        try: messagebox.showerror("Fatal Error", f"An unexpected error occurred:\n{e}\n\nCheck logs for details.")
        except: pass # If tkinter itself is broken
    finally:
        logging.info("Application closing.")