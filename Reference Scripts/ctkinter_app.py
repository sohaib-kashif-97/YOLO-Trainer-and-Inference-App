import customtkinter as ctk

# Set appearance mode and color theme
ctk.set_appearance_mode("dark")  # "dark", "light", or "system"
ctk.set_default_color_theme("blue")  # "blue", "green", or "dark-blue"

class SimpleApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Configure window
        self.title("Simple CTkinter App")
        self.geometry("500x400")
        self.resizable(False, False)
        
        # Create main frame
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(padx=20, pady=20, fill="both", expand=True)
        
        # Title label
        title_label = ctk.CTkLabel(
            main_frame,
            text="Welcome to CTkinter",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(pady=20)
        
        # Description label
        description_label = ctk.CTkLabel(
            main_frame,
            text="This is a simple CTkinter application",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        description_label.pack(pady=10)
        
        # Entry widget
        self.entry = ctk.CTkEntry(
            main_frame,
            placeholder_text="Enter your name...",
            width=300,
            height=40
        )
        self.entry.pack(pady=15)
        
        # Button frame
        button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        button_frame.pack(pady=20, fill="x")
        
        # Submit button
        submit_button = ctk.CTkButton(
            button_frame,
            text="Submit",
            command=self.on_submit,
            height=40,
            width=150
        )
        submit_button.pack(side="left", padx=10)
        
        # Clear button
        clear_button = ctk.CTkButton(
            button_frame,
            text="Clear",
            command=self.on_clear,
            height=40,
            width=150,
            fg_color="gray40"
        )
        clear_button.pack(side="left", padx=10)
        
        # Output label
        self.output_label = ctk.CTkLabel(
            main_frame,
            text="",
            font=ctk.CTkFont(size=12),
            text_color="lightgreen"
        )
        self.output_label.pack(pady=20)
    
    def on_submit(self):
        name = self.entry.get()
        if name:
            self.output_label.configure(text=f"Hello, {name}! 👋")
        else:
            self.output_label.configure(text="Please enter a name!", text_color="red")
    
    def on_clear(self):
        self.entry.delete(0, "end")
        self.output_label.configure(text="", text_color="lightgreen")

if __name__ == "__main__":
    app = SimpleApp()
    app.mainloop()
