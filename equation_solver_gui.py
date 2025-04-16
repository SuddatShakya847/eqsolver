import os
import base64
import json
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import io
import sys
import shutil
from main import main
from calculator import calculate

class EquationSolverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Equation Solver")
        self.root.geometry("900x600")
        self.root.resizable(True, True)
        
        # Set app style
        self.style = ttk.Style()
        self.style.configure("TButton", font=("Arial", 12), padding=10)
        self.style.configure("TLabel", font=("Arial", 12))
        
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Equation Solver", font=("Arial", 24, "bold"))
        title_label.pack(pady=10)
        
        # Info label
        info_label = ttk.Label(main_frame, text="Upload an image containing a handwritten equation to solve")
        info_label.pack(pady=5)
        
        # Image frame
        self.image_frame = ttk.LabelFrame(main_frame, text="Equation Image")
        self.image_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Image placeholder
        self.image_label = ttk.Label(self.image_frame, text="No image selected")
        self.image_label.pack(pady=50, expand=True)
        
        # Buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(pady=10, fill=tk.X)
        
        # Upload button
        upload_button = ttk.Button(buttons_frame, text="Upload Image", command=self.upload_image)
        upload_button.pack(side=tk.LEFT, padx=5)
        
        # Solve button
        self.solve_button = ttk.Button(buttons_frame, text="Solve Equation", command=self.solve_equation, state=tk.DISABLED)
        self.solve_button.pack(side=tk.LEFT, padx=5)
        
        # Clear button
        clear_button = ttk.Button(buttons_frame, text="Clear", command=self.clear)
        clear_button.pack(side=tk.LEFT, padx=5)
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Results")
        results_frame.pack(pady=10, fill=tk.BOTH)
        
        # Results area
        results_area = ttk.Frame(results_frame, padding=10)
        results_area.pack(fill=tk.BOTH)
        
        # Detected equation
        ttk.Label(results_area, text="Detected Equation:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.detected_var = tk.StringVar()
        detected_entry = ttk.Entry(results_area, textvariable=self.detected_var, width=40)
        detected_entry.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # Formatted equation
        ttk.Label(results_area, text="Formatted Equation:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.formatted_var = tk.StringVar()
        formatted_entry = ttk.Entry(results_area, textvariable=self.formatted_var, width=40)
        formatted_entry.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # Solution
        ttk.Label(results_area, text="Solution:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.solution_var = tk.StringVar()
        solution_entry = ttk.Entry(results_area, textvariable=self.solution_var, width=40)
        solution_entry.grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Initialize application state
        self.image_path = None
        self.image_data = None
        
    def upload_image(self):
        """Open file dialog to select an image"""
        file_types = [("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        file_path = filedialog.askopenfilename(title="Select Equation Image", filetypes=file_types)
        
        if file_path:
            self.image_path = file_path
            self.status_var.set(f"Image loaded: {os.path.basename(file_path)}")
            
            # Display the image
            self.display_image(file_path)
            
            # Enable solve button
            self.solve_button.config(state=tk.NORMAL)
            
            # Clear previous results
            self.clear_results()
    
    def display_image(self, file_path):
        """Display the selected image"""
        try:
            image = Image.open(file_path)
            
            # Calculate new dimensions while maintaining aspect ratio
            max_width = 400
            max_height = 300
            width, height = image.size
            
            if width > max_width or height > max_height:
                ratio = min(max_width / width, max_height / height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                image = image.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Update image label
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Keep a reference
            
            # Store image data for processing
            with open(file_path, "rb") as img_file:
                self.image_data = img_file.read()
        
        except Exception as e:
            self.status_var.set(f"Error loading image: {str(e)}")
            self.image_label.config(text="Error loading image", image="")
    
    def solve_equation(self):
        """Process the image and solve the equation"""
        if not self.image_path:
            return
        
        self.status_var.set("Processing image and solving equation...")
        self.root.update()  # Update the UI
        
        try:
            # Clean up previous runs
            for directory in ["internals", "segmented"]:
                if directory in os.listdir():
                    shutil.rmtree(directory)
            
            # Create necessary directory
            if "segmented" not in os.listdir():
                os.mkdir("segmented")
            
            # Create a BytesIO object from the image data
            image_bytes = io.BytesIO(self.image_data)
            
            # Process the image to get the equation
            operation = main(image_bytes)
            
            # Calculate the solution
            formatted_equation, solution = calculate(operation)
            
            # Display results
            self.detected_var.set(operation)
            self.formatted_var.set(formatted_equation)
            self.solution_var.set(str(solution))
            
            self.status_var.set("Equation solved successfully")
            
            # Clean up by moving files to internals directory
            os.mkdir("internals")
            shutil.move("segmented", "internals")
            shutil.move("input.png", "internals")
            if "segmented_characters.csv" in os.listdir():
                shutil.move("segmented_characters.csv", "internals")
        
        except Exception as e:
            self.status_var.set(f"Error solving equation: {str(e)}")
            self.clear_results()
    
    def clear_results(self):
        """Clear the result fields"""
        self.detected_var.set("")
        self.formatted_var.set("")
        self.solution_var.set("")
    
    def clear(self):
        """Reset the application state"""
        self.image_path = None
        self.image_data = None
        self.image_label.config(text="No image selected", image="")
        self.solve_button.config(state=tk.DISABLED)
        self.clear_results()
        self.status_var.set("Ready")

def main():
    root = tk.Tk()
    app = EquationSolverApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()