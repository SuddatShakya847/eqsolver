import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk, ImageOps, ImageFilter
import numpy as np
import cv2
import io
import re
import matplotlib.pyplot as plt
from solver import symbols, solve, sympify, Eq
from sympy.parsing.sympy_parser import parse_expr
import tempfile

class EquationSolverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Equation Solver")
        self.root.geometry("1000x700")
        self.root.resizable(True, True)
        
        # Set app style
        self.style = ttk.Style()
        self.style.configure("TButton", font=("Arial", 12), padding=10)
        self.style.configure("TLabel", font=("Arial", 12))
        
        # Initialize variables
        self.image_path = None
        self.processed_image = None
        self.segmented_chars = []
        self.debug_mode = tk.BooleanVar(value=False)
        self.temp_dir = tempfile.mkdtemp()
        
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Equation Solver", font=("Arial", 24, "bold"))
        title_label.pack(pady=10)
        
        # Top panel (images and visualization)
        top_panel = ttk.Frame(main_frame)
        top_panel.pack(fill=tk.BOTH, expand=True)
        
        # Original image frame
        self.original_frame = ttk.LabelFrame(top_panel, text="Original Image")
        self.original_frame.pack(side=tk.LEFT, padx=5, fill=tk.BOTH, expand=True)
        
        self.original_label = ttk.Label(self.original_frame)
        self.original_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Processed image frame
        self.processed_frame = ttk.LabelFrame(top_panel, text="Processed Image")
        self.processed_frame.pack(side=tk.LEFT, padx=5, fill=tk.BOTH, expand=True)
        
        self.processed_label = ttk.Label(self.processed_frame)
        self.processed_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Control panel
        control_panel = ttk.Frame(main_frame)
        control_panel.pack(fill=tk.X, pady=10)
        
        # Buttons
        upload_button = ttk.Button(control_panel, text="Upload Image", command=self.upload_image)
        upload_button.pack(side=tk.LEFT, padx=5)
        
        self.process_button = ttk.Button(control_panel, text="Process Image", 
                                         command=self.process_image, state=tk.DISABLED)
        self.process_button.pack(side=tk.LEFT, padx=5)
        
        self.solve_button = ttk.Button(control_panel, text="Solve Equation", 
                                       command=self.solve_equation, state=tk.DISABLED)
        self.solve_button.pack(side=tk.LEFT, padx=5)
        
        clear_button = ttk.Button(control_panel, text="Clear", command=self.clear)
        clear_button.pack(side=tk.LEFT, padx=5)
        
        # Debug checkbox
        debug_check = ttk.Checkbutton(control_panel, text="Debug Mode", 
                                      variable=self.debug_mode)
        debug_check.pack(side=tk.RIGHT, padx=5)
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Results")
        results_frame.pack(pady=10, fill=tk.BOTH)
        
        # Results display
        self.results_text = tk.Text(results_frame, height=10, wrap=tk.WORD, 
                                   font=("Courier", 12))
        self.results_text.pack(fill=tk.BOTH, padx=10, pady=10)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def upload_image(self):
        """Open file dialog to select an image"""
        file_types = [("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        file_path = filedialog.askopenfilename(title="Select Equation Image", 
                                               filetypes=file_types)
        
        if file_path:
            self.image_path = file_path
            self.status_var.set(f"Image loaded: {os.path.basename(file_path)}")
            
            # Display the original image
            self.display_original_image(file_path)
            
            # Enable process button
            self.process_button.config(state=tk.NORMAL)
            
            # Clear previous results
            self.results_text.delete(1.0, tk.END)
            self.processed_label.config(image="")
    
    def display_original_image(self, file_path):
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
            self.original_label.config(image=photo)
            self.original_label.image = photo  # Keep a reference
            
        except Exception as e:
            self.status_var.set(f"Error loading image: {str(e)}")
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def process_image(self):
        """Process the image to prepare for character segmentation"""
        if not self.image_path:
            return
        
        self.status_var.set("Processing image...")
        self.root.update()
        
        try:
            # Load image and convert to grayscale
            image = cv2.imread(self.image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding
            _, binary = cv2.threshold(gray, 0, 255, 
                                     cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Remove noise
            kernel = np.ones((2, 2), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Display the processed image
            self.processed_image = binary
            self.display_processed_image(binary)
            
            # Segment the characters
            self.segmented_chars = self.segment_characters(binary)
            
            # Enable solve button
            self.solve_button.config(state=tk.NORMAL)
            
            self.status_var.set("Image processed. Ready to solve.")
            
            # If debug mode is on, show segmented characters
            if self.debug_mode.get():
                self.show_segmented_characters()
                
        except Exception as e:
            self.status_var.set(f"Error processing image: {str(e)}")
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")
    
    def display_processed_image(self, binary_image):
        """Display the processed binary image"""
        # Convert OpenCV binary image to PIL format
        pil_img = Image.fromarray(binary_image)
        
        # Calculate new dimensions while maintaining aspect ratio
        max_width = 400
        max_height = 300
        width, height = pil_img.size
        
        if width > max_width or height > max_height:
            ratio = min(max_width / width, max_height / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(pil_img)
        
        # Update image label
        self.processed_label.config(image=photo)
        self.processed_label.image = photo  # Keep a reference
    
    def segment_characters(self, binary_image):
        """Segment the characters from the binary image"""
        # Find contours
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by x-coordinate (left to right)
        sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        
        # Filter out very small contours (noise)
        char_contours = [c for c in sorted_contours if cv2.contourArea(c) > 100]
        
        segmented_chars = []
        for i, contour in enumerate(char_contours):
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Extract character ROI with some padding
            padding = 5
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(binary_image.shape[1], x + w + padding)
            y2 = min(binary_image.shape[0], y + h + padding)
            
            # Extract character
            char_img = binary_image[y1:y2, x1:x2]
            
            # Save for later recognition
            segmented_chars.append({
                'image': char_img,
                'position': (x, y, w, h),
                'value': self.recognize_character(char_img)
            })
            
            # Save for debug
            if self.debug_mode.get():
                debug_path = os.path.join(self.temp_dir, f"char_{i}.png")
                cv2.imwrite(debug_path, char_img)
        
        return segmented_chars
    
    def recognize_character(self, char_image):
        """Simple character recognition based on template matching or other features"""
        # For simplicity, we'll use a very basic approach
        # In a real application, you'd use a trained model or OCR
        
        # Resize to standard size for recognition
        resized = cv2.resize(char_image, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Placeholder: Identify characters based on simple features
        # This is a very simplified placeholder - a real system would use ML
        h, w = resized.shape
        aspect_ratio = w / h
        pixel_density = np.sum(resized) / (h * w * 255)
        
        # Very basic character recognition based on features
        # This is oversimplified and would need to be replaced with a proper classifier
        if 0.4 < pixel_density < 0.6:
            if aspect_ratio < 0.8:
                return "1"
            else:
                return "+"
        elif pixel_density > 0.7:
            return "="
        elif 0.2 < pixel_density < 0.4:
            if aspect_ratio > 1.2:
                return "-"
            else:
                return "x"
        else:
            # Default digits based on pixel density
            densities = {
                (0.15, 0.25): "7",
                (0.25, 0.35): "4",
                (0.35, 0.45): "2",
                (0.45, 0.55): "3", 
                (0.55, 0.65): "5",
                (0.65, 0.75): "6",
                (0.75, 0.85): "8",
                (0.85, 0.95): "9",
                (0.05, 0.15): "0",
            }
            
            for (min_d, max_d), digit in densities.items():
                if min_d <= pixel_density <= max_d:
                    return digit
                    
        # If we can't determine, just use a placeholder
        return "?"
    
    def show_segmented_characters(self):
        """Show the segmented characters in a debug window"""
        if not self.segmented_chars:
            return
        
        num_chars = len(self.segmented_chars)
        rows = int(np.ceil(num_chars / 5))
        
        plt.figure(figsize=(10, 2 * rows))
        
        for i, char_data in enumerate(self.segmented_chars):
            plt.subplot(rows, 5, i + 1)
            plt.imshow(char_data['image'], cmap='gray')
            plt.title(f"Char: {char_data['value']}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def solve_equation(self):
        """Recognize the equation and solve it"""
        if not self.segmented_chars:
            self.status_var.set("Process the image first")
            return
        
        # Extract equation string from recognized characters
        equation_str = ''.join([char['value'] for char in self.segmented_chars])
        
        # Clean up the equation string
        equation_str = self.clean_equation_string(equation_str)
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Detected Equation: {equation_str}\n\n")
        
        try:
            # Check if it's a simple arithmetic expression or an equation
            if '=' in equation_str:
                # It's an equation, solve for the variable
                self.solve_algebraic_equation(equation_str)
            else:
                # It's an arithmetic expression, evaluate it
                self.evaluate_arithmetic(equation_str)
                
        except Exception as e:
            self.status_var.set(f"Error solving equation: {str(e)}")
            self.results_text.insert(tk.END, f"Error: {str(e)}")
    
    def clean_equation_string(self, equation_str):
        """Clean up the equation string to handle recognition errors"""
        # Replace common recognition errors
        equation_str = equation_str.replace('?', '7')  # Example replacement
        
        # Remove any characters that shouldn't be there
        equation_str = re.sub(r'[^0-9x+\-*/()=]', '', equation_str)
        
        # Handle 'x' as multiplication or variable based on context
        if 'x' in equation_str and '=' in equation_str:
            # Keep x as variable in equations
            pass
        else:
            # Replace x with * for multiplication in arithmetic expressions
            equation_str = equation_str.replace('x', '*')
            
        # Add missing 0 before decimal points
        equation_str = re.sub(r'(?<![0-9])\.', '0.', equation_str)
        
        return equation_str
    
    def solve_algebraic_equation(self, equation_str):
        """Solve an algebraic equation"""
        try:
            # Split the equation into left and right sides
            left_side, right_side = equation_str.split('=')
            
            # Parse with SymPy
            x = symbols('x')
            left_expr = parse_expr(left_side)
            right_expr = parse_expr(right_side)
            
            # Create the equation and solve
            equation = Eq(left_expr, right_expr)
            solution = solve(equation, x)
            
            # Format and display the result
            if solution:
                result_str = f"Solution: x = {solution[0]}"
                self.results_text.insert(tk.END, result_str)
                self.status_var.set("Equation solved successfully")
            else:
                self.results_text.insert(tk.END, "No solution found")
                self.status_var.set("No solution found")
                
        except Exception as e:
            self.results_text.insert(tk.END, f"Error solving equation: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
    
    def evaluate_arithmetic(self, expression_str):
        """Evaluate an arithmetic expression"""
        try:
            # Use eval() to calculate the result (safe for simple arithmetic)
            # In a production environment, you'd want to use a safer method
            result = eval(expression_str)
            
            # Format and display the result
            result_str = f"Result: {result}"
            self.results_text.insert(tk.END, result_str)
            self.status_var.set("Expression evaluated successfully")
            
        except Exception as e:
            self.results_text.insert(tk.END, f"Error evaluating expression: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
    
    def clear(self):
        """Reset the application state"""
        self.image_path = None
        self.processed_image = None
        self.segmented_chars = []
        
        # Clear UI elements
        self.original_label.config(image="")
        self.processed_label.config(image="")
        self.results_text.delete(1.0, tk.END)
        
        # Reset buttons
        self.process_button.config(state=tk.DISABLED)
        self.solve_button.config(state=tk.DISABLED)
        
        # Reset status
        self.status_var.set("Ready")
        
        # Clean temp directory
        self._clean_temp_dir()
    
    def _clean_temp_dir(self):
        """Clean temporary directory"""
        for file in os.listdir(self.temp_dir):
            file_path = os.path.join(self.temp_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

def main():
    root = tk.Tk()
    app = EquationSolverApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()