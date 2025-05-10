import os
import cv2 as cv
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk

from solver import solve_equation, process_equation
from processing import process_image, show_processed_images
from model import Model


DISPLAY_IMG_MAX_WIDTH = 800
DISPLAY_IMG_MAX_HEIGHT = 600


class EquationSolverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MathVision")
        self.root.configure(background="white")
        self.root.configure(background="white")
        self.root.geometry("1000x700")
        self.root.state("zoomed")
        self.root.resizable(True, True)

        # Set app style
        self.style = ttk.Style()
        self.style.configure(".", background="white")
        self.style.configure(
            "Accent.TButton",
            font=("Arial", 12, "bold"),
            padding=(20, 10),
            foreground="#4d63ec",
            background="#6366f1",
            borderwidth=0
        )
        self.style.map(
            "Accent.TButton",
            background=[("active", "#4f46e5"), ("disabled", "#e5e7eb")],
            foreground=[("disabled", "#9ca3af")]
        )
        self.style.configure(".", background="white")
        self.style.configure(
            "Accent.TButton",
            font=("Arial", 12, "bold"),
            padding=(20, 10),
            foreground="#4d63ec",
            background="#6366f1",
            borderwidth=0
        )
        self.style.map(
            "Accent.TButton",
            background=[("active", "#4f46e5"), ("disabled", "#e5e7eb")],
            foreground=[("disabled", "#9ca3af")]
        )
        self.style.configure("TButton", font=("Arial", 12), padding=10)
        self.style.configure("TLabel", font=("Arial", 12))

        # Initialize variables
        self.model = Model()
        self.image_path = None
        self.processed_image = None
        self.segmented_chars = []
        self.debug_mode = tk.BooleanVar(value=False)

        self.create_widgets()

    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(pady=10)

        title_line1 = ttk.Label(
            title_frame,
            text="Mathvision:",
            font=("Plus Jakarta Sans", 50, "bold"),
            foreground="#6495ED"
        )
        title_line1.pack()

        title_line2 = ttk.Label(
            title_frame,
            text="Quadratic Equation Solver",
            font=("Plus Jakarta Sans", 20),
            foreground="#6495ED"
        )
        title_line2.pack()

        # Top panel (images and visualization)
        control_panel = ttk.Frame(main_frame)
        control_panel.pack(fill=tk.X, pady=10)

        # Buttons
        upload_button = ttk.Button(
            control_panel, text="⬆ Upload Image", command=self.upload_image, style="Accent.TButton")
        upload_button.pack(side=tk.LEFT, padx=5)

        self.process_button = ttk.Button(
            control_panel, text="🛠 Process Image", command=self.process_image, state=tk.DISABLED, style="Accent.TButton")
        self.process_button.pack(side=tk.LEFT, padx=5)

        self.solve_button = ttk.Button(
            control_panel, text="🔍 Solve Equation", command=self.solve, state=tk.DISABLED, style="Accent.TButton")
        self.solve_button.pack(side=tk.LEFT, padx=5)

        clear_button = ttk.Button(
            control_panel, text="🗑 Clear", command=self.clear, style="Accent.TButton")
        clear_button.pack(side=tk.LEFT, padx=5)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var,
                               relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Debug checkbox
        debug_check = ttk.Checkbutton(control_panel, text="Debug Mode",
                                      variable=self.debug_mode)
        debug_check.pack(side=tk.RIGHT, padx=5)
        top_panel = ttk.Frame(main_frame)
        top_panel.pack(fill=tk.BOTH, expand=True)

        # Original image frame
        self.original_frame = tk.Frame(
            top_panel,
            bg="white",
            highlightbackground="#d1d5db",  # Tailwind gray-300
            highlightcolor="#d1d5db",
            highlightthickness=2,
            bd=0,
            relief="solid"
        )
        self.original_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.original_label = ttk.Label(self.original_frame)
        self.original_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        # Processed image frame
        self.processed_frame = ttk.LabelFrame(top_panel, text="Processed Image")
        self.processed_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.processed_label = tk.Label(
            self.processed_frame,
            text="No image to display",
            font=("Arial", 14),
            # bg="#e0f2fe",         # Light blue (Tailwind blue-100)
            fg="#2563eb",         # Blue text (Tailwind blue-600)
            anchor="center",
            justify="center"
        )
        self.processed_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)


        # Control panel

        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Results")
        results_frame.pack(pady=10, fill=tk.BOTH)

        # Results display
        self.results_text = tk.Text(results_frame, height=10, wrap=tk.WORD,
                                    font=("Courier", 12))
        self.results_text.pack(fill=tk.BOTH, padx=10, pady=10)

    def upload_image(self):
        """Open file dialog to select an image"""
        file_types = [("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        file_path = filedialog.askopenfilename(title="Select Equation Image",
                                               filetypes=file_types)

        self.process_button.config(state=tk.DISABLED)
        self.solve_button.config(state=tk.DISABLED)

        self.style.configure(
        "Blue.TButton",
        font=("Arial", 12),
        padding=10,
        foreground="#4d63ec",
        background="#2563eb",  # Tailwind's blue-600
    )
        self.style.map(
        "Blue.TButton",
        background=[("active", "#1e40af"), ("disabled", "#9ca3af")],  # darker on hover, gray when disabled
        foreground=[("disabled", "#e5e7eb")]
)

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

    def process_image(self):
        """Process the image to prepare for character segmentation"""
        if not self.image_path:
            return

        try:
            self.status_var.set("Processing image...")
            self.root.update()

            # Load image and convert to grayscale
            image = cv.imread(self.image_path)
            [binary, self.segmented_chars,
                processed_images] = process_image(image, self.debug_mode.get())

            # Display the processed image
            self.processed_image = binary
            self.display_processed_image()

            # Show segmented characters in debug mode
            if self.debug_mode.get():
                show_processed_images(self.segmented_chars, processed_images)

            # Enable solve button
            self.solve_button.config(state=tk.NORMAL)

            self.status_var.set("Image processed. Ready to solve.")

        except Exception as e:
            self.status_var.set(f"Error loading image: {str(e)}")
            messagebox.showerror(
                "Error", f"Failed to process equation: {str(e)}")

    def display_original_image(self, file_path):
        """Display the selected image"""
        try:
            image = Image.open(file_path)

            # Calculate new dimensions while maintaining aspect ratio
            max_width = DISPLAY_IMG_MAX_WIDTH
            max_height = DISPLAY_IMG_MAX_HEIGHT
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

    def display_processed_image(self):
        """Display the processed binary image"""
        try:
            # Convert OpenCV binary image to PIL format
            pil_img = Image.fromarray(self.processed_image)

            # Calculate new dimensions while maintaining aspect ratio
            max_width = DISPLAY_IMG_MAX_WIDTH
            max_height = DISPLAY_IMG_MAX_HEIGHT
            width, height = pil_img.size

            if width > max_width or height > max_height:
                ratio = min(max_width / width, max_height / height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                pil_img = pil_img.resize(
                    (new_width, new_height), Image.LANCZOS)

            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_img)

            # Update image label
            self.processed_label.config(image=photo)
            self.processed_label.image = photo

        except Exception as e:
            self.status_var.set(f"Error loading image: {str(e)}")
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def solve(self):
        """Recognize the equation and solve it"""
        if not self.segmented_chars:
            self.status_var.set("Process the image first")
            return

        try:
            self.status_var.set("Solving equation...")
            self.root.update()

            parsed_equation = []
            for chars in self.segmented_chars:
                x = chars["image"].reshape(28, 28, 1)
                # convert to (1, 28, 28, 3)
                input_img = np.concatenate(
                    (x, x, x), axis=2).reshape(1, 28, 28, 3)
                parsed_equation.append(self.model.predict(input_img))

            if len(parsed_equation) == 0:
                raise Exception("Unable to parse equation!")

            equation_str = process_equation(parsed_equation)

            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(
                tk.END, f"Detected Equation: {equation_str}\n")
            self.results_text.insert(tk.END, f"Solution: {solve_equation(equation_str)}\n")

            self.status_var.set("DONE.")

        except Exception as e:
            self.status_var.set(f"Error solving equation: {str(e)}")
            self.results_text.insert(tk.END, f"Error: {str(e)}")

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
