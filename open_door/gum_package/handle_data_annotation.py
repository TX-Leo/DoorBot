import os
import json
import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw
import numpy as np

import sys
root_dir = "../"
sys.path.append(root_dir)
from utils.lib_rgbd import *

class HandleDataAnnotator:
    def __init__(self, root_dir=None):
        """Initializes the annotation tool."""

        # Configuration parameters
        self.IMAGE_WIDTH = 1280
        self.IMAGE_HEIGHT = 1280
        self.CENTROID_RADIUS = 3
        self.POINT_RADIUS = 3
        self.CENTROID_COLOR = "red"
        self.P1_COLOR = "blue"
        self.P2_COLOR = "green"
        self.AUXILIARY_LINE_COLOR = "red"
        self.AUXILIARY_LINE_WIDTH = 2
        self.AUXILIARY_LINE_DASH = (2, 2)
        self.MOUSE_POINT_COLOR = "black"
        self.MOUSE_POINT_RADIUS = 3

        self.root_dir = root_dir
        self.image_files = []
        self.current_image_index = 0
        self.current_image = None
        self.annotation_data = {}
        self.p1 = None
        self.p2 = None
        self.orientation = None
        self.auxiliary_line_h = None
        self.auxiliary_line_v = None
        self.mouse_point = None

        self.window = tk.Tk()
        self.window.title("Handle Annotation Tool")

        # Styling using ttk themes
        style = ttk.Style()
        style.theme_use('clam') # clam; alt; default; classic

        style.configure("TButton", padding=6, relief="flat", background="#ccc")
        style.configure("TLabel", background="light gray")
        style.configure("TFrame", background="light gray")

        # Menu bar
        menubar = tk.Menu(self.window)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open Directory", command=self.open_directory)
        menubar.add_cascade(label="File", menu=filemenu)
        self.window.config(menu=menubar)

        # Top frame for buttons and status
        top_frame = ttk.Frame(self.window)
        top_frame.grid(row=0, column=0, sticky="ew")  

        # Status bar
        self.status_bar = ttk.Label(top_frame, text="", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.TOP, fill=tk.X)

        # Buttons
        button_frame = ttk.Frame(top_frame)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="Clear P1", command=self.clear_p1, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear P2", command=self.clear_p2, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Previous", command=self.previous_image, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Next", command=self.next_image, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Done", command=self.save_annotation, width=10).pack(side=tk.LEFT, padx=5)

        # Label to display mouse position
        self.mouse_position_label = ttk.Label(top_frame, text="") 
        self.mouse_position_label.pack()

        # Label to display image size
        self.image_size_label = ttk.Label(top_frame, text="")
        self.image_size_label.pack() 

        # Image frame
        self.image_frame = ttk.Frame(self.window) 
        self.image_frame.grid(row=1, column=0, sticky="nsew") 

        # Canvas for image display
        self.canvas = tk.Canvas(self.image_frame, width=self.IMAGE_WIDTH, height=self.IMAGE_HEIGHT, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # Configure grid weights
        self.window.grid_rowconfigure(1, weight=1)  # Image frame expands
        self.window.grid_columnconfigure(0, weight=1) 

        # Bind events
        self.canvas.bind("<Motion>", self.mouse_move)
        self.canvas.bind("<Button-1>", self.annotate_point)

        # Open directory if provided
        if self.root_dir:
            self.load_images()

    def open_directory(self):
        """Opens a directory selection dialog and loads the images."""
        self.root_dir = filedialog.askdirectory()
        if self.root_dir:
            self.load_images()

    def load_images(self):
        """Loads images and corresponding JSON data from the root directory."""
        self.image_files = sorted([
            f for f in os.listdir(self.root_dir) 
            if f.endswith('.jpg') and re.match(r'^\d+$', os.path.splitext(f)[0])
        ])
        self.annotation_data = {}
        for image_file in self.image_files:
            json_file = os.path.splitext(image_file)[0] + '.json'
            json_path = os.path.join(self.root_dir, json_file)
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    self.annotation_data[image_file] = data
        self.current_image_index = 0
        self.display_image()

    def display_image(self):
        """Displays the current image and annotations on the canvas."""
        image_file = self.image_files[self.current_image_index]
        self.status_bar.config(text=f"Image: {image_file}")
        image_path = os.path.join(self.root_dir, image_file)
        self.current_image = Image.open(image_path).convert("RGB")

        # Get and display image size 
        width, height = self.current_image.size
        self.image_size_label.config(text=f"Image Size: {width} x {height}") 

        # Resize image to fit canvas while maintaining aspect ratio
        self.photo = ImageTk.PhotoImage(self.current_image)
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        # Display centroid
        if image_file in self.annotation_data:
            self.canvas.delete("centroid") # Delete previous centroid
            Cx = self.annotation_data[image_file]['Cx']
            Cy = self.annotation_data[image_file]['Cy']
            self.orientation = self.annotation_data[image_file]['orientation']
            self.canvas.create_oval(Cx - self.CENTROID_RADIUS, Cy - self.CENTROID_RADIUS, 
                                   Cx + self.CENTROID_RADIUS, Cy + self.CENTROID_RADIUS, 
                                   fill=self.CENTROID_COLOR, outline=self.CENTROID_COLOR, tags="centroid")

        # Display P1 and P2 (if already annotated)
        self.load_annotations(image_file)

    def load_annotations(self, image_file):
        """Loads P1 and P2 annotations if they exist for the current image."""
        if image_file in self.annotation_data and 'p1_x' in self.annotation_data[image_file]:
            self.p1 = (
                self.annotation_data[image_file]['p1_x'],
                self.annotation_data[image_file]['p1_y']
            )
            self.canvas.create_oval(self.p1[0] - self.POINT_RADIUS, self.p1[1] - self.POINT_RADIUS, 
                                   self.p1[0] + self.POINT_RADIUS, self.p1[1] + self.POINT_RADIUS, 
                                   fill=self.P1_COLOR, outline=self.P1_COLOR, tags="p1")
        if image_file in self.annotation_data and 'p2_x' in self.annotation_data[image_file]:
            self.p2 = (
                self.annotation_data[image_file]['p2_x'],
                self.annotation_data[image_file]['p2_y']
            )
            self.canvas.create_oval(self.p2[0] - self.POINT_RADIUS, self.p2[1] - self.POINT_RADIUS, 
                                   self.p2[0] + self.POINT_RADIUS, self.p2[1] + self.POINT_RADIUS, 
                                   fill=self.P2_COLOR, outline=self.P2_COLOR, tags="p2")

    def mouse_move(self, event):
        """Updates mouse position label, auxiliary lines, and P2 preview."""
        x, y = event.x, event.y
        self.mouse_position_label.config(text=f"Mouse Position: ({x}, {y})")

        # Draw white dot for mouse position
        if self.mouse_point:
            self.canvas.delete(self.mouse_point)
        self.mouse_point = self.canvas.create_oval(x - self.MOUSE_POINT_RADIUS, y - self.MOUSE_POINT_RADIUS, 
                                               x + self.MOUSE_POINT_RADIUS, y + self.MOUSE_POINT_RADIUS, 
                                               fill=self.MOUSE_POINT_COLOR, outline=self.MOUSE_POINT_COLOR)

        self.draw_auxiliary_lines(x, y)

    def draw_auxiliary_lines(self, x, y):
        """Draws horizontal and vertical auxiliary lines through the mouse position."""
        if self.auxiliary_line_h:
            self.canvas.delete(self.auxiliary_line_h)
        self.auxiliary_line_h = self.canvas.create_line(0, y, self.canvas.winfo_width(), y, 
                                                       dash=self.AUXILIARY_LINE_DASH, fill=self.AUXILIARY_LINE_COLOR, 
                                                       width=self.AUXILIARY_LINE_WIDTH)

        if self.auxiliary_line_v:
            self.canvas.delete(self.auxiliary_line_v)
        self.auxiliary_line_v = self.canvas.create_line(x, 0, x, self.canvas.winfo_height(), 
                                                       dash=self.AUXILIARY_LINE_DASH, fill=self.AUXILIARY_LINE_COLOR, 
                                                       width=self.AUXILIARY_LINE_WIDTH)

        # Update P2 preview on the auxiliary line, but don't update self.p2
        if self.p1 is not None:
            if self.orientation == 'horizontal':
                p2_x = x
                p2_y = self.p1[1]
            elif self.orientation == 'vertical':
                p2_x = self.p1[0]
                p2_y = y

            if self.canvas.find_withtag("p2_preview"):
                self.canvas.coords("p2_preview", p2_x - self.POINT_RADIUS, p2_y - self.POINT_RADIUS, 
                                   p2_x + self.POINT_RADIUS, p2_y + self.POINT_RADIUS)
            else:
                self.canvas.create_oval(p2_x - self.POINT_RADIUS, p2_y - self.POINT_RADIUS, 
                                       p2_x + self.POINT_RADIUS, p2_y + self.POINT_RADIUS, 
                                       fill=self.P2_COLOR, outline=self.P2_COLOR, tags="p2_preview")

    def annotate_point(self, event):
        """Annotates a point on the image."""
        x, y = event.x, event.y
        if self.p1 is None:
            self.p1 = (x, y)
            self.canvas.create_oval(x - self.POINT_RADIUS, y - self.POINT_RADIUS, 
                                   x + self.POINT_RADIUS, y + self.POINT_RADIUS, 
                                   fill=self.P1_COLOR, outline=self.P1_COLOR, tags="p1")
        elif self.p2 is None:
            if self.orientation == 'horizontal':
                p2_x = x
                p2_y = self.p1[1]
            elif self.orientation == 'vertical':
                p2_x = self.p1[0]
                p2_y = y
            
            self.p2 = (p2_x, p2_y)
            self.canvas.create_oval(p2_x - self.POINT_RADIUS, p2_y - self.POINT_RADIUS, 
                                   p2_x + self.POINT_RADIUS, p2_y + self.POINT_RADIUS, 
                                   fill=self.P2_COLOR, outline=self.P2_COLOR, tags="p2")

    def clear_annotations(self):
        """Clears all annotation points and previews."""
        self.canvas.delete("p1")
        self.canvas.delete("p2")
        self.canvas.delete("p2_preview") # Clear any P2 preview 
        self.p1 = None
        self.p2 = None

    def clear_p1(self):
        """Clears the P1 annotation."""
        if self.p1:
            self.canvas.delete("p1")
            self.p1 = None
            self.clear_p2()  # Also clear P2 when clearing P1

    def clear_p2(self):
        """Clears the P2 annotation."""
        if self.p2:
            self.canvas.delete("p2")
            self.p2 = None

    def previous_image(self):
        """Displays the previous image."""
        if self.current_image_index > 0:
            self.current_image_index -= 1
        elif self.current_image_index == 0:
            self.current_image_index = len(self.image_files) - 1
        self.clear_annotations() # Clear annotations before loading the new image
        self.display_image()
    def next_image(self):
        """Displays the next image."""
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
        elif self.current_image_index == len(self.image_files) - 1:
            self.current_image_index = 0
        self.clear_annotations() # Clear annotations before loading the new image
        self.display_image()
    
    def save_annotation(self):
        """Saves the annotation data and annotated image."""
        image_file = self.image_files[self.current_image_index]
        if self.p1 is not None:
            Cx = self.annotation_data[image_file]['Cx']
            Cy = self.annotation_data[image_file]['Cy']
            dx = self.p1[0] - Cx
            dy = self.p1[1] - Cy
            
            # Save annotated image (using original image size)
            draw = ImageDraw.Draw(self.current_image)
            draw.ellipse((Cx - self.CENTROID_RADIUS, Cy - self.CENTROID_RADIUS, 
                          Cx + self.CENTROID_RADIUS, Cy + self.CENTROID_RADIUS), 
                         fill=self.CENTROID_COLOR, outline=self.CENTROID_COLOR)
            draw.ellipse((self.p1[0]  - self.POINT_RADIUS, 
                          self.p1[1]  - self.POINT_RADIUS, 
                          self.p1[0]  + self.POINT_RADIUS, 
                          self.p1[1]  + self.POINT_RADIUS), 
                         fill=self.P1_COLOR, outline=self.P1_COLOR)
            if self.p2 is not None:
                if self.orientation == 'horizontal':
                    R = self.p2[0] - self.p1[0]
                elif self.orientation == 'vertical':
                    R = self.p2[1] - self.p1[1]
                draw.ellipse((self.p2[0]  - self.POINT_RADIUS, 
                              self.p2[1]  - self.POINT_RADIUS, 
                              self.p2[0]  + self.POINT_RADIUS, 
                              self.p2[1]  + self.POINT_RADIUS), 
                             fill=self.P2_COLOR, outline=self.P2_COLOR)
            else:
                R = 0
            annotated_image_path = os.path.join(self.root_dir, os.path.splitext(image_file)[0] + '_annotated.png')
            self.current_image.save(annotated_image_path)
        else:
            dx, dy, R = 0, 0, 0
            
        # save to json
        self.annotation_data[image_file]['dx'] = dx
        self.annotation_data[image_file]['dy'] = dy
        self.annotation_data[image_file]['R'] = R
        json_file = os.path.splitext(image_file)[0] + '.json'
        json_path = os.path.join(self.root_dir, json_file)
        with open(json_path, 'w') as f:
            json.dump(self.annotation_data[image_file],f,indent=4)

        # save to txt
        box = self.annotation_data[image_file]['box']
        annotation_str = f"{box[0]} {box[1]} {box[2]} {box[3]} {dx} {dy} {R}"
        txt_file = os.path.splitext(image_file)[0] + '.txt'
        txt_path = os.path.join(self.root_dir, txt_file)
        with open(txt_path, 'w') as f:
            f.write(annotation_str)

        if R!=0:
            x1_2d,y1_2d=self.p1[0],self.p1[1]
            dx = dx
            dy = dy
            R  = R
            orientation = self.orientation
            angle=90
            img_path = os.path.join(self.root_dir, os.path.splitext(image_file)[0] + '.jpg')
            save_path = os.path.join(self.root_dir, os.path.splitext(image_file)[0] + '_vis.png')
            x2_2d,y2_2d,Ox,Oy = rotate_point(x1_2d,y1_2d,R,orientation,angle)
            vis_grasp(img_path,dx,dy,x1_2d,y1_2d,x2_2d,y2_2d,Ox,Oy,R,orientation,angle,save_path)

        # messagebox.showinfo("Annotation Saved", f"Annotation for {image_file} saved successfully!")

        def close_message():
            message_window.destroy() 

        message_window = tk.Toplevel(self.window)  # Create a new top-level window
        message_window.title("Annotation Saved")
        ttk.Label(message_window, text=f"Annotation for {image_file} saved successfully!").pack(pady=20, padx=20)
        message_window.after(1000, close_message)  # Close after 0.5 seconds (500 milliseconds)

        self.next_image()

    def run(self):
        """Starts the annotation tool GUI."""
        self.window.mainloop()

if __name__ == "__main__":
    root_dir = r'./data/lever2'
    handle_data_annotator = HandleDataAnnotator(root_dir=root_dir)
    handle_data_annotator.run()