import cv2
import numpy as np
import tkinter as tk
import os
from tkinter import filedialog
from PIL import Image, ImageTk
from tools.pfmutil import readPFM
from datasets.EXRloader import load_exr
from tools.utils import (
    disp_to_depth, 
    depth_to_disp
    )

default_calib_info = {
    # wheatstone scene 0912-2: 
    'cam0': np.array(
        [[312.96691817, 0.,             326.58117294],
         [  0.,         312.96691817,   320.55296326],
         [  0.,         0.,             1.          ]]
         ),
    'cam1': np.array(
        [[312.96691817, 0.,             326.58117294],
         [  0.,         312.96691817,   320.55296326],
         [  0.,         0.,             1.          ]]
         ),
    'baseline': 0.063, # meter
    'width': 640,
    'height': 640,
    'disp_max': 192, # disparity
    'disp_min': 0,
    'depth_min': 0.5, # meter
    'depth_max': 4.0, # meter
    }

class DispMapViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Disp/Depth/Distance Map Viewer")

        self.disp_map = None
        self.canvas_h, self.canvas_w = 640, 960
        self.board = 5

        ### Create a Canvas widget to display the depth map ###
        self.canvas = tk.Canvas(root, width=2*self.canvas_w + self.board , height=self.canvas_h)
        self.canvas.pack()

        ### Load Color Image Button ###
        load_button = tk.Button(root, text="Load Left Image", command=self.load_img)
        load_button.pack()
        
        ### Load Color Image Button ###
        load_button = tk.Button(root, text="Load Right Image", command=self.load_right_img)
        load_button.pack()
        
        ### Create Intrinsics button ###
        #self.save_button = tk.Button(root, text="Load Camrea Parameters", command=self.load_intrinsics)
        self.save_button = tk.Button(root, text="Load Camrea Parameters", command=self.read_calib)
        self.save_button.pack()

        ### Load Depth Map Button ###
        load_button = tk.Button(root, text="Load Disp Map", command=self.load_disp_map)
        load_button.pack()

        
        
        ### Create a label to display disparity value ###
        self.disp_label = tk.Label(root, text="Disparity Value:")
        self.disp_label.pack()

        ### Create a label to display depth value ###
        self.depth_label = tk.Label(root, text="Depth Value:")
        self.depth_label.pack()

        ### Create a label to display depth value ###
        self.dist_label = tk.Label(root, text="Distance Value:")
        self.dist_label.pack()

        ### Bind mouse click event to canvas ###
        self.canvas.bind("<Button-1>", self.display_depth_value)

    def load_disp_map(self):
        """ Load a depth map image using a file dialog
        """
        file_path = filedialog.askopenfilename(
            #initialdir=os.getcwd(),
            title="Select Disparity Files", 
            filetypes=[("Image Files", "*.png *.pfm *.exr")]
            )
        if file_path:
            if file_path.endswith(".png"):
                disp_map = cv2.imread(file_path, -1).astype(np.float32) / 256. # FIXME: assume normalization scale is 256.0 as in KITTI;
            elif file_path.endswith(".pfm"):
                disp_map = readPFM(file_path)
            elif file_path.endswith(".exr"):
                disp_map = load_exr(file_path)
            dispH, dispW = disp_map.shape[:]
            assert (self.img_w, self.img_h) == (dispW, dispH)
            disp_map[disp_map == np.inf] = .0 # set zero as a invalid disparity value;
            
            # do not forget to scale the disparity values accordingly;

            self.disp_map =  (self.canvas_w/dispW)* cv2.resize(disp_map, (self.canvas_w, self.canvas_h), 
                                                    interpolation=cv2.INTER_LINEAR
                                                    )
            # get depth from disparity
            self.depth_map = disp_to_depth(disp=self.disp_map, baseline=self.baseline, focal_length_px=self.fx)
    

     
    def load_img(self):
        """ Load a color map image using a file dialog
        """
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg")])
        if file_path:
            ## load image ##
            self.rgb_img = Image.open(file_path).convert('RGB')

            ## get raw size ##
            self.img_w, self.img_h = self.rgb_img.size

            ## resize to canvas shape ##
            self.rgb_img = self.rgb_img.resize((self.canvas_w, self.canvas_h))

            ## Display color map on the canvas ##
            self.rgb_img = ImageTk.PhotoImage(self.rgb_img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.rgb_img)
    
    def load_right_img(self):
        """ Load a color map image using a file dialog
        """
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg")])
        if file_path:
            ## load image ##
            self.rgb_img_right = Image.open(file_path).convert('RGB')

            ## get raw size ##
            self.img_w2, self.img_h2 = self.rgb_img_right.size

            ## resize to canvas shape ##
            self.rgb_img_right = self.rgb_img_right.resize((self.canvas_w, self.canvas_h))

            ## Display color map on the canvas ##
            self.rgb_img_right = ImageTk.PhotoImage(self.rgb_img_right)
            self.canvas.create_image(self.canvas_w + self.board, 0, anchor=tk.NW, image=self.rgb_img_right)
            #self.canvas.move(self.rgb_img_right, 0, 0)

    def load_intrinsics(self):
        """Load camera intrinsics using a file dialog
        Assume intrinsics file contains data with the following format

        fx 0  cx
        0  fy cy
        0  0  1
        """
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if file_path:
            K = np.loadtxt(file_path)
            self.fx = K[0,0] / self.img_w * self.canvas_w
            self.fy = K[1,1] / self.img_h * self.canvas_h
            self.cx = K[0,2] / self.img_w * self.canvas_w
            self.cy = K[1,2] / self.img_h * self.canvas_h
    
    """
    calib.txt
    cam0=[1870 0 297; 0 1870 277; 0 0 1]
    cam1=[1870 0 397; 0 1870 277; 0 0 1]
    doffs=100
    baseline=160
    width=694
    height=554
    ndisp=128
    isint=0
    vmin=32
    vmax=115
    dyavg=0
    dymax=0
    """
    def read_calib(self):
        calib_filename = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        calib_info = {}
        if calib_filename:
            with open(calib_filename) as f:
                lines = f.readlines()
                for l in lines:
                    line =  l.strip("\n").split("=")
                    key = line[0]
                    val = line[1]
                    if key in ["width", "height", "ndisp"]:
                        calib_info[key] = int(val)
                    elif key in ["baseline"]:
                        # camera baseline in mm;
                        calib_info[key] = float(val)
                    elif key in ['cam0', 'cam1']:
                        val = val[1:-1]
                        #print (key, val)
                        # E.g., cam0=[1758.23 0 953.34; 0 1758.23 552.29; 0 0 1];
                        # Split matrix string by semicolon to get rows
                        rows = [row.strip() for row in val.split(';') if row.strip()]
                        # Extract values from rows
                        matrix_values = [list(map(float, row.split())) for row in rows]
                        # Convert to numpy array
                        matrix = np.array(matrix_values)
                        calib_info[key] = matrix

            K = calib_info['cam0']
            #assert calib_info['width'] == self.img_w
            #assert calib_info['height'] == self.img_h

            self.fx = K[0,0] / self.img_w * self.canvas_w
            self.fy = K[1,1] / self.img_h * self.canvas_h
            self.cx = K[0,2] / self.img_w * self.canvas_w
            self.cy = K[1,2] / self.img_h * self.canvas_h 
            self.baseline = calib_info['baseline']


    def depth2dist(self, 
                   depth: float, 
                   fx: float, 
                   fy: float, 
                   cx: float, 
                   cy: float, 
                   x: float, 
                   y: float
                   ) -> float:
        """convert depth value to distance value

        Args:
            depth (float): depth value
            fx (float)   : focal x
            fy (float)   : focal y
            cx (float)   : principal point x
            cy (float)   : principal point y
            x (float)    : pixel location x
            y (float)    : pixel location y
        
        Returns:
            dist (float) : euclidean distance
        """
        ### Convert pixel coordinates to camera coordinates ###
        x_c = (x - cx) / fx * depth
        y_c = (y - cy) / fy * depth

        ### Calculate the Euclidean distance ###
        distance = np.sqrt(x_c**2 + y_c**2 + depth**2)

        return distance

    def display_depth_value(self, event):
        """Display depth/distance value
        """
        if self.depth_map is not None:
            x, y = event.x, event.y
            disp_value = self.disp_map[y, x]
            depth_value = self.depth_map[y, x]

            dist_value = self.depth2dist(depth_value, 
                                        fx=self.fx, 
                                        fy=self.fy,
                                        cx=self.cx,
                                        cy=self.cy,
                                        x=x,
                                        y=y)
            if disp_value > 0: 
                # Draw a circle at the clicked position
                radius = 5
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.rgb_img)
                self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, outline='red', width=2)
                
                # Draw a circle at the right image position
                radius = 5
                self.canvas.create_image(self.canvas_w + self.board, 0, anchor=tk.NW, image=self.rgb_img_right)
                x_right = int(max(x - disp_value, 0))
                x_right_pos = x_right + self.canvas_w + self.board
                y_right = y
                self.canvas.create_oval(x_right_pos - radius, y_right - radius, 
                                        x_right_pos + radius, y_right + radius, 
                                        outline='green', 
                                        width=2
                                        )

                # Display the depth value
                self.disp_label.config(text=f"Disparity Value: {disp_value:.03f}")
                self.depth_label.config(text=f"Depth Value: {depth_value:.03f}")
                self.dist_label.config(text=f"Distance  Value: {dist_value:.03f}")
                print (
                    f"==> Clicked! left pixel (red) @(x={x:03d}, y={y:03d}) | " \
                    f"paired right pixel (green) @(x={x_right:03d}, y={y_right:03d}) | " \
                    f"Disparity = {disp_value:.03f} | " \
                    f"Depth = {depth_value:.03f} | " \
                    f"Distance = {dist_value:.03f}"
                    ) 
            else:
                print ("No disparity/depth prediction at this point! Please try another!!!") 

"""
How to run this file:
- cd This_Project
- python3 -m depth_warp_viewer.disp_map_viewer
"""
if __name__ == "__main__":
    root = tk.Tk()
    app = DispMapViewer(root)
    root.mainloop()