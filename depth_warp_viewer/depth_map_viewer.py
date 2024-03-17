import cv2
import numpy as np
import tkinter as tk
import os
from tkinter import filedialog
from PIL import Image, ImageTk
from tools.pfmutil import readPFM
from tools.geometry_utils import adjust_cam_K, warp_one_pixel_from_ref_to_src
from tools.utils import (
    disp_to_depth, 
    depth_to_disp
    )

from tools.EXRloader import load_exr


os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"


class DepthMapViewer:
    def __init__(self, root, dataset_type='general'):
        self.root = root
        self.root.title("Depth Map Viewer")
        
        self.dataset_type = dataset_type
        self.depth_map = {}
        self.disp_map = {}
        self.K44 = {}
        self.invK44 = {}
        self.E44 = {}
        self.invE44 = {}
        self.fx = {}
        self.fy = {}
        self.cx = {}
        self.cy = {}
        self.baseline = {
            'irs': 0.1, # meter;
            'vkt2': -1,
            }
        self.dynamic_mask = {'ref': None, 'src': None}

        self.canvas_h, self.canvas_w = 360, 640
        self.canvas_h, self.canvas_w = 540, 960
        self.canvas_h, self.canvas_w = 640,640
        #self.canvas_h, self.canvas_w = 480, 640
        self.board = 5

        ### Create a Canvas widget to display the depth map ###
        self.canvas = tk.Canvas(root, width=2*self.canvas_w + self.board , height=self.canvas_h)
        self.canvas.pack()

        ### Load Color Image Button ###
        load_button = tk.Button(root, text="Load Ref Image", command=self.load_ref_img)
        load_button.pack()
        
        load_button = tk.Button(root, text="Load Src Image", command=self.load_src_img)
        load_button.pack()
        

        ### Load Depth Map Button ###
        load_button = tk.Button(root, text="Load Ref Depth Map", command= lambda: self.load_depth_map(image_tag='ref'))
        load_button.pack()
        
        load_button = tk.Button(root, text="Load Src Depth Map", command= lambda: self.load_depth_map(image_tag='src'))
        load_button.pack()

        ### Create Intrinsics button ###
        self.save_button = tk.Button(root, text="Load Ref Intrinsics", command= lambda: self.load_intrinsics(image_tag='ref'))
        self.save_button.pack()
        self.save_button = tk.Button(root, text="Load Src Intrinsics", command= lambda: self.load_intrinsics(image_tag='src'))
        self.save_button.pack()
        ### Create Extrinsics button ###
        self.save_button = tk.Button(root, text="Load Ref Extrinsics", command= lambda: self.load_pose(image_tag='ref'))
        self.save_button.pack()
        self.save_button = tk.Button(root, text="Load Src Extrinsics", command= lambda: self.load_pose(image_tag='src'))
        self.save_button.pack()

        
        ### Create a label to display depth value ###
        self.depth_label = tk.Label(root, text="Depth Value:")
        self.depth_label.pack()
        
        ### Create a label to display disparity value ###
        #self.disp_label = tk.Label(root, text="Disparity Value:")
        #self.disp_label.pack()

        ### Create a label to display depth value ###
        #self.dist_label = tk.Label(root, text="Distance Value:")
        #self.dist_label.pack()

        ### Bind mouse click event to canvas ###
        self.canvas.bind("<Button-1>", self.display_depth_value)
    
    """ from Dynamic Replica Dataloader """
    # > see: https://github.com/facebookresearch/dynamic_stereo/blob/c5077aa19f73cf3e51e363bf8a3796c85eb2eaa0/datasets/dynamic_stereo_datasets.py#L65
    def _load_16big_png_depth(self, depth_filepath):
        with Image.open(depth_filepath) as depth_pil:
            # the image is stored with 16-bit depth but PIL reads it as I (32 bit).
            # we cast it to uint16, then reinterpret as float16, then cast to float32
            depth = (
                np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
                .astype(np.float32)
                .reshape((depth_pil.size[1], depth_pil.size[0]))
            )
        return depth
    
    def _load_dynamic_instance_mask(self, instance_mask_path):
        instance_map = np.array(Image.open(instance_mask_path))
        # dynamic foreground with nonzere values;
        # remove the dynamic instance due to the violation of multi-view geometry (warping)
        dynamic_mask = instance_map > 0.1
        return dynamic_mask
    
    def read_scannet_png_depth(self, depth_png_filename):
        #NOTE: The depth map in milimeters can be directly loaded
        depth = cv2.imread(depth_png_filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)
        depth = depth / 1000.0 # change mm to meters;
        return depth
    
    def read_irs_exr_disparity2depth(self, disp_exr_filename):
        disp = load_exr(disp_exr_filename)
        baseline = 0.1
        focal_x = 480
        depth = baseline
        depth = disp_to_depth(disp, baseline, focal_x)
        return depth

    def load_depth_map(self, image_tag):
        """ Load a depth map image using a file dialog
        """
        assert image_tag in ['ref', 'src']
        file_path = filedialog.askopenfilename(
            #initialdir=os.getcwd(),
            title="Select PNG or PFM Depth Files", 
            filetypes=[("Image Files", "*.png *.pfm *.exr *.npy")]
            )
        if file_path:
            if file_path.endswith(".geometric.png"): # dynamic replica dataset
                depth_map = self._load_16big_png_depth(file_path)
                dynamic_mask = self.dynamic_mask[image_tag]
                if dynamic_mask is not None:
                    depth_map[dynamic_mask] = 0
                
            elif file_path.endswith(".pfm"):
                depth_map = readPFM(file_path)
            elif file_path.endswith(".exr"):
                depth_map = self.read_irs_exr_disparity2depth(file_path)
            elif file_path.endswith(".npy"):
                depth_map = depth = np.load(file_path)
            else:
                depth_map = self.read_scannet_png_depth(file_path)
            
            depth_map[depth_map == np.inf] = .0 # set zero as a invalid depth value;
            
            depth_map =  cv2.resize(depth_map, (self.canvas_w, self.canvas_h), 
                                            interpolation=cv2.INTER_LINEAR
                                )
            self.depth_map[image_tag] = depth_map
            #self.disp_map[image_tag] = self.depth_to_disparity(depth_map, self.baseline[image_tag], self.fx[image_tag])
    
    
    
    def get_instance_mask_filepath(self, img_file_path):
        if 'images' in img_file_path:
            tmp_path = img_file_path.replace('images','instance_id_maps')
            print ("tmp_path = ", tmp_path)
            if 'right-' in tmp_path:
                instance_mask_path = tmp_path.replace('right-','right_')
            else:
                instance_mask_path = tmp_path.replace('left-','left_')
            return instance_mask_path
        else:
            return None

    
    def load_ref_img(self):
        """ Load a color map image using a file dialog
        """
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg")])
        if file_path:

            ## load image ##
            self.ref_img = Image.open(file_path).convert('RGB')
            
            ## get raw size ##
            self.img_w, self.img_h = self.ref_img.size
            
            # get dynamic instance mask
            instance_mask_path = self.get_instance_mask_filepath(file_path)
            if instance_mask_path is not None and os.path.exists(instance_mask_path):
                dynamic_mask = self._load_dynamic_instance_mask(instance_mask_path)
                self.dynamic_mask['ref'] = dynamic_mask

                tmp_np = np.array(self.ref_img)
                tmp_np[:,:][dynamic_mask] = (255, 0, 0) # RGB;
                self.ref_img = Image.fromarray(tmp_np)
            
            ## resize to canvas shape ##
            self.ref_img = self.ref_img.resize((self.canvas_w, self.canvas_h))
            self.ref_img_np = np.array(self.ref_img)
            
                
                


            ## Display color map on the canvas ##
            self.ref_img = ImageTk.PhotoImage(self.ref_img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.ref_img)
    
    def load_src_img(self):
        """ Load a color map image using a file dialog
        """
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg")])
        if file_path:
            ## load image ##
            self.src_img = Image.open(file_path).convert('RGB')

            ## get raw size ##
            self.img_w2, self.img_h2 = self.src_img.size
            
            # get dynamic instance mask
            instance_mask_path = self.get_instance_mask_filepath(file_path)
            if instance_mask_path is not None and os.path.exists(instance_mask_path):
                dynamic_mask = self._load_dynamic_instance_mask(instance_mask_path)
                self.dynamic_mask['src'] = dynamic_mask

                tmp_np = np.array(self.src_img)
                tmp_np[:,:][dynamic_mask] = (255, 0, 0) # red;
                self.src_img = Image.fromarray(tmp_np)

            ## resize to canvas shape ##
            self.src_img = self.src_img.resize((self.canvas_w, self.canvas_h))
            self.src_img_np = np.array(self.src_img)

            ## Display color map on the canvas ##
            self.src_img = ImageTk.PhotoImage(self.src_img)
            self.canvas.create_image(self.canvas_w + self.board, 0, anchor=tk.NW, image=self.src_img)
            #self.canvas.move(self.src_img, 0, 0)

    def load_intrinsics(self, image_tag='ref'):
        """Load camera intrinsics using a file dialog
        Assume intrinsics file contains data with the following format

        fx 0  cx 0
        0  fy cy 0
        0  0  1  0
        0  0  0  1
        """
        assert image_tag in ['ref', 'src']
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if file_path:
            K44 = np.loadtxt(file_path).astype(np.float32).reshape(4,4)
            #print (f"{image_tag}: K original = {K44}")
            sx = self.canvas_w / self.img_w
            sy = self.canvas_h / self.img_h
            K44_new = adjust_cam_K(K44, scale_x=sx, scale_y=sy)
            self.K44[image_tag] = K44_new
            self.invK44[image_tag] = np.linalg.inv(K44_new)
            self.fx[image_tag] = self.K44[image_tag][0,0]
            self.fy[image_tag] = self.K44[image_tag][1,1] 
            self.cx[image_tag] = self.K44[image_tag][0,2] 
            self.cy[image_tag] = self.K44[image_tag][1,2]
    
    def load_pose(self, image_tag='ref'):
        assert image_tag in ['ref', 'src']
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if file_path:
            if "scannet" in file_path:
                is_E_not_invE = False
            elif "dyn_replica" in file_path:
                is_E_not_invE = True
            elif "irs" in file_path:
                is_E_not_invE = False
                #is_E_not_invE = True
                print (f"irs is_E={is_E_not_invE}")
            elif "tartanair" in file_path:
                is_E_not_invE = False
                print (f"TarTanAir is_E={is_E_not_invE}")
            else:
                is_E_not_invE = False
            
            if is_E_not_invE:
                #To directly load extrinsic, i.e., world-to-camera to map world points to camera points;
                world_T_cam_E_44 = np.loadtxt(file_path).astype(np.float32).reshape(4,4)
            
                #Camera-to-world to map cam points to world points
                cam_T_world_invE_44 = np.linalg.inv(world_T_cam_E_44) # invE
            else:
                cam_T_world_invE_44 = np.loadtxt(file_path).astype(np.float32).reshape(4,4) # invE 
                world_T_cam_E_44 = np.linalg.inv( cam_T_world_invE_44) # E
            self.E44[image_tag] = world_T_cam_E_44
            self.invE44[image_tag] = cam_T_world_invE_44


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
        if self.depth_map['ref'] is not None and self.depth_map['src'] is not None:
            print ("Please click ref depth on the left, not src depth (on the right)")
            x, y = event.x, event.y
            assert 0 <= x < self.canvas_w and 0 <= y < self.canvas_h, "Please click ref depth on the left!"
            
            ref_depth_value = self.depth_map['ref'][y, x]
            print (f" ref_depthh at (x={x},y={y}) = {ref_depth_value}")


            if ref_depth_value > 0: 
                # Draw a circle at the clicked position
                radius = 5
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.ref_img)
                self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, outline='red', width=2)

                # Draw a circle at the right image position
                radius = 5
                self.canvas.create_image(self.canvas_w + self.board, 0, anchor=tk.NW, image=self.src_img)

                
                x_src, y_src = warp_one_pixel_from_ref_to_src(
                        depth_ref = ref_depth_value, # scalr
                        pix_x_ref = x, # scalar
                        pix_y_ref = y, # scalr
                        invK_ref = self.invK44['ref'][:3,:3], # [3,3] 
                        invE_ref = self.invE44['ref'], #[4,4]
                        K_src = self.K44['src'][:3,:3], #[3,3]
                        E_src = self.E44['src'] #[4,4]
                        )
                
                x_right = int(min(max(x_src, 0), self.canvas_w-1))
                x_right_pos = x_right + self.canvas_w + self.board
                y_right = int(min(max(y_src, 0), self.canvas_h-1))
                y_right_pos = y_right
                self.canvas.create_oval(x_right_pos - radius, y_right_pos - radius, 
                                        x_right_pos + radius, y_right_pos + radius, 
                                        outline='green', 
                                        width=2
                                        )
                

                # Display the depth value
                #self.disp_label.config(text=f"Disparity Value: {disp_value:.03f}")
                ref_img_value = self.ref_img_np[y,x]
                src_img_value = self.src_img_np[y_right, x_right]
                src_depth_value = self.depth_map['src'][y_right, x_right]
                self.depth_label.config(text=
                                        f"Ref: {ref_depth_value:.03f}(ref_dep), {ref_img_value}(ref_img) \n" + \
                                        f"Src: {src_depth_value:.03f}(src_dep), {src_img_value}(src_img) \n"
                                        )
                print (
                    f"==> Clicked! Ref pixel (red) @(x={x:03d}, y={y:03d}) | " \
                    f"paired Src pixel (green) @(x={x_right:03d}, y={y_right:03d} | " \
                    f"Ref Depth = {ref_depth_value:.03f} | " \
                    f"Src Depth = {src_depth_value:.03f}"
                    ) 
            else:
                print ("No disparity/depth prediction at this point! Please try another!!!") 
            

"""
How to run this file:
- cd This_Project
- python3 -m depth_warp_viewer.depth_map_viewer
- python3.11 -m depth_warp_viewer.depth_map_viewer
"""
if __name__ == "__main__":
    root = tk.Tk()
    app = DepthMapViewer(root)
    root.mainloop()