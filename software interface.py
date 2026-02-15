import numpy as np
import tkinter as tk
from tkinter import filedialog
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class NiftiImageComparisonViewer:
    def __init__(self, master):
        self.master = master
        self.master.title("NIfTI Image Comparison")
        self.master.geometry("1200x800")
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(0, weight=0)
        self.master.grid_columnconfigure(1, weight=1)
        self.btn_frame = tk.Frame(self.master, bg="#F5F5F5", bd=1, relief=tk.RAISED)
        self.btn_frame.grid(row=0, column=0, sticky="ns", padx=5, pady=5)
        self.btn_frame.grid_rowconfigure(0, weight=1)
        self.btn_frame.grid_rowconfigure(1, weight=1)
        self.load_btn1 = tk.Button(self.btn_frame, text="Load Image 1", command=self.load_image1, width=12)
        self.load_btn1.grid(row=0, column=0, pady=(220, 0), sticky="n")
        self.load_btn2 = tk.Button(self.btn_frame, text="Load Image 2", command=self.load_image2, width=12)
        self.load_btn2.grid(row=1, column=0, pady=(0, 220), sticky="s")
        self.main_frame = tk.Frame(self.master, bg="#FFFFFF")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        self.fig, self.axes = plt.subplots(2, 3, figsize=(10, 6),
                                           gridspec_kw={
                                               'wspace': 0.2,
                                               'hspace': 0.3,
                                               'width_ratios': [1,1,1],
                                               'height_ratios': [1,1]
                                           })
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.slider_frame = tk.Frame(self.main_frame, bg="#FFFFFF")
        self.slider_frame.grid(row=1, column=0, sticky="ew", pady=5)
        self.slider_frame.grid_columnconfigure(0, weight=1)
        self.slider_frame.grid_columnconfigure(1, weight=1)
        self.slider_frame.grid_columnconfigure(2, weight=1)

        self.image1 = None
        self.image_array1 = None
        self.image2 = None
        self.image_array2 = None
        self.slices = [0, 0, 0]
        self.sliders = []

        for ax in self.axes.flatten():
            ax.axis('off')
            ax.set_aspect('equal')
        self.fig.tight_layout()

    def load_image1(self):
        filename = filedialog.askopenfilename(filetypes=(("NIfTI files", "*.nii *.nii.gz"), ("All files", "*.*")))
        if filename:
            self.image1 = sitk.ReadImage(filename)
            self.image_array1 = sitk.GetArrayFromImage(self.image1)
            self.init_slices()
            self.update_display()

    def load_image2(self):
        filename = filedialog.askopenfilename(filetypes=(("NIfTI files", "*.nii *.nii.gz"), ("All files", "*.*")))
        if filename:
            self.image2 = sitk.ReadImage(filename)
            self.image_array2 = sitk.GetArrayFromImage(self.image2)
            self.init_slices()
            self.update_display()

    def init_slices(self):
        if self.image1 is not None:
            self.slices = [
                self.image_array1.shape[0]//2,
                self.image_array1.shape[1]//2,
                self.image_array1.shape[2]//2
            ]
        elif self.image2 is not None:
            self.slices = [
                self.image_array2.shape[0]//2,
                self.image_array2.shape[1]//2,
                self.image_array2.shape[2]//2
            ]

    def update_display(self):
        for widget in self.slider_frame.winfo_children():
            widget.destroy()
        self.sliders.clear()
        self.display_images()

        for col in range(3):
            label = tk.Label(self.slider_frame, text=["Axial","Coronal","Sagittal"][col], bg="#FFFFFF")
            label.grid(row=0, column=col, pady=2)


            slider = tk.Scale(self.slider_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                              command=lambda v, c=col: self.update_slice(int(v), c),
                              length=int(self.axes[0,col].get_position().width * self.fig.dpi),
                              troughcolor="#DCDCDC", fg="#333")
            slider.grid(row=1, column=col, sticky="ew", padx=5)
            self.sliders.append(slider)

        self.update_slider_ranges()

    def display_images(self):
        for ax in self.axes.flatten():
            ax.clear()
            ax.axis('off')
            ax.set_aspect('equal')

        if self.image1 is not None:
            img1 = self.apply_window_level(self.image_array1)
            self.axes[0,0].imshow(img1[self.slices[0],:,:], cmap='gray')
            self.axes[0,1].imshow(img1[:,self.slices[1],:], cmap='gray')
            self.axes[0,2].imshow(img1[:,:,self.slices[2]], cmap='gray')
            self.axes[0,0].set_title("Image 1 - Axial", fontsize=9)
            self.axes[0,1].set_title("Image 1 - Coronal", fontsize=9)
            self.axes[0,2].set_title("Image 1 - Sagittal", fontsize=9)

        if self.image2 is not None:
            img2 = self.apply_window_level(self.image_array2)
            self.axes[1,0].imshow(img2[self.slices[0],:,:], cmap='gray')
            self.axes[1,1].imshow(img2[:,self.slices[1],:], cmap='gray')
            self.axes[1,2].imshow(img2[:,:,self.slices[2]], cmap='gray')
            self.axes[1,0].set_title("Image 2 - Axial", fontsize=9)
            self.axes[1,1].set_title("Image 2 - Coronal", fontsize=9)
            self.axes[1,2].set_title("Image 2 - Sagittal", fontsize=9)

        self.fig.tight_layout()
        self.canvas.draw()

    def update_slice(self, value, slice_type):
        self.slices[slice_type] = value
        self.display_images()
        self.update_slider_ranges()

    def update_slider_ranges(self):
        max_ranges = [0,0,0]
        if self.image1 is not None:
            max_ranges = [
                self.image_array1.shape[0]-1,
                self.image_array1.shape[1]-1,
                self.image_array1.shape[2]-1
            ]
        if self.image2 is not None:
            max_ranges = [
                max(max_ranges[0], self.image_array2.shape[0]-1),
                max(max_ranges[1], self.image_array2.shape[1]-1),
                max(max_ranges[2], self.image_array2.shape[2]-1)
            ]

        for i, slider in enumerate(self.sliders):
            slider.config(to=max_ranges[i])
            slider.set(min(self.slices[i], max_ranges[i]))

    def apply_window_level(self, image_array):
        window_center = 0
        window_width = 2000
        min_val = window_center - window_width/2
        max_val = window_center + window_width/2
        return np.clip((image_array - min_val)/(max_val - min_val)*255, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    root = tk.Tk()
    app = NiftiImageComparisonViewer(root)
    root.mainloop()