import sys
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
import cv2
import inpainting
import mouse

from PIL import Image, ImageTk

class MainApplication(ttk.Frame):

	def __init__(self, parent, *args, **kwargs):
		"""
		Initialize the system and all variables
		"""
		tk.Frame.__init__(self, parent, *args, **kwargs);
		self.parent = parent;
		self.parent.attributes("-zoomed", True);
		self.parent.title("Photoeditor");
		
		self.mainframe = ttk.Frame(self.parent, padding="2");
		self.mainframe.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S) );
		self.mainframe.columnconfigure(0, weight=1);
		self.mainframe.rowconfigure(0, weight=1);

		self.containerframe = ttk.Frame(self.mainframe, padding="2");
		self.containerframe.grid(column=1, row=1, sticky=(tk.N, tk.W, tk.E, tk.S));

		self.imgframe = ttk.Frame(self.containerframe, padding="2");
		self.imgframe.grid(column=1, row=2, sticky=(tk.N, tk.W, tk.E, tk.S));

		self.input_imgpanel = tk.Canvas(self.imgframe, bd=2, relief="ridge", width=560, height=560);
		self.input_imgpanel.grid(column=1,row=1,sticky=(tk.W, tk.E), padx=10, pady=10);

		self.output_imgpanel = tk.Canvas(self.imgframe, bd=2, relief="ridge", width=560, height=560);
		self.output_imgpanel.grid(column=2, row=1, sticky=(tk.W, tk.E), padx=10, pady=10);

		self.inputframe = ttk.Frame(self.mainframe, padding="2");
		self.inputframe.grid(column=1, row=2, sticky=(tk.N, tk.W, tk.E, tk.S));

		self.inputpath = tk.StringVar();
		self.outputpath = tk.StringVar();

		ttk.Label(self.inputframe, text="Input file").grid(column=1, row=1, sticky=(tk.W, tk.E), padx=2, pady=2);
		self.lbl2 =  ttk.Label(self.inputframe, text="Output file").grid(column=1, row=2, sticky=(tk.W, tk.E), padx=2, pady=2);
		self.ip_path_lbl = ttk.Label(self.inputframe, textvariable=self.inputpath).grid(column=2, row=1, sticky=(tk.W, tk.E), padx=2, pady=2);
		self.op_path_lbl = ttk.Label(self.inputframe, textvariable=self.outputpath).grid(column=2, row=2, sticky=(tk.W, tk.E), padx=2, pady=2);
		self.ip_button = ttk.Button(self.inputframe, text="Browse file", command=self.load_ipfile).grid(column=3, row=1, sticky=(tk.W, tk.E), padx=2, pady=2);
		self.op_button = ttk.Button(self.inputframe, text="Browse file", command=self.load_opfile).grid(column=3, row=2, sticky=(tk.W, tk.E), padx=2, pady=2);

		self.controlframe = ttk.Frame(self.mainframe, padding="10 20",width=10, border=2);
		self.controlframe.grid(column=2,row=1,sticky=(tk.N, tk.E, tk.S));

		self.function_var = tk.IntVar()

		self.rd_btn1 = ttk.Radiobutton(self.controlframe, text="Inpainting", variable=self.function_var, value=1).grid(row=1,column=1,sticky=(tk.W, tk.E));
		self.rd_btn2 = ttk.Radiobutton(self.controlframe, text="Red eye correction", variable=self.function_var, value=2).grid(row=2,column=1,sticky=(tk.W, tk.E));
		self.effect_btn = ttk.Button(self.controlframe, text="Apply effect", command=self.edit).grid(row=3,column=1,sticky=(tk.N, tk.W));

		for child in self.mainframe.winfo_children(): child.grid_configure(padx=5, pady=5);

		self.input_img = None;
		self.output_img = None;

	def update_inputimgpanel(self):
		"""
		This function updates the image input image panel and related 
		variables everytime user browses for new input image. 
		"""

		# default size of image is original size of the image
		size = (self.input_img.shape[1],self.input_img.shape[0]);
		print (size)

		# resize the image according to the panel
		if self.input_img.shape[1] > self.input_img.shape[0]:
			# if width is larger side and greater than panel width, resize
			if self.input_img.shape[1] > self.input_imgpanel.winfo_width():
				w = self.input_imgpanel.winfo_width();
				h = w * self.input_img.shape[0] / self.input_img.shape[1]; # maintain proportion
				h = int(h);

				size = (w,h);
				print("setting width equal");
		else :
			# if height is larger side and greater than panel height, resize
			if self.input_img.shape[0] > self.input_imgpanel.winfo_height():
				h = self.input_imgpanel.winfo_height();
				w = h * self.input_img.shape[1] / self.input_img.shape[0]; # maintain proportion
				w = int(w);

				size = (w,h);
				print("setting height equal");

		# store the resized image with correct channel order
		resized = Image.fromarray(self.bgr_to_rgb(self.input_img)).resize(size,Image.ANTIALIAS);
		# load the image in tk panel
		self.inputphoto = ImageTk.PhotoImage(resized);
		self.input_imgpanel.create_image(self.input_imgpanel.winfo_width()/2,self.input_imgpanel.winfo_height()/2,image=self.inputphoto, anchor=tk.CENTER, tags="IMG");
	
	def update_outputimgpanel(self):
		"""
		This function updates the image output image panel and related 
		variables everytime user applies an operation. 
		"""

		# default size of image is original size of the image
		size = (self.output_img.shape[1],self.output_img.shape[0]);
		# resize the image as in the case of input panel
		if self.output_img.shape[1] > self.output_img.shape[0]:
			if self.output_img.shape[1] > self.output_imgpanel.winfo_width():
				w = self.output_imgpanel.winfo_width();
				h = w * self.output_img.shape[0] / self.output_img.shape[1];
				h = int(h);
				size = (w,h);
				print("setting width equal");
		else :
			if self.output_img.shape[0] > self.output_imgpanel.winfo_height():
				h = self.output_imgpanel.winfo_height();
				w = h * self.output_img.shape[1] / self.output_img.shape[0];
				w = int(w);
				size = (w,h);
				print("setting height equal");
		print(size)
		resized = Image.fromarray(self.bgr_to_rgb(self.output_img)).resize(size,Image.ANTIALIAS);
		self.outputphoto = ImageTk.PhotoImage(resized);
		self.output_imgpanel.create_image(self.output_imgpanel.winfo_width()/2,self.output_imgpanel.winfo_height()/2,image=self.outputphoto, anchor=tk.CENTER, tags="IMG");

	def clear_outputimgpanel(self):
		"""
		This function clears the output image panel and 
		related variable whenever new input image is selected
		"""

		self.output_imgpanel.delete("IMG");
		self.output_img = None;
		self.outputpath.set("");

	def load_ipfile(self):
		"""
		This function loads input file and displays it on the input image panel
		It resizes the image based on panel dimensions
		"""
		# Open the input file dialog
		fname = tk.filedialog.askopenfilename(title = "Select image",filetypes = (("Image files",("*.jpg","*.jpeg","*.png","*.gif","*.tiff")),("JPEG files",("*.jpg","*jpeg")),("PNG files", "*.png"),("all files","*.*")));

		# Try to load the image if user selected a file
		if fname:
			try:
				self.inputpath.set(fname);
				self.input_img = cv2.imread(fname);
			except :
				# Show error if unable to open the image
				tk.messagebox.showerror("ERROR: Open file", "Failed to open file \n'%s'" %fname);

			self.update_inputimgpanel();

			# reset any image already loaded on output panel
			self.clear_outputimgpanel();

			
			# self.output_imgpanel.create_image(0,0,image=self.inputphoto, anchor=tk.NW, tags="IMG");
			# self.parent.bind("<Configure>", self.resize)

	def load_opfile(self):
		"""
		This function saves the output image to user selected
		location with user specified name
		"""
		if isinstance(self.output_img, np.ndarray):
			fname = tk.filedialog.asksaveasfilename(title = "Enter filename",filetypes = (("JPEG files",("*.jpg","*.jpeg")),("PNG files", "*.png"),("all files","*.*")));
			if fname:
				try:
					self.outputpath.set(fname);
					cv2.imwrite(fname,self.output_img);
				except cv2.error as e:
					# tk.messagebox.showerror("ERROR: Saving file", "Failed to save file \n'%s'" %fname);
					tk.messagebox.showerror("ERROR: Saving file", e);
		else:
			tk.messagebox.showerror("ERROR", "No output image available. Generate an image first.");

	
	def bgr_to_rgb(self,image):
		"""
		This function converts the default cv_BGR format file to PIL default RGB format
		"""
		
		# check if image has three channels
		if image.ndim == 3 and image.shape[2] == 3:
			return cv2.cvtColor(image,cv2.COLOR_BGR2RGB) # rotate the channels
		else:
			return image; # Default case: return the image with no change

	def redeye(self,img):
		"""
		This function takes a image having red eye defect and applies 
		correction it

		Keyword arguments:
		img: Image on which red eye correction to be done
		Return value:
		output_img - numpy array of dimension equal to input image
					representing the corrected red eye image

		Algorithm: 
		1. Detect eyes using the trained classifier
		2. Obtain a binary image using thresholding to separate red eye part
		3. Apply closing operation to remove any holes
		4. Find all connected regions
		5. Remove false positives using shape filter
		6. Grow the region using dilation and reducing the threshold value
		7. Use the binary image as a mask for colored image
		8. Equate red component of image to mean of blue and green in masked area
		"""

		# load the trained classifier weights from stored xml file
		eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml');
		# convert image to grayscale for eye detections
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
		# detect the eyes in the image
		eyes = eye_cascade.detectMultiScale(gray,1.22,8)
		# initialize the output image
		imgOut = img.copy();
		
		# loop over detected eyes to detect red eyes and correct them
		for (ex,ey,ew,eh) in eyes:
			
			eye = img[ey:ey+eh, ex:ex+ew];
			
			# load r,g,b component from image
			R = eye[:,:,2] / 255.0;
			G = eye[:,:,1] / 255.0;
			B = eye[:,:,0] / 255.0;

			# calculate redness by subtracting g,b channels from r
			redness = R - (G + B);

			# calculate redluminance (see references)
			redluminance = np.maximum( 0, redness - G);

			# use thresholding to generate a binary image
			thresh = 0.1;
			mask = redluminance > thresh;
			# binary image to accomodate lesser red regions 
			mask2 = redluminance > thresh*0.1;
			# change boolean to uint8 for make compatible with cv2 functions
			mask = mask.astype(np.uint8);
			mask2 = mask2.astype(np.uint8);

			# choose kernel size for dilation = 0.1* eye width			
			kernel_size = np.round(ew/10).astype(np.int);
			# choose closing kernel size
			closing_kernel_size = 3
			# apply closing operation
			mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((closing_kernel_size,closing_kernel_size)));
			mask2 = cv2.morphologyEx(mask2,cv2.MORPH_CLOSE,np.ones((closing_kernel_size,closing_kernel_size)));
			# get circular kernel for dilation
			kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size,kernel_size));
			# apply dilation
			mask = cv2.dilate(mask,kernel1,iterations=1);
			mask2 = cv2.dilate(mask2,kernel1,iterations=1);
			# find connected components in the mask
			output = cv2.connectedComponentsWithStats(255*mask.astype(np.uint8), 4, cv2.CV_32S)
			# initialize the eye region dimensions
			basewidth = output[2][0,cv2.CC_STAT_WIDTH];
			baseheight = output[2][0,cv2.CC_STAT_HEIGHT];
			# initialize min and max ratio of eye dimensions to 
			# red part dimensions to remove false positives
			kwmin = 0.15;
			kwmax = 0.4;
			# remove false positives
			for i in range(1,output[0]):
				noteye = -1;
				if output[2][i,cv2.CC_STAT_WIDTH] < kwmin*basewidth or output[2][i,cv2.CC_STAT_WIDTH] > kwmax*basewidth:
					noteye = i;
					# print("width does not satisfy")
					# print("Width ratio:",output[2][i,cv2.CC_STAT_WIDTH]/basewidth);
				if output[2][i,cv2.CC_STAT_HEIGHT] < kwmin*baseheight or output[2][i,cv2.CC_STAT_HEIGHT] > kwmax*baseheight:
					noteye = i;
					# print("height does not satisfy")
					# print("Height ratio:",output[2][i,cv2.CC_STAT_HEIGHT]/baseheight);

				if noteye != -1:
					# print("label ",noteye,"-",output[2][i,cv2.CC_STAT_HEIGHT],output[2][noteye,cv2.CC_STAT_WIDTH])
					mask[output[1]==noteye] = 0;
				# else:
				# 	print("Height ratio:",output[2][i,cv2.CC_STAT_HEIGHT]/baseheight);
				# 	print("Width ratio:",output[2][i,cv2.CC_STAT_WIDTH]/basewidth);
			
			# dilate the shape filtered mask to take into account more eye region
			mask = cv2.dilate(mask,kernel1,iterations=2);
			# remove any region that is not red
			mask = mask*mask2;

			# Calculate the mean channel by averaging the green and blue channels
			mean = 255*(B + G) / 2
			mask = mask.astype(np.bool)[:, :, np.newaxis]
			# mask = mask.astype(np.bool)
			mean = mean[:, :, np.newaxis]
			 
			# Copy the eye from the original image. 
			eyeOut = eye.copy()
			# Copy the mean image to the output image with mask applied
			np.copyto(eyeOut, mean.astype(np.uint8), where=mask)

			# Copy the fixed eye to the output image. 
			imgOut[ey:ey+eh, ex:ex+ew, :] = eyeOut

		return imgOut

	def exemplar_inpaint(self,img):
		
		removal_mask = mouse.crop(img.copy());
		imgOut = inpainting.inpaint(img.copy(),removal_mask);
		return imgOut;
	def edit(self) :
		"""
		This function is invoked when user clicks on apply function button
		It generates the output image by calling the appropriate function
		based on user input. Once output image is generated it loads it 
		to the output panel
		"""

		# print (self.function_var.get())
		self.clear_outputimgpanel();
		self.parent.config(cursor="gumby");
		self.parent.update();
		if self.function_var.get() == 1:
			self.output_img = self.exemplar_inpaint(self.input_img);
		elif self.function_var.get() == 2:
			self.output_img = self.redeye(self.input_img);

		# 	original = cv2.imread(self.input_img, cv2.IMREAD_COLOR);
		# 	noisy = original
		# 	# denoised = skimage.restoration.denoise_nl_means(noisy, patch_size = 4, patch_distance = 2, multichannel = True);
		# 	denoised = cv2.fastNlMeansDenoisingColored(img,templateWindowSize=7,conda create10,7,21);
		# 	sharpened = np.clip(denoised*1.5 - 0.5*skimage.filters.gaussian(denoised,sigma = 10, multichannel = True),0,1.0);
		# 	# sharpened = np.clip(original - skimage.filters.gaussian(denoised,sigma = 10, multichannel = True),0,1);
		# 	equalized = skimage.exposure.equalize_adapthist(sharpened,clip_limit=5e-3);
		# 	self.output_img = img_as_ubyte(equalized);


		self.update_outputimgpanel();
		self.parent.config(cursor="");
		# self.parent.update();


				

def main():
	root = tk.Tk();
	app = MainApplication(root);
	root.mainloop();


if __name__ == '__main__':
	main();