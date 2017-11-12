import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
def outlineImage(event,x,y,flags,param):

	global image,drawing,mask,prev_pt;

	cv2.imshow('image',image);

	brushwidth = 1

	if event == cv2.EVENT_LBUTTONDOWN:
		drawing = True
		prev_pt = x,y;

	elif event == cv2.EVENT_MOUSEMOVE:
		if drawing == True:
			curr_pt = x,y;
			# cv2.circle(image,(x,y),brushwidth,255,-1)
			# cv2.circle(mask,(x,y),brushwidth,255,-1)
			cv2.line(image,prev_pt,curr_pt,255);
			cv2.line(mask,prev_pt,curr_pt,255);
			prev_pt = curr_pt;

	elif event == cv2.EVENT_LBUTTONUP:
		drawing = False
		curr_pt = x,y
		# cv2.circle(image,(x,y),brushwidth,255,-1)
		# cv2.circle(mask,(x,y),brushwidth,255,-1)
		cv2.line(image,prev_pt,curr_pt,255)
		cv2.line(mask,prev_pt,curr_pt,255)
		prev_pt = curr_pt;

def crop(img):
	global drawing,prev_pt;
	global image,mask;
	drawing = False;
	prev_pt = None;
	image = img
	mask = np.zeros(image.shape[:2]).astype(np.uint8);
	clone = image.copy();
	cv2.namedWindow("image");
	cv2.setMouseCallback("image",outlineImage);

	while True:
		cv2.imshow("image",image);
		key = cv2.waitKey(1) & 0xFF;

		if key == ord('r'):
			image = clone.copy();
			mask = np.zeros(image.shape[:2]).astype(np.uint8);
		elif key == ord('c'):
			break;
	cv2.destroyAllWindows();
	ht,wt = mask.shape[:2];
	print(ht,wt);
	seed_pt = (wt - 1,ht - 1);
	flood_mask = np.zeros((ht+2,wt+2)).astype(np.uint8);
	cv2.floodFill(mask,flood_mask,seed_pt,255)
	mask = np.invert(mask);
	plt.imshow(mask,cmap="gray");
	cv2.imwrite("shroom_mask.jpg",mask);
	plt.show();

	return mask;

if __name__ == '__main__':
	img = cv2.imread(sys.argv[1]);
	crop(img);