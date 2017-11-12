import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
import threading
import timeit
import mouse


def getPatch(point,confidence):
	centerY, centerX = point;
	height, width = confidence.shape[:2]
	minX = max(centerX - halfPatchWidth, 0)
	maxX = min(centerX + halfPatchWidth, width - 1)
	minY = max(centerY - halfPatchWidth, 0)
	maxY = min(centerY + halfPatchWidth, height - 1)
	upperLeft = (minX, minY)
	lowerRight = (maxX, maxY)
	return upperLeft, lowerRight


def calculateGradients(work_image,tgt_region):
	'''
	This function uses Scharr operator to calculate derivatives/gradient
	in both x and y direction. 
	Parameters:
	-----------
	work_image: Image which is changed every iteration
	tgt_region: Boolean mask of pixels where True indicates object pixel 
	'''

	# convert to grayscale to calculate derivatives
	src_gray = cv2.cvtColor(work_image,cv2.COLOR_BGR2GRAY);

	# calculate x derivatives
	gradientX = cv2.Scharr(src_gray, cv2.CV_32F, 1, 0); # default parameter: scale shoule be 1
	# calculate y derivatives
	gradientY = cv2.Scharr(src_gray, cv2.CV_32F, 0, 1);
	
	# set non source region gradient to be zero
	gradientY[tgt_region] = 0;
	gradientX[tgt_region] = 0;
	
	# normalise the gradients
	gradientX /= 255;
	gradientY /= 255;
	return (gradientY,gradientX);


def computeFillFront(tgt_region,src_region):
	'''
	This function calculates the set of pixels which lie on the boundary,
	which are used to select the next patch to update
	Parameters:
	-----------
	tgt_region: A boolean map where True indicates that pixel belongs to object
	'''

	# intialize a laplacian kernel
	LAPLACIAN_KERNEL = np.ones((3, 3), dtype = np.float32)
	LAPLACIAN_KERNEL[1, 1] = -8
	# find the boundaries using laplacian kernel
	boundryMat = cv2.filter2D((255*tgt_region).astype(np.uint8), cv2.CV_32F, LAPLACIAN_KERNEL)
	boundryMat = boundryMat > 0; # create a boolean mask where boundary is present
	
	# initialize kernel to find the directional derivative of source region
	NORMAL_KERNELX = np.zeros((3, 3), dtype = np.float32);
	NORMAL_KERNELX[1, 0] = -1;
	NORMAL_KERNELX[1, 2] = 1;
	NORMAL_KERNELY = cv2.transpose(NORMAL_KERNELX);
	
	# update the gradients in both the directions
	sourceGradientX = cv2.filter2D((255*src_region).astype(np.uint8), cv2.CV_32F, NORMAL_KERNELX);
	sourceGradientY = cv2.filter2D((255*src_region).astype(np.uint8), cv2.CV_32F, NORMAL_KERNELY);
	
	# take all the points in fillfront where boundary is True
	fillFront = np.argwhere(boundryMat);
	
	# calculate the normals in both the directions
	normalX = sourceGradientY[boundryMat];
	normalY = -sourceGradientX[boundryMat];
	# calculate magnitude of normal vector
	norm_mag = np.sqrt(normalX**2 + normalY**2);
	pos_norm = norm_mag > 0;
	# calculate the unit vector by dividing with normal magnitude where it is +ve
	normalX[pos_norm] = normalX[pos_norm]/norm_mag[pos_norm];
	normalY[pos_norm] = normalY[pos_norm]/norm_mag[pos_norm];
	
	return (fillFront,normalX,normalY);

def computeConfidence(tgt_region,confidence,fillFront):
	'''
	This function updates the confidence matrix for the complete image
	Parameters:
	-----------
	tgt_region: Boolean mask where True represent object region
	confidence: Intialized confidence matrix
	'''
	# update confidence for every patch in fillfront
	for p in fillFront:
		py,px = p;
		(ax, ay), (bx, by) = getPatch(p,confidence);
		total = 0;
		# find where target region is false
		mapping = tgt_region[ay:by+1,ax:bx+1]==0;
		# take average of confidence over source region
		total = np.sum(confidence[ay:by+1,ax:bx+1][mapping]);
		confidence[py,px] = total / mapping.size;


def computeData(gradientY,gradientX,normalY,normalX,fillFront,data):
	'''
	This function computes data matrix as mentioned in the paper
	Parameters:
	-----------
	gradientY: Gradient along Y-direction
	gradientX: Gradient along X-direction
	normalY: normal along Y-direction
	normalX: normal along X-direction
	fillFront: list of pixels on the boundary
	data: data matrix to be updated
	'''
	y,x = fillFront[:,0],fillFront[:,1];
	# update data matrix
	data[y,x] = np.fabs(gradientX[y,x]*normalX + gradientY[y,x]*normalY)+0.001;


def computeTarget(data,confidence,fillFront):
	'''
	This function computes the maximum priority patch.
	Parameter:
	----------
	data: The data matrix as defined in the paper
	confidence: Calculated confidence value for every pixel
	fillFront: List of points on the boundary
	Return value:
	------------
	Point in [y,x] form where priority is maximum on fillFront
	'''
	# calculate the priority as product of confidence and data for all points in fillFront
	w = 0;
	rc = (1 - w) * confidence[fillFront[:,0],fillFront[:,1]] + w;
	priority = data[fillFront[:,0],fillFront[:,1]] * rc;
	return fillFront[priority.argmax()];


def computeBestPatch(data,confidence,fillFront,work_image,src_region,tgt_region,originalSourceRegion,gradientY,gradientX):
	'''
	This function computes the best patch using maximum priority and updates the 
	work_image and other matrices accordingly.
	Parameters:
	-----------
	'''
	global patch_list,coord_list,work_image_illus;
	currentPoint = computeTarget(data,confidence,fillFront);
	ty,tx = currentPoint;
	# print(currentPoint);
	(ax,ay),(bx,by) = getPatch(currentPoint,confidence);
	pHeight, pWidth = by-ay+1,bx-ax+1;
	height,width = work_image.shape[:2];
	split_x = int(width/pWidth)*pWidth;
	split_y = int(height/pHeight)*pHeight;
	bdr_patch = work_image[ay:ay+pHeight,ax:ax+pWidth].astype(np.float32);
	bdr_patch_mask = src_region[ay:ay+pHeight,ax:ax+pWidth];
	bdr_patch_mask = np.dstack((bdr_patch_mask,bdr_patch_mask,bdr_patch_mask));

	first_iter = True;

	sel_patch = None;
	if patch_list is None and (pHeight == 2*halfPatchWidth+1) and (pWidth == 2*halfPatchWidth+1):
		patch_list = [];
		coord_list = [];
		for i in range(height - pHeight):
			for j in range(width - pWidth):
				src_patch = originalSourceRegion[i:i+pHeight,j:j+pWidth];
				if np.sum(src_patch) == src_patch.size:
					patch_list.append(work_image[i:i+pHeight,j:j+pWidth]);
					coord_list.append((i,j));
		patch_list = np.array(patch_list);
		coord_list = np.array(coord_list);

	if (pHeight != 2*halfPatchWidth+1) or (pWidth != 2*halfPatchWidth+1):
		new_patch_list = [];
		new_coord_list = [];
		for i in range(height - pHeight):
			for j in range(width - pWidth):
				src_patch = originalSourceRegion[i:i+pHeight,j:j+pWidth];
				if np.sum(src_patch) == src_patch.size:
					new_patch_list.append(work_image[i:i+pHeight,j:j+pWidth]);
					new_coord_list.append((i,j));
		new_patch_list = np.array(new_patch_list);
		new_coord_list = np.array(new_coord_list);
	else:
		new_patch_list = patch_list;
		new_coord_list = coord_list;
	print(new_patch_list.shape,bdr_patch.shape,bdr_patch_mask.shape,pHeight,pWidth);
	start_time = timeit.default_timer();
	mse = (((new_patch_list - bdr_patch)**2)*bdr_patch_mask).sum(axis=(1,2,3))
	elp = timeit.default_timer();
	print("computedataTime:",start_time-elp);	
	min_errors_idx = np.argwhere(mse == mse.min());
	min_errors_idx = min_errors_idx.reshape(min_errors_idx.shape[0],);
	variance = np.var(new_patch_list[min_errors_idx],axis=(1,2,3));
	best_patch_arg = np.argmin(variance);
	best_patch_arg = min_errors_idx[best_patch_arg];
	sely,selx = new_coord_list[best_patch_arg];
	work_image_illus = work_image.copy();
	cv2.rectangle(work_image_illus,(ax,ay),(ax+pWidth,ay+pHeight),(255,0,0),1);
	cv2.rectangle(work_image_illus,(selx,sely),(selx+pWidth,sely+pHeight),(0,0,255),1);
	work_image[ay:ay+pHeight,ax:ax+pWidth][~bdr_patch_mask] = work_image[sely:sely+pHeight,selx:selx+pWidth][~bdr_patch_mask];

	gradientX[ay:ay+pHeight,ax:ax+pWidth][~bdr_patch_mask[:,:,1]] = gradientX[sely:sely+pHeight,selx:selx+pWidth][~bdr_patch_mask[:,:,1]];
	gradientY[ay:ay+pHeight,ax:ax+pWidth][~bdr_patch_mask[:,:,1]] = gradientY[sely:sely+pHeight,selx:selx+pWidth][~bdr_patch_mask[:,:,1]];
	confidence[ay:ay+pHeight,ax:ax+pWidth][~bdr_patch_mask[:,:,1]] = confidence[ty, tx];
	src_region[ay:ay+pHeight,ax:ax+pWidth] = True;
	tgt_region[ay:ay+pHeight,ax:ax+pWidth] = False;	


def inpaint(input_image,removal_mask):
	'''
	This is the main function to do the inpainting. It controls overall
	sequence of the program and calls other functions accordingly.
	Parameters
	-----------
	input_image: Image on which inpainting is to be done
	removal_mask: A binary mask of from where object is to be removed
	'''
	global work_image_illus,halfPatchWidth,patch_list,coord_list;
	work_image_illus = None;
	patch_list = None;
	coord_list = None;

	height,width = input_image.shape[:2];
	max_height = 400;
	max_width = 400;
	scale = 1;

	if height > max_height or width > max_width:
		scale = max_height / float(height);
		if max_width/float(width) < scale:
			scale = max_width / float(width);
		input_image = cv2.resize(input_image,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC);
		removal_mask = cv2.resize(removal_mask,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC);
		print("scale is",scale);

	fillFront = None;
	normalX = None;
	normalY = None;
	gradientY = None;
	gradientX = None;
	data = None;

	# input_image = cv2.imread(sys.argv[1]);
	# removal_mask = cv2.imread(sys.argv[2],0);

	# create a workimage which will be updated recursively
	work_image = np.copy(input_image);
	work_image_illus = work_image.copy();

	# threshold the mask to create a binary image
	removal_mask = removal_mask > 150;
	# initialize the confidence matrix
	confidence = (~removal_mask).astype(np.float32);
	# initialize the source region
	src_region= np.copy(~removal_mask);
	# copy of source region which will not be updated
	originalSourceRegion = np.copy(src_region)
	print(originalSourceRegion.sum());
	# initialize the target region
	tgt_region = removal_mask.copy();
	# initialize the target matrix
	data = np.zeros(shape=input_image.shape[:2],dtype=np.float32);
	# initalize the patch size
	patch_size = 4;
	halfPatchWidth = patch_size;

	# loop condition always true
	stay = True;
	# calculate the gradients
	gradientY,gradientX = calculateGradients(work_image,tgt_region);
	
	loop_iter = 0;
	# plt.imshow(removal_mask,cmap="gray");
	
	# start updating the image
	while stay and loop_iter<10000000:
		# calculate the points on boundary and normals
		fillFront,normalX,normalY = computeFillFront(tgt_region,src_region);
		if fillFront.size == 0:
			break;
		start_time = timeit.default_timer();
		computeConfidence(tgt_region,confidence,fillFront);
		elp = timeit.default_timer();
		# print("computeConfidenceTime:",start_time-elp);
		start_time = timeit.default_timer();
		computeData(gradientY,gradientX,normalY,normalX,fillFront,data);
		elp = timeit.default_timer();
		# print("computedataTime:",start_time-elp);
		start_time = timeit.default_timer();
		print(loop_iter)
		# if loop_iter > 29 and loop_iter < 31:
		# 	computeBestPatch(True);
		# else:
		computeBestPatch(data,confidence,fillFront,work_image,src_region,tgt_region,originalSourceRegion,gradientY,gradientX)
		elp = timeit.default_timer();
		print("#### longest time ######-:",start_time-elp);
		# if loop_iter%1==0:
		# 	cv2.imwrite("illus"+str(loop_iter)+".jpg",work_image_illus);
		loop_iter +=1;
	work_image = cv2.resize(work_image,None,fx=(1.0/scale),fy=(1.0/scale),interpolation=cv2.INTER_CUBIC);
	return work_image.copy()
def main():
	global work_image_illus,halfPatchWidth,patch_list,coord_list;
	work_image_illus = None;
	patch_list = None;
	coord_list = None;
	input_image = cv2.imread(sys.argv[1]);
	removal_mask = mouse.crop(input_image.copy());
	cv2.imwrite("test_images/test_mask.jpg",removal_mask);
	# removal_mask = cv2.imread(sys.argv[2],0);
	inpaint(input_image,removal_mask);
	plt.show()

if __name__ == '__main__':
	main()