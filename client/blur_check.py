import glob
import cv2
import numpy as np

pth = sorted(glob.glob('./new_imgs/*'))
for p in pth:
	im = cv2.imread(p,-1)
	image_canny = cv2.Canny(im, 100, 200)
	nonzero_ratio = np.count_nonzero(image_canny) * 1000.0 / image_canny.size
	print(nonzero_ratio)
	if nonzero_ratio < 10:
		print(p.split('/')[-1], 'Blur')
	else:
		print(p.split('/')[-1], 'Not Blur')
