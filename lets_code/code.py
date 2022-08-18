import cv2
import numpy as np
import skimage.exposure

# read image
img = cv2.imread('1.jpg')

# convert to gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# threshold to binary and invert so background is white and xxx are black
thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)[1]
thresh = 255 - thresh

# add black border around threshold image to avoid corner being largest distance
thresh2 = cv2.copyMakeBorder(thresh, 1, 1, 1, 1, cv2.BORDER_CONSTANT, (0))
h, w = thresh2.shape

# create zeros mask 2 pixels larger in each dimension
mask = np.zeros([h + 2, w + 2], np.uint8)

# apply distance transform
distimg = thresh2.copy()
distimg = cv2.distanceTransform(distimg, cv2.DIST_L2, 5)

# remove excess border
distimg = distimg[1:h-1, 1:w-1]

# get max value and location in distance image
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(distimg)

# scale distance image for viewing
distimg = skimage.exposure.rescale_intensity(
    distimg, in_range='image', out_range=(0, 255))
distimg = distimg.astype(np.uint8)

# draw circle on input
result = img.copy()
centx = max_loc[0]
centy = max_loc[1]
radius = int(max_val)
cv2.circle(result, (centx, centy), radius, (0, 0, 255), 1)
print('center x,y:', max_loc, 'center radius:', max_val)

# save image
cv2.imwrite('xxx_distance.png', distimg)
cv2.imwrite('xxx_radius.png', result)

# show the images
cv2.imshow("thresh", thresh)
cv2.imshow("thresh2", thresh2)
cv2.imshow("distance", distimg)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
