import selectivesearch
import numpy as np
import cv2
import time

def process(img):
    
    start_time = time.time()

    RESIZE_LENGTH = 100
    # loading lena image
    #img = skimage.data.lena()
    #img = cv2.resize(cv2.imread('fish-bike.jpg'), (448,448))

    resize_ratio_x = img.shape[1] / float(RESIZE_LENGTH)
    resize_ratio_y = img.shape[0] / float(RESIZE_LENGTH)

    img_resized = cv2.resize(img, (RESIZE_LENGTH, RESIZE_LENGTH))

    # perform selective search
    #img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=10)
    img_lbl, regions = selectivesearch.selective_search(
        img_resized, scale=500, sigma=0.9, min_size=10)

    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 2000 pixels
        if r['size'] < 0.03 * RESIZE_LENGTH**2:
            continue
        # distorted rects
        x, y, w, h = r['rect']

        if w / h > 1.2 or h / w > 1.2:
            continue

        #check if area is too large
        area = RESIZE_LENGTH **2
        if h*w > 0.8 * area:
            continue

        candidates.add(r['rect'])


    coordinates = []
    for x, y, w, h in candidates:
        x = int(x*resize_ratio_x)
        w = min(int(w*resize_ratio_x), img.shape[1]-x-1)
        y = int(y*resize_ratio_y)
        h = min(int(h*resize_ratio_y), img.shape[0]-y-1)
        coordinates.append([x,y,w,h])

    print("Selective search time %s s" % (time.time() - start_time))
 
    return coordinates


if __name__ == "__main__":
    main()
