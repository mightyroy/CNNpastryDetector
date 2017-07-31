import sys
sys.path.append('/usr/local/Cellar/opencv/2.4.11_1/lib/python2.7/site-packages')
import cv2
import roynet
import selectsearch
import numpy as np
import fastNMS
import time
import thresholding

labels = ["bananabread", "cinnamonroll", "croissant", "hotcross"]
prices = [1,1.50,2,1.20]

video_capture = cv2.VideoCapture(0)

#initiate roynet
r = roynet.RoyNet()

while True:

    totalPrice = 0
    nmsboxes = []


    # Capture frame-by-frame
    ret, frame = video_capture.read()

    #coordinates = selectsearch.process(frame)
    coordinates, thresholdImg = thresholding.processImage(frame)


    for coord in coordinates:
        x , y, w, h = coord

        croppedimg = frame[y:(y+h),x:(x+w)]

        resizedimg = cv2.resize(croppedimg,(448,448))

        #neuralnet prediction
        start_time = time.time()
        scores = r.predict(resizedimg)
        #print scores
        print("Neural net time %s s" % (time.time() - start_time))

        winnerArg = np.argmax(scores)
        
        #boxes with good probability of prediction to be processed by Non maximal suppresion
        if scores[0,winnerArg] > 0.9:

            nmsboxes.append([x,y,x+w,y+h, winnerArg])

         
    #nmsboxes = fastNMS.non_max_suppression_fast(np.asarray(nmsboxes), 0.8 ) 
    
    for box in nmsboxes:
        x1,y1,x2,y2,winnerArg = box
        #print x2-x1, y2-y1
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame,labels[winnerArg], (x1,y1+50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255),3)
        cv2.putText(frame,'$' + str(prices[winnerArg]), (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255),3)
        totalPrice += prices[winnerArg]

    cv2.putText(frame,'Total Bill: $' + str(totalPrice), (20,80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255),3)

   

    # Display the resulting frame
    frame = cv2.resize(frame,(600,400))
    cv2.imshow('Machine Cashier', frame)
    thresholdImg = cv2.resize(thresholdImg,(600,400))
    cv2.moveWindow('thresholding ROI',600, 10)
    cv2.imshow('thresholding ROI', thresholdImg)

    # if cv2.waitKey(0):
    #     break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
r.shutdown()