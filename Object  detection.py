import cv2 #importing opencv library
import time #import time lib.
import imutils #import imutils

cam = cv2.VideoCapture(0) #initialize camera
time.sleep(1) #giving 1 second delay

firstFrame=None #make first frame is nothing
area = 50
#threshold for how much change can be noticed in moving object

while True: #infinite loop
	_,img = cam.read() #reading frame from the camera
	text = "Normal" #no moving object detection
	#pre-processing
	img = imutils.resize(img, width=500) #resize the frame to 500 w
	grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert color to gray scale image
	gaussianImg = cv2.GaussianBlur(grayImg, (21, 21), 0) #smoothening

	#save the first frame, into the firstframe variable
	#from the 2nd iteration it wont go inside this if condition
	if firstFrame is None:
			firstFrame = gaussianImg
			continue
	
	imgDiff = cv2.absdiff(firstFrame, gaussianImg) #difference b/w first bg frame with current frame
	threshImg = cv2.threshold(imgDiff, 25, 255, cv2.THRESH_BINARY)[1] #detected region will be converted into binary
	threshImg = cv2.dilate(threshImg, None, iterations=2)
	cnts = cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	for c in cnts:
			if cv2.contourArea(c) < area:
					continue
			(x, y, w, h) = cv2.boundingRect(c)
			cv2.rectangle(img, (x, y), (x + w, y + h), (0, 200, 0), 2)
			text = "Moving Object detected"
	print(text)
	cv2.putText(img, text, (10, 20),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)
	cv2.imshow("cameraFeed",img)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

cam.release()
cv2.destroyAllWindows()
