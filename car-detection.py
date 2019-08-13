import numpy as np 
import cv2
import time
def main():
	
	car_classifier=cv2.CascadeClassifier("haarcascade_car.xml") #haarcascade_car file path
	cap=cv2.VideoCapture("file path")#full path of the video file

	while True:
		time.sleep(0.05)
		ret,frame=cap.read()
		frame=cv2.resize(frame,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_LINEAR)
		gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		cars=car_classifier.detectMultiScale(gray,1.2,2)

		for (x,y,w,h) in cars:
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
			cv2.imshow("Cars",frame)

		if cv2.waitKey(1)==27:
			break
	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
