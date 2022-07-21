import numpy as np
import cv2
face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
video = cv2.VideoCapture(0)

while(True):
    ret,frame=video.read()
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    roi=face.detectMultiScale(frame,1.3,5)
    for(x,y,w,h) in roi:
        frame[y:y+h,x:x+w]=cv2.medianBlur(frame[y:y+h,x:x+w],45)
    cv2.imshow("window",frame)
    key=cv2.waitKey(30)
    if(key==27):
        break
video.release()
cv2.destroyAllWindows()
//second commit

