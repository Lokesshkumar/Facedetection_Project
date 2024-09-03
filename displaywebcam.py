#read frame to frame from the videocam

import cv2

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

while True:

    ret,frame = cap.read()

    if ret==0:
        continue

    cv2.imshow('frame',frame)

    key_pressed = cv2.waitKey(1) & 0xFF
    
    if key_pressed==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()