#write a python script that captures images from webcam video stream
#extracts all faces from the image frame(using haarcascades)
#stores the face information into numpy arrays

# 1.Read and show the video stream, capture images
# 2.Detect faces and show bounding box(haarcascade)
# 3.Flatten the largest face image(gray scale image) and save it in numpy array
# 4.Repeat the above for multiple people to generate training data

import cv2
import numpy as np

#init camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
face_data = []
dataset_path = './data/'

file_name = input("Enter the name of the person : ")
while True:
    ret,frame = cap.read()

    if ret==False:
        continue

    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(frame,1.3,5)
    faces = sorted(faces,key =lambda f:f[2]*f[3])
    face_section=frame
    
    #pick the last face (because it is the largest face acc to area(f[2]))
    for face in faces[-1:]:
        x,y,w,h = face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

        #extract (crop out the required face) : Region of interest
        offset = 10
        face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))
        
        #store every 10th face
        skip +=1
        if skip%10==0:
            face_data.append(face_section)
            print(len(face_data))


    cv2.imshow('Frame',frame)        
    cv2.imshow("face section",face_section)

    
    key_pressed = cv2.waitKey(10) & 0xFF
    if key_pressed == ord('q'):
        break


#convert our face list array into a numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

#save this data into file system
np.save(dataset_path+file_name+'.npy',face_data)
print("Data Succesfully save at "+dataset_path+file_name+'.npy')

cap.release()
cv2.destroyAllWindows()