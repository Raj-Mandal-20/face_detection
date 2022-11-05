# print('Hello world')
# for i in range(1 ,10):
#     print(i)
import cv2
from random import randrange

trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')
# For detecting images
# img = cv2.imread('./friends.jpeg')

webcam = cv2.VideoCapture(0)

while True:

    successful_frame, frame = webcam.read()
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        
        
    face_coords = trained_face_data.detectMultiScale(grayscaled_img)
    print(face_coords)

    for (x, y, w, h) in face_coords:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(255),
                    randrange(255), randrange(255)), 7)
        
        
    cv2.imshow('AI WEBCAM', frame)
    cv2.waitKey(1)

# face_coords = trained_face_data.detectMultiScale(grayscaled_img)
# print(face_coords)

# for (x, y, w, h) in face_coords:
#     cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(255),
#                   randrange(255), randrange(255)), 7)

# cv2.imshow('Claver Programmer Face Detector', img)
# cv2.waitKey()

# print(cv2.__version__)
