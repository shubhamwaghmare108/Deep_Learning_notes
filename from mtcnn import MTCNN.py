from mtcnn import MTCNN
import cv2

detector = MTCNN()
x= cv2.VideoCapture(0)
ret, img = x.read()
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

faces = detector.detect_faces(img_rgb)

print(faces)