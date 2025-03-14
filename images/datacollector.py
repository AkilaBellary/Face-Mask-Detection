import cv2
video = cv2.VideoCapture(0)

facedetect = cv2.CascadeClassifier(r'C:\Users\akila\OneDrive\Desktop\My learnings\Projects\Face Mask Detection\images\haarcascade_frontalface_default.xml')

count=0


while True:
    ret, frame = video.read()
    faces = facedetect.detectMultiScale(frame,1.3,5)
    for x,y,w,h in faces:
        count = count+1
        name = './images/face_without_mask/'+ str(count) + '.jpg'
        print("Creating Images.........."+name)
        cv2.imwrite(name, frame[y:y+h,x:x+w])
        cv2.rectangle(frame, (x,y), (x+w, y+h),(0,255,0),3)
    cv2.imshow("WindowFrame",frame)
    k = cv2.waitKey(1)
    if k== ord('q'):
        break
video.release()
cv2.destroyAllWindows()
