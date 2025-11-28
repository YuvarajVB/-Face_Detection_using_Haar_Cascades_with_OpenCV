# Face Detection using Haar Cascades with OpenCV and Matplotlib

## Aim

To write a Python program using OpenCV to perform the following image manipulations:  
i) Extract ROI from an image.  
ii) Perform face detection using Haar Cascades in static images.  
iii) Perform eye detection in images.  
iv) Perform face detection with label in real-time video from webcam.

## Software Required

- Anaconda - Python 3.7 or above  
- OpenCV library (`opencv-python`)  
- Matplotlib library (`matplotlib`)  
- Jupyter Notebook or any Python IDE (e.g., VS Code, PyCharm)

## Algorithm

### I) Load and Display Images

- Step 1: Import necessary packages: `numpy`, `cv2`, `matplotlib.pyplot`  
- Step 2: Load grayscale images using `cv2.imread()` with flag `0`  
- Step 3: Display images using `plt.imshow()` with `cmap='gray'`

### II) Load Haar Cascade Classifiers

- Step 1: Load face and eye cascade XML files 
### III) Perform Face Detection in Images

- Step 1: Define a function `detect_face()` that copies the input image  
- Step 2: Use `face_cascade.detectMultiScale()` to detect faces  
- Step 3: Draw white rectangles around detected faces with thickness 10  
- Step 4: Return the processed image with rectangles  

### IV) Perform Eye Detection in Images

- Step 1: Define a function `detect_eyes()` that copies the input image  
- Step 2: Use `eye_cascade.detectMultiScale()` to detect eyes  
- Step 3: Draw white rectangles around detected eyes with thickness 10  
- Step 4: Return the processed image with rectangles  

### V) Display Detection Results on Images

- Step 1: Call `detect_face()` or `detect_eyes()` on loaded images  
- Step 2: Use `plt.imshow()` with `cmap='gray'` to display images with detected regions highlighted  

### VI) Perform Face Detection on Real-Time Webcam Video

- Step 1: Capture video from webcam using `cv2.VideoCapture(0)`  
- Step 2: Loop to continuously read frames from webcam  
- Step 3: Apply `detect_face()` function on each frame  
- Step 4: Display the video frame with rectangles around detected faces  
- Step 5: Exit loop and close windows when ESC key (key code 27) is pressed  
- Step 6: Release video capture and destroy all OpenCV windows

## Program:

### I) Load and Display Images
```py

import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread("image_01.png",0)
img2 = cv2.imread("image_02.png",0)
img3 = cv2.imread("image_03.png",0)

plt.figure(figsize=[20,20])
plt.subplot(131);plt.imshow(img1,cmap='gray');plt.title("Without Glass")
plt.subplot(132);plt.imshow(img2,cmap='gray');plt.title("With sunglass")
plt.subplot(133);plt.imshow(img3,cmap='gray');plt.title("Group Photo")
plt.show()

img1_resized = cv2.resize(img1,(1000,1000))
img2_resized = cv2.resize(img2,(1000,1000))
img3_resized = cv2.resize(img3,(1000,1000))

plt.figure(figsize=[20,20])
plt.subplot(131);plt.imshow(img1_resized,cmap='gray');plt.title("Without sunglass")
plt.subplot(132);plt.imshow(img2_resized,cmap='gray');plt.title("With Glass")
plt.subplot(133);plt.imshow(img3_resized,cmap='gray');plt.title("Group Photo")
plt.show()

```

### II) Load Haar Cascade Classifiers
```py
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
```

### III) Perform Face Detection in Images
```py

def detect_face(img):

    face_img = img.copy()

    face_rects = face_cascade.detectMultiScale(face_img, scaleFactor=1.1, minNeighbors=5)

    for (x,y,w,h) in face_rects:
        cv2.rectangle(face_img, (x,y), (x+w, y+h), (127,0,255), 10)
    return face_img


```

### IV) Perform Eye Detection in Images
```py
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def detect_eye(img):

    face_img = img.copy()

    face_rects = eye_cascade.detectMultiScale(face_img, scaleFactor=1.1, minNeighbors=5) # used to find out the location for face

    for (x,y,w,h) in face_rects:
        cv2.rectangle(face_img, (x,y), (x+w, y+h), (0,255,0), 2)
    return face_img

```

### V) Perform Face Detection on Real-Time Webcam Video
```py
cap = cv2.VideoCapture(0)

while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame")
            break
            
        result = detect_face(frame)
        cv2.imshow("Face Detection Through Webcam", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
plt.imshow(result[:,:,::-1])        
cap.release()
cv2.destroyAllWindows()

```

## Result
Face detection and eye detection has been successfully executed
