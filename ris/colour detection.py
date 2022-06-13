import cv2
import numpy as np
def empty(a):
    pass
print("package imported")
# img = cv2.imread("CVtask.jpg')
# cv2.imshow('task',img)
# cv2.waitKey(0)
# img = cv2.imread("CVtask.jpg")
# cv2.imshow("task",img)
# cv2.waitKey(0)
#read image 
img = cv2.imread('C:/Users/manis/OneDrive/Desktop/ris/CVtask.jpg')
print('original dimensions : ',img.shape)
scale_percent=65#percentage of original size
width = int(img.shape[1]*scale_percent/100)
height= int(img.shape[0]*scale_percent/100)
dim=(width,height)
#resize image
resized = cv2.resize(img,dim,interpolation =cv2.INTER_AREA)
print('resized dimensions : ',resized.shape)
#show image
cv2.imshow('task',img)
cv2.waitKey(0) # waits until a key is pressed
#cv2.destroyAllWindows() # destroys the window showing image
#showing resized image
cv2.imshow('resized image',resized)
cv2.waitKey(0)
#cv2.destroyAllWindows()
# to convert it to grey scale
imgGray =cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Image",imgGray)
cv2.waitKey(0) 
#cv2.destroyAllWindows()
resizehsv=cv2.cvtColor(resized,cv2.COLOR_BGR2HSV)
cv2.imshow('hsv image',resizehsv)
cv2.waitKey(0)
#cv2.destroyAllWindows()
cv2.namedWindow("trackbars")
cv2.resizeWindow("trackbars",640,240)
cv2.createTrackbar("hue min","trackbars",0,179,empty)
cv2.createTrackbar("hue max","trackbars",179,179,empty)
cv2.createTrackbar("sat min","trackbars",0,255,empty)
cv2.createTrackbar("sat max","trackbars",255,255,empty)
cv2.createTrackbar("val min","trackbars",0,255,empty)
cv2.createTrackbar("val max","trackbars",255,255,empty)
cv2.waitKey(0)
while True:
     h_min=cv2.getTrackbarPos("hue min","trackbars")
     h_max=cv2.getTrackbarPos("hue max","trackbars")
     s_min=cv2.getTrackbarPos("sat min","trackbars")
     s_max=cv2.getTrackbarPos("sat max","trackbars")
     v_min=cv2.getTrackbarPos("val min","trackbars")
     v_max=cv2.getTrackbarPos("val max","trackbars")

     print(h_min,h_max,s_min,s_max,v_min,v_max)
     lower=np.array([h_min,s_min,v_min])
     upper=np.array([h_max,s_max,v_max])
     mask = cv2.inRange(resizehsv,lower,upper)
     cv2.imshow("mask",mask)
     cv2.waitKey(1)
     #cv2.destroyAllWindows()