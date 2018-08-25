import cv2

vidcap = cv2.VideoCapture(0) # 0 is the camera index, modifying it changes which camera is used for video capture

while True:
    cv2.imshow("Window Name", vidcap.read()[1]) # the first returned variable indicates success or failure
    cv2.waitKey(20) # display frames at 1000 // 20 == 50 FPS

cv2.destroyAllWindows()
