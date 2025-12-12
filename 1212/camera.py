import cv2

cap = cv2.VideoCapture(0)
if cap.isOpened():
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow("camera", frame)
            if cv2.waitKey(10) != -1:
                cv2.imwrite("output/camera-capture.jpg", frame)
                break

cap.release()
cv2.destroyAllWindows()