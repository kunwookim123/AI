import cv2 as cv

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1000)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1000)


if not cap.isOpened():
    print('연결할 카메라가 없습니다.')
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print('불러올 이미지가 없습니다.')
        break

    cv.imshow("my camera", frame)

    if cv.waitKey(5000) == ord("q"):
        print('사용자에 의해 종료되었습니다.')
        break

cap.release()
cv.destroyAllWindows()