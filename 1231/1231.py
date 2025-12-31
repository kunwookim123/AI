import cv2

# 모델 로드
face_cascade = cv2.CascadeClassifier("1231/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("1231/haarcascade_eye.xml")

# 이미지 로드
img = cv2.imread("image/image2.jpg") # image2, image3도 동일하게 테스트
if img is None:
    print("이미지를 불러올 수 없습니다. 경로를 확인하세요.")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 1. 얼굴 탐지
faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

for (x, y, w, h) in faces:
    # 얼굴에 사각형 그리기
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # 2. 눈 탐지: 얼굴 영역(ROI) 안에서만 찾기
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    
    # 눈은 조금 더 유연하게 탐지 (minNeighbors를 3~4로 낮춤)
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3, minSize=(60, 60))
    
    for (ex, ey, ew, eh) in eyes:
        # 얼굴 영역 내 좌표이므로 roi_color에 그리기
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

cv2.imshow("Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()