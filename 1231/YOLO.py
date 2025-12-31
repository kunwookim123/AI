from ultralytics import YOLO
import cv2

model = YOLO('yolo11n.pt')

img = cv2.imread('1231/image1.jpg')

results = model.predict(img, conf=0.5)

annotated_frame = results[0].plot()


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("카메라를 열 수 없습니다.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 추론
    results = model.predict(frame, conf=0.5)

    # 바운딩 박스 + 라벨 그린 프레임
    annotated_frame = results[0].plot()
    
    cv2.imshow("YOLO Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()