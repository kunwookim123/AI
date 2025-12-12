import cv2

# 영상을 윈도우로 띄우기
cap = cv2.VideoCapture("video/dog.mp4")

# cap.get(int 속성의 ID)
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # 1280.0
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 720.0
print(cap.get(cv2.CAP_PROP_FPS)) # 25.0
print(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # 411.0
print(cap.get(cv2.CAP_PROP_POS_FRAMES)) # 0.0

cap.set(cv2.CAP_PROP_POS_FRAMES, 305) # 비디오 시작점 제어 가능
# cap.isOpened(): 정상적으로 파일이 열렸는지, 카메라 사용시 카메라가 연결됐는지 확인
# cap.read(): ret, frame 값을 반환
# ret: bool 값, 프레임을 정상적으로 읽었는지 반환
# frame: numpy 값, 읽어온 영상의 프레임 하나, ret가 false라면 none
# frame은 image 데이터이기 때문에 cv2.imshow()를 통해 화면에 보여줄 수 있음

# cap.release(): 영상 재생이 끝나고 메모리, 카메라 점유등 자원을 반납

if cap.isOpened():
    while True:
        ret, frame = cap.read()

        if not ret: # ret가 false라면 영상을 모두 재생시킨 것
            print("불러올 영상이 없습니다.")
            break

        cv2.imshow("dog video", frame)
        # waitKey의 숫자에 따라 영상 길이 조절 가능
        # cv2.CAP_PROP_FPS가 인자로 들어가면 원본 영상의 속도
        if cv2.waitKey(25) & 0xFF==ord('q'):
            break
else:
    print("비디오 파일 열기 불가능")

cap.release()
cv2.destroyAllWindows()