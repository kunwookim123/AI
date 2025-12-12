import cv2

##### 1. 이미지를 불러서 창으로 띄우는 작업

### 1-1 imread(이미지경로): 이미지 반환(numpy)
img = cv2.imread("image/bubble.jpg")
img2 = cv2.imread("image/bubble.jpg", cv2.IMREAD_GRAYSCALE)

# print("img:", img) # numpy 배열 출력

# 1-2 imshow(윈도우이름, 불러올이미지)
cv2.imshow("bubble", img)
cv2.imshow("bubble gray", img2)

# 1-3 waitkey(시간(밀리초)) : ASCII CODE로 반환
key = cv2.waitKey(0) # 무한대기

changeTOChar = chr(key)
changeToASCII = ord(changeTOChar)
print(f"문자: {changeTOChar}, ASCII CODE: {changeToASCII}")
print("key", key) # 사용자가 입력한 키의 ASCII CODE를 출력

cv2.destroyAllWindows()

##### 2. 이미지 저장, shape 속성 확인
gray_heather = cv2.imread("image/heather.jpg", cv2.IMREAD_GRAYSCALE)
color_winter = cv2.imread("image/winter.jpg", cv2.IMREAD_COLOR)

### 2-1 shape 속성: 세로, 가로, (채널값)을 튜플형태로 반환
# 채널은 grayscale 사진일 경우는 값이 없음
print("gray heather image shape", gray_heather.shape) # (427, 640)
print("color heather image shape", color_winter.shape) # (427, 640, 3)

cv2.imshow("cute cat gray", gray_heather)
cv2.imshow("cute cat color", color_winter)

h1, w1 = gray_heather.shape
print("gray height", h1)
print("gray width", w1)

h2, w2, c2 = color_winter.shape
print("color height", h2)
print("color width", w2)
print("color channel", c2)

h3, w3 = color_winter.shape[:2]
print("color height3", h3)
print("color width3", w3)

cv2.waitKey(0)
cv2.destroyAllWindows()

## imwrite("저장할 경로명", 저장할이미지)
cv2.imwrite("output/gray_heather.png", gray_heather)