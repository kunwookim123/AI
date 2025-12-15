import numpy as np

# 행렬
# 행렬(matrix) - 숫자들을 행과 열로 배열한 것

# 벡터 : 1차원 배열 [1,2,3]
# 행렬 : 2차원 배열 (벡터들의 모음)

# 텐서 : 3차원 이상의 배열



# 일상에서의 행렬
# 성적표
#           국어     영어     수학
# 학생1     [ 85      90      78]
# 학생2     [ 92      88      95]
# 학생3     [ 78      82      88]

# 이미지
# 28 x 28 픽셀 이미지 = 28행 x 28열 행렬
# 각 셀의 값 = 픽셀 밝기 (0 ~ 255)

matrix = np.array([
    [1,2,3],
    [4,5,6],
])

print(matrix)

zeros = np.zeros((3,4))
print(zeros)

# 행렬 형태(행, 열)
print(matrix.shape)
# 전체 원소 개수
print(matrix.size)
# 차원
print(matrix.ndim)

A = np.array([[1,2], [3,4]])
B = np.array([[5,6], [7,8]])

print(A + B)
print(A - B)
print(A * B)

# 행렬 곱셈(중요!)

# 행렬 곱셈 != 원소별 곱셈
# A x B의 규칙
# A의 열 개수 = B의 행 개수 => 계산 가능
# (2x3) x (3x2) = (2x2)
# (2x3) x (2x3) = 계산 불가

# A = [1 2]
#     [3 4]
# B = [5 6]
#     [7 8]

# A x B = [1x5 + 2x7 1x6 + 2x8] = [19 22]
#         [3x5 + 4x7 3x6 + 4x8]   [43 50]

A = np.array([[1,2], [3,4]])
B = np.array([[5,6], [7,8]])

# 방법 1: @ 연산자 사용(권장)
result = A @ B
print(result)

# 방법 2: np.dot() 함수 사용
result = np.dot(A, B)
print(result)

# 방법 3: np.matmul() 함수 사용
result = np.matmul(A, B)
print(result)


# 행렬 곱셈 주의사항
A = np.array([[1,2,3], [4,5,6]])
B = np.array([[1,2], [3,4], [5,6]])

# A @ B
print((A @ B).shape)

# B @ A
print((B @ A).shape)

# A @ B != B @ A
# 순서가 중요!

# 행렬의 전치와 형태 변환
# 전치(transpose)
# 행과 열을 바꿈
A = np.array([[1,2,3], [4,5,6]]) # 2x3

# 전치 행렬
A_T = A.T
# A_T = A.transpose()

print(A_T)


a = np.array([1,2,3,4,5,6])

b = a.reshape(3,2)
print(b)

d = a.reshape(2,-1) # 2행, 열은 자동
print(d)

# AI에서 행렬의 역할
# 신경망에서의 행렬
# 신경망의 핵심 = 행렬의 곱셈

# 입력 데이터(벡터) x 가중치(행렬) = 출력
# 입력: [x1, x2, x3]
# 가중치: [
#   [w11, w12]
#   [w21, w22]
#   [w31, w32]
# ]

# 출력 = 입력 @ 가중치 = [y1, y2]


# 이미지 데이터
# 이미지 = 행렬(또는 3차원 텐서)

# 흑백 이미지: 2차원 행렬

# 컬러 이미지: 3차원 텐서(높이, 너비, 채널)
