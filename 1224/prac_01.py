import torch

# # 1. 3x3 크기의 0으로 채워진 텐서
# zeros = torch.zeros(3, 3)
# print(f'영행렬:\n {zeros}')

# # 2. 2x4 크기의 1로 채워진 텐서
# ones = torch.ones(2, 4)
# print(f'일행렬:\n {ones}')

# # 3. 0부터9까지 들어있는 1차원 텐서
# numbers = torch.arange(0, 10, 1)
# print(f'0부터9까지 들어있는 1차원 텐서:\n {numbers}')

# # 4. 평균 0, 표준편차 1인 정규분포에서 샘플링한 5x5 텐서
# random_normal = torch.randn(5, 5)
# print(f'평균 0, 표준편차 1인 정규분포에서 샘플링한 5x5 텐서:\n {random_normal}')

# # 5. 3x3 단위행렬(대각선이 1인 행렬)
# identify = torch.eye(3)
# print(f'3x3 단위행렬:\n {identify}')

# x = torch.arange(24)
# print(f'원본:\n {x}')

# # 1. 2x12 형태로 변환
# shape_2_12 = x.reshape(2, 12)
# print(f'2x12 형태:\n {shape_2_12}')

# # 2. 3x8 형태로 변환
# shape_3_8 = shape_2_12.reshape(3, 8)
# print(f'3x8 형태:\n {shape_3_8}')

# # 3. 2x3x4 형태로 변환
# shape_2_3_4 = shape_3_8.reshape(2, 3, 4)
# print(f'2x3x4 형태:\n {shape_2_3_4}')

# # 4. 4x2x3 형태로 변환
# shape_4_2_3 = shape_2_3_4.reshape(4, 2, 3)
# print(f'4x2x3 형태:\n {shape_4_2_3}')

# # 5. 다시 1차원으로 펴기
# flattend = shape_4_2_3.reshape(24,)
# print(f'원본:\n {flattend}')

# 두 행렬
A = torch.tensor([
    [1,2,3],
    [4,5,6],
],dtype=torch.float32)

B = torch.tensor([
    [2,0,1],
    [1,3,2],
],dtype=torch.float32)

print(f'A:\n {A}')
print(f'B:\n {B}')

# 1. A와 B의 원소별 합
element_sum = A + B
print(f'A와 B의 원소별 합:\n {element_sum}')

# 2. A와 B의 원소별 곱
element_mul = A * B
print(f'A와 B의 원소별 곱:\n {element_mul}')

# 3. A의 모든 원소를 제곱
squared = A ** 2
print(f'A의 모든 원소를 제곱:\n {squared}')

# 4. A의 각 행의 합
row_sum = torch.sum(A, axis=1)
print(f'A의 각 행의 합:\n {row_sum}')

# 5. A의 각 열의 평균
col_mean = torch.mean(A, axis=0)
print(f'A의 각 열의 평균:\n {col_mean}')

# 6. A에서 3보다 큰 원소들 추출
greater_than_3 = A[A > 3]
print(f'A에서 3보다 큰 원소들 추출:\n {greater_than_3}')