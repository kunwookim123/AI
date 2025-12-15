import numpy as np

# 벡터 연산
# a = np.array([2, 4, 6])
# b = np.array([1, 3, 5])
# 크기 = np.linalg.norm(a)
# 단위벡터 = a / np.linalg.norm(a)

# print("a + b:", a + b)
# print("a - b:", a - b)
# print("a * 3:", a * 3)
# print("a의 크기:", 크기)
# print("a의 단위벡터:", 단위벡터)
# print("단위벡터의 크기:", np.linalg.norm(단위벡터))

A = np.array([[1,2], [3,4], [5,6]]) # 3 x 2
B = np.array([[1,2,3], [4,5,6]]) # 2 x 3
A_T = A.T
C = np.arange(12)

print("A의 shape:", A.shape)
print("A의 전치행렬:", A_T)
print("A @ B:", A @ B)
print("B @ A:", B @ A)
print(C.reshape(3,4))


