import torch

# 배치 데이터
batch = torch.randn(10, 5)
print(f'배치 데이터:\n {batch}')
print(f'배치 데이터 크기:\n {batch.shape}')

# 1. 모든 샘플에 벡터 [1, 2, 3, 4, 5] 더하기
vector = torch.tensor([1, 2, 3, 4, 5],dtype=torch.float32)
result1 = batch + vector
print(f'모든 샘플에 벡터 [1, 2, 3, 4, 5] 더하기:\n {result1}')

# 2. 각 특성(열)의 평균을 구하고, 각 샘플에서 해당 평균 빼기
col_mean = batch.mean(dim=0)
centered = batch - col_mean
print(f'각 특성(열)의 평균을 구하고, 각 샘플에서 해당 평균 빼기:\n {centered}')

# 3. 각 특성의 최솟값과 최댓값을 구하고, 0~1 범위로 정규화
# 공식 : (x - min) / (max - min)
min_vals = batch.min(dim=0).values
max_vals = batch.max(dim=0).values
normalized = (batch - min_vals) / (max_vals - min_vals)
print(f'최솟값: {min_vals}')
print(f'최댓값: {max_vals}')
print(f'각 특성의 최솟값과 최댓값을 구하고, 0~1 범위로 정규화:\n {normalized}')

# 실습
x = torch.tensor(3.0, requires_grad=True)

# y = (2x + 1)³
# 단계별로 분해해보자
# y = h(g(x))

# g(x) = 2x + 1, dg/dx = 2
# h(g) = g³, dh/dg = 3g²

# dy/dx 
#   = dh/dg * dg/dx 
#   = 3g² * 2 
#   = 3(2x + 1)² * 2
#   = 6(2x + 1)²

g = 2*x + 1
y = g ** 3

y.backward()
print(f'x = {x.item()}')
print(f'g = 2x + 1 = {g.item()}')
print(f'y = g³ = {y.item()}')
print(f'dy/dx = 6(2x + 1)² = {x.grad.item()}')
print(f'검증 6 * (2 * 3 + 1)² = {6 * ((2 * 3 + 1) ** 2)}')

