# 다중 선형 회귀
# 단순 선형 회귀
# y = wx + b
# 입력(x)이 1개
# 예) 면적 -> 집값


# 다중 선형 회귀
# y = w1x1 + w2x2 + w3x3 + ... + wnxn + b
# 입력(x)이 여러 개
# 예) 면적, 방 갯수, 역까지 거리 .. -> 집값

# 행렬 표현
# 벡터 표기:
# y = wx + b = wtx + b

# 행렬 표기(여러 데이터)
# Y = XW + b

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# 집값 데이터 생성
np.random.seed(42)
n_sample = 200

data = {
    "면적": np.random.randint(15, 50, n_sample),
    "방수": np.random.randint(1, 5, n_sample),
    "역거리": np.random.randint(0.1, 2, n_sample),
    "층수": np.random.randint(1, 25, n_sample),
    "건축년도": np.random.randint(1990, 2023, n_sample),
}

# 실제 관계: 집값 = 0.15x면적 + 0.5x방수 - 0.3x역거리 + 0.02x층수
집값 = (
    0.15 * data['면적'] +
    0.5 * data["방수"] +
    0.3 * data["역거리"] +
    0.02 * data["층수"]+
    2 +
    np.random.randn(n_sample) * 0.5
    )


df = pd.DataFrame(data)
df['집값']=집값

print(df.head())
print(df.describe())

# 모델 학습

# 특성과 타겟 분리
x = df[['면적','방수','역거리','층수','건축년도']]
y = df['집값']

# 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(
    x,y,test_size=0.2, random_state=42
)

# 모델 학습
model = LinearRegression()
model.fit(x_train, y_train)

# 결과 확인
print("===학습된 개수===")
for feature, coef in zip(x.columns, model.coef_):
    print(f'{feature}: {coef:.4f}')
print(f'절편: {model.intercept_:.4f}')

# 평가
# 예측
y_pred = model.predict(x_test)

# 평가 지표
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'R2 Score: {r2:.4f}')
print(f'RMSE: {rmse:.4f}억 원')

coef_df = pd.DataFrame({
    "특성": x.columns,
    '계수': model.coef_
})

coef_df = coef_df.sort_values('계수', key=abs, ascending=False)
print(coef_df)

from sklearn.datasets import fetch_california_housing


# 데이터 준비
housing = fetch_california_housing()

x = housing.data
y = housing.target
feature_names = housing.feature_names

# 데이터 확인
df = pd.DataFrame(x, columns=feature_names)
df['Target'] = y

print('데이터 크기:', x.shape)
print(df.describe())

# x.스케일링
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 학습/테스트 분할
# 다중 선형 회귀 모델 학습
# R² Score 확인
# 각 특성의 중요도 분석


