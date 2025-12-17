# import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score

# 광고비 = np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)
# 매출 = np.array([3,5,6,8,11,13,14,16,17,20])

# # 선형 회귀 모델 학습
# model = LinearRegression()
# model.fit(광고비, 매출)

# # 기울기, 절편 확인
# 기울기 = model.coef_[0]
# 절편 = model.intercept_

# print(f'기울기: {기울기:.3f}')
# print(f'절편: {절편:.3f}')

# # 광고비가 15백만원일 때 예상 매출
# x = np.array([[15]])
# y = model.predict(x)

# print(f'{y[0]:.2f}')

# # R² Score 계산
# z = model.predict(광고비)
# r2 = r2_score(매출, z)

# print(f'{r2:.3f}')

# R² Score가 1에 가까우면 어떤 의미를 지니는지
# 모델이 데이터를 잘 설명한다는 의미
# R² = 1 완벽한 예측
# R² = 0 평균으로 예측하는 것과 동일
# R² < 0 평균보다 못한 예측

# R² = 0.95면 모델이 데이터 변동의 95%를 설명

# 비전공자에게 나온 값들이 어떤 의미를 지니는지 어떻게 설명해야할까

# 예) 한식 뷔페 발주
# 나쁜 예시
# 요일, 메뉴, 날씨
# r2 = 0.81이고 MSE 23
# 다음주 요일 예측 방문 138명 입니다.

# 핵심 메시지(한 문장)
# 다음주 화요일은 평균보다 손님이 15 ~ 20% 많을 가능성이 높아서 재료를 기준 대비 10% 추가 발주가 안전합니다.

# 영양사
# 고기류 메뉴가 있을 때 방문객이 평균 18% 증가하는 경향을 보임
# 이번 주 메뉴 구성 기준으로 보면 고기 반찬 기준량을 1.2배 준비하는게 적절하다.

# 조리 담당
# 피크 시간대 12:10 ~ 12:40에 집중될 확률이 높아서 이 시간 전에 주력 메뉴 1차 준비를 끝내는게 좋을 것 같다.



# 예)태양광 발전소
# 나쁜 예
# 태양광 발전량 예측 모델의 MAE 3.2mwh이고 정확도 87% 입니다.

# 운영담당자
# 내일 오후 1시 부터 오후 4시까지는 발전량이 평소 대비 30% 낮을 가능성이 높아서 ESS방전 또는 외부 전력 보완이 필요하다.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# 1. 데이터 로드 및 준비
housing = fetch_california_housing()
x = housing.data
y = housing.target
feature_names = housing.feature_names

# 2. 학습/테스트 데이터 분할 (8:2 비율)
# random_state를 지정하여 결과의 재현성을 보장합니다.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 3. 데이터 스케일링 (Standardization)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 4. 다중 선형 회귀 모델 학습
model = LinearRegression()
model.fit(x_train_scaled, y_train)

# 5. 예측 및 R² Score 확인
y_pred = model.predict(x_test_scaled)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f'R² Score (결정계수): {r2:.4f}')
print(f'MSE (평균제곱오차): {mse:.4f}')

# 6. 각 특성의 중요도(회귀 계수) 분석
# 선형 회귀에서 계수(Coefficients)의 절대값이 클수록 타겟 변수에 미치는 영향이 큽니다.
coeff = pd.Series(data=model.coef_, index=feature_names).sort_values(ascending=False)

# 시각화
plt.figure(figsize=(10, 6))
sns.barplot(x=coeff.values, y=coeff.index, palette='viridis')
plt.title('Feature Importance (Regression Coefficients)')
plt.xlabel('Coefficient Value')
plt.ylabel('Features')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

print("\n--- 특성별 회귀 계수 상세 ---")
print(coeff)

# 학습/테스트 분할
# 다중 선형 회귀 모델 학습
# R² Score 확인
# 각 특성의 중요도 분석
