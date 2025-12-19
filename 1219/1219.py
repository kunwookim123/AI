# 결측치 삭제


import numpy as np
import pandas as pd


df = pd.DataFrame({
    'age': [25, 30, np.nan, 45, np.nan, 35],
    'income': [50000, np.nan, 60000, 80000, 55000, np.nan],
    'city': ['서울', '부산', np.nan, '대구', '서울', '인천']
})


# 방법2 채우기(Imputation) 가장 많이 사용
# 수치형 : 평균/중앙값/최빈값으로 채우기
df['age'].fillna(df['age'].median(), inplace=True) # 중앙값
df['income'].fillna(df['income'].mean(), inplace=True) # 평균

# 범주형 : 최빈값으로 채우기
df['city'].fillna(df['city'].mode()[0], inplace=True) # 최빈값
print(df.isnull().sum())

# 평균(Mean):
# 정규분포에 가까운 데이터
# 이상치가 많으면 부적합

# 중앙값(Median):
# 이상치에 강함
# 대부분의 수치형 데이터에 적합

# 최빈값(Mode):
# 범주형 데이터
# 가장 흔한 값으로 채우기

# 번주형 데이터 인코딩
# 문제 : 머신러닝 모델은 숫자만 이해
# '남성', '여성' -> 모델이 이해를 못함
# 0, 1 -> 이해함


# 방법 1. 수동 매핑(map)
df = pd.DataFrame({
    'sex': ['male', 'female', 'male', 'female']
})

# 수동으로 매핑
df['sex_encoding'] = df['sex'].map({'male': 0, 'female': 1})
print(df)

# 명확하고 제어가능
# 범주가 많으면 번거로움

# 방법 2. LabelEncoder
from sklearn.preprocessing import LabelEncoder

# 여러 범주가 있는 경우
df = pd.DataFrame({
    'city': ['서울', '부산', '대전', '대구', '서울', '인천']
})

le = LabelEncoder()
df['city_encoded'] = le.fit_transform(df['city'])

print(df)
# LabelEncoder는 순서를 부여
# 서울 = 3, 부산 = 2, 대전 = 1

# 문제 : 모델이 "서울" > "부산" > "대전"으로 오해할 가능성 있음
# 트리 기반 모델(랜덤 포레스트 등)은 괜찮음
# 회귀 모델은 One-Hot Encoding 권장

# One-Hot Encoding
# 순서가 없는 범주형 데이터
df = pd.DataFrame({
    'city': ['서울', '부산', '대전']
})

df_encoded = pd.get_dummies(df, columns=['city'])
print(df_encoded)


# 하이퍼파라미터 튜닝
# 하이퍼파라미터
# 학습 전에 사람이 정해주는 설정값
# 예 : 랜덤 포레스트의 트리 개수, 최대 깊이 등

# 파라미터 vs 하이퍼파라미터
# 파라미터 : 모델이 학습으로 찾는 값(가중치)
# 하이퍼파라미터 : 사람이 설정하는 값

# 방법 1 : 수동
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)

mode2 = RandomForestClassifier(n_estimators=200)

mode3 = RandomForestClassifier(n_estimators=300)
# -> 너무 오래 걸림

# 방법 2 : GridSearchCv 자동으로 최적값 찾기

from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# 데이터 로드
iris = load_iris()
X, y = iris.data, iris.target

# 시도할 하이퍼파라미터 조합
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5]
}
# 총 3 x 3 x 2 = 18가지 조합

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    model,              # 모델
    param_grid,         # 시도할 파라미터
    cv=5,               # 5-fold 교차 검증
    scoring='accuracy', # 평가 지표
    n_jobs=1            # 모든 CPU 사용
)

# 자동으로 모든 조합 시도
grid_search.fit(X, y)

# 결과 확인
print(f'최적의 파라미터: {grid_search.best_params_}')
print(f'최고 점수: {grid_search.best_score_:.2%}')
print(f'최적 모델: {grid_search.best_estimator_}')


# GridSearchCv가 하는 일:
# 1. param_grid의 모든 조합 생성
# 2. 각 조합 마다:
# 5-fold 교차 검증 수행
# 평균 점수 계산

# 3. 가장 높은 점수의 조합 선택
# 4. 전체 데이터로 최적 모델 재학습


# 주의 사항
# 장점 : 자동으로 최적값 찾기
# 단점 : 시간이 오래 걸림

# 팁
# 처음엔 넓은 범위로 탐색
# 그 다음 좁은 범위로 세밀하게 조정


# 누가 생존했는가? 예측하는 분류 모델을 처음부터 끝까지 만들어보기
# 1단계 : 데이터 탐색
# 데이터 불러오기
# 결측치가 어디에 얼마나 있는지 확인
# 생존/사망 비율 파악
# 성별, 객실 등급별 생존을 시각화
# 특성 간 상관관계 히트맵 그리기

# 2단계 : 데이터 전처리
# 필요한 특성만 선택
# 결측치 채우기
# 범주형 데이터를 숫자로 변환
# 훈련/테스트 세트 분할

# 3단계 : 모델 학습 및 비교
# 3가지 모델 훈련 : 로지스틱 회귀, 결정 트리, 랜덤 포레스트
# 교차 검증으로 성능 비교
# GridSearchCv로 랜덤 포레스트 하이퍼파라미터 튜닝

# 4단계 : 평가
# 테스트 세트로 최종 정확도 측정
# 혼동 행렬이 어디서 틀렸는지 확인
# 특성 중요도 확인(어떤 특성이 생존 예측에 중요했나?)
