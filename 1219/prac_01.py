import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# 1단계 : 데이터 탐색

df = pd.read_csv('Titanic.csv')

# 결측치 확인
print("--- [결측치 확인] ---")
print(df.isnull().sum())

# 생존/사망 비율 파악
print("\n--- [생존/사망 비율] ---")
print(df['Survived'].value_counts(normalize=True))

# 성별, 객실 등급별 생존 시각화
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.barplot(x='Sex', y='Survived', data=df, errorbar=None)
plt.title("Survival Rate by Sex")

plt.subplot(1, 2, 2)
sns.barplot(x='Pclass', y='Survived', data=df, errorbar=None)
plt.title("Survival Rate by Pclass")
plt.show()

# 특성 간 상관관계 히트맵 (성별 포함)
temp_df = df.copy()
le_temp = LabelEncoder()
temp_df['Sex'] = le_temp.fit_transform(temp_df['Sex']) # 성별을 숫자로 임시 변환

plt.figure(figsize=(10, 8))
numeric_df = temp_df.select_dtypes(include=['number']) # 이제 Sex도 포함됨
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap (Including Sex)")
plt.show()

# 2단계 : 데이터 전처리

# 핵심 특성 선택
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
X = df[features].copy()
y = df['Survived']

# 결측치 채우기 (Age, Fare)
X['Age'] = X['Age'].fillna(X['Age'].median())
X['Fare'] = X['Fare'].fillna(X['Fare'].median())

# 범주형 데이터를 숫자로 변환 (Sex)
le = LabelEncoder()
X['Sex'] = le.fit_transform(X['Sex'])

# 훈련/테스트 세트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 로지스틱 회귀용 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3단계 : 모델 학습 및 비교

print("\n--- [모델별 교차 검증 점수] ---")

# 로지스틱 회귀 (스케일링 데이터)
lr = LogisticRegression(random_state=42)
lr_scores = cross_val_score(lr, X_train_scaled, y_train, cv=5)
print(f"Logistic Regression: {lr_scores.mean():.4f}")

# 결정 트리
dt = DecisionTreeClassifier(random_state=42)
dt_scores = cross_val_score(dt, X_train, y_train, cv=5)
print(f"Decision Tree: {dt_scores.mean():.4f}")

# 랜덤 포레스트
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
print(f"Random Forest (Best CV): {grid_search.best_score_:.4f}")

# 4단계 : 평가

y_pred = best_rf.predict(X_test)
print("\n--- [최종 모델(RF) 평가 결과] ---")
print(f"Final Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# 혼동 행렬
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (Final Model)")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 특성 중요도
plt.figure(figsize=(8, 6))
importances = pd.Series(best_rf.feature_importances_, index=features)
importances.sort_values().plot(kind='barh', color='skyblue')
plt.title("Feature Importance")
plt.show()