import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. 데이터 로드
df = pd.read_csv('tested.csv')

# 2. 필요한 특성만 선택 및 결측치 처리
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
df = df[features + ['Survived']]
df = df.dropna()

# 3. 성별을 숫자로 변환
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# 4. 특성(X)과 타겟(y) 분리
X = df[features]
y = df['Survived']

# 5. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. 모델 학습
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 7. 평가
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"모델 정확도: {accuracy:.2%}")

# 8. 시각화
plt.figure(figsize=(10, 10))
plot_tree(model, 
          feature_names=features, 
          class_names=['perished', 'survived'],
          filled=True, 
          rounded=True)
plt.show()