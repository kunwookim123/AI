# 결정 트리
# 스무고개 게임

# 이 과일 뭔지 맞추기
# Q1. 빨간색인가요?
# Yes -> Q2. 작은가요?
            # Yes -> 체리
            # No -> 사과

# No -> Q3. 노란색인가요?

# 컴퓨터가 하는 일은 어떤 질문을 어떤 순서로 할지 데이터에서 자동으로 찾는 것

# 트리 구조 용어
#           [루트 노드]
#           빨간색인가요?
#           /          \
#          yes         no
#         /              \
#    [내부 노드]      [내부 노드]
#     작은가요?       노란색인가요?
#     /      \       /      \
#   yes      no    yes      no
#    |        |     |        |
#  [리프]   [리프] [리프]    [리프]
# 

# 깊이(Depth) 루트에서 리프까지 거치는 질문 수

# 좋은 질문 vs 나쁜 질문
# 핵심은 잘 나누는 질문을 찾는 것

# 데이터: 사과 10개, 오렌지 10개를 구문하고 싶음
# 나쁜 질문: "무게가 100g" 이상인가?

# 좋은 질문: "빨간색인가요?"

# 각 그룹은 순수 해지도록 나누기

# 순수도

# 지니 불순도(Gini Impurity)
# Gini = 1 - (각 클래스 비율의 제곱의 합)
#      = 1 - ∑(pi²)

# 직관적 의미: 랜덤을 뽑아서 랜덤으로 라벨을 붙이면 틀릴 확률

# 예시 1: 상자에 [사과 10개]만 있음
# 사과 비율 = 10 / 10 = 1.0
# Gini = 1 - (1.0²) = 1 - 1 = 0
# 완전히 순수

# 예시 2: 상자에 [사과 5개 오렌지 5개]가 있음
# 사과 비율 = 0.5, 오렌지 비율 = 0.5
# Gini = 1 - (0.5² + 0.5²) = 1 - 0.5 = 0.5
# 최대로 불순 (반반이라 가장 헷갈림)

# 예시 3: 상자에 [사과 9개 오렌지 1개]가 있음
# 사과 비율 = 0.9, 오렌지 비율 = 0.1
# Gini = 1 - (0.9² + 0.1²) = 1 - 0.82 = 0.18
# 꽤 순수함

# 이진 분류에서는 0.5가 가장 크지만
# 0 ~ 0.5까지

# 다중 클래스에서는
# 3개 0.666
# 4개 0.75

# 엔트로피(Entropy)
# Entropy = -∑(pi x log₂(pi))
# 직관적 의미: "얼마나 혼란스러운가?" (정보 이론 개념)

# 예시 1: [사과 10개]
# Entropy = -(1.0 x log₂(1.0)) = 0
# 전혀 혼란스럽지 않음

# 예시 2: [사과 5개 오렌지 5개]
# Entropy = -(0.5 x log₂(0.5)) x 2 = 1
# 최대로 혼란스러움

# 예시 3: 상자에 [사과 9개 오렌지 1개]
# Entropy = 0.47
# 약간 혼란스러움

# 0 ~ 1까지

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 데이터 로드
iris = load_iris()
X, y = iris.data, iris.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 모델 학습
model = DecisionTreeClassifier(
    criterion='gini', # 분할 기준 : 'gini' 또는 'entropy'
    max_depth=5, # 최대 깊이
    min_samples_split=10, # 분할을 위한 최소 샘플의 수
    min_samples_leaf=5, # 리프 노드 최소 샘플 수
    max_features=None, # 분할에 사용할 특성 수
    random_state=42                       
)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)
print(f'정확도: {accuracy_score(y_test, y_pred):.2%}')

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


# 한글 폰트 설정 추가
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
# plt.rcParams['font.family'] = 'AppleGothic'  # Mac
plt.rcParams['axes.unicode_minus'] = False


plt.figure(figsize=(20, 10))
plot_tree(model,
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True,
          rounded=True,
          fontsize=10)
plt.title('붓꽃 분류 결정 트리')
plt.show()

# 텍스트로 출력
from sklearn.tree import export_text

tree_rules = export_text(model,
                         feature_names=list(iris.feature_names))
print(tree_rules)


# 1. 데이터 로드(csv 파일 경로)
# df = 

# 2. 필요한 특성만 선택 (dropna())

# 3. 성별을 숫자로 변환

# 4. 특성과 타겟 분리

# 5. 분할

# 6. 모델 학습

# 7. 평가

# 8. 시각화
