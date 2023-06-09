아이펠캠퍼스 온라인4기 피어코드리뷰

- 코더 : 이효준
- 리뷰어 : 소용현

----------------------------------------------
## 2-1. 프로젝트 1: 손수 설계하는 선형회귀, 당뇨병 수치를 맞춰보자!

PRT(PeerReviewTemplate)

- [O] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
- [O] 주석을 보고 작성자의 코드가 이해되었나요?
```
'''
입력 데이터 개수에 맞는 가중치 w와 b를 준비하기 전에 df_X의 shpae이 (442,10)임
10개의 피쳐가 있기에 w1, w2, w3, ... w10까지 존재하고
앞서 편향 b는 상수이기에 하나의 w0으로 생각할 수 있다.
''' 
# 10개의 w와 1개의 b를 생성한다.
W = np.random.rand(10)
b = np.random.rand()
# 모델 함수 구현하기
def model(X, W, b):
    predictions = 0
    for i in range(X.shape[1]):
        predictions += X[:, i] * W[i]
    predictions += b
    return predictions
```
위와 같이 상세한 설명이 주석으로 작성되어 있다.
- [X] 코드가 에러를 유발할 가능성이 있나요?
#위 항목에 대한 근거 작성 필수
- [O] 코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기)
loss를 낮추기 위한 시도들과 randomstate와 min_max스케일링을 통한 loss 낮추기를 보다 코드를 제대로 이해하고 작성한 것 같다.
- [O] 코드가 간결한가요?
```
def gradient(X, W, b, y):
    # N : 전체 정답 데이터 개수
    N = len(y)
    
    # y_pred 준비
    y_pred = model(X, W, b)
    
    # 공식에 맞게 gradient 계산
    dW = 1/N * 2 * X.T.dot(y_pred - y)
        
    # b의 gradient 계산
    db = 2 * (y_pred - y).mean()
    return dW, db
```
수식을 간결하게 나타내었다

----------------------------------------------
## 2-2. 프로젝트 2: 날씨 좋은 우러요일 오후 세시, 자전거 타는 사람은 몇 명?
PRT(PeerReviewTemplate)

- [O] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
- [O] 주석을 보고 작성자의 코드가 이해되었나요?
```
# 자전거 타기에 중요한 요소, 날씨, 온도, 체감온도, 습도, 주말유무, 시간대, 자전거 대여 횟수
features = ['weather', 'temp', 'atemp', 'humidity','windspeed', 'holiday', 'hour', 'count']
new_df = pd.DataFrame()

for feature in features:
    new_df[feature] = df[feature]

# new_df에 결측치(null) 확인
new_df.isnull().any()
```
- [o] 코드가 에러를 유발할 가능성이 있나요?
```
train_csv_path = os.getenv('HOME')+'/data/data/bike-sharing-demand/train.csv'
df = pd.read_csv(train_csv_path)
```
데이터가 없거나, 운영체제에 따라 에러가 발생할 수 있다.

*실습임을 감안

- [o] 코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기)
데이터에 대해 범용성을위한 선택, 무의미한 데이터 선별등의 작업을 이해하고 작성하였다.

- [o] 코드가 간결한가요?
```
# y_test(정답)과 predictions(위에서 구한 모델 예측 값)를 MSE, RMSE를 통해 비교한다.
mse = mean_squared_error(y_test, predictions)
```
----------------------------------------------
참고 링크 및 코드 개선
- 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
- 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
