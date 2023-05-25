# AIFFEL Campus Online 4th Code Peer Review Templete
- 코더 : 이효준
- 리뷰어 : 소용현


# PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.
- [x] 1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
```
X_val = X_train[:49157]
y_val = y_train[:49157]

partial_X_train = X_train[49157:]  
partial_y_train = y_train[49157:]
```
validation셋을 위와 같이 나누었는데 실제 학습에서는 
```
history = model.fit(X_train,
                    y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(X_val, y_val),
                    verbose=1)
```
트레인셋이 잘못적용된 부분이 있어 validation loss 관측에 문제가 있다.  

>최종 정확도가 85%를 넘지 못했다 ㅠㅠ
- [o] 2.주석을 보고 작성자의 코드가 이해되었나요?
```
    # 1차원의 list 형태로 변환
    # Before : [['가', '가', '나', '다'], ['가', '라', '마'], ['마', '사']]
    # After  : ['가', '가', '나', '다', '가', '라', '마', '마', '사']
    words = np.concatenate(X_train).tolist()

    # 위에서 변환된 ['가', '가', '나', '다', '가', '라', '마', '마', '사'] list가 있다고 할때
    # Counter()를 거치고 나면 Counter({'가': 3, '마': 2, '나': 1, '다': 1, '라': 1, '사': 1}) 객체가 된다.
    # Counter객체를 most_common(3)함수를 사용하면 최빈값 3번째 까지 리스트에 담긴 튜플 형태로 반환된다.
    # output : [('가', 3), ('마', 2), ('나', 1)]
    counter = Counter(words)
    counter = counter.most_common(num_words-4)
    
    # counter = [('가', 3), ('마', 2), ('나', 1)] 일때
    # [k for k, _ in counter]의 output은 ['가', '마', '나']이 된다.
    # 같은 차원의 리스트 합 연산은 ['', ''] + ['가', '마', '나'] => ['', '', '가', '마', '나']형태가 된다.
    vocab = ['', '', '', ''] + [key for key, _ in counter]
    word_to_index = {word:index for index, word in enumerate(vocab)}
```
상세한 주석이 되어 있다
- [x] 3.코드가 에러를 유발할 가능성이 있나요?
  
- [o] 4.코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 시각화 구현을 위한 데이터 가공등을 보면 이해하고 작성한 것으로 판단된다
- [x] 5.코드가 간결한가요?
  > 반복적으로 사용되는 코드를 조금 더 줄였으면 좋을 것 같다.
