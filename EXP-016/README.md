# 아이펠캠퍼스 온라인4기 피어코드리뷰

- 코더 : 이효준
- 리뷰어 : 심재형

PRT(PeerReviewTemplate)
----------------------------------------------

### 코드가 정상적으로 동작하고 주어진 문제를 해결했나요? (세모)
전부 정상적으로 진행됩니다.<br>
```python
source_data = pd.read_csv('data/ChatbotData.csv')
print(f"원천 데이터 불러오기 완료 \n총 길이 : {len(source_data)}")
print("---<각 컬럼별 결측치 개수 확인>---")
print(source_data.isnull().sum())
```
<br>데이터를 불러오고 결측치를 확인하는 방법은 전처리 하기전 좋은 방법인 것 같습니다!
```python
# 길이 분포 출력
import matplotlib.pyplot as plt
import numpy as np

questions_len = [len(s.split()) for s in source_data['Q']]
answers_len = [len(s.split()) for s in source_data['A']]

print('Qestion의 최소 길이 : {}'.format(np.min(questions_len)))
print('Qestion의 최대 길이 : {}'.format(np.max(questions_len)))
print('Qestion의 평균 길이 : {}'.format(np.mean(questions_len)))
print('Answer의 최소 길이 : {}'.format(np.min(answers_len)))
print('Answer의 최대 길이 : {}'.format(np.max(answers_len)))
print('Answer의 평균 길이 : {}'.format(np.mean(answers_len)))

plt.subplot(1,2,1)
plt.boxplot(questions_len)
plt.title('Questions')
plt.subplot(1,2,2)
plt.boxplot(answers_len)
plt.title('Answers')
plt.tight_layout()
plt.show()

plt.title('Questions')
plt.hist(questions_len, bins = 40)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

plt.title('Answers')
plt.hist(answers_len, bins = 40)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()
```
<br>길이 분포를 출력해서 최적의 최대길이 값을 찾는것도 좋은거같아요!

### 주석을 보고 작성자의 코드가 이해되었나요? (O)
```python
#샘플의 최대 허용 길이 또는 패딩 후의 최종 길이
MAX_LENGTH = 21
print(MAX_LENGTH)

# 정수 인코딩, 최대 길이를 초과하는 샘플 제거, 패딩
def tokenize_and_filter(inputs, outputs):
    tokenized_inputs, tokenized_outputs = [], []
  
    for (sentence1, sentence2) in zip(inputs, outputs):
    # 정수 인코딩 과정에서 시작 토큰과 종료 토큰을 추가
        sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
        sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN

    # 최대 길이 25 이하인 경우에만 데이터셋으로 허용
        if len(sentence1) <= MAX_LENGTH and len(sentence2) <= MAX_LENGTH:
            tokenized_inputs.append(sentence1)
            tokenized_outputs.append(sentence2)
  
  # 최대 길이 25으로 모든 데이터셋을 패딩
    tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
      tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
    tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
      tokenized_outputs, maxlen=MAX_LENGTH, padding='post')
  
    return tokenized_inputs, tokenized_outputs
```
<br>각각의 인코딩 과정들을 주석으로 설명 해주심으로 써 더 직관적으로 이해하기 편했습니다

### 코드가 에러를 유발할 가능성이 있나요? (X)
코드 자체에는 에러를 유발할 가능성이 보이지 않습니다.
```python
def load_chat():
    questions, answers = [], []
    _question, _answer = list(source_data['Q']), list(source_data['A'])
    
    for i in range(len(source_data)):
        questions.append(preprocess_sentence(_question[i]))
        answers.append(preprocess_sentence(_answer[i]))
        
    return questions, answers
```
<br> 전처리 과정도 깔끔하게 정리되어 문제는 없습니다!

### 코드 작성자가 코드를 제대로 이해하고 작성했나요? (O)
```python
def decoder_inference(sentence):
    sentence = preprocess_sentence(sentence)

    # 입력된 문장을 정수 인코딩 후, 시작 토큰과 종료 토큰을 앞뒤로 추가.
    # ex) Where have you been? → [[8331   86   30    5 1059    7 8332]]
    sentence = tf.expand_dims(
      START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

    # 디코더의 현재까지의 예측한 출력 시퀀스가 지속적으로 저장되는 변수.
    # 처음에는 예측한 내용이 없음으로 시작 토큰만 별도 저장. ex) 8331
    output_sequence = tf.expand_dims(START_TOKEN, 0)

    # 디코더의 인퍼런스 단계
    for i in range(MAX_LENGTH):
        # 디코더는 최대 MAX_LENGTH의 길이만큼 다음 단어 예측을 반복합니다.
        predictions = model(inputs=[sentence, output_sequence], training=False)
        predictions = predictions[:, -1:, :]

        # 현재 예측한 단어의 정수
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # 만약 현재 예측한 단어가 종료 토큰이라면 for문을 종료
        if tf.equal(predicted_id, END_TOKEN[0]):
            break

        # 예측한 단어들은 지속적으로 output_sequence에 추가됩니다.
        # 이 output_sequence는 다시 디코더의 입력이 됩니다.
        output_sequence = tf.concat([output_sequence, predicted_id], axis=-1)

    return tf.squeeze(output_sequence, axis=0)
    
def sentence_generation(sentence):
    # 입력 문장에 대해서 디코더를 동작 시켜 예측된 정수 시퀀스를 리턴받습니다.
    prediction = decoder_inference(sentence)

    # 정수 시퀀스를 다시 텍스트 시퀀스로 변환합니다.
    predicted_sentence = tokenizer.decode(
      [i for i in prediction if i < tokenizer.vocab_size])

    print('입력 : {}'.format(sentence))
    print('출력 : {}'.format(predicted_sentence))

    return predicted_sentence
```
<br>일련의 많은 과정들을 함수화 시키며 챗봇화 하여 사용자 편의성에 중점을 둔 코드인거 같아요! 완벽히 이해하고 계시기에 가능한 방법이라고 생각합니다!
![image](https://github.com/7eta/Aiffel_EXPLORATION_RS4/assets/65104209/95454d91-fa19-4dec-8c22-50014e827b36)<br>
단순한 전처리 과정에서도 상황에 맞게 전처리를 하는 방식도 좋은거같아요!
### 코드가 간결한가요? (O)
![image](https://github.com/7eta/Aiffel_EXPLORATION_RS4/assets/65104209/6bdfe0f9-ed66-4a51-bd06-76b102a298aa)
<br>간단한 함수화로 훨씬 간결하고 직관적으로 이해됩니다!

----------------------------------------------

## 참고 링크 및 코드 개선
