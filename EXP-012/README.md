# AIFFEL Campus Online 4th Code Peer Review Templete
- 코더 : 이효준
- 리뷰어 : 소용현


# PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.
- [o] 1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
- [o] 2.주석을 보고 작성자의 코드가 이해되었나요?
  > 
  ```
  # summarize의 ratio를 조절하면 글자가 보이기도 안보이기도 한다. 이 비율을 잘 조정해야겠다.
  # ratio의 범위는 0 ~ 1 사이의 값을 갖는다.
  for i in range(10):
      print("실제 요약 : ", news_csv_data2['headlines'][i])
      print("예측 요약 : ", summarize(news_csv_data2['text'][i], ratio=0.5))
      print("\n")
  ```
  summarize ratio를 파라미터로 넣으느 이유가 잘 설명되어 있다.
  
- [o] 3.코드가 에러를 유발할 가능성이 있나요?
  > 
  ```
  clean_text_path = 'data/clean_text.csv'
  if os.path.isfile(clean_text_path):
      with open(clean_text_path, 'r') as file:
          clean_text = list(csv.reader(file))
          file.close()
  else:
      clean_text = []

      for text in notebook.tqdm(news_csv_data['text']):
          _cleaned = preprocess_sentence(text, remove_stopwords=True)
          clean_text.append(_cleaned)

      with open(clean_text_path, 'w') as file:
          write = csv.writer(file)
          write.writerow(clean_text)
          file.close()
  ```
  텍스트 데이터의 csv저장의 경우 delimeter가 텍스트에 포함되어 있을 수도 있어, 정제되지 않은 긴 텍스트일 경우 유의해야 한다.
  *전처리를 거쳤기에 문제는 없을 것으로 보이지만, csv보다는 오브젝트로 내보내고 불러오는 습관이 되면 좋을 것 같다.
- [o] 4.코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 인터뷰 결과 추출적 요약이 잘 되지 않는 이유를 탐구하고, 작성하였다.
  > 패딩의 의의를 이해하고 있다.
  > 정수형 변환의 의의를 이해하고 있다.
- [o] 5.코드가 간결한가요?
  > 
  ```
  # inplace=True 를 설정하면 DataFrame 타입 값을 return 하지 않고 data 내부를 직접적으로 바꿉니다
  news_csv_data.drop_duplicates(subset = ['text'], inplace=True)
  ```
  중복처리를 한줄로 잘 표현하였다.
