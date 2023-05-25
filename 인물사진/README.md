# AIFFEL Campus Online 4th Code Peer Review Templete
- 코더 : 이효준
- 리뷰어 : 소용현


# PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.
- [o] 1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
- [o] 2.주석을 보고 작성자의 코드가 이해되었나요?
  > 
  ```
    # cv2.bitwise_and()을 사용하면 배경만 있는 영상을 얻을 수 있습니다.
    # 0과 어떤 수를 bitwise_and 연산을 해도 0이 되기 때문에 
    # 사람이 0인 경우에는 사람이 있던 모든 픽셀이 0이 됩니다. 결국 사람이 사라지고 배경만 남아요!
    img_bg_blur = cv2.bitwise_and(img_orig_blur, img_bg_mask)
    # np.where(조건, 참일때, 거짓일때)
    # 세그멘테이션 마스크가 255인 부분만 원본 이미지 값을 가지고 오고 
    # 아닌 영역은 블러된 이미지 값을 사용합니다.
    if bg_img is None:
        img_concat = np.where(img_mask_color==255, _img, img_bg_blur)
    else:
        #원본 size(w, h)를 합성배경의 size와 맞춰야 한다.
        _h, _w, _ = img_bg_blur.shape
        _bg = cv2.imread(bg_img)
        _bg = cv2.resize(_bg, (_w, _h))
        img_concat = np.where(img_mask_color==255, _img, _bg)
        ```
        상세한 주석으로 이해하기 편했다.
- [o] 3.코드가 에러를 유발할 가능성이 있나요?
  > 
  ```
      _img = cv2.imread(img_path)#img_path의 이미지가 없으면 에러가 발생할 수 있다. 예외처리를 해주면 좋을 것 같다.
      #class_name을 잘못 입력하면 에러가 발생할 수 있다. 예외처리를 해주며 좋을 것 같다.
    _class_id = LABEL_NAMES.index(class_name)
    
    #계속해서 사용되는 모델 선언은 함수 밖에 해주면 좋을 것 같다.
    model = semantic_segmentation()
    model.load_pascalvoc_model(model_path)
    
    # 
    output, segmap = model.segmentAsPascalvoc(img_path)
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(_img, cv2.COLOR_BGR2RGB))
    plt.title("A raw image")
    
    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(segmap, cv2.COLOR_BGR2RGB))
    plt.title("A semantic segmentation image")

        
    _class_name = []
    for class_id in output['class_ids']:
        _class_name.append(LABEL_NAMES[class_id])
    print(f"위 사진에서 다음 클래스를 찾았습니다. {_class_name}")
    colormap = np.zeros((256, 3), dtype = int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3
        
    #colormap[Integer]의 결과는 BGR순이므로 RGB순으로 바꾸어야 함
    colormap_rgb = tuple(colormap[_class_id][::-1])
    seg_map = np.all(segmap==colormap_rgb, axis=-1) 
    ```
    사용자 입력값에 따른 예외처리를 해주면 좋을 것 같다.
- [o] 4.코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 
  ```
      img_orig_blur = cv2.blur(_img, (50,50))
    # cv2.cvtColor(입력 이미지, 색상 변환 코드): 입력 이미지의 색상 채널을 변경
    # cv2.COLOR_BGR2RGB: 원본이 BGR 순서로 픽셀을 읽다보니
    # 이미지 색상 채널을 변경해야함 (BGR 형식을 RGB 형식으로 변경) 
    img_mask_color = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)

    # cv2.bitwise_not(): 이미지가 반전됩니다. 배경이 0 사람이 255 였으나
    # 연산을 하고 나면 배경은 255 사람은 0입니다.
    img_bg_mask = cv2.bitwise_not(img_mask_color)

    # cv2.bitwise_and()을 사용하면 배경만 있는 영상을 얻을 수 있습니다.
    # 0과 어떤 수를 bitwise_and 연산을 해도 0이 되기 때문에 
    # 사람이 0인 경우에는 사람이 있던 모든 픽셀이 0이 됩니다. 결국 사람이 사라지고 배경만 남아요!
    img_bg_blur = cv2.bitwise_and(img_orig_blur, img_bg_mask)
    # np.where(조건, 참일때, 거짓일때)
    # 세그멘테이션 마스크가 255인 부분만 원본 이미지 값을 가지고 오고 
    # 아닌 영역은 블러된 이미지 값을 사용합니다.
    if bg_img is None:
        img_concat = np.where(img_mask_color==255, _img, img_bg_blur)
    else:
        #원본 size(w, h)를 합성배경의 size와 맞춰야 한다.
        _h, _w, _ = img_bg_blur.shape
        _bg = cv2.imread(bg_img)
        _bg = cv2.resize(_bg, (_w, _h))
        img_concat = np.where(img_mask_color==255, _img, _bg)
  ```
  이미지 합성 부분을 제대로 이해하고 작성하였다.
- [o] 5.코드가 간결한가요?
  > 계속 사용되는 부분은 함수로 선언하여 간결하게 작성하였다.

- 프로그램에 부하가 생기면서 반복사용되는 모델load같은 경우에는 전역변수처럼 사용하면 좋을 것 같다. 
