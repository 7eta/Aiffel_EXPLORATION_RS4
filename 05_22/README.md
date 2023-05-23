# AIFFEL Campus Online 4th Code Peer Review Templete
- 코더 : 소용현
- 리뷰어 : 이효준


# PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.
- [o] 1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
- [o] 2.주석을 보고 작성자의 코드가 이해되었나요?
  > # _img[row범위,col범위] = img_sticker의 값이 0이면, 합성 dst를 아니면 _img
        _img[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]] = \
    np.where(rotated_sticker==0,dst,sticker_area).astype(np.uint8)
  > 필터 방식을통한 이미지 합성방법 주석으로 설명  
- [x] 3.코드가 에러를 유발할 가능성이 있나요?
  > 위 항목에 대한 근거 작성 필수
- [o] 4.코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 스티커합성방법에 대한 이해 : 원본 이미지와 합성이미지를 만들고, 필터 방식을 통해 합성이미지가 완전 검은색(0)인 경우에만 합성이미지를 사용한다는 방식을 이해하고 있다.
- [o] 5.코드가 간결한가요?
  > 함수를 사용하여 간결한 호출이 가능하도록 코딩되어 있다.
```
img_ori_bgr = cv2.imread('images/timcook_0.jpg')
img_ori_rgb = cv2.cvtColor(img_ori_bgr, cv2.COLOR_BGR2RGB)
plt.subplot(3,3,1).set_title("original image")
plt.imshow(cv2.cvtColor(transpose_put_sticker(img_ori_rgb), cv2.COLOR_BGR2RGB))
```

# 예시
1. 코드의 작동 방식을 주석으로 기록합니다.
2. 코드의 작동 방식에 대한 개선 방법을 주석으로 기록합니다.     
```
img_ori_bgr = cv2.imread('images/timcook_p60.jpg')
img_ori_rgb = cv2.cvtColor(img_ori_bgr, cv2.COLOR_BGR2RGB)
plt.subplot(3,3,4).set_title("brightness 60% up image")
plt.imshow(cv2.cvtColor(transpose_put_sticker(img_ori_rgb), cv2.COLOR_BGR2RGB))
```
faceDetecting은 밝기조절된 이미지로 영역과 이목구비를 찾은 후에, 원본 이미지에 스티커를 합성하는 방식이 스티커앱의 컨셉에는 좀 더 좋을 것 같다.

```
        sticker_area = _img[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]]
        
        dst = cv2.addWeighted(sticker_area, 0.3, img_sticker, 0.8, 0)
        
        # _img[row범위,col범위] = img_sticker의 값이 0이면, 합성 dst를 아니면 _img
        _img[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]] = \
    np.where(img_sticker==0,dst,sticker_area).astype(np.uint8)
```
스티커 이미지의 투명도 값을 활용하여 합성해보는 것도 좋을 것 같다.
크기 조절이 되면서 스티커 이미지는 단색이 아니게 변하고, 스티커 이미지가 단색이 아닌 경우에도 사용할 수 있도록 투명도를 활용해 보는 것도 좋을 것 같다.
```
        sticker_area = img[refined_y:refined_y+img_sticker_rotate.shape[0], refined_x:refined_x+img_sticker_rotate.shape[1]]
        # PNG 이미지의 알파 채널을 추출합니다.
        alpha_channel = img_sticker_rotate[:, :, 3]
        # PNG 이미지의 BGR 채널을 추출합니다.
        bgr_channels = img_sticker_rotate[:, :, :3]
        # 알파 채널을 3채널로 확장합니다.
        expanded_alpha = cv2.cvtColor(alpha_channel, cv2.COLOR_GRAY2BGR)

        # JPG 이미지와 같은 크기의 배경을 생성합니다.
        background = cv2.multiply((255 - expanded_alpha) / 255, sticker_area, dtype = cv2.CV_32F)

        # PNG 이미지와 같은 크기의 전경을 생성합니다.
        foreground = cv2.multiply(bgr_channels, expanded_alpha / 255, dtype = cv2.CV_32F)
        # 전경과 배경을 합성하여 최종 이미지를 생성합니다.
        result = cv2.add(background, foreground, dtype=cv2.CV_32F)

        #img_show 합성부분을 result로 교체
        img[refined_y:refined_y+img_sticker_rotate.shape[0], refined_x:refined_x+img_sticker_rotate.shape[1]] = \
                result.astype(np.uint8)
```
```
 sticker_path = 'images/cat-whiskers.png'
    img_sticker = cv2.imread(sticker_path)
```
스티커 이미지와 같이 프로그램에서 고정적으로 사용하는 DATA는 한번만 호출하도록 바꿔주는 것이 좋을 것 같다.

3. 참고한 링크 및 ChatGPT 프롬프트 명령어가 있다면 주석으로 남겨주세요.


# 참고 링크 및 코드 개선
```python
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```
