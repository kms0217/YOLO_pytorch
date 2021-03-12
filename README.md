# YOLO (You Only Look Once)

### Introduction
- Detection과 Classification문제를 하나의 regression 문제로 정의
  - 하나의 신경망으로 탐지와 분류를 동시에 수행 (Unified Detection)
    
    ![image/yolo1.png](/image/yolo1.png)
- 기존 실시간 Detection 모델에 비해 높은 성능 + 빠른 속도 (45 frame pre second)
- 다른 도메인에서도 좋은 성능을 보여준다.
  - Pascal Voc 2007로 학습한 YOLO를 사용해 예술 작품 데이터셋으로 테스트했을 때 정확도가 다른 Detection모델에 비해서 적게 떨어짐

### Unified Detection
- YOLO는 Input Image를 S x S의 Grid 로 나눈다. 
  - 물체의 중심을 포함하는 Grid Cell이 해당 물체를 감지한다.
- 각 Grid Cell은 B개의 Bounding Box를 가진다.
  - 각 Bounding box는 다음 5개의 예측으로 구성된다.
    1. x, y : 예측한 bounding box의 중심좌표 (grid cell에 상대적, 0.5 0.5라면 grid의 중심)
    2. w, h : 예측한 bounding box의 width, height (전체 이미지 크기에 대한 비율)
    3. confidence score : Pr(object) * IOU(ground truth, predict)
- 각 Grid Cell은 n개의 클래스를 예측하는 확률을 가지고있다. 
- 논문에서 S는 7, B는 2, n은 20이다.
  - Output Tensor가 (7 x 7 x 30)인데 7 x 7은 각 Grid Cell을 의미한다.
  - 30은 각 Grid Cell이 예측한 정보로 2개의 bounding box의 예측, class 확률을 나타낸다.
  - 즉 [x, y, w, h, con][x, y, w, h, con][20개 클래스에 대한 확률] 이다.
- Network Design
  - 24개의 Conv layer
  - Input : 448 x 448 x 3
  - Output : 7 x 30 x 30
   
    ![image/yolo2.png](image/yolo2.png)

- Train
  - Pretrain
    - Network의 앞의 20개의 Conv Layer는 Avg Pool Layer, Linear Layer를 붙혀 ImageNet 1000 Classification으로 Pretrain한다.
    - Pretrain이 끝난 뒤 Avg pool, Linear Layer를 제거하고 Conv Layer 4개와 2개의 Linear Layer을 붙혀 yolo모델을 학습한다.
  - Dataset
    - Pascal Voc 2007
    - Pascal Voc 2012
  - Activation Function
     -  마지막 출력과 각 Layer 사이마다 leakyReLU(alpha = 0.1)를 사용
  - Avoid overfitting
    - Dropout(0.5)을 첫 번째 connceted Layer 뒤에 적용한다.
    - data augmentation
      - random scaling and translations of up to 20% of the original image size
      -  randomly adjust the exposure and saturation of the image by up to a factor of 1.5 in the HSV color space.
   - Learning rate
     - 총 135 epoch 중 75 epoch까지는 1e-2, 75 ~ 105 epoch까지는 1e-3, 105 ~ 135 epoch까지는 1e-4를 사용

- Loss function
    
    ![image/yolo3.png](image/yolo3.png)
    - 총 5개의 식으로 loss를 계산한다.
    - 각 loss들은 SSE를 사용한다.
    - Loss를 계산할 때 B개의 box를 모두 사용하지 않고 predictor로 선택된 bounding box만 사용한다.
      - predictor란 Ground Truth와 IOU가 더 큰 box를 말한다. 
    - 보통 object가 있는 Grid Cell보다 없는 Grid Cell이 더 많다.
      - 이를 해결하기 위해 lambda_coord(5), lambda_noobj(0.5)를 제시하였다.
    - Localization loss
        
        ![image/yolo4.png](image/yolo4.png)
      - 위의 식의 indicator는 i번 Cell에 j번 bounding box가 predictor라면 1, 아니면 0인 값이다. 
      - w, h에 루트를 사용하는 이유는 큰 box에 대해서 작은 분산을 반영하기 위해서다.
    - Confidence loss

        ![image/yolo5.png](image/yolo5.png)
      - 오브젝트가 존재하는곳, 오브젝트가 존재하지 않는 곳의 confidence loss를 계산한다.
    - Classification loss

        ![image/yolo6.png](image/yolo6.png)
      - 오브젝트가 존재하는 곳에 대해서만 Classification loss를 계산한다.

- Comparison to Other Detection System

    ![image/yolo7.png](image/yolo7.png)

    - Real-Time Detector에 대해서 성능 속도면에서 더 좋은것을 볼 수 있다.
      - Fast Yolo의 경우 앞의 20개의 Conv Layer대신 9만 사용한 모델이다.
