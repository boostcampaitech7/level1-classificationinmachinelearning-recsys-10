# 비트코인 가격 예측 프로젝트

1시간 뒤 비트코인 가격의 등락율을 예측하는 프로젝트입니다. <br/>
구체적으로, 모델은 크립토퀀트에서 1차적으로 정제한 마켓 데이터와 네트워크 데이터를 입력으로 받아
1시간 뒤 비트코인의 등락율을 아래와 같이 클래스로 구분하여 예측합니다.

||0.5% 이상 상승|0 ~ 0.5% 상승|0 ~ 0.5% 하락|0.5% 이상 하락|
|:---:|:---:|:---:|:---:|:---:|
|class|3|2|1|0|

<br/>

# Team

|<img src="https://avatars.githubusercontent.com/u/20788198?v=4" width="150" height="150"/>|<img src="https://avatars.githubusercontent.com/u/81938013?v=4" width="150" height="150"/>|<img src="https://avatars.githubusercontent.com/u/112858891?v=4" width="150" height="150"/>|<img src="https://avatars.githubusercontent.com/u/103016689?v=4" width="150" height="150"/>|<img src="https://avatars.githubusercontent.com/u/176903280?v=4" width="150" height="150"/>|
|:-:|:-:|:-:|:-:|:-:|
|곽정무<br/>[@jkwag](https://github.com/jkwag)|박준하<br/>[@joshua5301](https://github.com/joshua5301)|박태지<br/>[@spsp4755](https://github.com/spsp4755)|신경호<br/>[@Human3321](https://github.com/Human3321)|이효준<br/>[@Jun9096](https://github.com/Jun9096)|

<br/>

# Prerequisites and Installation
- Python 3.11
- pip

```bash
pip install -r requirements.txt
```

<br/>

# Usage

### 1\. 프로젝트 최상위 폴더에 'data/raw' 폴더를 생성하고 안에 원시 데이터를 넣습니다.

### 2\. 'feature_engineering.ipynb'를 실행하여 데이터 전처리를 진행합니다. <br/>
전처리된 데이터는 'data/preprocessed' 폴더에 저장됩니다.

### 3\. 'process_model.ipynb'를 실행하여 모델 학습 및 검증, 테스트를 진행합니다. <br/>
Load Model 부분에서 module_name을 변경함으로써 사용할 모델을 선택할 수 있습니다. <br/>
테스트 결과는 최상위 폴더에 output.csv로 저장됩니다.

<br/>

# Models

### - Classification model

단일 LightGBM 모델을 사용한 단순 분류 모델입니다.

### - Regression model

단일 LightGBM 모델을 사용한 등락폭 회귀 모델입니다.

### - Soft labeling model

Soft labeling을 적용해 classification으로 인해 손실되는 정보를 완화시킨 모델입니다.

### - One-vs-Rest model

One-vs-Rest 방식을 적용한 다중 분류 모델입니다.

### - Direction-strength model

등락의 방향을 예측하는 모델과 등락의 강도를 예측하는 모델을 결합한 모델입니다.

### - Voting model

Soft voting을 적용한 앙상블 모델입니다.