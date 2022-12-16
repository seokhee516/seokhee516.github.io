---
layout: post
title: "RNN, LSTM, GRU"
date: 2022-10-12
categories:
 - Deep Learning
tags: [NLP, DL, RNN, LSTM, GRU]
description: "순환 신경망 RNN과 LSTM 및 GRU"
mathjax: true
---
# 순환 신경망 (RNN, Recurrent Neural Network)

- RNN: 시계열 데이터를 다루기에 최적화된 인공신경망
- 시계열 데이터란 시간축을 중심으로 현재 시간의 데이터가 앞, 뒤 시간의 데이터와 연관 관계를 가지고 있는 데이터를 의미. ex) 오늘의 주식 가격은 어제의 주식 가격과 연관이 있고, 내일의 주식 가격은 오늘의 주식 가격과 연관이 있음. 따라서 주식 가격은 시계열 데이터
- 주식 가격 이외에도 파형으로 표현되는 음성 데이터, 앞뒤 문맥을 가진 단어들의 집합으로 표현되는 자연어 데이터 등이 대표적인 시계열 데이터

![Untitled](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/072369de-f1f9-46e7-a367-03baf93aadb1/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221216%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221216T140310Z&X-Amz-Expires=86400&X-Amz-Signature=35493700c078d1a6d2d12bedefefe969c22ac3a8c24df6df0774677e4cd82cc6&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)

![Untitled](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/a79bde44-c1dc-4bde-8a51-b7d788cbbb17/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221216%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221216T140314Z&X-Amz-Expires=86400&X-Amz-Signature=2d0a9f375f6c325348e4ea748b43463458d9a5614fc860bc788c0e0a6b8a8a21&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)

## 연속형 데이터 (Sequential Data)

- 어떤 순서로 오느냐에 따라서 단위의 의미가 달라지는 데이터
- RNN 은 연속형(Sequential) 데이터를 잘 처리하기 위해 고안된 신경망

## RNN의 구조

![Untitled](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/7b2a4f7b-0b83-409d-9c43-839804f6894f/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221216%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221216T140318Z&X-Amz-Expires=86400&X-Amz-Signature=eea89dd6622fc28164f818ac0ca0589d91ea1af6619f329978601c9ea9e175e1&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)

### 기본 네트워크

등호 왼쪽을 보면 3개의 화살표가 있다.

1. 입력 벡터가 은닉층에 들어가는 것을 나타내는 화살표
2. 은닉층로부터 출력 벡터가 생성되는 것을 나타내는 화살표
3. **은닉층에서 나와 다시 은닉층으로 입력**되는 것을 나타내는 화살표.
    - 3번 화살표는 기존 신경망에서는 없었던 과정
    - 특정 시점에서의 은닉 벡터가 다음 시점의 입력 벡터로 다시 들어가는 과정
    - 출력 벡터가 다시 입력되는 특성 때문에 **'순환(Recurrent) 신경망'** 이라는 이름이 붙음

time-step 별로 펼쳐서 RNN 알아보기
- 신경망을 시점에 따라 펼쳐보면 오른쪽 그림처럼 나타낼 수 있음

## 다양한 형태의 RNN

실제로 다양한 형태의 RNN이 있다. 아래 그림에서 가장 왼쪽에 위치한 one-to-one은 실질적으로 순환이 적용되지는 않은 형태이다.

![Untitled](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/25b854bd-8148-47d4-b038-ac8831a72350/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221216%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221216T140322Z&X-Amz-Expires=86400&X-Amz-Signature=ea40c6e1001965b6343d5544874e60e9a36cb20f3ba3619ad3fec438bad3ef0b&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)

1. one-to-many : 1개의 벡터를 받아 Sequential한 벡터를 반환. 이미지를 입력받아 이를 설명하는 문장을 만들어내는 **이미지 캡셔닝(Image captioning)**에 사용
2. many-to-one : Sequential 벡터를 받아 1개의 벡터를 반환. 문장이 긍정인지 부정인지를 판단하는 **감성 분석(Sentiment analysis)**에 사용
3. many-to-many(1) : Sequential 벡터를 모두 입력받은 뒤 Sequential 벡터를 출력. **시퀀스-투-시퀀스(Sequence-to-Sequence, Seq2Seq) 구조**라고도 부름. 번역할 문장을 입력받아 번역된 문장을 내놓는 **기계 번역(Machine translation)**에 사용
4. many-to-many(2) : Sequential 벡터를 입력받는 즉시 Sequential 벡터를 출력. **비디오를 프레임별로 분류(Video classification per frame)**하는 곳에 사용

## RNN의 장점과 단점

### **RNN의 장점**

- 이론적으로 모델이 간단
- 어떤 길이의 sequential 데이터라도 처리할 수 있음

### RNN의 단점

- **병렬화(parallelization) 불가능**
    - RNN 기반 모델은 **단어 벡터가 순차적으로 입력**되기 때문에 병렬화가 어려워 연산 시간이 오래 걸림
    - 이는 sequential 데이터 처리를 가능하게 해주는 요인이지만, 이러한 구조는 GPU 연산의 장점인 병렬화를 불가능
    - RNN 기반의 모델은 GPU 연산을 하였을 때 이점이 거의 없다는 단점
- **기울기 폭발(Exploding Gradient), 기울기 소실(Vanishing Gradient)**
    - 활성화 함수 thanh 미분했을 때, 최댓값인 1이거나 대부분 0에 가까운 값을 가짐
    - 이를 역전파 과정에서 반복적으로 곱해줄 경우, 기울기가 거의 전단되지 않는 **기울기 소실(Vanishing Gradient)** 문제가 생기거나, **기울기가 과하게 전달되는 기울기 폭발(Exploding Gradient)** 문제를 가짐
    - 🔍 **tanh 미분한 함수**
        
        ![Untitled](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/4eae65fa-c1d7-43e4-9e50-0f07892080ca/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221216%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221216T140326Z&X-Amz-Expires=86400&X-Amz-Signature=00b057e446754151ba0c7f712b9e1de39138e1aa740a559962cfb1061e5b314a&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)
        
        위 그래프에서 최댓값이 1이고, (-4,4) 이외의 범위에서는 거의 0에 가까운 값을 나타내는 것을 알 수 있다. 문제는 역전파 과정에서 이 값을 반복해서 곱해주어야 한다는 점이다.이 Recurrent가 10회, 100회 반복된다고 보면, 이 값의 10제곱, 100제곱이 식 내부로 들어가게 된다. 만약 이 값이 0.9 일 때 10제곱이 된다면 0.349가 됨. 이렇게 되면 시퀀스 앞쪽에 있는 hidden-state 벡터에는 역전파 정보가 거의 전달되지 않게 됨. 반대로 이 값이 1.1 이면 10제곱만해도 2.59배로 커지게 됨. 이렇게 되면 시퀀스 앞쪽에 있는 hidden-state 벡터에는 역전파 정보가 과하게 전달.
        
- **장기 의존성(Long-term dependency)** 문제
    - 장기 의존성 문제란 긴 문장(시퀀스)을 처리할 때 앞쪽에 입력된 단어의 의미가 사라지게 되는 현상.

# LSTM (Long Term Short Memory, 장단기기억망)

RNN에서 "기울기 정보의 크기를 적절하게 조정하여 줄 수 있다면 문제를 해결할 수 있지 않을까?"라는 생각을 해볼 수 있다. 이런 아이디어에서, **기울기 정보 크기를 조절하기 위한 Gate를 추가한 모델을 LSTM**이라고 한다. 언어 모델 뿐만 아니라 신경망을 활용한 시계열 알고리즘에는 대부분 LSTM을 사용하고 있다.

## **LSTM의 구조**

![Untitled](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/4eae65fa-c1d7-43e4-9e50-0f07892080ca/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221216%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221216T140326Z&X-Amz-Expires=86400&X-Amz-Signature=00b057e446754151ba0c7f712b9e1de39138e1aa740a559962cfb1061e5b314a&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)

LSTM은 기울기 소실 문제를 해결하기 위해 **3가지 게이트(gate)**를 추가

1. forget gate ($f_t$): 과거 정보를 얼마나 유지할 것인가?
2. input gate ($i_t$) : 새로 입력된 정보는 얼마만큼 활용할 것인가?
3. output gate ($o_t$) : 두 정보를 계산하여 나온 출력 정보를 얼마만큼 넘겨줄 것인가?

hidden-state 말고도, **활성화 함수를 직접 거치지 않는 상태**인 **cell-state** 가 추가. cell-state는 역전파 과정에서 활성화 함수를 거치지 않아 정보 손실이 없기 때문에 **뒷쪽 시퀀스의 정보에 비중을 결정**할 수 있으면서 동시에 **앞쪽 시퀀스의 정보를 완전히 잃지 않을 수 있다**.

# GRU (Gated Recurrent Unit)

LSTM의 단순화한 버전인 GRU

2개의 gate

![Untitled](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/1096f1de-21dc-432e-92eb-738e45c9b7c0/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221216%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221216T140330Z&X-Amz-Expires=86400&X-Amz-Signature=bbbf5e584db2691d464a1df661196cf4d9fdc290dd47e2ef63f3856556054249&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)

1. LSTM에서 있었던 cell-state가 사라졌다.
    - cell-state 벡터 $c_t$ 와 hidden-state 벡터 $h_t$가 하나의 벡터 $h_t$로 통일
2. 하나의 Gate $z_t$가 forget, input gate를 모두 제어한다. → reset gate 
    - $z_t$가 1이면 forget 게이트가 열리고, input 게이트가 닫히게 되는 것과 같은 효과
    - 반대로 $z_t$가 0이면 input 게이트만 열리는 것과 같은 효과
3. GRU 셀에서는 output 게이트가 없어졌다. → update gate
    - 대신에 전체 상태 벡터 $h_t$ 가 각 time-step에서 출력되며, 이전 상태의 $h_t$$_-$$_1$의 어느 부분이 출력될 지 새롭게 제어하는 Gate인 $r_t$ 가 추가