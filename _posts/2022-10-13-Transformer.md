---
layout: post
title: "Attention Is All You Need - Transformer"
date: 2022-10-13
categories:
 - Deep Learning
tags: [NLP, DL, Transformer, Attention, Encoder, Decoder]
description: "Encoder와 Decoder, Attention, Attention Is All You Need 논문 기반 Transformer 구조"
mathjax: true
---

# **기존 RNN 기반(LSTM, GRU) 번역 모델의 단점**

RNN이 가진 가장 큰 단점 중 하나는 기울기 소실로부터 나타나는 **장기 의존성(Long-term dependency)** 문제이다
![seq2seq](https://camo.githubusercontent.com/c176c39d796b63ab7826371a862d97a0845311740eabe6c93380a18247a0a9a8/68747470733a2f2f757365722d696d616765732e67697468756275736572636f6e74656e742e636f6d2f34353337373838342f38363034303939352d66323762343830302d626137662d313165612d386361312d3637623235313735373365622e676966)

위 구조는 기계 번역에서 RNN 기반의 모델(LSTM, GRU)이 단어를 처리하는 방법이다. 이러한 구조의 문제는 고정 길이의 **hidden-state 벡터에 모든 단어의 의미를 담아야 한다**는 점이다.
아무리 LSTM, GRU가 장기 의존성 문제를 개선하였더라도 **문장이 매우 길어지면(30-50 단어) 모든 단어 정보를 고정 길이의 hidden-state 벡터에 담기 어렵다.**
이런 문제를 해결하기 위해서 고안된 방법이 바로 **Attention(어텐션)**이다.

# Attention
![att](https://camo.githubusercontent.com/bfd2b9ee05decdd01d7bf36eb9a3e124ce7fcb456768f4ab3d727ab8b8a4a78b/68747470733a2f2f757365722d696d616765732e67697468756275736572636f6e74656e742e636f6d2f34353337373838342f38363034303837332d62393432643830302d626137662d313165612d396635392d6565323339323366373737652e676966)
개선포인트는, decoder가 encoder 의 LSTM 계층의 마지막 hidden-state만을 이용하는 것이 아닌, hidden-state를 전부 활용할 수 있도록 만드는 것이다. 즉, Attention은 각 **인코더의 Time-step 마다 생성되는 hidden-state 벡터를 모두 간직**한다.
- 입력 단어가 N개라면 N개의 hidden-state 벡터를 모두 간직
- 모든 단어가 입력되면 생성된 hidden-state 벡터를 모두 디코더에 넘겨줌

> encoder: 입력된 데이터를 압축된 데이터(context vector)로 만드는 것

> decoder: 원본이 나오도록 하는 것

## 디코더에서 Attention이 동작하는 방법 

- **디코더**의 각 time-step 마다의 **hidden-state 벡터**는 **쿼리(query)**로 작용
- **인코더**에서 넘어온 **N개의 hidden-state 벡터**를 **키(key)**로 여기고 이들과의 **연관성**을 계산
- 이 때 계산은 **내적(dot-product)**을 사용하고 **내적의 결과를 Attention 가중치**로 사용

- 즉, **Attention**이란 인코더에 **입력된 문장의 단어**와 **디코더가 생성하려는 단어가 연관된 정도**를 나타내는 **가중치**

### 디코더 "I"에 대한 어텐션 가중치가 구해지는 과정

![Untitled](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/a54e3bd2-d0a1-44cd-8bd0-718fb39b6899/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221216%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221216T134439Z&X-Amz-Expires=86400&X-Amz-Signature=d880be4fbabc896553c13c39f44e1f1245f7c246996b95c7dbb526718b6ed639&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)

1. **쿼리(Query)인 디코더의 hidden-state 벡터**, **키(Key)인 인코더에서 넘어온 hidden-state 벡터**를 준비
2. 각각의 벡터를 **내적**한 값을 구함
3. 이 값에 **소프트맥스(softmax) 함수**를 취해줌
4. 소프트맥스를 취하여 나온 값에 **밸류(Value)**에 해당하는 인코더에서 넘어온 hidden-state 벡터를 곱해줌
5. 이 벡터를 모두 **더해준다**. 이 벡터의 성분 중에는 **쿼리-키 연관성이 높은** 밸류 벡터의 성분이 더 많이 들어있게 됨
6. (그림에는 나와있지 않지만) 최종적으로 5에서 **생성된 벡터**와 **디코더의 hidden-state 벡터를 사용하여 출력 단어를 결정**

디코더는 인코더에서 넘어온 모든 Hidden state 벡터에 대해 계산한다. 그렇기 때문에 Time-step마다 출력할 단어가 어떤 인코더의 어떤 단어 정보와 연관되어 있는지, 즉 어떤 단어에 집중(Attention)할 지를 알 수 있게 된다.

# Trasformer
이후 RNN이나 LSTM을 사용하지 않고, **Atention만 사용하여 적용된 모델이 Transformer**이다.

![1](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/3de55a8e-7c07-4018-8f8c-4ba0d8c41baf/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221216%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221216T120731Z&X-Amz-Expires=86400&X-Amz-Signature=0f1f397c4742a591b2039245677fa660c9e9d6c82abed74cad564da3b99d9441&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)

- 인코더 블록은 **2개의 sub-layer Multi-Head (Self) Attention, Feed Forward**로 나눌 수 있음

- 디코더 블록은 **3개의 sub-layer Masked Multi-Head (Self) Attention, Multi-Head (Encoder-Decoder) Attention, Feed Forward**로 나눌 수 있음

## Positional Encoding

![2](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/83f3bf0c-58c2-4980-9d57-349baf07e076/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221216%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221216T120937Z&X-Amz-Expires=86400&X-Amz-Signature=8ab7afba30397658dfd32e585fecfa2749a8245383229ad284bc9f9b088e2c98&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)

- 트랜스포머에서는 병렬화를 위해 모든 단어 벡터를 동시에 입력받음
- 컴퓨터는 어떤 단어가 어디에 위치하는지 알 수 없게 됨

- 그래서 컴퓨터가 이해할 수 있도록 단어의 위치 정보를 제공하기 위한 벡터를 따로 제공해주어야 함
- 단어의 상대적인 위치 정보를 제공하기 위한 벡터를 만드는 과정을 **Positional Encoding** 이라 함

## Self-Attention

**Self-Attention**은 번역하려는 문장 내부 요소의 관계를 잘 파악하기 위해서, 문장 자신에 대해 어텐션 메커니즘을 적용하는 것을 말한다.
**Self-Attention**은 세 가지 가중치 벡터를 대상으로 어텐션을 적용. 적용하는 방식은 기존 Attention 메커니즘과 거의 동일하다.

1. **가중치 행렬 $W^Q$, $W^K$, $W^V$** 로부터 각 단어의 **쿼리, 키, 밸류(q, k, v) 벡터**를 만들어낸다.
    - 내적을 통해 나오는 값이 Attention 스코어(Score)가 됨
2. 분석하고자 하는 단어의 **쿼리 벡터(q)**와 문장 내 모든 단어(자신 포함)의 **키 벡터(k)**를 **내적**하여 **각 단어와 관련 정도**를 구한다.
    - Score가 높을 수록 연관성 높음
3. 다음으로 **$\sqrt{d_k}$로 나누어 준 뒤 Softmax**를 취해준다.
    - 계산값을 안정적으로 만들어주기 위한 계산 보정
    - Softmax를 통해 쿼리에 해당하는 단어와 문장 내 다른 단어가 가지는 관계의 비율을 구할 수 있음
4. **Softmax의 출력값** 과 **밸류 벡터(v)**를 **곱해준 뒤 더해준다.** 
- 해당 **단어에 대한 Self-Attention 출력값**을 얻을 수 있음
5. 하나의 벡터에 대해서만 살펴보았지만 **실제 Attention 계산은 행렬 단위로 병렬 계산**

## Multi-Head Attention

![3](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/41fbdd7c-184c-42d5-8551-a08748ce32bb/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221216%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221216T130048Z&X-Amz-Expires=86400&X-Amz-Signature=90e53b6ae454d62f3ed65c4b7c436967274e2291cf610ff1951a9c8629b67adb&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)

- Multi-Head Attention 이란 **Self-Attention을 동시에 병렬적으로 실행하는 것**
- 각 Head 마다 다른 Attention 결과를 내어주기 때문에 앙상블과 유사한 효과
- 논문에서는 8개의 Head를 사용
- 8번의 Self-Attention을 실행하여 각각의 출력 행렬 $Z_0,Z_1,⋯,Z_7$ 을 만들어냄

![4](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/6df3111e-71ff-4bab-86e2-fc443ef3e1f4/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221216%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221216T130719Z&X-Amz-Expires=86400&X-Amz-Signature=a3897ad657a889a0e96fef0dbfcf11d096405e38b0bbdd354d2c0e6ea849f05e&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)

- 출력된 행렬 $Z_n$(n=0,⋯,7)을**이어붙여짐(Concatenate)**.
- 또 다른 파라미터 행렬인 $W^o$ 와의 내적을 통해 Multi-Head Attention의 최종 결과인 행렬 Z를 만들어냅니다. 여기서 행렬 Wo의 요소 역시 학습을 통해 갱신
- **최종적으로 생성된 행렬 Z**는 토큰 벡터로 이루어진 행렬 X와 **동일한 크기(Shape)**가 됨

![5](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/b76266b2-8898-4356-a1c9-8b1c118cd912/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221216%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221216T130818Z&X-Amz-Expires=86400&X-Amz-Signature=4ddcd2844dca5652a11f536599cb18773c0872dedf4cabe7f7b7d488ba9cebe4&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)

## Layer Normalization & Skip Connection

![6](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/3d69a644-7517-449f-aaa6-ab4cd7f9b5af/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221216%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221216T130849Z&X-Amz-Expires=86400&X-Amz-Signature=486bdb2f987beb72c9bc684da8a3dd8e3fbf8f1cad454940a073f0f61f892937&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)

트랜스포머의 모든 sub-layer에서 출력된 벡터는 **Layer normalization**과 **Skip connection**을 거치게 된다.

- Layer normalization의 효과는 Batch normalization과 유사
- 학습이 훨씬 빠르고 잘 되도록 함
- Skip connection(혹은 Residual connection)은 역전파 과정에서 정보가 소실되지 않도록 함

## Feed Forward Neural Network

![7](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/91631f6e-f342-4bc8-85dd-091179c9ed1a/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221216%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221216T130947Z&X-Amz-Expires=86400&X-Amz-Signature=79d94f0f8e7bafa95f8fd1819d6cb74e024cc2272172c426e57d56f6566684a5&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)

- **FFNN(Feed forward neural network)**는 은닉층의 차원이 늘어났다가 다시 원래 차원으로 줄어드는 단순한 2층 신경망
- 활성화 함수(Activation function)으로 ReLU를 사용

$$\text{FFNN}(x) = \max(0, W_1x + b_1) W_2 +b_2$$

## Masked Self-Attention

![8](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/cd7f845d-57f9-4e05-bf87-5df756e3a9d7/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221216%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221216T131121Z&X-Amz-Expires=86400&X-Amz-Signature=42d06370dcc810489fcf4cd6ca6bf1570b07fbb2725f9191639da67a6fd4be95&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)

**Masked Self-Attention**은 디코더 블록에서 사용되는 특수한 Self-Attention
- 디코더는 Auto-Regressive(왼쪽 단어를 보고 오른쪽 단어를 예측)하게 단어를 생성하기 때문에 타깃 단어 이후 단어를 보지 않고 단어를 예측해야 함
- 따라서 타깃 단어 뒤에 위치한 단어는 Self-Attention에 영향을 주지 않도록 **마스킹(masking)** 해주어야 함


**_Self-Attention (without Masking) vs Masked Self-Attention_**

![9](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/a6e6eb2b-21ec-4f1c-8be9-c5fde8470938/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221216%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221216T131200Z&X-Amz-Expires=86400&X-Amz-Signature=97ac57bf111bc346d02b4cfab229666b8419aabebdfc9c067c2e9253dd71e1a4&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)

- Self-Attention 메커니즘은 쿼리 행렬(Q)와 키 행렬(K)의 내적
- 결과로 나온 행렬을 차원의 $\sqrt{d_k}$로 나누어 준 다음, Softmax를 취해주고 밸류 행렬(V)과 내적

![10](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/0fcb246d-292d-4b33-a0e7-e5ab2ce79498/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221216%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221216T131416Z&X-Amz-Expires=86400&X-Amz-Signature=0217303f76ee5b835267a0816d42385a97247231f96aab7d98b29f8615f7f9fb&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)
![11](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/7c8c1ce2-3803-4970-9166-36f9470e127c/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221216%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221216T131435Z&X-Amz-Expires=86400&X-Amz-Signature=6bc43c9bd2133ac4806bb7b76d3598d1f725458eb2dd264ec299a859f18d685a&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)
- **마스킹(Masking)**: **Masked Self-Attention** 에서는 Softmax를 취해주기 전, 가려주고자 하는 요소에만 −∞ 에 해당하는 매우 작은 수를 더해줌
- 마스킹된 값은 Softmax를 취해 주었을 때 0이 나오므로 Value 계산에 반영되지 않음

## Encoder-Decoder Attention

![12](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/2bb59e27-831b-48e9-9b63-86bbe6beeeed/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221216%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221216T131454Z&X-Amz-Expires=86400&X-Amz-Signature=3f418f75511053b6e41cb85c435de5148c3686a4f03a8a1f8a1f8ce2586dd67f&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)

- 디코더에서 Masked Self-Attention 층을 지난 벡터는 **Encoder-Decoder Attention** 층으로 들어감
- 좋은 번역을 위해서는 **번역할 문장과 번역된 문장 간의 관계** 역시 중요
- **번역할 문장과 번역되는 문장의 정보 관계**를 **엮어주는 부분**이 바로 이 부분
- 이 층에서는 **디코더 블록의** Masked Self-Attention으로부터 출력된 벡터를 **쿼리(Q)**벡터로 사용
- **키(K)와 밸류(V)** 벡터는 최상위(=6번째) 인코더 블록에서 사용했던 값을 그대로 가져와서 사용

## Linear & Softmax Layer

![13](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/91515418-ff68-4658-aaba-cf6f62a79139/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221216%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221216T131552Z&X-Amz-Expires=86400&X-Amz-Signature=aed15acc3e2eddc056d359f52d0f27bc5cf468fceb1881e514d526b016a761cc&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)

디코더의 최상층을 통과한 벡터들은 Linear 층을 지난 후 Softmax를 통해 예측할 단어의 확률을 구하게 됨
-   Linear : 소프트맥스에 입력값으로 들어갈 logit 생성
-   Sofrmax : 모델이 알고 있는 모든 단어들에 대한
이후 Label Smoothing을 통해 모델 BLEU accuracy 향상
-   Label Smoothing이란 0과 1에 가까운 값으로 변화해주는 기술. 모델이 학습 데이터에 치중하여 학습하지 못하는 것을 막아줌
-   Label이 Noisy한 경우, 즉 같은 입력값인데 다른 출력값들이 학습데이터에 많을 경우 Label Smoothing이 도움이 됨

[Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)